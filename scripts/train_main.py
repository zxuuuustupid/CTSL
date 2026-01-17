import os
import sys
import yaml
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from copy import deepcopy
from itertools import cycle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataloader import get_dataloader
from src.models.encoder import MechanicEncoder
from src.models.decoder import MechanicDecoder
from src.models.classifier import MechanicClassifier


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_teacher(config, device):
    """加载冻结的教师模型"""
    cfg = config['model']
    encoder = MechanicEncoder(cfg['input_channels'], cfg['base_filters'], cfg['feature_dim']).to(device)
    decoder = MechanicDecoder(cfg['feature_dim'], cfg['input_channels'], cfg['base_filters']).to(device)
    classifier = MechanicClassifier(cfg['feature_dim'], cfg['num_classes'], cfg['dropout']).to(device)
    
    # 确保加载具体的pth文件
    ckpt_path = os.path.join(config['teacher']['checkpoint'], config['data']['dataset_name'], 'best_model.pth')
    if os.path.exists(ckpt_path) and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        classifier.load_state_dict(ckpt['classifier_state_dict'])
        decoder.load_state_dict(ckpt['decoder_state_dict'])
        print(f"教师模型已加载: {ckpt_path}")
    else:
        raise FileNotFoundError(f"教师模型文件未找到: {ckpt_path}")
    
    encoder.eval()
    classifier.eval()   
    decoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
    for p in decoder.parameters(): p.requires_grad = False
    
    return encoder, classifier, decoder


def build_student(config, device):
    """构建学生模型"""
    cfg = config['model']
    encoder = MechanicEncoder(cfg['input_channels'], cfg['base_filters'], cfg['feature_dim']).to(device)
    # classifier = MechanicClassifier(cfg['feature_dim'], cfg['num_classes'], cfg['dropout']).to(device)
    # return encoder, classifier
    return encoder


def get_infinite_loader(loader):
    """将DataLoader转换为无限循环生成器"""
    while True:
        for batch in loader:
            yield batch


def load_all_wc_data(config):
    """加载所有目标工况数据，返回无限迭代器"""
    data_cfg = config['data']
    batch_size = config['training']['batch_size']
    
    wc_iters = {}
    for wc in data_cfg['target_wcs']:
        path = os.path.join(data_cfg['root_dir'], wc, 'train')
        loader = get_dataloader(path, batch_size, shuffle=True)
        wc_iters[wc] = get_infinite_loader(loader)
    
    return wc_iters


def compute_loss(encoder_s, classifier, encoder_t, decoder_t, x, labels, config):
    """计算通用损失函数 (L_AC + L_CC + L_LC)"""
    loss_cfg = config['loss']
    
    # 1. 学生前向传播
    feat_s = encoder_s(x)
    logits = classifier(feat_s)
    
    # 2. 教师前向传播 (作为Ground Truth，不传梯度)
    with torch.no_grad():
        feat_t = encoder_t(x).detach()
    
    # L_AC: 对抗/特征一致性
    l_ac = nn.MSELoss()(feat_s, feat_t)
    
    # L_CC: 循环一致性 (特征 -> 教师解码 -> 教师编码)
    x_recon = decoder_t(feat_s, target_length=x.shape[-1])
    with torch.no_grad():
        feat_cycle = encoder_t(x_recon).detach()
    l_cc = nn.MSELoss()(feat_cycle, feat_s)
    
    # L_LC: 标签一致性
    l_lc = nn.CrossEntropyLoss()(logits, labels)
    
    total = loss_cfg['lambda_ac'] * l_ac + loss_cfg['lambda_cc'] * l_cc + loss_cfg['lambda_lc'] * l_lc
    return total, {'ac': l_ac.item(), 'cc': l_cc.item(), 'lc': l_lc.item()}


def inner_update(encoder, loss, inner_lr, first_order=True):
    """内循环更新: θ' = θ - α * ∇L_support"""
    encoder_prime = deepcopy(encoder)
    
    # 计算梯度 (retain_graph=True 确保不释放计算图，以便后续计算Meta Loss)
    grads = grad(loss, 
                 encoder.parameters(), 
                 create_graph=not first_order, 
                 retain_graph=True,
                 allow_unused=True)
    
    # 更新临时参数
    for p, g in zip(encoder_prime.parameters(), grads):
        if g is not None:
            p.data = p.data - inner_lr * g
    
    return encoder_prime


def meta_train_step(wc_iters, encoder_s, classifier, encoder_t, decoder_t, config, device):
    """
    标准的元学习步骤 (DG-Meta / MLDG 风格):
    1. Task Sampling: 将工况划分为 Meta-Train (Support) 和 Meta-Test (Query)
    2. Inner Loop: 在 Support Set 上计算损失并获得临时参数 θ'
    3. Outer Loop: 在 Query Set 上使用 θ' 计算损失，并结合 Support Loss 进行最终更新
    """
    meta_cfg = config['meta']
    wc_list = list(wc_iters.keys())
    
    # 随机划分工况任务
    random.shuffle(wc_list)
    # N-1个工况用于内循环更新 (模拟已知工况)
    support_wcs = wc_list[:-1]
    # 1个工况用于外循环测试 (模拟未知工况)
    query_wc = wc_list[-1]
    
    # ========== 1. Meta-Train / Support Set 阶段 ==========
    # 从支持集工况中随机抽取一个 Batch
    train_wc = random.choice(support_wcs)
    x_sup, y_sup = next(wc_iters[train_wc])
    x_sup, y_sup = x_sup.to(device), y_sup.to(device)
    
    # 计算 Support Loss
    l_sup, _ = compute_loss(encoder_s, classifier, encoder_t, decoder_t, x_sup, y_sup, config)
    
    # 获取临时参数 θ' (Fast Weights)
    encoder_prime = inner_update(encoder_s, l_sup, meta_cfg['inner_lr'], meta_cfg['first_order'])
    
    # ========== 2. Meta-Test / Query Set 阶段 ==========
    # 从查询集工况中抽取一个 Batch
    x_qry, y_qry = next(wc_iters[query_wc])
    x_qry, y_qry = x_qry.to(device), y_qry.to(device)
    
    # 使用临时参数 θ' 计算 Query Loss
    # 注意：这里使用 encoder_prime (θ')，但分类器 classifier 共享 (或视具体算法而定)
    l_qry, metrics = compute_loss(encoder_prime, classifier, encoder_t, decoder_t, x_qry, y_qry, config)
    
    # ========== 3. 最终 Meta Loss ==========
    # MLDG 常用: Total Loss = L_support + beta * L_query
    l_total = l_sup + meta_cfg['beta'] * l_qry
    
    return l_total, l_sup.item(), l_qry.item()


def evaluate(encoder, classifier, config, device):
    """在测试集上评估泛化能力"""
    data_cfg = config['data']
    batch_size = config['training']['batch_size']
    
    results = {}
    total_acc = 0
    
    encoder.eval()
    classifier.eval()
    
    for wc in data_cfg['test_wcs']:
        path = os.path.join(data_cfg['root_dir'], wc, 'test')
        loader = get_dataloader(path, batch_size, shuffle=False)
        
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                feat = encoder(x)
                pred = classifier(feat).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        acc = 100. * correct / total
        results[wc] = acc
        total_acc += acc
        print(f"  工况 {wc}: {acc:.2f}%")
    
    avg_acc = total_acc / len(data_cfg['test_wcs'])
    print(f"  平均准确率: {avg_acc:.2f}%")
    return avg_acc


def main(config_path):
    config = load_config(config_path)
    set_seed(config['seed'])
    
    device = torch.device(f"cuda:{config['device']['gpu_id']}" 
                          if config['device']['use_cuda'] and torch.cuda.is_available() 
                          else "cpu")
    print(f"训练设备: {device}")
    
    os.makedirs(config['output']['save_dir'], exist_ok=True)
    
    # 模型初始化
    encoder_t, classifier_t, decoder_t = load_teacher(config, device)
    # encoder_s, classifier = build_student(config, device)

    encoder_s = MechanicEncoder(
        input_channels=config['model']['input_channels'],
        base_filters=config['model']['base_filters'],
        output_feature_dim=config['model']['feature_dim']
    ).to(device)
# 分类器直接用 teacher 的，或者把 teacher 的分类器提取出来
    classifier = classifier_t
    classifier.eval() # 必须 Eval 模式
    for p in classifier.parameters():
        p.requires_grad = False # 必须冻结

    
    # 数据加载 (使用无限迭代器，摒弃 source_loader)
    wc_iters = load_all_wc_data(config)
    
    # 优化器
    # params = list(encoder_s.parameters()) + list(classifier.parameters())
    params = encoder_s.parameters()  # 只优化编码器参数
    optimizer = optim.Adam(params, lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    
    # 训练参数
    iterations_per_epoch = config['training'].get('iterations_per_epoch', 100) # 每 Epoch 迭代次数
    best_acc = 0
    
    print("开始 MCID 元学习训练...")
    
    for epoch in range(1, config['training']['epochs'] + 1):
        encoder_s.train()
        # classifier.train()
        
        total_loss, total_sup, total_qry = 0, 0, 0
        
        for _ in range(iterations_per_epoch):
            optimizer.zero_grad()
            
            # 执行一步元训练
            loss, l_sup, l_qry = meta_train_step(
                wc_iters,
                encoder_s, classifier, encoder_t, decoder_t, config, device
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_sup += l_sup
            total_qry += l_qry
        
        # 日志
        print(f"Epoch {epoch}: Loss={total_loss/iterations_per_epoch:.4f}, "
              f"Support={total_sup/iterations_per_epoch:.4f}, "
              f"Query={total_qry/iterations_per_epoch:.4f}")
        
        # 评估与保存
        if epoch % 5 == 0:
            print(f"Epoch {epoch} 评估:")
            avg_acc = evaluate(encoder_s, classifier, config, device)
            
            if avg_acc > best_acc:
                best_acc = avg_acc
                save_path = os.path.join(config['output']['save_dir'], 
                                       f"{config['data']['dataset_name']}_mcid_best.pth")
                torch.save({
                    'encoder_state_dict': encoder_s.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'best_acc': best_acc
                }, save_path)
                print(f"保存最佳模型: {save_path} (Acc: {best_acc:.2f}%)")
    
    print(f"\n训练结束，最佳平均准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mcid.yaml")
    args = parser.parse_args()
    main(args.config)