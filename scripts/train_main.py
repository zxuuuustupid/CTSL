"""
MCID 元学习训练
- 教师模型看源域干净数据
- 学生模型从多个目标工况学习跨工况不变特征
- Meta-Train: N-1 个工况, Meta-Test: 1 个工况
"""
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
    
    ckpt_path = os.path.join(config['teacher']['checkpoint'], config['data']['dataset_name'] , 'best_model.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        print(f"教师模型已加载: {ckpt_path}")
    else:
        print(f"警告: 教师模型不存在 {ckpt_path}")
    
    encoder.eval()
    decoder.eval()
    for p in encoder.parameters(): p.requires_grad = False
    for p in decoder.parameters(): p.requires_grad = False
    
    return encoder, decoder


def build_student(config, device):
    """构建学生模型"""
    cfg = config['model']
    encoder = MechanicEncoder(cfg['input_channels'], cfg['base_filters'], cfg['feature_dim']).to(device)
    classifier = MechanicClassifier(cfg['feature_dim'], cfg['num_classes'], cfg['dropout']).to(device)
    return encoder, classifier


def load_all_wc_data(config):
    """加载所有目标工况数据"""
    data_cfg = config['data']
    batch_size = config['training']['batch_size']
    
    wc_loaders = {}
    for wc in data_cfg['target_wcs']:
        path = os.path.join(data_cfg['root_dir'], wc, 'train')
        wc_loaders[wc] = get_dataloader(path, batch_size, shuffle=True)
    
    return wc_loaders


def compute_loss(feat_s, feat_t, logits, labels, x_clean, decoder_t, encoder_t, config):
    """计算 MCID 损失: L_AC + L_CC + L_LC"""
    loss_cfg = config['loss']
    
    # L_AC: 对抗一致性 (特征蒸馏)
    l_ac = nn.MSELoss()(feat_s, feat_t)
    
    # L_CC: 循环一致性 (学生特征 -> 教师解码 -> 教师编码)
    x_recon = decoder_t(feat_s, target_length=x_clean.shape[-1])
    feat_cycle = encoder_t(x_recon)
    l_cc = nn.MSELoss()(feat_cycle, feat_s)
    
    # L_LC: 标签一致性
    l_lc = nn.CrossEntropyLoss()(logits, labels)
    
    total = loss_cfg['lambda_ac'] * l_ac + loss_cfg['lambda_cc'] * l_cc + loss_cfg['lambda_lc'] * l_lc
    return total, {'ac': l_ac.item(), 'cc': l_cc.item(), 'lc': l_lc.item()}


def inner_update(encoder, loss, inner_lr, first_order=True):
    """内循环更新，返回临时模型"""
    encoder_prime = deepcopy(encoder)
    grads = grad(loss, 
                 encoder.parameters(), 
                 create_graph=not first_order, 
                 retain_graph=True,
                 allow_unused=True)
    
    for p, g in zip(encoder_prime.parameters(), grads):
        if g is not None:
            p.data = p.data - inner_lr * g
    
    return encoder_prime


def meta_train_step(x_source, y_source, wc_loaders, wc_iters, 
                    encoder_s, classifier, encoder_t, decoder_t, config, device):
    """
    一步元训练:
    1. 随机分割工况为 train_wcs (N-1) 和 test_wc (1)
    2. Meta-Train: 用 train_wcs 计算损失，得到临时模型
    3. Meta-Test: 用 test_wc 评估临时模型
    4. 返回总损失
    """
    meta_cfg = config['meta']
    wc_list = list(wc_loaders.keys())
    
    # 随机分割工况
    random.shuffle(wc_list)
    train_wcs = wc_list[:-1]
    test_wc = wc_list[-1]
    
    # 教师特征 (看源域干净数据)
    with torch.no_grad():
        feat_t = encoder_t(x_source).detach()
    
    # ========== Meta-Train ==========
    # 从训练工况中采样一个batch
    train_wc = random.choice(train_wcs)
    try:
        x_train, y_train = next(wc_iters[train_wc])
    except StopIteration:
        wc_iters[train_wc] = iter(wc_loaders[train_wc])
        x_train, y_train = next(wc_iters[train_wc])
    x_train, y_train = x_train.to(device), y_train.to(device)
    
    feat_s = encoder_s(x_train)
    logits = classifier(feat_s)
    
    # 用源域标签对齐（假设不同工况下标签一致）
    l_mt, _ = compute_loss(feat_s, feat_t, logits, y_source, x_source, decoder_t, encoder_t, config)
    
    # 内循环更新
    encoder_prime = inner_update(encoder_s, l_mt, meta_cfg['inner_lr'], meta_cfg['first_order'])
    
    # ========== Meta-Test ==========
    try:
        x_test, y_test = next(wc_iters[test_wc])
    except StopIteration:
        wc_iters[test_wc] = iter(wc_loaders[test_wc])
        x_test, y_test = next(wc_iters[test_wc])
    x_test, y_test = x_test.to(device), y_test.to(device)
    
    feat_s_prime = encoder_prime(x_test)
    logits_prime = classifier(feat_s_prime)
    
    l_test, _ = compute_loss(feat_s_prime, feat_t, logits_prime, y_source, x_source, decoder_t, encoder_t, config)
    
    # 总损失
    l_total = l_mt + meta_cfg['beta'] * l_test
    
    return l_total, l_mt.item(), l_test.item()


def evaluate(encoder, classifier, config, device):
    """在所有工况上评估"""
    data_cfg = config['data']
    batch_size = config['training']['batch_size']
    
    results = {}
    total_acc = 0
    
    for wc in data_cfg['test_wcs']:
        path = os.path.join(data_cfg['root_dir'], wc, 'test')
        loader = get_dataloader(path, batch_size, shuffle=False)
        
        correct, total = 0, 0
        encoder.eval()
        classifier.eval()
        
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
        print(f"  {wc}: {acc:.2f}%")
    
    avg_acc = total_acc / len(data_cfg['test_wcs'])
    print(f"  平均: {avg_acc:.2f}%")
    return avg_acc


def main(config_path):
    config = load_config(config_path)
    set_seed(config['seed'])
    
    device = torch.device(f"cuda:{config['device']['gpu_id']}" 
                          if config['device']['use_cuda'] and torch.cuda.is_available() 
                          else "cpu")
    print(f"设备: {device}")
    
    os.makedirs(config['output']['save_dir'], exist_ok=True)
    
    # 加载模型
    encoder_t, decoder_t = load_teacher(config, device)
    encoder_s, classifier = build_student(config, device)
    
    # 加载数据
    source_path = os.path.join(config['data']['root_dir'], config['data']['source_wc'], 'train')
    source_loader = get_dataloader(source_path, config['training']['batch_size'], shuffle=True)
    wc_loaders = load_all_wc_data(config)
    wc_iters = {wc: iter(loader) for wc, loader in wc_loaders.items()}
    
    # 优化器
    params = list(encoder_s.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    
    # 训练
    best_acc = 0
    for epoch in range(1, config['training']['epochs'] + 1):
        encoder_s.train()
        classifier.train()
        
        total_loss, total_mt, total_test = 0, 0, 0
        
        for x_src, y_src in source_loader:
            x_src, y_src = x_src.to(device), y_src.to(device)
            
            optimizer.zero_grad()
            loss, l_mt, l_test = meta_train_step(
                x_src, y_src, wc_loaders, wc_iters,
                encoder_s, classifier, encoder_t, decoder_t, config, device
            )
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mt += l_mt
            total_test += l_test
        
        n = len(source_loader)
        print(f"Epoch {epoch}: Loss={total_loss/n:.4f}, MT={total_mt/n:.4f}, Test={total_test/n:.4f}")
        
        # 评估
        if epoch % 5 == 0:
            print("评估:")
            avg_acc = evaluate(encoder_s, classifier, config, device)
            
            if avg_acc > best_acc:
                best_acc = avg_acc
                torch.save({
                    'encoder_state_dict': encoder_s.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                }, os.path.join(config['output']['save_dir'],config['data']['dataset_name'] + '_best_model.pth'))
                print(f"保存最佳模型: {best_acc:.2f}%")
    
    print(f"\n训练完成，最佳准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mcid.yaml")
    args = parser.parse_args()
    main(args.config)
