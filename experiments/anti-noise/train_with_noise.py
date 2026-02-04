"""
带噪声训练的学生网络测试脚本
在训练过程中对数据添加不同强度的高斯白噪声，评估模型的抗噪性能
所有输出保存在 experiments/anti-noise 目录下
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
from tqdm import tqdm
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch.utils.data import ConcatDataset, DataLoader
from src.data.dataloader import NpyDataset, get_dataloader
from src.models.encoder import MechanicEncoder
from src.models.decoder import MechanicDecoder
from src.models.classifier import MechanicClassifier

# 输出目录
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 训练噪声等级
# TRAIN_SNR_LEVELS = [None, 50,20,15,10, 8, 5]  # None表示无噪声
TRAIN_SNR_LEVELS = [None, 10,9,8,7,6, 5]  # None表示无噪声

def add_gaussian_noise(signal, snr_db):
    """添加高斯白噪声"""
    if snr_db is None:
        return signal
    signal_power = torch.mean(signal ** 2, dim=-1, keepdim=True)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(signal) * torch.sqrt(noise_power)
    return signal + noise


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def load_data_split(config):
    """加载数据"""
    data_cfg = config['data']
    batch_size = config['training']['batch_size']

    source_wcs = data_cfg['source_wc']
    if isinstance(source_wcs, str):
        source_wcs = [source_wcs]

    source_datasets = []
    for wc in source_wcs:
        path = os.path.join(data_cfg['root_dir'], wc, 'train')
        if not os.path.exists(path):
            raise FileNotFoundError(f"源工况路径不存在: {path}")
        source_datasets.append(NpyDataset(path))

    combined_source = ConcatDataset(source_datasets)
    source_loader = DataLoader(combined_source, batch_size=batch_size, shuffle=True,
                                pin_memory=True, num_workers=1)
    source_iter = get_infinite_loader(source_loader)

    target_iters = {}
    target_wcs = data_cfg['target_wcs']

    for wc in target_wcs:
        path = os.path.join(data_cfg['root_dir'], wc, 'train')
        if not os.path.exists(path):
            print(f"警告: 目标工况路径不存在，跳过: {path}")
            continue
        dataset = NpyDataset(path)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           pin_memory=True, num_workers=1)
        target_iters[wc] = get_infinite_loader(loader)

    return source_iter, target_iters


def load_teacher(config, device):
    """加载冻结的教师模型"""
    cfg = config['model']
    encoder = MechanicEncoder(cfg['input_channels'], cfg['base_filters'], cfg['feature_dim']).to(device)
    decoder = MechanicDecoder(cfg['feature_dim'], cfg['input_channels'], cfg['base_filters']).to(device)
    classifier = MechanicClassifier(cfg['feature_dim'], cfg['num_classes'], cfg['dropout']).to(device)

    source_list = config['data']['source_wc']
    nums = sorted(["".join(filter(str.isdigit, wc)) for wc in source_list], key=int)
    wc_tag = "_".join(nums)
    filename = f"train_{wc_tag}_best_model.pth"
    ckpt_path = os.path.join(config['teacher']['checkpoint'], config['data']['dataset_name'], filename)

    if os.path.exists(ckpt_path) and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        classifier.load_state_dict(ckpt['classifier_state_dict'])
        decoder.load_state_dict(ckpt['decoder_state_dict'])
    else:
        raise FileNotFoundError(f"教师模型文件未找到: {ckpt_path}")

    encoder.eval()
    classifier.eval()
    decoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False

    return encoder, classifier, decoder


def compute_loss(encoder_s, classifier, encoder_t, decoder_t, x_s, y_s, x_t, y_t, config):
    """计算损失函数"""
    loss_cfg = config['loss']

    feat_s = encoder_s(x_s)
    logits = classifier(feat_s)

    with torch.no_grad():
        feat_t = encoder_t(x_t).detach()

    # L_AC
    l_ac = torch.tensor(0.0, device=x_s.device)
    valid_classes = 0
    classes_s = torch.unique(y_s)
    classes_t = torch.unique(y_t)
    common_classes = [c for c in classes_s if c in classes_t]

    for c in common_classes:
        proto_s = feat_s[y_s == c].mean(dim=0)
        proto_t = feat_t[y_t == c].mean(dim=0)
        l_ac += nn.MSELoss()(proto_s, proto_t)
        valid_classes += 1

    if valid_classes > 0:
        l_ac = l_ac / valid_classes

    # L_CC
    x_recon = decoder_t(feat_s, target_length=x_s.shape[-1])
    with torch.no_grad():
        feat_cycle = encoder_t(x_recon).detach()

    if feat_cycle.shape[0] == feat_t.shape[0]:
        l_cc = nn.MSELoss()(feat_cycle, feat_t)
    else:
        l_cc = torch.tensor(0.0, device=feat_cycle.device)

    # L_LC
    l_lc = nn.CrossEntropyLoss()(logits, y_s)

    total = loss_cfg['lambda_ac'] * l_ac + loss_cfg['lambda_cc'] * l_cc + loss_cfg['lambda_lc'] * l_lc
    return total, {'ac': l_ac.item(), 'cc': l_cc.item(), 'lc': l_lc.item()}


def inner_update(encoder, loss, inner_lr, first_order=True):
    """内循环更新"""
    encoder_prime = deepcopy(encoder)
    grads = grad(loss, encoder.parameters(), create_graph=not first_order,
                 retain_graph=True, allow_unused=True)

    for p, g in zip(encoder_prime.parameters(), grads):
        if g is not None:
            p.data = p.data - inner_lr * g

    return encoder_prime


def meta_train_step_with_noise(source_iter, target_iters, encoder_s, classifier,
                                encoder_t, decoder_t, config, device, train_snr):
    """带噪声的元训练步骤（只对随机一个工况加噪声）"""
    meta_cfg = config['meta']
    wc_list = list(target_iters.keys())
    random.shuffle(wc_list)

    query_wc = wc_list[-1]
    support_wcs = wc_list[:-1]

    # 随机选择一个工况加噪声（可以是 support 或 query）
    all_wcs = support_wcs + [query_wc]
    noise_wc = random.choice(all_wcs)

    total_sup_loss = 0

    for wc in support_wcs:
        x_s, y_s = next(target_iters[wc])
        x_s, y_s = x_s.to(device), y_s.to(device)
        # 只对选中的工况添加噪声
        if wc == noise_wc:
            x_s = add_gaussian_noise(x_s, train_snr)

        x_t, y_t = next(source_iter)
        x_t, y_t = x_t.to(device), y_t.to(device)

        l_step, _ = compute_loss(encoder_s, classifier, encoder_t, decoder_t,
                                  x_s, y_s, x_t, y_t, config)
        total_sup_loss += l_step

    if len(support_wcs) > 0:
        total_sup_loss = total_sup_loss / len(support_wcs)
    else:
        total_sup_loss = torch.tensor(0.0, device=device)

    if isinstance(total_sup_loss, torch.Tensor) and total_sup_loss.item() != 0:
        encoder_prime = inner_update(encoder_s, total_sup_loss, meta_cfg['inner_lr'], meta_cfg['first_order'])
    else:
        encoder_prime = encoder_s

    x_q_s, y_q_s = next(target_iters[query_wc])
    x_q_s, y_q_s = x_q_s.to(device), y_q_s.to(device)
    # 只对选中的工况添加噪声
    if query_wc == noise_wc:
        x_q_s = add_gaussian_noise(x_q_s, train_snr)

    x_q_t, y_q_t = next(source_iter)
    x_q_t, y_q_t = x_q_t.to(device), y_q_t.to(device)

    l_qry, _ = compute_loss(encoder_prime, classifier, encoder_t, decoder_t,
                            x_q_s, y_q_s, x_q_t, y_q_t, config)

    l_total = total_sup_loss + meta_cfg['beta'] * l_qry

    return l_total, total_sup_loss.item() if isinstance(total_sup_loss, torch.Tensor) else 0, l_qry.item()


def evaluate(encoder, classifier, config, device):
    """评估（无噪声）"""
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

    avg_acc = total_acc / len(data_cfg['test_wcs'])
    return results, avg_acc


def train_with_noise_level(config, device, train_snr):
    """使用指定噪声等级训练"""
    snr_label = "clean" if train_snr is None else f"{train_snr}dB"
    print(f"\n{'='*60}")
    print(f"开始训练 - 训练噪声: {snr_label}")
    print(f"{'='*60}")

    # 重新设置随机种子保证公平比较
    set_seed(config['seed'])

    # 加载教师模型
    encoder_t, classifier_t, decoder_t = load_teacher(config, device)

    # 构建学生模型
    encoder_s = MechanicEncoder(
        input_channels=config['model']['input_channels'],
        base_filters=config['model']['base_filters'],
        output_feature_dim=config['model']['feature_dim']
    ).to(device)
    encoder_s.load_state_dict(encoder_t.state_dict())

    classifier = classifier_t
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    # 加载数据
    source_iter, target_iters = load_data_split(config)

    # 优化器
    optimizer = optim.Adam(encoder_s.parameters(), lr=config['training']['lr'],
                           weight_decay=config['training']['weight_decay'])

    iterations_per_epoch = config['training'].get('iterations_per_epoch', 100)
    best_acc = 0
    best_results = {}

    # 减少训练轮数用于快速测试
    epochs = min(config['training']['epochs'], 10)

    for epoch in range(1, epochs + 1):
        encoder_s.train()
        total_loss = 0

        pbar = tqdm(range(iterations_per_epoch), desc=f"Epoch {epoch}/{epochs}", leave=False)
        for _ in pbar:
            optimizer.zero_grad()
            loss, _, _ = meta_train_step_with_noise(
                source_iter, target_iters, encoder_s, classifier,
                encoder_t, decoder_t, config, device, train_snr
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 每轮评估
        results, avg_acc = evaluate(encoder_s, classifier, config, device)
        print(f"  Epoch {epoch}: Loss={total_loss/iterations_per_epoch:.4f}, Acc={avg_acc:.2f}%")

        if avg_acc > best_acc:
            best_acc = avg_acc
            best_results = results

    return best_acc, best_results


def main(config_path):
    config = load_config(config_path)

    device = torch.device(f"cuda:{config['device']['gpu_id']}"
                          if config['device']['use_cuda'] and torch.cuda.is_available()
                          else "cpu")
    print(f"测试设备: {device}")

    dataset_name = config['data']['dataset_name']

    # 存储所有噪声等级的结果
    all_results = {}

    for train_snr in tqdm(TRAIN_SNR_LEVELS, desc="训练噪声等级"):
        snr_label = "clean" if train_snr is None else f"{train_snr}dB"
        best_acc, per_wc_results = train_with_noise_level(config, device, train_snr)

        all_results[snr_label] = {
            'best_acc': best_acc,
            'per_wc': per_wc_results
        }

    # === 保存结果 ===
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 保存 YAML（带时间戳）
    yaml_path = os.path.join(OUTPUT_DIR, f"train_noise_{dataset_name}_{timestamp}.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            'dataset': dataset_name,
            'train_snr_levels': [str(s) if s is not None else 'clean' for s in TRAIN_SNR_LEVELS],
            'results': all_results
        }, f, allow_unicode=True)
    print(f"\nYAML 已保存至: {yaml_path}")

    # 2. 保存 CSV（可覆盖）
    csv_path = os.path.join(OUTPUT_DIR, f"train_noise_{dataset_name}.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        test_wcs = config['data']['test_wcs']
        header = "Train_SNR," + ",".join(test_wcs) + ",Avg\n"
        f.write(header)
        for snr_label, data in all_results.items():
            row = [snr_label]
            for wc in test_wcs:
                row.append(f"{data['per_wc'].get(wc, 0):.2f}")
            row.append(f"{data['best_acc']:.2f}")
            f.write(",".join(row) + "\n")
    print(f"CSV 已保存至: {csv_path}")

    # 打印汇总表格
    print("\n" + "=" * 60)
    print("训练噪声实验结果汇总")
    print("=" * 60)
    print(f"{'Train SNR':<12} | {'Best Acc':<10}")
    print("-" * 25)
    for snr_label, data in all_results.items():
        print(f"{snr_label:<12} | {data['best_acc']:.2f}%")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="带噪声训练测试脚本")
    parser.add_argument("--config", default="configs/mcid_PU_train_1_meta_2_4.yaml",
                        help="配置文件路径")
    args = parser.parse_args()

    main(args.config)
