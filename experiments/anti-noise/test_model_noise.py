"""
Anti-Noise 测试脚本
对测试样本添加不同强度的高斯白噪声进行抗噪性评估
"""
import os
import sys
import yaml
import argparse
import random
import numpy as np
import torch
import datetime
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from torch.utils.data import DataLoader
from src.data.dataloader import NpyDataset
from src.models.encoder import MechanicEncoder
from src.models.classifier import MechanicClassifier

# 噪声等级 (SNR dB)：无噪声, 10dB, 5dB, 0dB, -5dB, -10dB
SNR_LEVELS = [None, 10,9,8,7,6, 5]


def add_gaussian_noise(signal, snr_db):
    """
    添加高斯白噪声
    Args:
        signal: 输入信号 [B, C, L]
        snr_db: 信噪比 (dB)
    Returns:
        加噪后的信号
    """
    if snr_db is None:
        return signal

    # 计算信号功率
    signal_power = torch.mean(signal ** 2, dim=-1, keepdim=True)
    # SNR = 10 * log10(P_signal / P_noise) => P_noise = P_signal / 10^(SNR/10)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    # 生成噪声
    noise = torch.randn_like(signal) * torch.sqrt(noise_power)
    return signal + noise


def get_test_dataloader(path, batch_size=32):
    """专为测试优化的 DataLoader"""
    dataset = NpyDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,  # Windows 兼容
        pin_memory=True
    )

# 输出目录
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(config, device, ckpt_path=None):
    """加载训练好的模型"""
    cfg = config['model']
    encoder = MechanicEncoder(cfg['input_channels'], cfg['base_filters'], cfg['feature_dim']).to(device)
    classifier = MechanicClassifier(cfg['feature_dim'], cfg['num_classes'], cfg['dropout']).to(device)

    if ckpt_path is None:
        # 默认加载 MCID 模型
        src_list = config['data']['source_wc']
        src_nums = sorted(["".join(filter(str.isdigit, x)) for x in src_list], key=int)
        src_tag = "_".join(src_nums)

        tgt_list = config['data']['target_wcs']
        tgt_nums = sorted(["".join(filter(str.isdigit, x)) for x in tgt_list], key=int)
        tgt_tag = "_".join(tgt_nums)

        file_name = f"mcid_train_{src_tag}_meta_{tgt_tag}_best.pth"
        ckpt_path = os.path.join(config['output']['save_dir'], config['data']['dataset_name'], file_name)

    print(f"正在加载模型: {ckpt_path}")

    if os.path.exists(ckpt_path) and os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        classifier.load_state_dict(ckpt['classifier_state_dict'])
        print(f"模型已加载: {ckpt_path}")
        if 'best_acc' in ckpt:
            print(f"训练时最佳准确率: {ckpt['best_acc']:.2f}%")
    else:
        raise FileNotFoundError(f"模型文件未找到: {ckpt_path}")

    encoder.eval()
    classifier.eval()
    return encoder, classifier


def evaluate_with_noise(encoder, classifier, config, device, test_wcs=None, snr_db=None):
    """在测试集上评估模型（支持添加噪声）"""
    data_cfg = config['data']
    batch_size = config['training']['batch_size']

    if test_wcs is None:
        test_wcs = data_cfg['test_wcs']

    results = {}
    total_acc = 0

    encoder.eval()
    classifier.eval()

    snr_label = "无噪声" if snr_db is None else f"SNR={snr_db}dB"

    for wc in test_wcs:
        path = os.path.join(data_cfg['root_dir'], wc, 'test')
        if not os.path.exists(path):
            print(f"警告: 测试路径不存在，跳过: {path}")
            continue

        loader = get_test_dataloader(path, batch_size)
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                # 添加噪声
                x_noisy = add_gaussian_noise(x, snr_db)
                feat = encoder(x_noisy)
                pred = classifier(feat).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = 100. * correct / total
        results[wc] = acc
        total_acc += acc

    avg_acc = total_acc / len(test_wcs) if len(test_wcs) > 0 else 0
    return results, avg_acc


def save_results(all_results, config):
    """保存所有噪声等级测试结果到单个文件"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = config['data']['dataset_name']
    save_name = f"anti_noise_{dataset_name}_{timestamp}.yaml"
    save_path = os.path.join(OUTPUT_DIR, save_name)

    # 构建结果数据
    result_data = {
        'dataset': dataset_name,
        'test_wcs': config['data']['test_wcs'],
        'timestamp': timestamp,
        'snr_levels': [str(s) if s is not None else 'clean' for s in SNR_LEVELS],
        'results': all_results,
        # 生成汇总表格
        'summary': {snr: data['avg_acc'] for snr, data in all_results.items()}
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(result_data, f, allow_unicode=True, default_flow_style=False)

    # === 保存 CSV (可覆盖，无时间戳) ===
    csv_path = os.path.join(OUTPUT_DIR, f"anti_noise_{dataset_name}.csv")
    with open(csv_path, 'w', encoding='utf-8') as f:
        # 表头：SNR, 各工况, 平均
        test_wcs = config['data']['test_wcs']
        header = "SNR," + ",".join(test_wcs) + ",Avg\n"
        f.write(header)
        # 数据行
        for snr_label, data in all_results.items():
            row = [snr_label]
            for wc in test_wcs:
                row.append(f"{data['per_wc'].get(wc, 0):.2f}")
            row.append(f"{data['avg_acc']:.2f}")
            f.write(",".join(row) + "\n")
    print(f"CSV 已保存至: {csv_path}")

    return save_path


def main(config_path, ckpt_path=None, test_wcs=None):
    config = load_config(config_path)
    set_seed(config['seed'])

    device = torch.device(f"cuda:{config['device']['gpu_id']}"
                          if config['device']['use_cuda'] and torch.cuda.is_available()
                          else "cpu")
    print(f"测试设备: {device}")

    # 加载模型
    encoder, classifier = load_model(config, device, ckpt_path)

    # 解析测试工况
    if test_wcs is not None:
        test_wc_list = test_wcs.split(',')
    else:
        test_wc_list = config['data']['test_wcs']

    print(f"\n开始抗噪声测试，目标工况: {test_wc_list}")
    print(f"噪声等级 (SNR): {SNR_LEVELS}")
    print("=" * 60)

    # 存储所有噪声等级的结果
    all_results = {}

    for snr in tqdm(SNR_LEVELS, desc="噪声等级"):
        snr_label = "clean" if snr is None else f"{snr}dB"
        results, avg_acc = evaluate_with_noise(encoder, classifier, config, device, test_wc_list, snr)
        all_results[snr_label] = {
            'per_wc': results,
            'avg_acc': avg_acc
        }
        print(f"  [{snr_label:>6}] 平均准确率: {avg_acc:.2f}%")

    # 保存结果
    save_path = save_results(all_results, config)

    print("=" * 60)
    print("抗噪声测试完成！")
    print(f"结果已保存至: {save_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anti-Noise 模型测试脚本")
    parser.add_argument("--config", default="configs/mcid_PU_train_1_meta_2_4.yaml",
                        help="配置文件路径")
    parser.add_argument("--ckpt", default=None,
                        help="模型检查点路径（可选，默认根据配置自动推断）")
    parser.add_argument("--test_wcs", default=None,
                        help="测试工况列表，逗号分隔，如 WC2,WC3,WC4（可选，默认使用配置文件中的 test_wcs）")
    args = parser.parse_args()

    main(args.config, args.ckpt, args.test_wcs)
