import os
import glob
import pandas as pd
import numpy as np
import random
import csv

# ================= 核心配置区域 =================

# PU数据集路径
RAW_DATA_ROOT = r"F:\Project\mid\德国数据集\领域泛化\PUdata_1_csv"
OUTPUT_ROOT = r"F:\Project\mid\S-MID\data\PU"  # 建议换个目录名区分时域数据

# 工况文件夹 (WC1, WC2, WC3, WC4)
WORKING_CONDITIONS = ["WC1", "WC2", "WC3", "WC4"]

# 故障类型映射关系
FAULT_TYPE_MAP = {
    "K001": 0,  # 正常状态
    "KA15": 1,  # 内圈故障1
    "KA04": 2,  # 内圈故障2
    "KI18": 3,  # 外圈故障2
    "KI21": 4,  # 外圈故障3
    "KB27": 5,  # 滚动体故障1
    "KB23": 6,  # 滚动体故障2
    "KB24": 7,  # 滚动体故障3
}

# 样本数量配置
TRAIN_NUM = 1000
TEST_NUM = 200
WINDOW_SIZE = 1024 # 时域输入建议 1024 或 2048
OVERLAP_RATIO = 0.8
STRIDE = int(WINDOW_SIZE * (1 - OVERLAP_RATIO))

# ================= 信号处理模块 (仅时域) =================

def advanced_signal_process(sample):
    """
    输入: (Channels, Window_Size)
    处理: 1. 去直流 2. Z-score 标准化 (保持时域特征)
    输出: (Channels, Window_Size)
    """
    processed_sample = []

    for ch in range(sample.shape[0]):
        sig = sample[ch, :]

        # 1. 去直流分量 (使信号中心对齐0)
        sig = sig - np.mean(sig)

        # 2. 样本内归一化 (Z-score)
        # 这是时域模型（如WDCNN, 1D-CNN）最常用的处理方式
        std = np.std(sig)
        if std > 1e-6:
            sig = (sig - np.mean(sig)) / std
        else:
            sig = sig - np.mean(sig) # 防止全0信号

        # 如果你希望将信号限制在 [-1, 1] 之间，可以取消下面三行的注释
        # max_val = np.max(np.abs(sig)) + 1e-6
        # sig = sig / max_val

        processed_sample.append(sig)

    return np.array(processed_sample)

# ================= 主逻辑部分 =================

def sliding_window_with_process(data_matrix, window_size, stride):
    n_channels, n_points = data_matrix.shape
    if n_points < window_size:
        return np.array([])

    n_samples = (n_points - window_size) // stride + 1
    samples = []

    for i in range(n_samples):
        start = i * stride
        end = start + window_size
        slice_data = data_matrix[:, start:end]

        # 调用仅包含时域处理的函数
        processed_data = advanced_signal_process(slice_data)
        samples.append(processed_data)

    return np.array(samples)

def process_one_file(csv_path):
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            first_row = next(reader)

        has_header = any(not item.replace('.', '', 1).isdigit() and item.strip() != '' for item in first_row)

        if has_header:
            df = pd.read_csv(csv_path, header=0)
            if df.shape[1] > 0:
                signal_column = df.columns[0]
                data_values = df[signal_column].values
            else:
                data_values = df.values.flatten()
        else:
            df = pd.read_csv(csv_path, header=None)
            data_values = df.values.flatten()

        data = data_values.reshape(1, -1).astype(np.float32)
        return data
    except Exception as e:
        print(f"    [读取失败] {csv_path}: {e}")
        return None

def main():
    print(f"开始处理Paderborn University数据集 (时域模式)...")
    print(f"数据输出目录: {OUTPUT_ROOT}")

    for wc_name in WORKING_CONDITIONS:
        wc_path = os.path.join(RAW_DATA_ROOT, wc_name)
        if not os.path.exists(wc_path):
            continue

        print(f"\n处理工况: {wc_name}")
        train_dir = os.path.join(OUTPUT_ROOT, wc_name, "train")
        test_dir = os.path.join(OUTPUT_ROOT, wc_name, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for fault_folder, label in FAULT_TYPE_MAP.items():
            fault_path = os.path.join(wc_path, fault_folder)
            if not os.path.exists(fault_path):
                continue

            csv_files = glob.glob(os.path.join(fault_path, "*.csv"))
            if not csv_files:
                continue

            csv_path = csv_files[0]
            raw_data = process_one_file(csv_path)
            if raw_data is None: continue

            # 时域信号直接进行窗口切割
            samples = sliding_window_with_process(raw_data, WINDOW_SIZE, STRIDE)

            if len(samples) == 0: continue

            if len(samples) < (TRAIN_NUM + TEST_NUM):
                actual_train = min(TRAIN_NUM, int(len(samples) * 0.8))
                actual_test = min(TEST_NUM, len(samples) - actual_train)
            else:
                actual_train = TRAIN_NUM
                actual_test = TEST_NUM

            random.seed(42)
            indices = list(range(len(samples)))
            random.shuffle(indices)
            train_indices = indices[:actual_train]
            test_indices = indices[actual_train:actual_train + actual_test]

            final_train = samples[train_indices]
            final_test = samples[test_indices]

            np.save(os.path.join(train_dir, f"{label}.npy"), final_train)
            np.save(os.path.join(test_dir, f"{label}.npy"), final_test)

            print(f"    [完成] {fault_folder} -> 样本数: {len(samples)} (Train:{actual_train}, Test:{actual_test})")

    print(f"\n所有工况时域数据处理完成！")

if __name__ == "__main__":
    main()
