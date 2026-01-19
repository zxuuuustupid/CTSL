import os
import glob
import pandas as pd
import numpy as np
from scipy.fftpack import fft
import random
import csv

# ================= 核心配置区域 =================

# PU数据集路径
RAW_DATA_ROOT = r"F:\Project\mid\德国数据集\领域泛化\PUdata_1_csv"
OUTPUT_ROOT = r"F:\Project\mid\S-MID\data\PU"

# 工况文件夹 (WC1, WC2, WC3, WC4)
WORKING_CONDITIONS = ["WC1", "WC2", "WC3", "WC4"]

# 故障类型映射关系 - 根据您的文件夹名设置
FAULT_TYPE_MAP = {
    "K001": 0,  # 正常状态
    "KA15": 1,  # 内圈故障1
    "KA04": 2,  # 内圈故障2
    # "KI16": 3,  # 外圈故障1
    "KI18": 3,  # 外圈故障2
    "KI21": 4,  # 外圈故障3
    "KB27": 5,  # 滚动体故障1
    "KB23": 6,  # 滚动体故障2
    "KB24": 7,  # 滚动体故障3
}

# 样本数量配置
TRAIN_NUM = 1000
TEST_NUM = 200
WINDOW_SIZE = 2048
OVERLAP_RATIO = 0.995
STRIDE = int(WINDOW_SIZE * (1 - OVERLAP_RATIO))

# ================= 信号处理模块 =================

def advanced_signal_process(sample):
    """
    输入: (Channels, Window_Size) - 对于PU数据集，Channels=1
    处理: 1. 去直流 2. FFT 3. 取幅值 4. 归一化
    输出: (Channels, Window_Size) 形状保持不变
    """
    processed_sample = []

    for ch in range(sample.shape[0]):
        sig = sample[ch, :]

        # 1. 去直流分量
        sig = sig - np.mean(sig)

        # 2. 样本内归一化 (Z-score)
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)

        # 3. 傅里叶变换
        fft_res = fft(sig)
        mag = np.abs(fft_res)

        # 4. 对数压缩: 增强微弱特征
        mag = np.log1p(mag)

        # 5. 归一化到0-1范围
        mag = (mag - np.min(mag)) / (np.max(mag) - np.min(mag) + 1e-6)

        processed_sample.append(mag)

    return np.array(processed_sample)

# ================= 主逻辑部分 =================

def sliding_window_with_process(data_matrix, window_size, stride):
    """
    对单通道数据进行滑动窗口切片并处理
    """
    n_channels, n_points = data_matrix.shape
    if n_points < window_size:
        return np.array([])

    n_samples = (n_points - window_size) // stride + 1
    samples = []

    for i in range(n_samples):
        start = i * stride
        end = start + window_size
        slice_data = data_matrix[:, start:end]

        # 信号处理
        processed_data = advanced_signal_process(slice_data)
        samples.append(processed_data)

    return np.array(samples)

def process_one_file(csv_path):
    """
    处理单个CSV文件（单列数据），自动检测是否有header
    """
    try:
        # 方法1：尝试读取第一行判断是否有header
        with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            first_row = next(reader)

        # 检查第一行是否包含非数字内容（如"vibration_signal"）
        has_header = any(not item.replace('.', '', 1).isdigit() and item.strip() != '' for item in first_row)

        if has_header:
            print(f"      检测到header行，跳过第一行")
            # 读取CSV，跳过header
            df = pd.read_csv(csv_path, header=0)
            # 假设第一列是振动信号
            if df.shape[1] > 0:
                signal_column = df.columns[0]
                data_values = df[signal_column].values
            else:
                data_values = df.values.flatten()
        else:
            # 无header，直接读取
            df = pd.read_csv(csv_path, header=None)
            data_values = df.values.flatten()

        # 转换为单通道数据 (1, n_points)
        data = data_values.reshape(1, -1).astype(np.float32)
        return data

    except Exception as e:
        print(f"    [读取失败] {csv_path}: {e}")
        return None

def get_fault_type_mapping():
    """
    自动获取故障类型映射关系
    遍历所有工况文件夹，收集所有故障类型文件夹名
    """
    fault_types = set()

    for wc in WORKING_CONDITIONS:
        wc_path = os.path.join(RAW_DATA_ROOT, wc)
        if not os.path.exists(wc_path):
            continue

        # 获取所有子文件夹（故障类型）
        fault_folders = [f for f in os.listdir(wc_path)
                        if os.path.isdir(os.path.join(wc_path, f))]

        for folder in fault_folders:
            fault_types.add(folder)

    # 创建映射关系
    mapping = {fault_type: idx for idx, fault_type in enumerate(sorted(fault_types))}
    return mapping

def main():
    print(f"开始处理Paderborn University数据集...")
    print(f"数据根目录: {RAW_DATA_ROOT}")

    # 验证FAULT_TYPE_MAP
    if not FAULT_TYPE_MAP:
        print("FAULT_TYPE_MAP为空，正在自动获取故障类型映射关系...")
        auto_mapping = get_fault_type_mapping()
        print("自动检测到的故障类型映射关系:")
        for fault_type, label in auto_mapping.items():
            print(f"  '{fault_type}': {label}")
        print("请将上述映射关系复制到FAULT_TYPE_MAP配置中，然后重新运行程序")
        return

    print(f"使用的故障类型映射关系: {FAULT_TYPE_MAP}")

    for wc_name in WORKING_CONDITIONS:
        wc_path = os.path.join(RAW_DATA_ROOT, wc_name)

        if not os.path.exists(wc_path):
            print(f"  [警告] 工况文件夹不存在: {wc_path}")
            continue

        print(f"\n处理工况: {wc_name}")
        train_dir = os.path.join(OUTPUT_ROOT, wc_name, "train")
        test_dir = os.path.join(OUTPUT_ROOT, wc_name, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # 遍历所有故障类型文件夹
        for fault_folder, label in FAULT_TYPE_MAP.items():
            fault_path = os.path.join(wc_path, fault_folder)

            if not os.path.exists(fault_path):
                print(f"    [跳过] 故障类型文件夹不存在: {fault_path}")
                continue

            # 查找CSV文件
            csv_files = glob.glob(os.path.join(fault_path, "*.csv"))

            if not csv_files:
                print(f"    [跳过] 未找到CSV文件: {fault_path}")
                continue

            # 取第一个CSV文件
            csv_path = csv_files[0]
            print(f"    处理: {fault_folder} -> 标签:{label}")
            print(f"      文件: {os.path.basename(csv_path)}")

            # 处理文件
            raw_data = process_one_file(csv_path)
            if raw_data is None:
                continue

            print(f"      原始数据形状: {raw_data.shape}")

            # 检查数据是否有效
            if raw_data.shape[1] == 0:
                print(f"      [跳过] 无效数据，无数据点")
                continue

            # 滑动窗口切片并处理
            samples = sliding_window_with_process(raw_data, WINDOW_SIZE, STRIDE)

            if len(samples) == 0:
                print(f"      [跳过] 无法生成有效样本")
                continue

            print(f"      生成样本数量: {len(samples)}")

            if len(samples) < (TRAIN_NUM + TEST_NUM):
                print(f"      [警告] 样本数量不足，调整数量")
                actual_train = min(TRAIN_NUM, int(len(samples) * 0.8))
                actual_test = min(TEST_NUM, len(samples) - actual_train)
                print(f"      调整后: 训练集={actual_train}, 测试集={actual_test}")
            else:
                actual_train = TRAIN_NUM
                actual_test = TEST_NUM

            if len(samples) < (actual_train + actual_test):
                print(f"      [跳过] 样本数量仍然不足")
                continue

            # 随机打乱
            random.seed(42)
            indices = list(range(len(samples)))
            random.shuffle(indices)
            train_indices = indices[:actual_train]
            test_indices = indices[actual_train:actual_train + actual_test]

            final_train = samples[train_indices]
            final_test = samples[test_indices]

            # 保存
            np.save(os.path.join(train_dir, f"{label}.npy"), final_train)
            np.save(os.path.join(test_dir, f"{label}.npy"), final_test)

            print(f"      [完成] 训练集: {final_train.shape}, 测试集: {final_test.shape}")

        print(f"  [工况完成] {wc_name}")

if __name__ == "__main__":
    main()
