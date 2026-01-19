import os
import glob
import pandas as pd
import numpy as np
from scipy.fftpack import fft

# ================= 核心配置区域 =================

# 路径配置
RAW_DATA_ROOT = r"F:\Project\CZSL\code\Disentangling-before-Composing\加拿大轴承数据集\data"
OUTPUT_ROOT = r"F:\Project\mid\S-MID\data\Ottawa"

# 映射关系定义
WC_MAP = {'A': 'WC1', 'B': 'WC2', 'C': 'WC3', 'D': 'WC4'}
STATE_MAP = {'H': 0, 'IF': 1, 'OF': 2, 'BF': 3, 'CF': 4}

# 采样与切片配置
TRAIN_NUM = 1000
TEST_NUM = 200
WINDOW_SIZE = 2048
OVERLAP_RATIO = 0.8
STRIDE = int(WINDOW_SIZE * (1 - OVERLAP_RATIO))

# ================= 信号处理模块 =================

def advanced_signal_process(sample):
    """
    输入: (Channels, Window_Size)
    处理: 1. Z-score 2. FFT 3. 幅值取对数 4. 归一化
    """
    processed_sample = []
    for ch in range(sample.shape[0]):
        sig = sample[ch, :]
        # 1. 样本内标准化
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)
        # 2. 傅里叶变换
        fft_res = fft(sig)
        mag = np.abs(fft_res)
        # 3. 对数压缩
        mag = np.log1p(mag)
        # 4. 归一化到 0-1
        mag = (mag - np.min(mag)) / (np.max(mag) - np.min(mag) + 1e-6)
        processed_sample.append(mag)
    return np.array(processed_sample)

# ================= 数据切片模块 =================

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
        processed_data = advanced_signal_process(slice_data)
        samples.append(processed_data)

        # 达到最大需求数即可停止，节省内存
        if len(samples) >= (TRAIN_NUM + TEST_NUM + 10):
            break

    return np.array(samples)

# ================= 主逻辑部分 =================

def main():
    print(f"开始处理 Ottawa 数据集...")

    # 1. 获取所有 CSV 文件
    search_pattern = os.path.join(RAW_DATA_ROOT, "*.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"错误：在路径 {RAW_DATA_ROOT} 下未找到任何 CSV 文件！")
        return

    for csv_path in csv_files:
        # 获取文件名（例如 A-BF.csv -> A-BF）
        file_name = os.path.splitext(os.path.basename(csv_path))[0]

        # 解析文件名：假设格式始终为 "工况-故障"
        try:
            parts = file_name.split('-')
            wc_key = parts[0]      # A, B, C, D
            state_key = parts[1]   # H, IF, OF, BF, CF

            wc_name = WC_MAP[wc_key]
            save_label = STATE_MAP[state_key]
        except (IndexError, KeyError):
            print(f" [跳过] 无法解析文件名格式: {file_name}")
            continue

        print(f" 正在处理: {file_name} -> {wc_name} Label:{save_label}")

        # 创建输出目录
        train_dir = os.path.join(OUTPUT_ROOT, wc_name, "train")
        test_dir = os.path.join(OUTPUT_ROOT, wc_name, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # 2. 读取数据
        try:
            # Ottawa 数据集通常第一列是时间或索引，之后是传感器数据
            # 如果 CSV 没有表头，请将 header=0 改为 None
            df = pd.read_csv(csv_path)
            # 转置为 (Channels, Length)
            raw_data = df.values.astype(np.float32).T
        except Exception as e:
            print(f" [读取失败] {file_name}: {e}")
            continue

        # 3. 滑动窗切片与 FFT 处理
        samples = sliding_window_with_process(raw_data, WINDOW_SIZE, STRIDE)

        if len(samples) < (TRAIN_NUM + TEST_NUM):
            print(f" [警告] {file_name} 样本数不足 (只有 {len(samples)})，请检查 WINDOW_SIZE 或 STRIDE")
            continue

        # 4. 打乱并保存
        np.random.seed(42)
        np.random.shuffle(samples)

        final_train = samples[:TRAIN_NUM]
        final_test = samples[TRAIN_NUM : TRAIN_NUM + TEST_NUM]

        np.save(os.path.join(train_dir, f"{save_label}.npy"), final_train)
        np.save(os.path.join(test_dir, f"{save_label}.npy"), final_test)
        print(f" [DONE] 保存成功 | 训练集: {final_train.shape}")

if __name__ == "__main__":
    main()
