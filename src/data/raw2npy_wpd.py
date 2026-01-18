import os
import glob
import pandas as pd
import numpy as np
from scipy.fftpack import fft

# ================= 核心配置区域 =================
RAW_DATA_ROOT = r"F:\Project\TripletLoss\BJTU-RAO Bogie Datasets\Data\BJTU_RAO_Bogie_Datasets"
OUTPUT_ROOT = r"F:\Project\mid\S-MID\data\gearbox" # 建议换个名字

FOLDER_PATTERN = "M0_G{}_LA0_RA0"
STATE_MAP = {0: 0, 3: 1, 7: 2, 8: 3}
SAMPLE_INDICES = range(1, 10)

TRAIN_NUM = 1000
TEST_NUM = 200
WINDOW_SIZE = 2048
OVERLAP_RATIO = 0.8
STRIDE = int(WINDOW_SIZE * (1 - OVERLAP_RATIO))

# ================= 信号处理模块 =================

def advanced_signal_process(sample):
    """
    输入: (Channels, Window_Size) e.g., (6, 2048)
    处理: 1. 去直流 2. FFT 3. 取幅值 4. 归一化
    输出: (Channels, Window_Size) 形状保持不变
    """
    processed_sample = []

    for ch in range(sample.shape[0]):
        sig = sample[ch, :]

        # 1. 样本内归一化 (Z-score): 解决不同工况幅值差异的核心
        sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)

        # 2. 傅里叶变换
        # 我们取整个 WINDOW_SIZE 的 FFT
        fft_res = fft(sig)
        mag = np.abs(fft_res)

        # 3. 对数压缩: 增强微弱特征
        mag = np.log1p(mag)

        # 4. 再次标准化 (将频域数值限制在 0-1 附近)
        mag = (mag - np.min(mag)) / (np.max(mag) - np.min(mag) + 1e-6)

        processed_sample.append(mag)

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

        # --- 注入专业信号处理 ---
        processed_data = advanced_signal_process(slice_data)
        samples.append(processed_data)

    return np.array(samples)

def process_one_file(csv_path):
    try:
        # 只读取有用的通道（如果是 gearbox，通常是多轴向传感器）
        df = pd.read_csv(csv_path, header=0)
        data = df.values.astype(np.float32).T
        return data
    except Exception as e:
        print(f"    [读取失败] {csv_path}: {e}")
        return None

def main():
    print(f"开始专业预处理... 模式: FFT + Per-sample Normalization")

    for s_idx in SAMPLE_INDICES:
        wc_name = f"WC{s_idx}"
        train_dir = os.path.join(OUTPUT_ROOT, wc_name, "train")
        test_dir = os.path.join(OUTPUT_ROOT, wc_name, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for raw_state, save_label in STATE_MAP.items():
            state_folder_name = FOLDER_PATTERN.format(raw_state)
            target_folder = os.path.join(RAW_DATA_ROOT, state_folder_name, f"Sample_{s_idx}")
            search_pattern = os.path.join(target_folder, "data_gearbox*.csv")
            csv_files = glob.glob(search_pattern)

            if not csv_files: continue
            csv_path = csv_files[0]

            raw_data = process_one_file(csv_path)
            if raw_data is None: continue

            # 调用带信号处理的切片函数
            samples = sliding_window_with_process(raw_data, WINDOW_SIZE, STRIDE)

            if len(samples) < (TRAIN_NUM + TEST_NUM):
                print(f"  [跳过] WC{s_idx} G{raw_state} 样本数不足")
                continue

            np.random.seed(42)
            np.random.shuffle(samples)

            final_train = samples[:TRAIN_NUM]
            final_test = samples[TRAIN_NUM : TRAIN_NUM + TEST_NUM]

            np.save(os.path.join(train_dir, f"{save_label}.npy"), final_train)
            np.save(os.path.join(test_dir, f"{save_label}.npy"), final_test)
            print(f"  [DONE] {wc_name} Label:{save_label} | Shape:{final_train.shape}")

if __name__ == "__main__":
    main()
