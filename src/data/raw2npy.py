import os
import glob
import pandas as pd
import numpy as np

# ================= 核心配置区域 =================

# 1. 原始数据根目录
RAW_DATA_ROOT = r"F:\Project\TripletLoss\BJTU-RAO Bogie Datasets\Data\BJTU_RAO_Bogie_Datasets"

# 2. 输出保存根目录 (脚本会自动创建 WC1-WC9 子文件夹)
OUTPUT_ROOT = r"F:\Project\mid\S-MID\data\gearbox"

# 3. 状态映射 (Folder名字中的关键字 -> 保存的标签)
FOLDER_PATTERN = "M0_G{}_LA0_RA0" 
STATE_MAP = {
    0: 0,  # G0 -> 0.npy
    3: 1,  # G3 -> 1.npy
    7: 2,  # G7 -> 2.npy
    8: 3   # G8 -> 3.npy
}

# 4. 工况设置 (Sample 1-9 对应 WC 1-9)
SAMPLE_INDICES = range(1, 10) # 1 到 9

# 5. 样本数量
TRAIN_NUM = 1000
TEST_NUM = 200

# 6. 滑窗参数
WINDOW_SIZE = 2048      # !!! 请确认长度，不够切会报错 !!!
OVERLAP_RATIO = 0.8     # 80% 重叠
STRIDE = int(WINDOW_SIZE * (1 - OVERLAP_RATIO)) 

# ====================================================

def sliding_window(data_matrix, window_size, stride):
    """
    输入: (Channels, Time_Steps)
    输出: (N, Channels, Window_Size)
    """
    n_channels, n_points = data_matrix.shape
    if n_points < window_size:
        return np.array([])
    
    n_samples = (n_points - window_size) // stride + 1
    
    samples = []
    for i in range(n_samples):
        start = i * stride
        end = start + window_size
        samples.append(data_matrix[:, start:end])
    
    return np.array(samples)

def process_one_file(csv_path):
    try:
        # 读取CSV，排除表头
        df = pd.read_csv(csv_path, header=0)
        # 转置为 (Channels, Length)
        data = df.values.astype(np.float32).T 
        return data
    except Exception as e:
        print(f"    [读取失败] {csv_path}: {e}")
        return None

def main():
    print(f"开始处理... 窗口: {WINDOW_SIZE}, 步长: {STRIDE}")
    print(f"目标: 生成 WC1 - WC9，每个含 train(1000) 和 test(200)")

    # === 第一层循环：工况 (Sample_1 -> WC1) ===
    for s_idx in SAMPLE_INDICES:
        wc_name = f"WC{s_idx}"
        print(f"\n=========================================")
        print(f"正在处理工况: Sample_{s_idx}  --->  生成 {wc_name}")
        print(f"=========================================")

        # 创建输出目录: .../gearbox/WC1/train 和 .../gearbox/WC1/test
        train_dir = os.path.join(OUTPUT_ROOT, wc_name, "train")
        test_dir = os.path.join(OUTPUT_ROOT, wc_name, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # === 第二层循环：状态 (G0, G3, G7, G8) ===
        for raw_state, save_label in STATE_MAP.items():
            
            # 1. 构造原始文件夹路径
            state_folder_name = FOLDER_PATTERN.format(raw_state)
            # 路径: .../M0_G0_LA0_RA0/Sample_1/
            target_folder = os.path.join(RAW_DATA_ROOT, state_folder_name, f"Sample_{s_idx}")
            
            # 2. 搜索 CSV (data_gearbox_....csv)
            search_pattern = os.path.join(target_folder, "data_gearbox*.csv")
            csv_files = glob.glob(search_pattern)

            if not csv_files:
                print(f"  [警告] G{raw_state} 在 Sample_{s_idx} 下未找到CSV文件！跳过。")
                continue

            # 默认取第一个
            csv_path = csv_files[0]
            
            # 3. 读取与切片
            raw_data = process_one_file(csv_path)
            if raw_data is None: continue

            samples = sliding_window(raw_data, WINDOW_SIZE, STRIDE)
            total_extracted = len(samples)

            # 4. 检查数量是否足够
            needed = TRAIN_NUM + TEST_NUM
            if total_extracted < needed:
                print(f"  [严重警告] G{raw_state} (Sample_{s_idx}) 样本不足！")
                print(f"    现有: {total_extracted}, 需要: {needed}。")
                print(f"    该文件将被跳过生成，请减小 overlap 或 窗口大小。")
                continue

            # 5. 打乱并切分
            # 使用固定种子保证每次生成结果一致，但shuffle保证训练测试分布同构
            np.random.seed(42 + s_idx + raw_state) 
            np.random.shuffle(samples)

            final_train = samples[:TRAIN_NUM]
            final_test = samples[TRAIN_NUM : TRAIN_NUM + TEST_NUM]

            # 6. 保存
            # train/0.npy, test/0.npy
            np.save(os.path.join(train_dir, f"{save_label}.npy"), final_train)
            np.save(os.path.join(test_dir, f"{save_label}.npy"), final_test)

            print(f"  [成功] G{raw_state} -> {save_label}.npy | Train:{final_train.shape}, Test:{final_test.shape}")

    print("\n所有工况处理完毕！")

if __name__ == "__main__":
    main()