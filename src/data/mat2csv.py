import os
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
import traceback

# ================= 配置区域 =================
# 建议只保留标准长度的数据（约256,000点），16,001点的短数据会严重干扰DG（领域泛化）效果
MIN_DATA_LENGTH = 200000
VIBRATION_KEYWORDS = ['vibration', 'vibration_1', 'vib_1', 'acc', 'acceleration']
# ===========================================

def mat_to_csv_paderborn(source_root, target_root):
    """
    针对 Paderborn University 轴承数据集的高可靠性 MAT 转 CSV 工具
    """
    Path(target_root).mkdir(parents=True, exist_ok=True)
    mat_files = []
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.lower().endswith('.mat'):
                mat_files.append(os.path.join(root, file))

    print(f"🚀 找到 {len(mat_files)} 个 MAT 文件，准备开始转换...")

    success_count = 0
    fail_count = 0
    skip_count = 0

    for mat_path in mat_files:
        try:
            relative_path = os.path.relpath(os.path.dirname(mat_path), source_root)
            target_dir = os.path.join(target_root, relative_path)
            Path(target_dir).mkdir(parents=True, exist_ok=True)

            mat_filename = os.path.basename(mat_path)
            base_name = os.path.splitext(mat_filename)[0]
            csv_path = os.path.join(target_dir, base_name + '.csv')

            # 1. 加载 MAT 文件
            mat_data = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

            # 2. 提取信号
            # 尝试根据文件名提取主变量，如果失败则全局搜索
            signal_data = None
            if base_name in mat_data:
                signal_data = extract_vibration_from_struct(mat_data[base_name])

            if signal_data is None:
                signal_data = search_vibration_globally(mat_data)

            # 3. 校验并转换
            if signal_data is not None:
                # 扁平化处理
                signal_data = signal_data.flatten()

                # 长度过滤：核心修改，防止训练 12.5% 的元凶
                if signal_data.size < MIN_DATA_LENGTH:
                    print(f"  ⚠️ 跳过 {base_name}: 长度过短 ({signal_data.size} 点)")
                    skip_count += 1
                    continue

                # 转换为 DataFrame
                df = pd.DataFrame(signal_data, columns=['vibration_signal'])

                # 检查是否存在 NaN
                if df['vibration_signal'].isnull().any():
                    df = df.fillna(method='ffill')

                df.to_csv(csv_path, index=False)
                print(f"  ✓ 成功: {base_name} (Length: {signal_data.size})")
                success_count += 1
            else:
                print(f"  ✗ 失败: {base_name} 未找到振动信号字段")
                fail_count += 1

        except Exception as e:
            print(f"  ✗ 严重错误 {mat_path}: {str(e)}")
            fail_count += 1

    print("\n" + "=" * 60)
    print("✨ 处理总结:")
    print(f"  - 成功转换: {success_count}")
    print(f"  - 长度不足跳过: {skip_count}")
    print(f"  - 提取失败: {fail_count}")
    print(f"  - 保存根目录: {target_root}")
    print("=" * 60)

def extract_vibration_from_struct(struct_obj):
    """
    深度优先搜索结构体中的振动信号字段
    """
    # 场景 1: PU 数据集标准的 X 字段（通常是个 Cell 数组或嵌套结构）
    if hasattr(struct_obj, 'X'):
        x_field = struct_obj.X
        # 如果 X 是数组/列表，遍历查找包含 Name='vibration' 的元素
        if isinstance(x_field, np.ndarray):
            # 针对 struct_as_record=False 产生的对象数组
            for item in x_field.flat:
                if hasattr(item, 'Name') and any(k in str(item.Name).lower() for k in VIBRATION_KEYWORDS):
                    if hasattr(item, 'Data'):
                        return item.Data
                # 如果没有 Name 属性，但 Data 很大，可能是它
                if hasattr(item, 'Data') and isinstance(item.Data, np.ndarray) and item.Data.size > MIN_DATA_LENGTH:
                    return item.Data

    # 场景 2: 递归查找所有属性
    for attr in dir(struct_obj):
        if any(k in attr.lower() for k in VIBRATION_KEYWORDS):
            val = getattr(struct_obj, attr)
            if isinstance(val, np.ndarray) and val.size > 1000:
                return val
    return None

def search_vibration_globally(mat_dict):
    """
    在 MAT 字典中全局搜索大型数值数组
    """
    potential_signals = []
    for key, value in mat_dict.items():
        if key.startswith('__'): continue

        # 如果是数值数组
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            if value.size >= MIN_DATA_LENGTH:
                # 如果名字里带 vibration，最高优先级
                if any(k in key.lower() for k in VIBRATION_KEYWORDS):
                    return value
                potential_signals.append(value)

        # 深度递归查找嵌套结构
        if hasattr(value, '__dict__') or isinstance(value, np.ndarray):
            res = extract_vibration_from_struct(value)
            if res is not None: return res

    # 如果没找到带名字的，返回最大的数组
    if potential_signals:
        return max(potential_signals, key=lambda x: x.size)
    return None

if __name__ == "__main__":
    # === 路径配置 ===
    # 请确保源目录中包含各个工况（WC1, WC2等）的文件夹
    SOURCE_DIR = r'F:\Project\mid\德国数据集\领域泛化\PUdata_1'
    TARGET_DIR = r'F:\Project\mid\德国数据集\领域泛化\PUdata_1_csv'

    print("--- Paderborn University (PU) Dataset Preprocessing Tool ---")
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ 错误: 找不到源目录 {SOURCE_DIR}")
    else:
        mat_to_csv_paderborn(SOURCE_DIR, TARGET_DIR)
