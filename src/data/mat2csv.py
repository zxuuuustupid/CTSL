import os
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
import traceback

def mat_to_csv_preserve_structure(source_root, target_root):
    """
    专门针对PUdata数据集复杂结构体格式的MAT到CSV转换
    """
    # 确保目标根目录存在
    Path(target_root).mkdir(parents=True, exist_ok=True)

    # 遍历源目录
    mat_files = []
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.lower().endswith('.mat'):
                mat_files.append(os.path.join(root, file))

    print(f"找到 {len(mat_files)} 个MAT文件")

    success_count = 0
    fail_count = 0

    for mat_path in mat_files:
        try:
            # 计算相对路径
            relative_path = os.path.relpath(os.path.dirname(mat_path), source_root)

            # 构建目标文件夹路径
            target_dir = os.path.join(target_root, relative_path)
            Path(target_dir).mkdir(parents=True, exist_ok=True)

            # 构建CSV文件名
            mat_filename = os.path.basename(mat_path)
            base_name = os.path.splitext(mat_filename)[0]
            csv_filename = base_name + '.csv'
            csv_path = os.path.join(target_dir, csv_filename)

            print(f"\n处理: {mat_path}")
            print(f"  -> {csv_path}")
            print(f"    预期变量名: '{base_name}'")

            # 读取MAT文件
            try:
                mat_data = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
                print("  ✓ scipy读取成功 (struct_as_record=False)")
            except Exception as e:
                print(f"  ✗ scipy读取失败: {str(e)}")
                fail_count += 1
                continue

            # 检查变量是否存在
            if base_name in mat_data:
                print(f"  ✓ 找到目标变量: '{base_name}'")
                main_struct = mat_data[base_name]

                # 提取振动信号数据
                signal_data = extract_pudata_complex_structure(main_struct)

                if signal_data is not None and signal_data.size > 0:
                    # 转换为DataFrame
                    df = convert_to_dataframe(signal_data)

                    if df is not None:
                        # 保存为CSV
                        df.to_csv(csv_path, index=False)
                        print(f"  ✓ 转换成功！数据形状: {df.shape}")
                        print(f"    保存到: {csv_path}")
                        success_count += 1
                    else:
                        print(f"  ✗ 无法转换为DataFrame")
                        save_debug_info(mat_path, mat_data, csv_path)
                        fail_count += 1
                else:
                    print(f"  ✗ 无法提取有效的振动信号数据")
                    save_debug_info(mat_path, mat_data, csv_path)
                    fail_count += 1
            else:
                print(f"  ✗ 未找到变量 '{base_name}'")
                # 尝试查找其他可能的振动信号变量
                signal_data = find_vibration_signal_in_mat(mat_data)
                if signal_data is not None:
                    df = convert_to_dataframe(signal_data)
                    if df is not None:
                        df.to_csv(csv_path, index=False)
                        print(f"  ✓ 通过替代方法找到振动信号！")
                        success_count += 1
                    else:
                        save_debug_info(mat_path, mat_data, csv_path)
                        fail_count += 1
                else:
                    save_debug_info(mat_path, mat_data, csv_path)
                    fail_count += 1

        except Exception as e:
            print(f"  ✗ 处理失败: {str(e)}")
            print(f"    详细错误: {traceback.format_exc()}")
            fail_count += 1

    print("\n" + "=" * 60)
    print(f"转换完成！统计结果:")
    print(f"  成功: {success_count} 个文件")
    print(f"  失败: {fail_count} 个文件")
    print(f"  总计: {success_count + fail_count} 个文件")
    print(f"CSV文件保存在: {target_root}")
    return success_count, fail_count

def extract_pudata_complex_structure(main_struct):
    """专门提取PUdata复杂结构体中的振动信号数据"""
    try:
        print(f"  分析PUdata复杂结构体")

        # 情况1: 直接访问X字段
        if hasattr(main_struct, 'X'):
            print(f"    ✓ 找到X字段")
            X_data = main_struct.X

            # 情况1a: X是结构体数组
            if isinstance(X_data, np.ndarray) and X_data.dtype.names is not None:
                print(f"      X是结构体数组，字段: {X_data.dtype.names}")

                # 查找vibration_1字段
                vibration_signal = find_vibration_in_x_struct(X_data)
                if vibration_signal is not None:
                    return vibration_signal

            # 情况1b: X是cell数组
            elif isinstance(X_data, np.ndarray) and X_data.dtype == 'object':
                print(f"      X是cell数组，尝试提取")
                vibration_signal = find_vibration_in_x_cell(X_data)
                if vibration_signal is not None:
                    return vibration_signal

        # 情况2: 通过Y字段查找
        if hasattr(main_struct, 'Y'):
            print(f"    ⚠️ 尝试通过Y字段查找")
            vibration_signal = find_vibration_in_y_field(main_struct.Y)
            if vibration_signal is not None:
                return vibration_signal

        # 情况3: 遍历所有字段
        print(f"    ⚠️ 遍历所有字段查找振动信号...")
        vibration_signal = search_all_fields_for_vibration(main_struct)
        if vibration_signal is not None:
            return vibration_signal

        print(f"    ✗ 未找到振动信号数据")
        return None

    except Exception as e:
        print(f"    提取失败: {str(e)}")
        return None

def find_vibration_in_x_struct(X_struct):
    """在X结构体数组中查找振动信号"""
    try:
        # 遍历X结构体的每个元素
        for i in range(X_struct.size):
            element = X_struct.flat[i]

            # 检查是否有Name和Data字段
            if hasattr(element, 'Name') and hasattr(element, 'Data'):
                name = element.Name
                data = element.Data

                # 检查Name是否包含vibration
                if isinstance(name, str) and 'vibration' in name.lower():
                    print(f"        ✓ 找到振动信号: '{name}'")
                    if isinstance(data, np.ndarray) and data.size > 1000:
                        print(f"          数据形状: {data.shape}")
                        return data

                # 检查常见的振动信号名称
                vibration_names = ['vibration_1', 'vibration_2', 'vibration', 'vib', 'acc', 'acceleration']
                if isinstance(name, str) and any(vib_name in name.lower() for vib_name in vibration_names):
                    print(f"        ✓ 找到振动信号: '{name}'")
                    if isinstance(data, np.ndarray) and data.size > 1000:
                        print(f"          数据形状: {data.shape}")
                        return data

        # 如果没有找到，尝试查看每个字段
        for field_name in X_struct.dtype.names:
            field_data = getattr(X_struct, field_name)
            if isinstance(field_data, np.ndarray) and field_data.dtype.names is not None:
                result = find_vibration_in_x_struct(field_data)
                if result is not None:
                    return result

        return None

    except Exception as e:
        print(f"      在X结构体中查找失败: {str(e)}")
        return None

def find_vibration_in_x_cell(X_cell):
    """在X cell数组中查找振动信号"""
    try:
        # 遍历cell数组
        for i in range(X_cell.size):
            cell_item = X_cell.flat[i]

            # 如果是结构体
            if hasattr(cell_item, 'dtype') and cell_item.dtype.names is not None:
                vibration_signal = find_vibration_in_x_struct(cell_item)
                if vibration_signal is not None:
                    return vibration_signal

            # 如果是数组且较大
            if isinstance(cell_item, np.ndarray) and cell_item.size > 1000:
                if np.issubdtype(cell_item.dtype, np.number):
                    print(f"        ✓ 在cell[{i}]找到大型数值数组")
                    return cell_item

        return None
    except Exception as e:
        print(f"      在X cell中查找失败: {str(e)}")
        return None

def find_vibration_in_y_field(Y_field):
    """在Y字段中查找振动信号"""
    try:
        # Y字段通常包含故障信息，但也可能包含信号
        if isinstance(Y_field, np.ndarray) and Y_field.size > 1000:
            if np.issubdtype(Y_field.dtype, np.number):
                print(f"      ✓ Y字段包含大型数值数组，可能为信号")
                return Y_field
        return None
    except:
        return None

def search_all_fields_for_vibration(struct_obj):
    """递归搜索所有字段查找振动信号"""
    try:
        # 如果是numpy数组
        if isinstance(struct_obj, np.ndarray):
            if struct_obj.dtype.names is not None:
                # 结构体数组
                for field_name in struct_obj.dtype.names:
                    field_value = getattr(struct_obj, field_name)
                    result = search_all_fields_for_vibration(field_value)
                    if result is not None:
                        return result
            elif struct_obj.dtype == 'object':
                # cell数组
                for i in range(struct_obj.size):
                    cell_item = struct_obj.flat[i]
                    result = search_all_fields_for_vibration(cell_item)
                    if result is not None:
                        return result
            elif np.issubdtype(struct_obj.dtype, np.number) and struct_obj.size > 1000:
                print(f"      ✓ 找到大型数值数组，形状: {struct_obj.shape}")
                return struct_obj

        # 如果是对象，检查属性
        if hasattr(struct_obj, '__dict__'):
            for attr_name in dir(struct_obj):
                if attr_name.startswith('__') or attr_name in ['size', 'shape', 'dtype']:
                    continue

                try:
                    attr_value = getattr(struct_obj, attr_name)
                    if attr_value is None:
                        continue

                    # 检查属性名是否包含vibration
                    if 'vibration' in attr_name.lower() or 'vib' in attr_name.lower() or 'acc' in attr_name.lower():
                        if isinstance(attr_value, np.ndarray) and attr_value.size > 1000:
                            if np.issubdtype(attr_value.dtype, np.number):
                                print(f"      ✓ 在属性 '{attr_name}' 找到振动信号")
                                return attr_value

                    # 递归搜索
                    result = search_all_fields_for_vibration(attr_value)
                    if result is not None:
                        return result
                except:
                    continue

        return None
    except:
        return None

def find_vibration_signal_in_mat(mat_data):
    """在整个MAT数据中搜索振动信号"""
    try:
        vibration_names = ['vibration', 'vib', 'acc', 'acceleration', 'signal', 'data']

        for var_name, var_value in mat_data.items():
            if var_name.startswith('__'):
                continue

            # 检查变量名是否包含振动相关词汇
            if any(vib_name in var_name.lower() for vib_name in vibration_names):
                if isinstance(var_value, np.ndarray) and var_value.size > 1000:
                    if np.issubdtype(var_value.dtype, np.number):
                        print(f"    ✓ 在变量 '{var_name}' 找到振动信号")
                        return var_value

            # 递归搜索
            result = search_all_fields_for_vibration(var_value)
            if result is not None:
                print(f"    ✓ 在变量 '{var_name}' 的嵌套结构中找到振动信号")
                return result

        return None
    except:
        return None

def convert_to_dataframe(signal_data):
    """将信号数据转换为DataFrame"""
    try:
        # 确保是numpy数组
        if not isinstance(signal_data, np.ndarray):
            signal_data = np.array(signal_data)

        # 处理空数组
        if signal_data.size == 0:
            print(f"    ✗ 空数组，无法转换")
            return None

        # 处理非数值数据
        if not np.issubdtype(signal_data.dtype, np.number):
            print(f"    ⚠️ 非数值数据类型: {signal_data.dtype}，尝试转换")
            try:
                # 尝试提取数值部分
                if hasattr(signal_data, 'item'):
                    numeric_value = float(signal_data.item())
                    signal_data = np.array([numeric_value])
                else:
                    signal_data = signal_data.astype(float)
                print(f"    ✓ 转换成功")
            except Exception as e:
                print(f"    ✗ 转换失败: {str(e)}")
                return None

        print(f"    处理前形状: {signal_data.shape}")

        # 确保数据是1D或2D
        if signal_data.ndim > 2:
            print(f"    ⚠️ 高维数组 (ndim={signal_data.ndim})，展平为2D")
            # 找到最大的维度作为时间维度
            time_dim = np.argmax(signal_data.shape)
            other_dims = [d for i, d in enumerate(signal_data.shape) if i != time_dim]
            num_features = np.prod(other_dims)

            # 重塑为 (time_steps, features)
            signal_data = signal_data.reshape(signal_data.shape[time_dim], num_features, order='F')
            print(f"      重塑后形状: {signal_data.shape}")

        # 处理1D数组
        if signal_data.ndim == 1:
            print(f"    ✓ 1D振动信号，创建单列")
            return pd.DataFrame(signal_data, columns=['vibration_signal'])

        # 处理2D数组
        elif signal_data.ndim == 2:
            rows, cols = signal_data.shape
            print(f"    2D数组: {rows}行 × {cols}列")

            # 如果列数远大于行数，转置
            if cols > rows * 10:
                print(f"      ⚠️ 列数远大于行数，转置处理")
                signal_data = signal_data.T
                rows, cols = signal_data.shape

            # 如果是单通道信号（1列）
            if cols == 1:
                print(f"      ✓ 单通道振动信号")
                return pd.DataFrame(signal_data, columns=['vibration_signal'])

            # 如果是多通道信号（2-8列）
            elif 2 <= cols <= 8:
                print(f"      ✓ 多通道振动信号 ({cols}通道)")
                column_names = [f'vibration_channel_{i+1}' for i in range(cols)]
                return pd.DataFrame(signal_data, columns=column_names)

            # 如果是时间序列格式（时间在行，特征在列）
            else:
                print(f"      ✓ 多特征振动数据 ({cols}特征)")
                # 只取前4个特征以避免CSV过大
                if cols > 4:
                    print(f"        ⚠️ 特征过多，只取前4个通道")
                    signal_data = signal_data[:, :4]
                    cols = 4

                column_names = [f'vibration_feature_{i+1}' for i in range(cols)]
                return pd.DataFrame(signal_data, columns=column_names)

        print(f"    ✗ 无法处理的维度: {signal_data.ndim}")
        return None

    except Exception as e:
        print(f"    转换失败: {str(e)}")
        return None

def save_debug_info(mat_path, mat_data, csv_path):
    """保存详细的调试信息"""
    debug_path = csv_path.replace('.csv', '_debug.txt')
    with open(debug_path, 'w', encoding='utf-8') as f:
        f.write(f"PUdata详细调试信息\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"源文件: {mat_path}\n")
        f.write(f"时间: {pd.Timestamp.now()}\n\n")

        f.write("变量列表及结构:\n")
        f.write("-" * 40 + "\n")

        for var_name in mat_data.keys():
            if var_name.startswith('__'):
                continue

            f.write(f"变量: {var_name}\n")
            var_value = mat_data[var_name]

            if isinstance(var_value, np.ndarray):
                f.write(f"  类型: numpy.ndarray\n")
                f.write(f"  形状: {var_value.shape}\n")
                f.write(f"  数据类型: {var_value.dtype}\n")

                # 如果是结构体数组
                if var_value.dtype.names is not None:
                    f.write(f"  结构体字段: {var_value.dtype.names}\n")

                    # 显示前几个元素的概览
                    num_elements = min(3, var_value.size)
                    f.write(f"  前 {num_elements} 个元素概览:\n")

                    for i in range(num_elements):
                        if i >= var_value.size:
                            break
                        element = var_value.flat[i]
                        f.write(f"    元素 {i}:\n")

                        for field_name in var_value.dtype.names:
                            try:
                                field_value = getattr(element, field_name)
                                f.write(f"      字段 '{field_name}':\n")

                                if isinstance(field_value, np.ndarray):
                                    f.write(f"        形状: {field_value.shape}\n")
                                    f.write(f"        数据类型: {field_value.dtype}\n")
                                    # 显示小数组的内容
                                    if field_value.size < 10:
                                        f.write(f"        值: {field_value}\n")
                                else:
                                    f.write(f"        值: {field_value}\n")
                            except Exception as e:
                                f.write(f"      字段 '{field_name}' 访问失败: {str(e)}\n")

            else:
                f.write(f"  类型: {type(var_value)}\n")
                try:
                    f.write(f"  值: {var_value}\n")
                except:
                    f.write(f"  值: 无法显示\n")

            f.write("\n")

    print(f"    详细调试信息已保存到: {debug_path}")

if __name__ == "__main__":
    # 配置路径
    source_root = r'F:\Project\mid\德国数据集\领域泛化\PUdata_1'
    target_root = r'F:\Project\mid\德国数据集\领域泛化\PUdata_1_csv'

    print("PUdata终极版MAT到CSV转换工具")
    print("=" * 60)
    print(f"源目录: {source_root}")
    print(f"目标目录: {target_root}")
    print("=" * 60)

    # 检查依赖
    try:
        import scipy
        print(f"✅ scipy版本: {scipy.__version__}")
    except ImportError:
        print("❌ scipy未安装，请运行: pip install scipy numpy pandas")
        exit(1)

    # 执行转换
    success_count, fail_count = mat_to_csv_preserve_structure(source_root, target_root)

    # 成功后的提示
    if success_count > 0:
        print("\n" + "=" * 60)
        print("✨ 转换完成！")
        print(f"成功转换: {success_count} 个文件")
        print(f"转换失败: {fail_count} 个文件")
        print(f"所有CSV文件保存在: {target_root}")
        print("\n💡 提示：")
        print("1. 每个CSV文件包含振动信号数据")
        print("2. 失败的文件有_debug.txt调试信息")
        print("3. 如果仍有失败，可以根据_debug.txt信息进一步优化")
