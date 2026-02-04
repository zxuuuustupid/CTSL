"""
从 teacher_choice 实验文件夹提取 total_accuracy 数据
"""
import os
import json
import csv
import re

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_missing_wc(folder_name):
    """从文件夹名字中解析缺失的 WC 编号（即测试的 WC）"""
    # 格式: train_X_meta_Y_Z_时间戳
    # 例如: train_1_meta_2_3_xxx -> 缺失 4
    match = re.match(r'train_(\d+)_meta_(\d+)_(\d+)_', folder_name)
    if match:
        nums = {int(match.group(1)), int(match.group(2)), int(match.group(3))}
        all_nums = {1, 2, 3, 4}
        missing = all_nums - nums
        if missing:
            return list(missing)[0]
    return None

def main():
    results = []

    # 遍历所有子文件夹
    for folder in os.listdir(EXPERIMENT_DIR):
        folder_path = os.path.join(EXPERIMENT_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        # 查找 metrics 开头的 json 文件
        for f in os.listdir(folder_path):
            if f.startswith('metrics') and f.endswith('.json'):
                json_path = os.path.join(folder_path, f)
                with open(json_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                    total_acc = data.get('Total_Accuracy', None)

                    # 解析缺失的 WC
                    missing_wc = get_missing_wc(folder)

                    results.append({
                        'folder': folder,
                        'test_wc': missing_wc,
                        'total_accuracy': total_acc
                    })

    # 按缺失的 WC 编号排序
    results.sort(key=lambda x: (x['test_wc'] if x['test_wc'] else 99))

    # 写入 CSV
    csv_path = os.path.join(EXPERIMENT_DIR, 'result.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Folder', 'Test_WC', 'Total_Accuracy'])
        for r in results:
            writer.writerow([r['folder'], f"WC{r['test_wc']}", r['total_accuracy']])

    print(f"已保存至: {csv_path}")
    print(f"共提取 {len(results)} 条记录")

    # 打印结果
    print("\n" + "=" * 60)
    print(f"{'Test WC':<10} | {'Accuracy':<10} | Folder")
    print("-" * 60)
    for r in results:
        print(f"WC{r['test_wc']:<8} | {r['total_accuracy']:<10.2f} | {r['folder']}")

if __name__ == "__main__":
    main()
