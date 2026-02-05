"""
从 error/log 文件夹提取实验结果
对每个任务取最近三次实验的 Total_Accuracy
输出到 CSV 文件
"""
import os
import json
import re
from datetime import datetime
from collections import defaultdict
import csv

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_folder_name(folder_name):
    """
    解析文件夹名，提取任务名和时间戳
    例如：train_1_meta_2_3_20260205_003713
    返回：(task_name, timestamp)
    """
    # 匹配最后的时间戳 YYYYMMDD_HHMMSS
    match = re.match(r'^(.+)_(\d{8}_\d{6})$', folder_name)
    if match:
        task_name = match.group(1)
        timestamp_str = match.group(2)
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            return task_name, timestamp
        except ValueError:
            return None, None
    return None, None


def extract_total_accuracy(json_path):
    """从 metrics JSON 文件提取 Total_Accuracy"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('Total_Accuracy', None)
    except Exception as e:
        print(f"读取失败: {json_path}, 错误: {e}")
        return None


def main():
    # 存储结果：{(数据集, 任务名): [(时间戳, {WC: acc, ...}), ...]}
    results = defaultdict(list)

    # 遍历数据集
    datasets = [d for d in os.listdir(LOG_DIR)
                if os.path.isdir(os.path.join(LOG_DIR, d))]

    for dataset in sorted(datasets):
        dataset_path = os.path.join(LOG_DIR, dataset)

        # 遍历任务文件夹
        task_folders = [f for f in os.listdir(dataset_path)
                        if os.path.isdir(os.path.join(dataset_path, f))]

        for folder in task_folders:
            task_name, timestamp = parse_folder_name(folder)
            if task_name is None:
                continue

            folder_path = os.path.join(dataset_path, folder)

            # 查找所有 metrics_WC*.json 文件
            wc_accuracies = {}
            for file in os.listdir(folder_path):
                if file.startswith('metrics_WC') and file.endswith('.json'):
                    wc_match = re.match(r'metrics_(WC\d+)\.json', file)
                    if wc_match:
                        wc_name = wc_match.group(1)
                        json_path = os.path.join(folder_path, file)
                        acc = extract_total_accuracy(json_path)
                        if acc is not None:
                            wc_accuracies[wc_name] = acc

            if wc_accuracies:
                results[(dataset, task_name)].append((timestamp, wc_accuracies, folder))

    # 对每个任务，取最近三次实验
    final_results = []

    for (dataset, task_name), experiments in sorted(results.items()):
        # 按时间戳降序排序，取最近三次
        experiments_sorted = sorted(experiments, key=lambda x: x[0], reverse=True)
        latest_three = experiments_sorted[:3]

        for timestamp, wc_accs, folder in latest_three:
            # 计算平均准确率
            if wc_accs:
                avg_acc = sum(wc_accs.values()) / len(wc_accs)
            else:
                avg_acc = 0

            # 获取所有 WC 名称并排序
            wc_names = sorted(wc_accs.keys(), key=lambda x: int(x.replace('WC', '')))

            final_results.append({
                'Dataset': dataset,
                'Task': task_name,
                'Timestamp': timestamp.strftime('%Y%m%d_%H%M%S'),
                'WC_Accuracies': wc_accs,
                'WC_Names': wc_names,
                'Avg_Accuracy': avg_acc,
                'Folder': folder
            })

    # 收集所有可能的 WC 列
    all_wc_names = set()
    for r in final_results:
        all_wc_names.update(r['WC_Names'])
    all_wc_names = sorted(all_wc_names, key=lambda x: int(x.replace('WC', '')))

    # 写入 CSV
    csv_path = os.path.join(OUTPUT_DIR, 'error_results.csv')

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        # 表头
        fieldnames = ['Dataset', 'Task', 'Timestamp'] + all_wc_names + ['Avg']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in final_results:
            row = {
                'Dataset': r['Dataset'],
                'Task': r['Task'],
                'Timestamp': r['Timestamp'],
                'Avg': f"{r['Avg_Accuracy']:.2f}"
            }
            # 填充各 WC 列
            for wc in all_wc_names:
                if wc in r['WC_Accuracies']:
                    row[wc] = f"{r['WC_Accuracies'][wc]:.2f}"
                else:
                    row[wc] = ''

            writer.writerow(row)

    print(f"已保存: {csv_path}")

    # 打印统计
    print("\n" + "=" * 60)
    print("提取结果统计")
    print("=" * 60)
    print(f"数据集数量: {len(datasets)}")
    print(f"任务数量: {len(results)}")
    print(f"记录数量: {len(final_results)}")

    # 按数据集分组显示
    current_dataset = None
    for r in final_results:
        if r['Dataset'] != current_dataset:
            current_dataset = r['Dataset']
            print(f"\n[{current_dataset}]")
        print(f"  {r['Task']} ({r['Timestamp']}): Avg={r['Avg_Accuracy']:.2f}%")


if __name__ == "__main__":
    main()
