"""
合并重复任务结果
计算 (max + min) / 2 ± (max - min) / 2
"""
import os
import csv
from collections import defaultdict

INPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'error_results.csv')
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'error_results_merged.csv')


def main():
    # 读取原始数据
    # {(Dataset, Task): [Avg1, Avg2, Avg3]}
    task_results = defaultdict(list)

    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row['Dataset']
            task = row['Task']
            avg = float(row['Avg'])
            task_results[(dataset, task)].append(avg)

    # 计算合并结果
    merged_results = []

    for (dataset, task), accs in sorted(task_results.items()):
        if len(accs) >= 2:
            max_acc = max(accs)
            min_acc = min(accs)
            center = (max_acc + min_acc) / 2
            error = (max_acc - min_acc) / 2
        else:
            center = accs[0]
            error = 0

        merged_results.append({
            'Dataset': dataset,
            'Task': task,
            'Accuracy': f"{center:.2f}±{error:.2f}",
            'Center': center,
            'Error': error,
            'N': len(accs),
            'Raw': accs
        })

    # 写入 CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Dataset', 'Task', 'Accuracy', 'N']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in merged_results:
            writer.writerow({
                'Dataset': r['Dataset'],
                'Task': r['Task'],
                'Accuracy': r['Accuracy'],
                'N': r['N']
            })

    print(f"已保存: {OUTPUT_CSV}")

    # 打印结果
    print("\n" + "=" * 60)
    print("合并结果（平均值 ± 误差）")
    print("=" * 60)

    current_dataset = None
    for r in merged_results:
        if r['Dataset'] != current_dataset:
            current_dataset = r['Dataset']
            print(f"\n[{current_dataset}]")
        print(f"  {r['Task']}: {r['Accuracy']} (n={r['N']})")


if __name__ == "__main__":
    main()
