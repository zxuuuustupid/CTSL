"""
训练噪声实验结果可视化 - IEEE 顶刊风格
针对 train_noise_*.csv 文件
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator

# 使用 IEEE 风格设置
matplotlib.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_train_noise_csv(csv_path):
    """加载训练噪声 CSV 结果文件"""
    snr_levels = []
    accuracies = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            snr = row['Train_SNR']
            # 转换 SNR 标签
            if snr == 'clean':
                snr_levels.append('Clean')
            else:
                snr_levels.append(snr.replace('dB', ''))
            accuracies.append(float(row['Avg']))

    return snr_levels, accuracies


def plot_train_noise_line(csv_path, method_name="S-MID", save_name=None):
    """绘制训练噪声折线图"""
    snr_levels, accuracies = load_train_noise_csv(csv_path)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    x = np.arange(len(snr_levels))

    ax.plot(x, accuracies, 'o-', color='#c0392b', linewidth=2,
            markersize=8, markerfacecolor='white', markeredgewidth=2,
            label=method_name)

    ax.set_xlabel('Training SNR (dB)', fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(snr_levels)

    y_min = max(0, min(accuracies) - 10)
    y_max = min(100, max(accuracies) + 5)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(MultipleLocator(10))

    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', framealpha=0.9, edgecolor='gray')

    plt.tight_layout()

    if save_name is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        save_name = base_name

    svg_path = os.path.join(OUTPUT_DIR, f"{save_name}.svg")
    png_path = os.path.join(OUTPUT_DIR, f"{save_name}.png")

    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"已保存: {svg_path}")
    print(f"已保存: {png_path}")


def plot_train_noise_bar(csv_path, method_name="S-MID", save_name=None):
    """绘制训练噪声柱状图+折线图组合"""
    snr_levels, accuracies = load_train_noise_csv(csv_path)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(snr_levels))
    width = 0.6

    # 使用红色渐变（科研常见配色）
    colors = plt.cm.Reds(np.linspace(0.75, 0.35, len(snr_levels)))

    # 柱状图
    bars = ax.bar(x, accuracies, width, color=colors, edgecolor='#922b21',
                  linewidth=0.8, alpha=0.85, zorder=2)

    # 折线图
    ax.plot(x, accuracies, 'o-', color='#922b21', linewidth=2,
            markersize=7, markerfacecolor='white', markeredgecolor='#922b21',
            markeredgewidth=2, zorder=3)

    # 数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Training SNR (dB)', fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(snr_levels)

    y_min = max(0, min(accuracies) - 15)
    ax.set_ylim(y_min, 105)

    ax.grid(True, axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_name is None:
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        save_name = f"{base_name}_bar"

    svg_path = os.path.join(OUTPUT_DIR, f"{save_name}.svg")
    png_path = os.path.join(OUTPUT_DIR, f"{save_name}.png")

    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"已保存: {svg_path}")
    print(f"已保存: {png_path}")


if __name__ == "__main__":
    # 查找训练噪声 CSV 文件
    csv_files = [f for f in os.listdir(OUTPUT_DIR)
                 if f.startswith('train_noise_') and f.endswith('.csv')]

    if not csv_files:
        print("未找到 train_noise_*.csv 结果文件！请先运行 train_with_noise.py")
    else:
        for csv_file in csv_files:
            csv_path = os.path.join(OUTPUT_DIR, csv_file)
            print(f"\n处理: {csv_file}")
            print("-" * 40)

            # 1. 折线图
            plot_train_noise_line(csv_path, method_name="S-MID")

            # 2. 柱状图+折线图
            plot_train_noise_bar(csv_path, method_name="S-MID")

        print("\n" + "=" * 40)
        print("训练噪声图表生成完成！")
