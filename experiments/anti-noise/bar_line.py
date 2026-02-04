"""
抗噪声实验结果可视化 - IEEE 顶刊风格
生成 SVG 和 PNG 格式的结果图
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


def load_csv_results(csv_path):
    """加载 CSV 结果文件"""
    snr_levels = []
    accuracies = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            snr = row['SNR']
            # 转换 SNR 标签为数值（用于绘图）
            if snr == 'clean':
                snr_levels.append('Clean')
            else:
                snr_levels.append(snr.replace('dB', ''))
            accuracies.append(float(row['Avg']))

    return snr_levels, accuracies


def plot_anti_noise_results(csv_path, method_name="Ours", save_name=None):
    """
    绘制单方法抗噪声结果图
    """
    snr_levels, accuracies = load_csv_results(csv_path)

    # 创建图形
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # X轴位置
    x = np.arange(len(snr_levels))

    # 绘制折线图
    ax.plot(x, accuracies, 'o-', color='#1f77b4', linewidth=2,
            markersize=8, markerfacecolor='white', markeredgewidth=2,
            label=method_name)

    # 设置轴标签
    ax.set_xlabel('SNR (dB)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')

    # X轴刻度
    ax.set_xticks(x)
    ax.set_xticklabels(snr_levels)

    # Y轴范围（留出空间）
    y_min = max(0, min(accuracies) - 10)
    y_max = min(100, max(accuracies) + 5)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(MultipleLocator(10))

    # 网格
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # 图例
    ax.legend(loc='lower left', framealpha=0.9, edgecolor='gray')

    # 紧凑布局
    plt.tight_layout()

    # 保存
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

    return svg_path, png_path


def plot_multi_method_comparison(results_dict, save_name="anti_noise_comparison"):
    """
    绘制多方法对比图（IEEE顶刊标准风格）

    Args:
        results_dict: dict, {方法名: csv文件路径} 或 {方法名: (snr_levels, accuracies)}
    """
    # 定义颜色和标记样式（适合打印和彩色显示）
    styles = [
        {'color': '#d62728', 'marker': 'o', 'linestyle': '-'},   # 红色圆形
        {'color': '#1f77b4', 'marker': 's', 'linestyle': '--'},  # 蓝色方形
        {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},  # 绿色三角
        {'color': '#ff7f0e', 'marker': 'D', 'linestyle': ':'},   # 橙色菱形
        {'color': '#9467bd', 'marker': 'v', 'linestyle': '-'},   # 紫色倒三角
        {'color': '#8c564b', 'marker': 'p', 'linestyle': '--'},  # 棕色五边形
    ]

    fig, ax = plt.subplots(figsize=(5, 3.8))

    all_accs = []
    x = None

    for idx, (method_name, data) in enumerate(results_dict.items()):
        style = styles[idx % len(styles)]

        # 支持直接传入数据或CSV路径
        if isinstance(data, str):
            snr_levels, accuracies = load_csv_results(data)
        else:
            snr_levels, accuracies = data

        if x is None:
            x = np.arange(len(snr_levels))

        all_accs.extend(accuracies)

        ax.plot(x, accuracies,
                marker=style['marker'],
                linestyle=style['linestyle'],
                color=style['color'],
                linewidth=1.8,
                markersize=7,
                markerfacecolor='white',
                markeredgewidth=1.5,
                label=method_name)

    # 设置轴
    ax.set_xlabel('SNR (dB)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(snr_levels)

    # Y轴范围
    y_min = max(0, min(all_accs) - 10)
    y_max = min(100, max(all_accs) + 5)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(MultipleLocator(10))

    # 网格
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # 图例 - 放在图外或图内最佳位置
    ax.legend(loc='lower left', framealpha=0.95, edgecolor='gray',
              fancybox=False, ncol=1)

    plt.tight_layout()

    # 保存
    svg_path = os.path.join(OUTPUT_DIR, f"{save_name}.svg")
    png_path = os.path.join(OUTPUT_DIR, f"{save_name}.png")

    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"已保存: {svg_path}")
    print(f"已保存: {png_path}")

    return svg_path, png_path


def plot_bar_chart(csv_path, method_name="S-MID", save_name=None):
    """
    绘制柱状图风格的抗噪声结果（另一种常见展示方式）
    """
    snr_levels, accuracies = load_csv_results(csv_path)

    fig, ax = plt.subplots(figsize=(5, 3.5))

    x = np.arange(len(snr_levels))
    width = 0.6

    # 使用渐变色表示噪声强度
    colors = plt.cm.Blues(np.linspace(0.8, 0.3, len(snr_levels)))

    bars = ax.bar(x, accuracies, width, color=colors, edgecolor='black', linewidth=0.8)

    # 在柱子上方添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('SNR (dB)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(snr_levels)

    y_min = max(0, min(accuracies) - 15)
    ax.set_ylim(y_min, 105)

    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
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
    # 查找当前目录下的 CSV 文件
    csv_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv')]

    if not csv_files:
        print("未找到 CSV 结果文件！请先运行 test_model.py")
    else:
        for csv_file in csv_files:
            csv_path = os.path.join(OUTPUT_DIR, csv_file)
            print(f"\n处理: {csv_file}")
            print("-" * 40)

            # 1. 绘制折线图
            plot_anti_noise_results(csv_path, method_name="S-MID")

            # 2. 绘制柱状图
            plot_bar_chart(csv_path, method_name="S-MID")

        # 3. 如果需要多方法对比，可以这样使用：
        # plot_multi_method_comparison({
        #     'S-MID (Ours)': 'anti_noise_PU.csv',
        #     'Baseline': ([...], [...]),  # 或直接传入数据
        # })

        print("\n" + "=" * 40)
        print("所有图表生成完成！")
