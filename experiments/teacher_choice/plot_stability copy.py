"""
教师网络选择稳定性分析 - 箱线图可视化
IEEE 顶刊风格
"""
import os
from tkinter import font
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# IEEE 顶刊风格设置
# IEEE 顶刊风格设置
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',  # <--- 【新增】关键修改：将数学公式字体设为 stix (类似 Times New Roman)
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
})

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_teacher_stability():
    # 读取数据
    csv_path = os.path.join(OUTPUT_DIR, 'result.csv')
    df = pd.read_csv(csv_path)

    # 按 Test_WC 分组
    groups = df.groupby('Test_WC')['Total_Accuracy'].apply(list).to_dict()

    # 排序（WC1, WC2, WC3, WC4）
    wc_order = ['WC1', 'WC2', 'WC3', 'WC4']
    data = [groups[wc] for wc in wc_order]

    # 四种不同的科研配色（低饱和度蓝红绿紫）
    colors = ['#7d6b91', '#b85450', '#5a8f7b', '#4a6fa5']  # 灰蓝、灰红、灰绿、灰紫
    edge_colors = ['#5a4a6b', '#8b3a36', '#3d6354','#2d4a6f' ]

    # 创建图形
    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=300)

    # 绘制箱线图
    bp = ax.boxplot(data,
                    positions=range(1, len(wc_order) + 1),
                    widths=0.5,
                    patch_artist=True,
                    showmeans=False,
                    medianprops=dict(color='#2c3e50', linewidth=1.5),
                    whiskerprops=dict(color='#2c3e50', linewidth=1.2),
                    capprops=dict(color='#2c3e50', linewidth=1.2),
                    flierprops=dict(marker='o', markerfacecolor='#e74c3c',
                                    markeredgecolor='#c0392b', markersize=5))

    # 为每个箱子设置不同颜色
    for patch, color, edge in zip(bp['boxes'], colors, edge_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(edge)
        patch.set_linewidth(1.2)
        patch.set_alpha(0.75)

    # 添加数据点（抖动显示，与箱子颜色对应）
    for i, (d, color) in enumerate(zip(data, edge_colors)):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.8, color=color, s=40, zorder=5,
                   edgecolors='white', linewidths=0.8)

    # 设置坐标轴
    ax.set_xticks(range(1, len(wc_order) + 1))
    # 使用 LaTeX 公式显示 T_1^C, T_2^C, T_3^C, T_4^C
    ax.set_xticklabels([r'$T_1^C$', r'$T_2^C$', r'$T_3^C$', r'$T_4^C$'], fontsize=12, fontfamily='Times New Roman')
    ax.set_xlabel('Test Task', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')

    # Y轴范围
    ax.set_ylim(60, 100)

    # 移除上方和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 网格线
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # 保存
    svg_path = os.path.join(OUTPUT_DIR, 'teacher_stability_boxplot.svg')
    pdf_path = os.path.join(OUTPUT_DIR, 'teacher_stability_boxplot.pdf')
    png_path = os.path.join(OUTPUT_DIR, 'teacher_stability_boxplot.png')

    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"已保存: {svg_path}")
    print(f"已保存: {pdf_path}")
    print(f"已保存: {png_path}")

    # 打印统计信息
    print("\n" + "=" * 50)
    print("稳定性统计分析")
    print("=" * 50)
    for wc, d in zip(wc_order, data):
        print(f"{wc}: Mean={np.mean(d):.2f}%, Std={np.std(d):.2f}%, Range={max(d)-min(d):.2f}%")


if __name__ == "__main__":
    plot_teacher_stability()
