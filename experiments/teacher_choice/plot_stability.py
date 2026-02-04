"""
教师网络选择稳定性分析 - 源工况着色版 (Source-WC Coloring)
【逻辑修正 - 逆序版】：
1. 映射关系调整：T1->WC4, T2->WC3, T3->WC2, T4->WC1。
2. 颜色逻辑：点的颜色代表它背后的【教师工况/源工况】。
3. 视觉风格：IEEE 顶刊半透明质感 + 深色描边。
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc

# --- IEEE 顶刊风格设置 ---
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'legend.fontsize': 10,
})

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def darken_color(color, amount=0.7):
    """辅助函数：加深颜色用于描边"""
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = mc.to_rgb(c)
    c = [max(0, x * amount) for x in c]
    return mc.to_hex(c)

def plot_teacher_stability_reverse_order():
    # --- 1. 数据准备 ---
    csv_path = os.path.join(OUTPUT_DIR, 'result.csv')
    if not os.path.exists(csv_path):
        print(f"错误：找不到文件 {csv_path}")
        return
    df = pd.read_csv(csv_path)

    # 确保有索引列
    if 'Source_Index' not in df.columns:
        df['Source_Index'] = df.groupby('Test_WC').cumcount()

    # 所有可能的工况定义
    all_wcs_sorted = ['WC1', 'WC2', 'WC3', 'WC4']

    # 【核心修正】：X轴的任务顺序 (从 WC4 到 WC1)
    # T1对应WC4, T2对应WC3...
    target_wc_order = ['WC4', 'WC3', 'WC2', 'WC1']

    np.random.seed(42)

    # --- 2. 绘图设置 ---

    # 定义 4 个工况的专属颜色
    # WC1:紫, WC2:红, WC3:绿, WC4:蓝
    wc_colors = {
        'WC1': '#88719d',
        'WC2': '#c8564b',
        'WC3': '#5a9c80',
        'WC4': '#5681b8'
    }

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)

    # =================================================================
    # 【绘制逻辑】
    # =================================================================

    jitter_amount = 0.1

    # 遍历任务 T1~T4 (对应 target_wc_order 里的 WC4~WC1)
    for i, target_wc in enumerate(target_wc_order):
        # 取出当前目标工况的数据
        group = df[df['Test_WC'] == target_wc]

        # 【逻辑推断】：
        # 如果当前测试 WC4，那么源工况一定是 WC1, WC2, WC3 (按顺序)
        possible_sources = sorted([w for w in all_wcs_sorted if w != target_wc])

        for _, row in group.iterrows():
            idx = row['Source_Index']

            # 找到对应的源工况颜色
            if idx < len(possible_sources):
                source_wc_name = possible_sources[idx] # 例如 'WC1'

                curr_fill = wc_colors[source_wc_name]
                curr_edge = darken_color(curr_fill, 0.7)

                # 水平微移
                offset = np.random.uniform(-jitter_amount, jitter_amount)

                ax.scatter(
                    x=i + offset,
                    y=row['Total_Accuracy'],
                    marker='o',
                    s=110,
                    facecolor=curr_fill,
                    edgecolor=curr_edge,
                    linewidth=1.0,
                    alpha=0.9,
                    zorder=10
                )

    # =================================================================

    # --- 3. 坐标轴与美化 ---
    ax.set_xticks(range(len(target_wc_order)))
    ax.set_xticklabels([r'$T_1^C$', r'$T_2^C$', r'$T_3^C$', r'$T_4^C$'], fontsize=14)

    # 这里的Label含义：测试任务 (目标域)
    ax.set_xlabel('Test Task', fontweight='bold', labelpad=10)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', labelpad=10)

    y_min = df['Total_Accuracy'].min()
    y_max = df['Total_Accuracy'].max()
    margin = (y_max - y_min) * 0.3
    ax.set_ylim(y_min - margin, y_max + margin)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', color='gray', linewidth=0.5, alpha=0.5, zorder=0)

    # --- 4. 图例 ---
    # 显示颜色代表哪个【源工况】
    legend_elements = []
    # 图例按 WC1~WC4 顺序排列
    for wc_name in all_wcs_sorted:
        color = wc_colors[wc_name]
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=color,
                       markeredgecolor=darken_color(color, 0.7),
                       label=f'{wc_name[1:]}',
                       markersize=9)
        )

    ax.legend(handles=legend_elements,
              title="Teacher Domain",
              loc='upper center',
              bbox_to_anchor=(0.5, 1.15),
              ncol=4,
              frameon=False,
              handletextpad=0.1)

    plt.tight_layout()

    # --- 5. 保存 ---
    save_name = 'teacher_stability_reverse_order'
    png_path = os.path.join(OUTPUT_DIR, f'{save_name}.png')
    pdf_path = os.path.join(OUTPUT_DIR, f'{save_name}.pdf')

    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"逆序逻辑版已保存至: {png_path}")

if __name__ == "__main__":
    plot_teacher_stability_reverse_order()
