"""
分组柱状图可视化 - IEEE 顶刊风格
一个x点对应多个条形，用于对比不同场景/方法
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
    'axes.labelsize': 16,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
})

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_test_noise_csv(csv_path):
    """加载测试噪声 CSV（SNR 列）"""
    snr_levels = []
    accuracies = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            snr = row['SNR']
            if snr == 'clean':
                snr_levels.append('Clean')
            else:
                snr_levels.append(snr.replace('dB', ''))
            accuracies.append(float(row['Avg']))
    return snr_levels, accuracies


def load_train_noise_csv(csv_path):
    """加载训练噪声 CSV（Train_SNR 列）"""
    snr_levels = []
    accuracies = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            snr = row['Train_SNR']
            if snr == 'clean':
                snr_levels.append('Clean')
            else:
                snr_levels.append(snr.replace('dB', ''))
            accuracies.append(float(row['Avg']))
    return snr_levels, accuracies


# def plot_grouped_bar(data_dict, snr_labels, save_name="grouped_comparison"):
#     """
#     绘制分组柱状图

#     Args:
#         data_dict: dict, {场景名: [准确率列表]}
#         snr_labels: list, SNR 标签列表
#         save_name: str, 保存文件名
#     """
#     n_groups = len(snr_labels)
#     n_scenarios = len(data_dict)

#     # 配色与参考程序一致：蓝色（测试噪声）、红色（训练噪声）
#     # Test Noise: Blues 渐变, 边框 #2c3e50, 折线 #1a5276
#     # Train Noise: Reds 渐变, 边框 #922b21, 折线 #922b21
#     color_configs = [
#         {'cmap': plt.cm.Blues, 'edge': '#2c3e50', 'line': '#1a5276'},  # 蓝色系
#         {'cmap': plt.cm.Reds, 'edge': '#922b21', 'line': '#922b21'},   # 红色系
#     ]

#     fig, ax = plt.subplots(figsize=(7, 4))

#     # 柱宽和位置
#     bar_width = 0.35
#     x = np.arange(n_groups)

#     # 找到 Clean 的索引
#     clean_idx = snr_labels.index('Clean') if 'Clean' in snr_labels else -1

#     # 绘制每个场景的柱状图和折线图
#     scenario_list = list(data_dict.items())
#     for i, (scenario_name, accuracies) in enumerate(scenario_list):
#         cfg = color_configs[i % len(color_configs)]

#         # 渐变色
#         colors = cfg['cmap'](np.linspace(0.8, 0.3, n_groups))

#         # Clean 只画第一个场景（居中），其他 SNR 画分组
#         if i == 0:
#             # 第一个场景：Clean 居中，其他偏左
#             positions = []
#             plot_accs = []
#             plot_colors = []
#             for j in range(n_groups):
#                 if j == clean_idx:
#                     positions.append(x[j])  # Clean 居中
#                     plot_colors.append('#5a8f7b')  # Clean 用灰绿色表示干净数据
#                 else:
#                     positions.append(x[j] - bar_width / 2)  # 偏左
#                     plot_colors.append(colors[j])
#                 plot_accs.append(accuracies[j])

#             # Clean 柱子用特殊边框色
#             edge_colors_list = ['#3d6354' if j == clean_idx else cfg['edge'] for j in range(n_groups)]

#             bars = ax.bar(positions, plot_accs, bar_width,
#                           color=plot_colors,
#                           edgecolor=edge_colors_list,
#                           linewidth=0.8, alpha=0.85,
#                           label=scenario_name, zorder=2)

#             # 折线图
#             ax.plot(positions, plot_accs, 'o-', color=cfg['line'], linewidth=2,
#                     markersize=7, markerfacecolor='white', markeredgecolor=cfg['line'],
#                     markeredgewidth=2, zorder=3)

#             # 数值标签
#             for bar, acc in zip(bars, plot_accs):
#                 height = bar.get_height()
#                 ax.annotate(f'{acc:.1f}',
#                             xy=(bar.get_x() + bar.get_width() / 2, height),
#                             xytext=(0, 8),
#                             textcoords="offset points",
#                             ha='center', va='bottom', fontsize=7, fontweight='bold')
#         else:
#             # 其他场景：跳过 Clean 柱子，其他偏右
#             positions = []
#             plot_accs = []
#             plot_colors = []
#             for j in range(n_groups):
#                 if j == clean_idx:
#                     continue  # 跳过 Clean 柱子
#                 positions.append(x[j] + bar_width / 2)  # 偏右
#                 plot_accs.append(accuracies[j])
#                 plot_colors.append(colors[j])

#             bars = ax.bar(positions, plot_accs, bar_width,
#                           color=plot_colors,
#                           edgecolor=cfg['edge'],
#                           linewidth=0.8, alpha=0.85,
#                           label=scenario_name, zorder=2)

#             # 折线图：包含 Clean 点（连接到居中位置）
#             line_positions = []
#             line_accs = []
#             for j in range(n_groups):
#                 if j == clean_idx:
#                     line_positions.append(x[j])  # Clean 连到居中位置
#                 else:
#                     line_positions.append(x[j] + bar_width / 2)  # 偏右
#                 line_accs.append(accuracies[j])

#             ax.plot(line_positions, line_accs, 'o-', color=cfg['line'], linewidth=2,
#                     markersize=7, markerfacecolor='white', markeredgecolor=cfg['line'],
#                     markeredgewidth=2, zorder=3)

#             # 数值标签
#             for bar, acc in zip(bars, plot_accs):
#                 height = bar.get_height()
#                 ax.annotate(f'{acc:.1f}',
#                             xy=(bar.get_x() + bar.get_width() / 2, height),
#                             xytext=(0, 8),
#                             textcoords="offset points",
#                             ha='center', va='bottom', fontsize=7, fontweight='bold')

#     # 设置坐标轴
#     ax.set_xlabel('SNR (dB)', fontweight='bold')
#     ax.set_ylabel('Accuracy (%)', fontweight='bold')
#     ax.set_xticks(x)
#     ax.set_xticklabels(snr_labels)

#     # Y轴范围
#     all_accs = [acc for accs in data_dict.values() for acc in accs]
#     y_min = max(0, min(all_accs) - 15)
#     y_max = min(110, max(all_accs) + 15)
#     ax.set_ylim(y_min, y_max)

#     # 网格
#     ax.grid(True, axis='y', linestyle='--', alpha=0.4, zorder=0)
#     ax.set_axisbelow(True)

#     # 图例（自定义颜色）
#     import matplotlib.patches as mpatches
#     legend_handles = []
#     # Clean
#     if clean_idx != -1:
#         legend_handles.append(mpatches.Patch(facecolor='#5a8f7b', edgecolor='#3d6354',
#                                              label='Clean', linewidth=0.8))
#     # Test Noise
#     legend_handles.append(mpatches.Patch(facecolor=plt.cm.Blues(0.55), edgecolor='#2c3e50',
#                                          label='Target Noise', linewidth=0.8))
#     # Train Noise
#     legend_handles.append(mpatches.Patch(facecolor=plt.cm.Reds(0.55), edgecolor='#922b21',
#                                          label='Source Noise', linewidth=0.8))
#     ax.legend(handles=legend_handles, loc='lower left', framealpha=0.95, edgecolor='gray')

#     # 移除上方和右侧边框
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

#     plt.tight_layout()

#     # 保存
#     svg_path = os.path.join(OUTPUT_DIR, f"{save_name}.svg")
#     png_path = os.path.join(OUTPUT_DIR, f"{save_name}.png")
#     pdf_path = os.path.join(OUTPUT_DIR, f"{save_name}.pdf")

#     plt.savefig(svg_path, format='svg', bbox_inches='tight')
#     plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
#     plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
#     plt.close()

#     print(f"已保存: {svg_path}")
#     print(f"已保存: {png_path}")
#     print(f"已保存: {pdf_path}")


# def plot_grouped_bar(data_dict, snr_labels, save_name="grouped_comparison"):
def plot_grouped_bar(data_dict, snr_labels, save_name="grouped_comparison"):
    """
    绘制分组柱状图 (修复版 2.0：红线连接 Clean 点 + 差值标注)
    """
    n_groups = len(snr_labels)
    n_scenarios = len(data_dict)

    # 配色配置
    color_configs = [
        {'cmap': plt.cm.Blues, 'edge': '#2c3e50', 'line': '#1a5276'},  # 蓝色系
        {'cmap': plt.cm.Reds, 'edge': '#922b21', 'line': '#922b21'},   # 红色系
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    bar_width = 0.35
    x = np.arange(n_groups)

    clean_idx = snr_labels.index('Clean') if 'Clean' in snr_labels else -1
    scenario_coords = {}

    # === 1. 绘制柱状图 AND 折线图 ===
    scenario_list = list(data_dict.items())
    for i, (scenario_name, accuracies) in enumerate(scenario_list):
        cfg = color_configs[i % len(color_configs)]
        colors = cfg['cmap'](np.linspace(0.8, 0.3, n_groups))

        # --- 分别存储用于画柱子(bar)和画线(line)的数据 ---
        bar_pos, bar_heights, bar_colors, bar_edges = [], [], [], []
        line_pos, line_heights = [], [] # 【关键】：折线数据单独存

        coords = [] # 用于差值计算

        for j in range(n_groups):
            # 基础位置计算
            if j == clean_idx:
                pos = x[j] # Clean 永远居中
                col = '#5a8f7b'
                edge = '#3d6354'
            else:
                pos = x[j] - bar_width/2 if i == 0 else x[j] + bar_width/2
                col = colors[j]
                edge = cfg['edge']

            # --- 数据分流逻辑 ---

            # 1. 折线数据：所有点都要加，绝不跳过！
            line_pos.append(pos)
            line_heights.append(accuracies[j])

            # 2. 柱子数据：如果是第二个场景的 Clean，跳过不画柱子
            if not (j == clean_idx and i != 0):
                bar_pos.append(pos)
                bar_heights.append(accuracies[j])
                bar_colors.append(col)
                bar_edges.append(edge)

            # 3. 记录坐标用于差值
            coords.append((pos, accuracies[j]))

        scenario_coords[i] = coords

        # A. 画柱子 (使用 bar_pos)
        bars = ax.bar(bar_pos, bar_heights, bar_width,
                      color=bar_colors, edgecolor=bar_edges,
                      linewidth=0.8, alpha=0.85, label=scenario_name, zorder=2)

        # B. 画折线 (使用 line_pos - 这里包含了 Clean 点)
        ax.plot(line_pos, line_heights,
                marker='o', linestyle='-', linewidth=1.5,
                color=cfg['line'],
                markeredgecolor='white',
                markeredgewidth=1.0,
                markersize=6,
                label=None,
                zorder=3)

        # C. 标数值 (只标在柱子上)
        for bar, acc in zip(bars, bar_heights):
            height = bar.get_height()
            ax.annotate(f'{acc:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, fontweight='bold')

    # === 2. 添加差值标注 ===
    if n_scenarios == 2:
        coords1 = scenario_coords[0]
        coords2 = scenario_coords[1]

        for j in range(n_groups):
            if j == clean_idx: continue

            x1, y1 = coords1[j]
            x2, y2 = coords2[j]
            diff = y1 - y2

            text_y = max(y1, y2) + 5
            text_x = (x1 + x2) / 2

            text_color = '#d62728' if diff < 0 else '#2ca02c'
            sign = '+' if diff > 0 else ''

            ax.plot([x1, x1, x2, x2], [y1 + 1, max(y1, y2)+3, max(y1, y2)+3, y2 + 1],
                    color='gray', linewidth=1.5, alpha=0.5)

            ax.text(text_x, text_y, f"{sign}{diff:.1f}%",
                    ha='center', va='bottom', fontsize=12,
                    color=text_color, fontweight='bold')

    # === 设置坐标轴 ===
    ax.set_xlabel('SNR (dB)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(snr_labels)

    all_accs = [acc for accs in data_dict.values() for acc in accs]
    y_min = max(0, min(all_accs) - 10)
    y_max = min(104, max(all_accs) + 20)
    ax.set_ylim(y_min, y_max)

    ax.grid(True, axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # 图例
    import matplotlib.patches as mpatches
    legend_handles = []
    if clean_idx != -1:
        legend_handles.append(mpatches.Patch(facecolor='#5a8f7b', edgecolor='#3d6354', label='Clean'))

    names = list(data_dict.keys())
    legend_handles.append(mpatches.Patch(facecolor=plt.cm.Blues(0.55), edgecolor='#2c3e50', label=names[0]))
    if len(names) > 1:
        legend_handles.append(mpatches.Patch(facecolor=plt.cm.Reds(0.55), edgecolor='#922b21', label=names[1]))

    ax.legend(handles=legend_handles, loc='upper right', framealpha=0.95, ncol=3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # 保存
    svg_path = os.path.join(OUTPUT_DIR, f"{save_name}.svg")
    png_path = os.path.join(OUTPUT_DIR, f"{save_name}.png")

    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"已保存(修复连线版): {png_path}")

def plot_test_vs_train_noise(test_csv, train_csv, save_name="test_vs_train_noise"):
    """
    对比测试噪声和训练噪声的结果
    """
    snr_labels_test, acc_test = load_test_noise_csv(test_csv)
    snr_labels_train, acc_train = load_train_noise_csv(train_csv)

    # 使用测试噪声的 SNR 标签（假设两者 SNR 等级相同）
    data_dict = {
        'Target Noise': acc_test,
        'Source Noise': acc_train,
    }

    plot_grouped_bar(data_dict, snr_labels_test, save_name)


def plot_multi_scenario_bar(scenarios_dict, save_name="multi_scenario_comparison"):
    """
    绘制多场景对比分组柱状图

    Args:
        scenarios_dict: dict, {场景名: csv文件路径} 或 {场景名: (snr_labels, accuracies)}
    """
    data_dict = {}
    snr_labels = None

    for scenario_name, data in scenarios_dict.items():
        if isinstance(data, str):
            # 自动判断 CSV 类型
            with open(data, 'r', encoding='utf-8') as f:
                header = f.readline()
                if 'Train_SNR' in header:
                    labels, accs = load_train_noise_csv(data)
                else:
                    labels, accs = load_test_noise_csv(data)
        else:
            labels, accs = data

        data_dict[scenario_name] = accs
        if snr_labels is None:
            snr_labels = labels

    plot_grouped_bar(data_dict, snr_labels, save_name)


if __name__ == "__main__":
    # 示例用法
    print("=" * 50)
    print("分组柱状图生成工具")
    print("=" * 50)

    # 查找 CSV 文件
    test_csvs = [f for f in os.listdir(OUTPUT_DIR)
                 if f.startswith('anti_noise_') and f.endswith('.csv')]
    train_csvs = [f for f in os.listdir(OUTPUT_DIR)
                  if f.startswith('train_noise_') and f.endswith('.csv')]

    # 如果同时有测试噪声和训练噪声的结果，生成对比图
    if test_csvs and train_csvs:
        # 尝试匹配数据集
        for test_csv in test_csvs:
            # 提取数据集名称
            dataset = test_csv.replace('anti_noise_', '').replace('.csv', '')
            train_csv = f"train_noise_{dataset}.csv"

            if train_csv in train_csvs:
                print(f"\n生成对比图: {dataset}")
                print("-" * 40)
                plot_test_vs_train_noise(
                    os.path.join(OUTPUT_DIR, test_csv),
                    os.path.join(OUTPUT_DIR, train_csv),
                    save_name=f"noise_comparison_{dataset}"
                )

    # 也可以手动指定多个场景对比
    # plot_multi_scenario_bar({
    #     'S-MID': 'anti_noise_PU.csv',
    #     'Baseline': 'baseline_PU.csv',
    # }, save_name='method_comparison')

    print("\n" + "=" * 50)
    print("分组柱状图生成完成！")
