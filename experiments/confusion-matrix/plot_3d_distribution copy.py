import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from glob import glob


# ================= 1. 顶刊绘图风格配置 =================
def set_pub_style():
    """配置 Matplotlib 以符合顶刊发表标准"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 14,
        'axes.linewidth': 1.5,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

set_pub_style()


# ================= 2. 配置四个PU任务 =================
# 任务名使用LaTeX格式
tasks = [
    (r"${T}_{1}^{C}$", "train_1_meta_2_3", "WC4"),
    (r"${T}_{2}^{C}$", "train_1_meta_2_4", "WC3"),
    (r"${T}_{3}^{C}$", "train_1_meta_3_4", "WC2"),
    (r"${T}_{4}^{C}$", "train_2_meta_3_4", "WC1"),
]

# PU数据集8类标签
PU_CLASS_NAMES = ['K001', 'KA15', 'KA04', 'KI18', 'KI21', 'KB27', 'KB23', 'KB24']

log_root = "log/PU"


# ================= 3. 辅助函数 =================
def get_latest_log_dir(pattern):
    """获取最新的日志目录"""
    dirs = glob(os.path.join(log_root, f"{pattern}_*"))
    if not dirs:
        raise FileNotFoundError(f"未找到匹配的日志目录: {pattern}")
    dirs.sort()
    return dirs[-1]


def load_confusion_matrix(log_dir, wc):
    """从 JSON 文件加载混淆矩阵"""
    json_path = os.path.join(log_dir, f"metrics_{wc}.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到指标文件: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    cm = np.array(data['Confusion_Matrix'])
    acc = data['Total_Accuracy']
    return cm, acc


# 顶刊蓝色系配色
from matplotlib.colors import LinearSegmentedColormap

# 创建顶刊风格蓝色渐变 (浅蓝->深蓝)
blue_colors = ['#E6F2FF', '#B3D9FF', '#66B3FF', '#3399FF', '#0066CC', "#024C96"]
blue_cmap = LinearSegmentedColormap.from_list('pub_blues', blue_colors, N=256)


def plot_3d_bar(ax, data, title, x_labels=None, y_labels=None, cmap=None):
    """在子图上绘制3D柱状分布图"""
    rows, cols = data.shape

    if x_labels is None:
        x_labels = [f"C{i}" for i in range(cols)]
    if y_labels is None:
        y_labels = [f"C{i}" for i in range(rows)]

    # 归一化
    data_normalized = data.astype('float') / (data.sum(axis=1, keepdims=True) + 1e-6) * 100

    # 创建网格
    xpos, ypos = np.meshgrid(np.arange(cols), np.arange(rows))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos, dtype=float)

    dx = dy = 0.6
    dz = data_normalized.flatten()

    # 使用顶刊蓝色配色
    colormap = blue_cmap
    norm = plt.Normalize(vmin=0, vmax=100)
    colors = colormap(norm(dz))

    # 绘制
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.9,
             edgecolor='gray', linewidth=0.3, zsort='average')

    # 柱顶标注
    for i in range(len(dz)):
        if dz[i] > 5:
            ax.text(xpos[i] + dx/2, ypos[i] + dy/2, dz[i] + 3,
                   f'{dz[i]:.0f}', ha='center', va='bottom',
                   fontsize=16, fontweight='bold', color='#555555')  # 浅灰色

    # 坐标轴
    # ax.set_xticks(np.arange(cols) + dx/2)
    # ax.set_yticks(np.arange(rows) + dy/2)

    # ================= 核心修改：让 True 和 Predicted 也倾斜 =================

    # 1. 设置刻度标签 (Ticks) - 保持之前的倾斜
    ax.set_xticks(np.arange(cols) + dx/2)
    ax.set_yticks(np.arange(rows) + dy/2)

    # 2. 设置轴标题 (Labels) - 新增 rotation 参数
    # Predicted (X轴): 向左倾斜，为了贴合 X 轴的视觉延伸线
    ax.set_xlabel('Predicted', fontsize=20, fontweight='bold', labelpad=12, rotation=-15)

    # True (Y轴): 向右倾斜，为了贴合 Y 轴的视觉延伸线
    ax.set_ylabel('True', fontsize=20, fontweight='bold', labelpad=12, rotation=40)

    # ======================================================================




    # ax.set_xticklabels(x_labels, fontsize=9)
    # ax.set_yticklabels(y_labels, fontsize=9)


        # 调整 X轴 (Predicted): 负角度旋转，向右对齐
    ax.set_xticklabels(x_labels, fontsize=12, rotation=30, ha='left', va='center')



    ax.tick_params(axis='x', pad=8)



    # 调整 Y轴 (True): 正角度旋转，向左对齐 (这样看起来就像贴在轴线上)
    ax.set_yticklabels(y_labels, fontsize=12, rotation=-30, ha='left', va='center')


    ax.tick_params(axis='y', pad=1)

    # 调整轴标题的距离 (labelpad)，防止被旋转后的标签挡住
    # ax.set_xlabel('Predicted', fontsize=16, fontweight='bold', labelpad=15)
    # ax.set_ylabel('True', fontsize=16, fontweight='bold', labelpad=15)

    # ax.set_xlabel('Predicted', fontsize=16, fontweight='bold', labelpad=8)
    # ax.set_ylabel('True', fontsize=16, fontweight='bold', labelpad=8)
    ax.set_zlabel('Recall (%)', fontsize=20, fontweight='bold', labelpad=6)
    ax.set_zlim(0, 110)

    # ax.set_title(title, fontsize=14, fontweight='bold', pad=8)
    ax.view_init(elev=30, azim=135)  # 调整视角

    # 去掉背景网格
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)

    return colormap, norm


# ================= 4. 主绘图逻辑 (2x2 四张图) =================
def main():
    fig = plt.figure(figsize=(16, 14))

    cmap_used, norm_used = None, None

    for idx, (task_name, pattern, wc) in enumerate(tasks):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')

        try:
            log_dir = get_latest_log_dir(pattern)
            print(f"[{task_name}] 加载日志: {log_dir}")

            cm, acc = load_confusion_matrix(log_dir, wc)
            # title = f"{task_name} (Test: {wc}, Acc: {acc:.1f}%)"
            title = task_name
            cmap_used, norm_used = plot_3d_bar(ax, cm, title, PU_CLASS_NAMES, PU_CLASS_NAMES)

        except FileNotFoundError as e:
            print(f"[{task_name}] 错误: {e}")
            ax.set_title(f"{task_name} - 数据缺失", fontsize=12, color='red')

    # 添加全局Colorbar
    if cmap_used and norm_used:
        sm = plt.cm.ScalarMappable(cmap=cmap_used, norm=norm_used)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Recall (%)', fontsize=14, fontweight='bold')
        cbar.ax.tick_params(labelsize=12)

    plt.subplots_adjust(right=0.9, wspace=0.05, hspace=0.1)

    # 保存
    output_dir = "experiments/confusion-matrix"
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(os.path.join(output_dir, "Fig_PU_3D_Distribution.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "Fig_PU_3D_Distribution.pdf"), dpi=300)
    plt.savefig(os.path.join(output_dir, "Fig_PU_3D_Distribution.svg"), dpi=300)

    print("\n3D分布图绘制完成！已保存到 experiments/confusion-matrix/")
    plt.show()


if __name__ == "__main__":
    main()
