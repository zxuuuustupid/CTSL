import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from glob import glob


# ================= 1. 顶刊绘图风格配置 =================
def set_pub_style():
    """配置 Matplotlib 以符合顶刊发表标准"""
    plt.rcParams.update({
        # 字体设置
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 12,

        # 线条与轴
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        # 输出
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

set_pub_style()


# ================= 2. 配置四个PU任务 =================
# 任务配置: (任务名, 日志目录模式, 测试工况)
tasks = [
    ("PU123", "train_1_meta_2_3", "WC4"),
    ("PU124", "train_1_meta_2_4", "WC3"),
    ("PU134", "train_1_meta_3_4", "WC2"),
    ("PU234", "train_2_meta_3_4", "WC1"),
]

log_root = "log/PU"


# ================= 3. 辅助函数 =================
def get_latest_log_dir(pattern):
    """获取最新的日志目录 (按时间戳排序)"""
    dirs = glob(os.path.join(log_root, f"{pattern}_*"))
    if not dirs:
        raise FileNotFoundError(f"未找到匹配的日志目录: {pattern}")
    # 按目录名排序 (时间戳在最后), 取最新的
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


def plot_confusion_matrix(ax, cm, title, class_names=None):
    """在子图上绘制3D立体混淆矩阵"""
    num_classes = cm.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # 归一化 (行归一化，表示召回率)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6) * 100

    # 创建网格
    xpos, ypos = np.meshgrid(np.arange(num_classes), np.arange(num_classes))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos, dtype=float)

    # 柱体尺寸
    dx = dy = 0.6
    dz = cm_normalized.flatten()

    # 颜色映射 - 使用 viridis (越低越蓝，越高越黄)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=100)
    colors = cmap(norm(dz))

    # 绘制3D柱状图
    bars = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, alpha=0.9,
                    edgecolor='gray', linewidth=0.3, zsort='average')

    # 在每个柱子顶部标注数值
    for i in range(len(dz)):
        if dz[i] > 5:  # 只标注大于5%的值，避免太密集
            ax.text(xpos[i] + dx/2, ypos[i] + dy/2, dz[i] + 2,
                   f'{dz[i]:.0f}', ha='center', va='bottom',
                   fontsize=7, fontweight='bold', color='black')

    # 设置坐标轴
    ax.set_xticks(np.arange(num_classes) + dx/2)
    ax.set_yticks(np.arange(num_classes) + dy/2)
    ax.set_xticklabels(class_names, fontsize=9, rotation=0)
    ax.set_yticklabels(class_names, fontsize=9)

    # 标签
    ax.set_xlabel('Predicted', fontsize=11, labelpad=8)
    ax.set_ylabel('True', fontsize=11, labelpad=8)
    ax.set_zlabel('Recall (%)', fontsize=11, labelpad=5)

    # Z轴范围
    ax.set_zlim(0, 110)

    # 标题
    ax.set_title(title, fontsize=13, fontweight='bold', pad=0)

    # 调整视角
    ax.view_init(elev=25, azim=45)

    # 返回用于创建colorbar的映射
    return cmap, norm


# ================= 4. 主绘图逻辑 =================
def main():
    # 创建 2x2 画布，使用3D子图
    fig = plt.figure(figsize=(15, 13))

    # 类别名称 (PU数据集8类)
    class_names = [f"C{i}" for i in range(8)]

    cmap_used = None
    norm_used = None

    for idx, (task_name, pattern, wc) in enumerate(tasks):
        # 创建3D子图
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')

        try:
            # 获取最新日志目录
            log_dir = get_latest_log_dir(pattern)
            print(f"[{task_name}] 加载日志: {log_dir}")

            # 加载混淆矩阵
            cm, acc = load_confusion_matrix(log_dir, wc)

            # 绘制3D混淆矩阵
            title = f"{task_name} (Test: {wc}, Acc: {acc:.1f}%)"
            cmap_used, norm_used = plot_confusion_matrix(ax, cm, title, class_names)

        except FileNotFoundError as e:
            print(f"[{task_name}] 错误: {e}")
            ax.set_title(f"{task_name} - 数据缺失", fontsize=14, color='red')

    # 添加全局Colorbar
    if cmap_used is not None and norm_used is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap_used, norm=norm_used)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Recall (%)', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

    plt.subplots_adjust(right=0.9, wspace=0.1, hspace=0.15)

    # ================= 5. 保存图片 =================
    output_dir = "experiments/confusion-matrix"
    os.makedirs(output_dir, exist_ok=True)

    # 保存为 PDF
    pdf_path = os.path.join(output_dir, "Fig_PU_Confusion_Matrix.pdf")
    plt.savefig(pdf_path, format='pdf', dpi=300)

    # 保存为 SVG
    svg_path = os.path.join(output_dir, "Fig_PU_Confusion_Matrix.svg")
    plt.savefig(svg_path, format='svg', dpi=300)

    # 保存为高清 PNG
    png_path = os.path.join(output_dir, "Fig_PU_Confusion_Matrix.png")
    plt.savefig(png_path, format='png', dpi=300)

    print(f"\n混淆矩阵绘制完成！已保存三种格式：")
    print(f"1. [PDF] {pdf_path}")
    print(f"2. [SVG] {svg_path}")
    print(f"3. [PNG] {png_path}")

    plt.show()


if __name__ == "__main__":
    main()
