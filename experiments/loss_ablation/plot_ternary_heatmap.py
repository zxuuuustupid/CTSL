"""
三元热图可视化 - 损失函数权重对准确率的影响
简洁版 - 顶刊风格
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap

# 全局字体设置
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.linewidth': 1,
})


def ternary_to_cartesian(a, b, c):
    """三元坐标转笛卡尔坐标"""
    total = a + b + c
    if total == 0:
        return 0.5, np.sqrt(3) / 6
    a, b, c = a / total, b / total, c / total
    x = 0.5 * (2 * b + c)
    y = (np.sqrt(3) / 2) * c
    return x, y


def in_triangle(x, y):
    """检查点是否在三角形内"""
    v0, v1, v2 = np.array([0, 0]), np.array([1, 0]), np.array([0.5, np.sqrt(3)/2])

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    p = np.array([x, y])
    d1, d2, d3 = sign(p, v0, v1), sign(p, v1, v2), sign(p, v2, v0)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def plot_ternary_heatmap(csv_path: str, save_path: str = None):
    """绘制简洁版三元热图"""

    # 读取数据
    df = pd.read_csv(csv_path)
    ac, cc, lc = df['ac'].values, df['cc'].values, df['lc'].values
    accuracy = df['Accuracy'].values

    # 坐标转换
    coords = [ternary_to_cartesian(a, b, c) for a, b, c in zip(ac, cc, lc)]
    x_coords, y_coords = np.array([c[0] for c in coords]), np.array([c[1] for c in coords])

    # 创建图形
    fig, ax = plt.subplots(figsize=(6, 5.5))

    # 插值 - 使用三次样条插值获得更平滑的过渡
    triang = tri.Triangulation(x_coords, y_coords)
    xi, yi = np.linspace(0, 1, 400), np.linspace(0, np.sqrt(3)/2, 400)
    Xi, Yi = np.meshgrid(xi, yi)

    # 使用CubicTriInterpolator实现平滑插值
    interpolator = tri.CubicTriInterpolator(triang, accuracy, kind='geom')
    Zi = interpolator(Xi, Yi)

    # Mask三角形外区域
    for i in range(len(xi)):
        for j in range(len(yi)):
            if not in_triangle(Xi[j, i], Yi[j, i]):
                Zi[j, i] = np.nan

    # 配色 - 柔和自然的绿黄红渐变
    colors = ['#1a9850', '#66bd63', '#a6d96a', '#d9ef8b', '#ffffbf',
              '#fee08b', '#fdae61', '#f46d43', '#d73027']
    cmap = LinearSegmentedColormap.from_list('RdYlGn_r', colors, N=256)
    # 使用中位数作为色阶中心，让红绿占比更均衡
    vcenter = np.median(accuracy)
    norm = TwoSlopeNorm(vmin=accuracy.min(), vcenter=vcenter, vmax=accuracy.max())

    # 绘制热图 - 使用更多levels实现平滑过渡
    contourf = ax.contourf(Xi, Yi, Zi, levels=100, cmap=cmap, norm=norm)

    # 三角形边框
    triangle = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    ax.plot(triangle[:, 0], triangle[:, 1], 'k-', linewidth=1.5)

    # 数据点 - 弱化显示，仅作为参考
    ax.scatter(x_coords, y_coords, c='white', s=8, edgecolors='gray',
               linewidths=0.3, alpha=0.5, zorder=5)

    # 最优点
    best_idx = np.argmax(accuracy)
    ax.scatter(x_coords[best_idx], y_coords[best_idx],
               facecolors='none', edgecolors='k', s=100, linewidths=1.5, zorder=6)

    # 标注最优点的取值
    label_text = f'{accuracy[best_idx]:.2f}%\n({ac[best_idx]:.2f}, {cc[best_idx]:.2f}, {lc[best_idx]:.2f})'
    ax.annotate(label_text,
                xy=(x_coords[best_idx], y_coords[best_idx]),
                xytext=(x_coords[best_idx] - 0.20, y_coords[best_idx] + 0.12),
                fontsize=8, ha='center', va='bottom',
                bbox=dict(facecolor='white', edgecolor='k', linewidth=0.8, pad=2),
                arrowprops=dict(arrowstyle='-', color='k', lw=0.6))

    # 顶点标签
    ax.text(-0.06, -0.06, '$\\lambda_{ac}$', fontsize=12, ha='center', va='center')
    ax.text(1.06, -0.06, '$\\lambda_{cc}$', fontsize=12, ha='center', va='center')
    ax.text(0.5, np.sqrt(3)/2 + 0.06, '$\\lambda_{lc}$', fontsize=12, ha='center', va='center')

    # 刻度
    n_ticks = 5
    for i in range(1, n_ticks):
        t = i / n_ticks
        # 底边
        ax.plot([t, t], [0, -0.015], 'k-', lw=0.8)
        ax.text(t, -0.04, f'{t:.1f}', fontsize=7, ha='center', va='top', color='#444')

        # 左边
        x, y = ternary_to_cartesian(1-t, 0, t)
        ax.text(x - 0.04, y, f'{t:.1f}', fontsize=7, ha='right', va='center', color='#444')

        # 右边
        x, y = ternary_to_cartesian(0, 1-t, t)
        ax.text(x + 0.04, y, f'{t:.1f}', fontsize=7, ha='left', va='center', color='#444')

    # 网格线
    for i in range(1, n_ticks):
        t = i / n_ticks
        x1, y1 = ternary_to_cartesian(1-t, 0, t)
        x2, y2 = ternary_to_cartesian(0, 1-t, t)
        ax.plot([x1, x2], [y1, y2], color='gray', lw=0.3, alpha=0.5)

        x1, y1 = ternary_to_cartesian(t, 1-t, 0)
        x2, y2 = ternary_to_cartesian(t, 0, 1-t)
        ax.plot([x1, x2], [y1, y2], color='gray', lw=0.3, alpha=0.5)

        x1, y1 = ternary_to_cartesian(1-t, t, 0)
        x2, y2 = ternary_to_cartesian(0, t, 1-t)
        ax.plot([x1, x2], [y1, y2], color='gray', lw=0.3, alpha=0.5)

    # 颜色条
    cbar = plt.colorbar(contourf, ax=ax, shrink=0.7, aspect=20, pad=0.02)
    cbar.set_label('Accuracy (%)', fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # 设置
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, np.sqrt(3)/2 + 0.12)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()

    # 保存
    if save_path:
        base = save_path.rsplit('.', 1)[0]
        for fmt in ['png', 'pdf', 'svg']:
            plt.savefig(f'{base}.{fmt}', dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ {base}.{fmt}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "loss_results.csv")
    save_path = os.path.join(script_dir, "ternary_heatmap.png")
    plot_ternary_heatmap(csv_path, save_path)
