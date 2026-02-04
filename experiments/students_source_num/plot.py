from tkinter import font
import matplotlib.pyplot as plt
import numpy as np
import os

# 0. 确保保存路径存在
save_dir = 'experiments/students_source_num'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 1. 准备数据
source_counts = ['2', '3', '4', '5', '6', '7']
accuracies = [93.71, 99.8, 98.75, 99.83, 100.0, 100.0]

# 2. 顶刊风格全局配置 (Times New Roman + MathJax 风格)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 10.5,
    'axes.linewidth': 1.0,        # 坐标轴线宽
    'axes.labelsize': 11,         # 轴标题字号
    'xtick.labelsize': 10,        # X轴刻度字号
    'ytick.labelsize': 10,        # Y轴刻度字号
    'xtick.direction': 'in',      # 刻度朝内
    'ytick.direction': 'in',      # 刻度朝内
    'xtick.major.size': 4,        # 刻度线长度
    'ytick.major.size': 4,
    'figure.dpi': 300,            # 默认分辨率
    'savefig.dpi': 300            # 保存分辨率
})

# 创建画布 (宽度 5-6 英寸适合论文单栏或半页)
fig, ax = plt.subplots(figsize=(5.5, 4.2))

# 3. 绘制柱状图 (使用更沉稳的学术蓝)
# 颜色参考：Science/Nature 常用色系
bar_color = '#4a7ebb'  # 柔和的深蓝色
bar_width = 0.55

bars = ax.bar(source_counts, accuracies,
              color=bar_color,
              width=bar_width,
              edgecolor='black',    # 加上极细的黑边增加质感
              linewidth=0.5,
              alpha=0.85,           # 轻微透明度
              zorder=2)             # 位于网格之上

# 4. 叠加趋势线
line_color = '#d62728' # 砖红色，形成对比，或使用深灰 '#333333'
ax.plot(np.arange(len(source_counts)), accuracies,
        color=line_color,
        linewidth=1.5,
        marker='o',
        markersize=6,
        markerfacecolor='white',   # 白心圆点
        markeredgecolor=line_color,
        markeredgewidth=1.2,
        linestyle='--',            # 虚线表示趋势
        dashes=(5, 3),             # 虚线样式
        zorder=3)

# 5. 数值标注
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    # 动态调整文本位置，对于100%防止溢出
    xy_text = (0, 4) if acc < 99.9 else (0, 3)

    ax.annotate(f'{acc:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=xy_text,
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9,
                color='black',
                # fontweight='normal'
                fontweight='bold'
                ) # 保持字体清爽，不必粗体

# 6. 坐标轴精细设置
ax.set_ylabel('Accuracy (%)', labelpad=8)
ax.set_xlabel('Number of Source Domains', labelpad=8)

# Y轴范围控制：给予上方一定留白
y_min = 92
y_max = 101.5
ax.set_ylim(y_min, y_max)

# Y轴刻度：确保不显示超过100的刻度
ticks = np.arange(92, 101, 2) # 2% 为间隔更清爽
ax.set_yticks(ticks)

# 美化边框 (Spines)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# 稍微加粗左和下边框
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)

# 添加水平网格线 (灰色虚线，置于底层)
ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.3, zorder=0)

# 7. 布局调整与保存
plt.tight_layout()

# 导出文件
svg_path = os.path.join(save_dir, 'source_domain_count_impact.svg')
pdf_path = os.path.join(save_dir, 'source_domain_count_impact.pdf')

plt.savefig(svg_path, format='svg', bbox_inches='tight')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')

print(f"Figure saved to:\n- {svg_path}\n- {pdf_path}")
plt.show()
