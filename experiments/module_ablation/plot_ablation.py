import matplotlib.pyplot as plt
import numpy as np
import os

# 0. 确保保存路径存在
save_dir = 'experiments/module_ablation'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 1. 准备数据
tasks = [r'$T_1^C$', r'$T_2^C$', r'$T_3^C$', r'$T_4^C$']
with_simulation = [84.1, 93.9, 73.3, 93.8]
without_simulation = [74.12, 90.44, 68.1, 93.2]

# 2. IEEE顶刊风格全局配置
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.labelsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

# 创建画布 (IEEE单栏宽度)
fig, ax = plt.subplots(figsize=(4.5, 3.5))

# 3. 柱状图配置
x = np.arange(len(tasks))
bar_width = 0.35

# 配色：浅蓝紫 + 浅黄
color_with = '#91aadc'      # 浅蓝紫 - Ours (完整方法)
color_without = '#ffe699'   # 浅黄色 - w/o Simulation (消融)

# 绘制分组柱状图
bars1 = ax.bar(x - bar_width/2, with_simulation, bar_width,
               label='Ours',
               color=color_with,
               edgecolor='white',
               linewidth=0.3,
               zorder=2)

bars2 = ax.bar(x + bar_width/2, without_simulation, bar_width,
               label='w/o Generalization Simulation',
               color=color_without,
               edgecolor='white',
               linewidth=0.3,
               zorder=2)

# 4. 数值标注
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.8,
            f'{height:.1f}', ha='center', va='bottom',
            fontsize=8, color='#5a7cb8', fontweight='medium')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.8,
            f'{height:.1f}', ha='center', va='bottom',
            fontsize=8, color='#c9a000', fontweight='medium')

# 5. 增益标注 (简洁数值)
for i in range(len(tasks)):
    diff = with_simulation[i] - without_simulation[i]
    if abs(diff) > 0.1:
        mid_x = x[i]
        max_h = max(with_simulation[i], without_simulation[i])
        sign = '+' if diff > 0 else ''
        # 简洁的增益标注
        ax.text(mid_x+0.2, max_h + 1.5, f'({sign}{diff:.2f})',
                ha='center', va='bottom',
                fontsize=7.5, color='#444444', style='italic')

# 6. 坐标轴配置
ax.set_ylabel('Accuracy (%)', labelpad=6)
ax.set_xlabel('Experimental Tasks', labelpad=6)
ax.set_xticks(x)
ax.set_xticklabels(tasks)

# Y轴范围
ax.set_ylim(60, 105)
ax.set_yticks(np.arange(60, 101, 10))

# 7. 美化边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#333333')
ax.spines['bottom'].set_color('#333333')

# 8. 网格线
ax.yaxis.grid(True, linestyle='--', which='major', color='#d0d0d0', linewidth=0.5, zorder=0)
ax.set_axisbelow(True)

# 9. 图例
legend = ax.legend(loc='upper left',
                   frameon=True,
                   framealpha=0.95,
                   edgecolor='#999999',
                   fontsize=9,
                   borderpad=0.3,
                   handlelength=1.0,
                   handletextpad=0.4)
legend.get_frame().set_linewidth(0.4)

# 10. 布局调整与保存
plt.tight_layout()

svg_path = os.path.join(save_dir, 'module_ablation.svg')
pdf_path = os.path.join(save_dir, 'module_ablation.pdf')
png_path = os.path.join(save_dir, 'module_ablation.png')

plt.savefig(svg_path, format='svg', bbox_inches='tight')
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)

print(f"Figure saved to:\n- {svg_path}\n- {pdf_path}\n- {png_path}")
# plt.show()
plt.close()
