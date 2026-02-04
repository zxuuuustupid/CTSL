"""
消融实验结果可视化 - IEEE 顶刊风格
水平条形图展示
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 准备数据（按准确率排序，最好的在最上面）
data = {
    'Ablation Setting': ['LC+CC+DC', 'LC+CC', 'LC+DC', 'LC', 'CC+DC', 'DC', 'CC'],
    'Accuracy': [93.94, 93.68, 92.25, 91.68, 89.25, 87.19, 86.56]
}
df = pd.DataFrame(data)
# 按准确率排序（从低到高，这样最高的显示在最上面）
df = df.sort_values('Accuracy', ascending=True).reset_index(drop=True)

# 2. 设置绘图风格
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 10,
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# 3. 配色方案：紫色系（SCI常见的专业配色）
# 完整模型用深紫色突出，其他用紫色渐变
n = len(df)
base_colors = plt.cm.Purples(np.linspace(0.35, 0.75, n))
# 找到完整模型 (LC+CC+DC) 的索引，用深紫色突出
full_idx = df[df['Ablation Setting'] == 'LC+CC+DC'].index[0]
colors = [base_colors[i] if i != full_idx else '#5b2c6f' for i in range(n)]

# 4. 开始绘图
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

# 绘制水平条形图
y_pos = np.arange(len(df))
bars = ax.barh(y_pos, df['Accuracy'], height=0.65, color=colors,
               edgecolor='#4a235a', linewidth=0.8, zorder=3)

# 5. 精细化调整
# 设置坐标轴范围
ax.set_xlim(84, 96)

# 添加数值标签（在条形右侧）
for i, (bar, acc) in enumerate(zip(bars, df['Accuracy'])):
    width = bar.get_width()
    # 完整模型用加粗显示
    weight = 'bold' if df.iloc[i]['Ablation Setting'] == 'LC+CC+DC' else 'normal'
    ax.text(width + 0.3, bar.get_y() + bar.get_height()/2,
            f'{acc:.2f}%', ha='left', va='center',
            fontsize=9, fontweight=weight, color='#333333')

# 设置Y轴标签
ax.set_yticks(y_pos)
ax.set_yticklabels(df['Ablation Setting'], fontweight='bold')

# 移除上方和右侧的边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置标签
ax.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax.set_ylabel('Loss Combination', fontsize=11, fontweight='bold')

# 设置网格线（仅X轴，虚线）
ax.grid(axis='x', linestyle='--', alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# 添加参考线标注完整模型性能
ax.axvline(x=93.94, color='#5b2c6f', linestyle=':', linewidth=1.2, alpha=0.7)

# 6. 保存结果
plt.tight_layout()
plt.savefig('experiments/ablation/ablation_results.svg', format='svg', bbox_inches='tight')
plt.savefig('experiments/ablation/ablation_results.pdf', format='pdf', bbox_inches='tight')
plt.savefig('experiments/ablation/ablation_results.png', format='png', bbox_inches='tight', dpi=300)
plt.close()

print("图像已保存:")
print("  - experiments/ablation/ablation_results.svg")
print("  - experiments/ablation/ablation_results.pdf")
print("  - experiments/ablation/ablation_results.png")
