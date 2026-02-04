import matplotlib.pyplot as plt
import pandas as pd

# 1. 准备数据
data = {
    'Ablation Setting': ['LC+CC+DC', 'LC+DC', 'LC+CC', 'CC+DC', 'LC', 'DC', 'CC'],
    'Accuracy': [93.94, 92.25, 93.68, 89.25, 91.68, 87.19, 86.56]
}
df = pd.DataFrame(data)

# 2. 设置绘图风格 (IEEE 常用 Times New Roman 或 Arial)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.unicode_minus'] = False

# 配色方案：深蓝色突出 Baseline，浅灰色淡化对比项，显得非常专业
# IEEE 顶刊常用深蓝: #08519c, 浅灰: #d9d9d9 或浅蓝: #9ecae1
colors = ['#08519c'] + ['#9ecae1'] * (len(df) - 1)

# 3. 开始绘图
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

# 绘制柱状图 (zorder=3 让柱子在网格线上方)
bars = ax.bar(df['Ablation Setting'], df['Accuracy'], color=colors,
              width=0.6, edgecolor='black', linewidth=0.8, zorder=3)

# 4. 精细化调整
# 设置坐标轴范围 (适当留白)
ax.set_ylim(80, 98)

# 添加数值标签 (加粗以清晰可见)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{height:.2f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#333333')

# 移除上方和右侧的边框 (Top & Right Spines)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置标签
ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax.set_xlabel('Loss', fontsize=11, fontweight='bold')

# 设置网格线 (仅Y轴，虚线)
ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# 修正：这里是 set_axisbelow
ax.set_axisbelow(True)

# 优化刻度显示
plt.xticks(rotation=20, fontsize=9)
plt.yticks(fontsize=9)

# 5. 保存结果
plt.tight_layout()
# 建议同时保存为 pdf 和 svg，pdf 在 LaTeX 中更好用
plt.savefig('experiments/ablation/ablation_results.svg', format='svg', bbox_inches='tight')
plt.savefig('experiments/ablation/ablation_results.pdf', format='pdf', bbox_inches='tight')
plt.show()

print("图像已保存为 experiments/ablation/ablation_results.svg 和 experiments/ablation/ablation_results.pdf")
