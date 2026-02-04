import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
cases = ['1', '2', '3', '4', '5', '6', '7', '8']
accuracies = [93.3, 91.1, 86.8, 88.4, 87.3, 67.3, 62.4, 47.9]

# 2. 设置绘图环境 (强制使用 Times New Roman)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

# 3. 绘制分区背景阴影 (保持原有的高级感分区)
# 区域1：全部故障类型覆盖 (淡蓝色)
ax.axvspan(-0.5, 4.5, color='#e6f2ff', alpha=0.5, label='Full Fault Coverage', zorder=1)
# 区域2：存在缺失故障类型 (淡红色)
ax.axvspan(4.5, 7.5, color='#ffe6e6', alpha=0.5, label='Missing Fault Types', zorder=1)

# 4. 绘制主体折线
# 使用深蓝色 #003366，白底圆点
ax.plot(cases, accuracies, color='#003366', linewidth=2.5, marker='o',
        markersize=9, markerfacecolor='white', markeredgewidth=2, label='Avg. Accuracy', zorder=5)

# 5. 【新增】显示每个点的数据数值
for i, acc in enumerate(accuracies):
    # 根据数据点位置自动调整偏移，防止文字重叠
    offset = 1.5 if acc > 60 else 2.5
    ax.text(i, acc + offset, f'{acc:.1f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#003366', zorder=6)

# 6. 精细化修饰
# 添加中间的分界虚线
ax.axvline(x=4.5, color='#666666', linestyle='--', linewidth=1.2, alpha=0.8, zorder=2)

# 阶段性文字标注
ax.text(2, 96, 'Class-Balanced/Imbalanced\n(All Classes Known)',
        ha='center', va='bottom', fontsize=10, fontweight='bold', color='#003366')
ax.text(6, 78, 'Partial Class Missing\n(Teacher-Only Classes)',
        ha='center', va='bottom', fontsize=10, fontweight='bold', color='#990000')

# 坐标轴标签和范围
ax.set_ylim(40, 105) # 稍微调高上限给文字留空
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Degree of Source Domain Imbalance', fontsize=12, fontweight='bold')

# 移除冗余边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置网格
ax.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# 7. 保存结果
plt.tight_layout()
plt.savefig('experiments/students_unbalance/imbalance_impact_with_values.svg', format='svg', bbox_inches='tight')
plt.savefig('experiments/students_unbalance/imbalance_impact_with_values.pdf', format='pdf', bbox_inches='tight')
plt.show()

print("包含数据标注的高级趋势图已保存。")
