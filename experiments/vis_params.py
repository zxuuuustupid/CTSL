import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ================= 1. 顶刊绘图风格配置 =================
def set_pub_style():
    """配置 Matplotlib 以符合顶刊发表标准"""
    plt.rcParams.update({
        # 字体设置
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],  # 强制衬线字体为新罗马
        'mathtext.fontset': 'stix',         # 数学公式字体接近 Times
        'font.size': 14,                    # 全局基础字号

        # 线条与轴
        'axes.linewidth': 1.5,              # 坐标轴线宽
        'lines.linewidth': 2.5,             # 曲线线宽
        'xtick.major.width': 1.5,           # 刻度线宽
        'ytick.major.width': 1.5,
        'xtick.direction': 'in',            # 刻度朝内 (物理/工程常用)
        'ytick.direction': 'in',

        # 图例
        'legend.frameon': True,             # 显示图例边框
        'legend.edgecolor': 'black',        # 图例边框颜色
        'legend.fancybox': False,           # 直角边框
        'legend.framealpha': 1.0,           # 不透明

        # 输出
        'savefig.bbox': 'tight',            # 自动切除白边
        'savefig.pad_inches': 0.1           # 留一点边距
    })

set_pub_style()

# ================= 2. 数据加载 =================
try:
    baseline = np.load("experiments/PU/sparsity_data_baseline.npy")
    mid = np.load("experiments/PU/sparsity_data_mid.npy")
except FileNotFoundError:
    print("错误：未找到数据文件，请先运行稀疏性分析脚本！")
    exit()

# 数据对齐 (取最小长度，防止长度不一致报错)
min_len = min(len(baseline), len(mid))
baseline = baseline[:min_len]
mid = mid[:min_len]
epochs = min_len
x_axis = range(1, epochs + 1)

# ================= 3. 高级绘图 =================
# 创建画布，黄金比例或 4:3 比例适合单栏/双栏插图
fig, ax = plt.subplots(figsize=(8, 6))

# --- 配色方案 (Nature/Science 风格) ---
# Baseline: 暖色调 (代表过拟合/激进)，使用深砖红
color_base = '#D62728'
# MID: 冷色调 (代表冷静/稀疏)，使用深钢蓝
color_mid = '#1F77B4'

# --- 绘制曲线 ---
# zorder控制绘图顺序，保证曲线在网格之上
ax.plot(x_axis, baseline, label='Baseline (Standard)', color=color_base, linestyle='-', alpha=0.9, zorder=3)
ax.plot(x_axis, mid, label='MID (Ours)', color=color_mid, linestyle='-', alpha=0.9, zorder=3)

# --- 细节修饰 ---
# 1. 网格线 (灰色虚线，置于底层)
ax.grid(True, which='major', linestyle='--', linewidth=0.75, color='gray', alpha=0.3, zorder=0)

# 2. 坐标轴标签 (使用 LaTeX 格式增强数学符号美感，如果没有装LaTeX环境会自动回退到stix字体)
ax.set_xlabel('Training Epochs', fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel(r'Norm of Parameters ($\|\theta\|_2^2$)', fontsize=16, fontweight='bold', labelpad=10)

# 3. 标题 (可选，正式发Paper时通常写在LaTeX的caption里，这里为了完整性加上)
# ax.set_title('Comparison of Model Sparsity', fontsize=18, pad=15)

# 4. 坐标轴刻度优化
ax.tick_params(axis='both', which='major', labelsize=14, length=6)
# 设置x轴间隔 (根据epoch数量自动调整，防止太挤)
if epochs > 10:
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))

# 5. 图例优化
# loc='best' 自动寻找空白处
legend = ax.legend(fontsize=14, loc='center right', shadow=False)
legend.get_frame().set_linewidth(1.0)

# 6. 填充区域 (可选：增加视觉层次感，显示差异)
# ax.fill_between(x_axis, baseline, mid, color='gray', alpha=0.1, label='Sparsity Gap')

# ================= 4. 保存矢量图 =================
output_dir = "experiments/PU"
os.makedirs(output_dir, exist_ok=True)

# 保存为 PDF (LaTeX 编译最佳伴侣)
pdf_path = os.path.join(output_dir, "Fig10_Comparison.pdf")
plt.savefig(pdf_path, format='pdf', dpi=300)

# 保存为 SVG (Visio/Illustrator 可无限编辑)
svg_path = os.path.join(output_dir, "Fig10_Comparison.svg")
plt.savefig(svg_path, format='svg', dpi=300)

# 保存为高清 PNG (预览用)
png_path = os.path.join(output_dir, "Fig10_Comparison.png")
plt.savefig(png_path, format='png', dpi=300)

print(f"绘图完成！已保存三种格式：")
print(f"1. [PDF] {pdf_path} (推荐 LaTeX 使用)")
print(f"2. [SVG] {svg_path} (推荐 PPT/AI 编辑)")
print(f"3. [PNG] {png_path} (快速预览)")

plt.show()
