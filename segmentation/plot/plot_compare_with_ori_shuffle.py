import matplotlib.pyplot as plt
import numpy as np

# 横坐标：分段个数，从4到256
segments = ["4", "8", "16", "32", "64", "128", "256"]

# 纵坐标：比特匹配率
fixed_segment_matching_rates = [
    0.972325103, 0.978514552, 0.977929043, 0.98236755, 0.987916667, 0.992489141, 0.989420573
]

# (设置阈值为2的)
adaptive_segment_matching_rates = [
    0.998148148, 0.998421197, 0.998813944, 0.998799669, 1.0, 1.0, 1.0]

# 将匹配率转换为不匹配率
fixed_segment_non_matching_rates = [1 - rate for rate in fixed_segment_matching_rates]
adaptive_segment_non_matching_rates = [1 - rate for rate in adaptive_segment_matching_rates]

# 设置柱状图的位置
x = np.arange(len(segments))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
bars1 = ax.bar(x - width / 2, fixed_segment_non_matching_rates, width, label='Fixed Segmentation')
bars2 = ax.bar(x + width / 2, adaptive_segment_non_matching_rates, width, label='Adaptive Segmentation')

# 添加一些文本用于标注
ax.set_xlabel('Number of Episode (Episode Length = 4)')
ax.set_ylabel('Bit Mismatch Rate')
ax.set_title('BMR by Number of Episodes and Shuffling Method')
ax.set_xticks(x)
ax.set_xticklabels(segments)
ax.legend()

# 在柱状图上添加标签
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.3f}', ha='center',
             va='bottom')  # ha: horizontal alignment

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.3f}', ha='center',
             va='bottom')  # ha: horizontal alignment

fig.tight_layout()

plt.savefig('compare_shuffle.svg', dpi=1200, bbox_inches='tight')
# plt.show()
