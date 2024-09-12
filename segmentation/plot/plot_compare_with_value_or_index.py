import matplotlib.pyplot as plt
import numpy as np

# 横坐标：分段个数，从4到128
segments = ["4 * 4", "4 * 8", "4 * 16", "4 * 32", "4 * 64", "4 * 128"]

# 纵坐标：比特匹配率
value_matching_rates = [0.996570506, 0.987068286, 0.988187937, 0.996064714, 1.0, 0.966997368]
index_matching_rates = [0.997098774, 0.993865658, 0.98802444, 0.989979322, 0.951114972, 0.878863135]
both_matching_rates = [0.996506986, 0.993461353, 0.995854107, 0.996230357, 0.993700279, 0.996930589]

# 将匹配率转换为不匹配率
value_non_matching_rates = [1 - rate for rate in value_matching_rates]
index_non_matching_rates = [1 - rate for rate in index_matching_rates]
both_non_matching_rates = [1 - rate for rate in both_matching_rates]

# 设置柱状图的位置
x = np.arange(len(segments))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
bars1 = ax.bar(x - width, value_non_matching_rates, width, label='Value')
bars2 = ax.bar(x, index_non_matching_rates, width, label='Index')
bars3 = ax.bar(x + width, both_non_matching_rates, width, label='Both')

# 添加一些文本用于标注
ax.set_xlabel('Number of Elements')
ax.set_ylabel('Bit Mismatch Rate')
ax.set_title('BMR with Different Measurements')
ax.set_xticks(x)
ax.set_xticklabels(segments)
ax.legend()

# 在柱状图上添加标签
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.3f}', ha='center', va='bottom')  # ha: horizontal alignment

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.3f}', ha='center', va='bottom')  # ha: horizontal alignment

for bar in bars3:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.3f}', ha='center', va='bottom')  # ha: horizontal alignment

fig.tight_layout()

plt.savefig('compare_index_value.svg', dpi=1200, bbox_inches='tight')
plt.show()
