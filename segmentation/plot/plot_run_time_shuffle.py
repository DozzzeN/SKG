import matplotlib.pyplot as plt

# 数据总量
data_sizes = ['4', '8', '16', '32', '64', '128', '256', '512']
data_sizes_numeric = [16, 32, 64, 128, 256, 512, 1024, 2048]  # 用于折线图的横坐标

# 分段划分时间
partition_times = [0.000806322, 0.003297302, 0.00588669, 0.010083566, 0.019033967, 0.044465895, 0.084037329, 0.169576044]

# 分段搜索时间
search_times = [0.000683481, 0.00257327, 0.008782883, 0.056133576, 0.20980961, 0.97261216, 3.699442134, 12.541222581]

# 总时间
total_times = [p + s for p, s in zip(partition_times, search_times)]

# 绘制折线图
plt.figure(figsize=(10, 6))

plt.plot(data_sizes_numeric, partition_times, label='Shuffling Time', marker='o')
plt.plot(data_sizes_numeric, search_times, label='Search Time', marker='o')
plt.plot(data_sizes_numeric, total_times, label='Total Time', marker='o')

# 在总时间的点上标注值
for i, txt in enumerate(total_times):
    plt.text(data_sizes_numeric[i], total_times[i], f'{txt:.2f}', ha='right', va='bottom')

# 设置对数刻度
plt.yscale('log')

plt.xlabel('Number of Episode (Episode Length = 4)')
plt.ylabel('Time (seconds)')
plt.title('Shuffling Time, Search Time, and Total Time')
plt.xticks(data_sizes_numeric, data_sizes, rotation=45)
plt.legend()

plt.grid(True)
plt.tight_layout()

plt.savefig('runtime_shuffle.svg', dpi=1200, bbox_inches='tight')
# plt.show()
