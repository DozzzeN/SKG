import matplotlib.pyplot as plt

# 数据总量
data_sizes = ['4 * 4', '4 * 8', '4 * 16', '4 * 32', '4 * 64', '4 * 128', '4 * 256', '4 * 512']
data_sizes_numeric = [16, 32, 64, 128, 256, 512, 1024, 2048]  # 用于折线图的横坐标

# 分段划分时间(设置阈值的)
partition_times = [0.000294267, 0.000475894, 0.001455412, 0.015392925, 0.167750993, 1.107278596, 4.968781267, 20.375933]

# 分段搜索时间
search_times = [0.001423173, 0.011382241, 0.056643441, 0.138525744, 0.279532869, 0.68996917, 2.258261679, 8.8179526]

# 总时间
total_times = [p + s for p, s in zip(partition_times, search_times)]

# 绘制折线图
plt.figure(figsize=(10, 6))

plt.plot(data_sizes_numeric, partition_times, label='Partition Time', marker='o')
plt.plot(data_sizes_numeric, search_times, label='Search Time', marker='o')
plt.plot(data_sizes_numeric, total_times, label='Total Time', marker='o')

# 在总时间的点上标注值
for i, txt in enumerate(total_times):
    plt.text(data_sizes_numeric[i], total_times[i], f'{txt:.2f}', ha='right', va='bottom')

# 设置对数刻度
plt.yscale('log')

plt.xlabel('Data (Number of Elements)')
plt.ylabel('Time (seconds)')
plt.title('Partitioning Time, Search Time, and Total Time')
plt.xticks(data_sizes_numeric, data_sizes, rotation=45)
plt.legend()

plt.grid(True)
plt.tight_layout()

plt.savefig('runtime.svg', dpi=1200, bbox_inches='tight')
# plt.show()
