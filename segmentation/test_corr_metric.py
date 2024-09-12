import numpy as np
from dtw import accelerated_dtw

def dtw_metric(data1, data2):
    distance = lambda x, y: np.abs(x - y)
    data1 = np.array(data1)
    data2 = np.array(data2)
    # return dtw(data1, data2, dist=distance)[0]
    return accelerated_dtw(data1, data2, dist=distance)

# 仿真分析：用相关系数、欧式距离、DTW距离作为度量进行分段匹配
# 创建两个数据集
M = 1024
N = 4
data1 = np.random.normal(0, 1, (M, N))  # M个分段，每个分段N个值
SNR_dB = 30  # 期望的信噪比（dB）
# 计算信号的能量
signal_energy = np.var(data1)

# 计算噪声的能量
noise_energy = signal_energy / (10**(SNR_dB / 10))  # 根据信噪比计算噪声能量

# 生成噪声数据（使用正态分布，均值为0，标准差为噪声能量的平方根）
noise_data = np.random.normal(0, np.sqrt(noise_energy), (M, N))

# print(np.sqrt(noise_energy))
# 生成具有指定信噪比的数据
data2 = data1 + noise_data

index = np.random.permutation(M)
data2 = data2[index]

# 初始化一个列表用于存储互相关结果
cross_correlation = []

# 对每对子序列计算互相关
est_index = np.array([])
for i in range(M):
    cross_correlation.append([])
    for j in range(M):
        cross_correlation[-1].append(np.correlate(data2[i], data1[j])[0])
        # cross_correlation[-1].append(pearsonr(data2[i], data1[j])[0])
    est_index = np.append(est_index, np.argmax(cross_correlation[-1]))

# 将结果转换为数组
cross_correlation = np.array(cross_correlation)

# 用欧式距离度量相关性
euclidean = []
est_index_euclidean = np.array([])
for i in range(M):
    euclidean.append([])
    for j in range(M):
        euclidean[-1].append(np.linalg.norm(data2[i] - data1[j]))
        # euclidean[-1].append(np.abs(np.mean(data2[i]) - np.mean(data1[j])))
        # euclidean[-1].append(dtw_metric(data2[i], data1[j])[0])

    est_index_euclidean = np.append(est_index_euclidean, np.argmin(euclidean[-1]))


# 用DTW度量
dtw = []
est_index_dtw = np.array([])
for i in range(M):
    dtw.append([])
    for j in range(M):
        dtw[-1].append(dtw_metric(data2[i], data1[j])[0])
    est_index_dtw = np.append(est_index_dtw, np.argmin(dtw[-1]))

# print("估计的索引:")
# print(est_index)
# print("真实索引:")
# print(index)
print("索引匹配率:")
print(np.mean(est_index == index))

# print("估计的索引:")
# print(est_index_euclidean)
# print("真实索引:")
# print(index)
print("索引匹配率:")
print(np.mean(est_index_euclidean == index))


print("索引匹配率:")
print(np.mean(est_index_dtw == index))