import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def smooth(x, window_len=11, window='hanning'):
    # ndim返回数组的维度
    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser'")

    # np.r_拼接多个数组，要求待拼接的多个数组的列数必须相同
    # 切片[开始索引:结束索引:步进长度]
    # 使用算术平均矩阵来平滑数据
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        # 元素为float，返回window_len个1.的数组
        w = np.ones(window_len, 'd')
    elif window == 'kaiser':
        beta = 5
        w = eval('np.' + window + '(window_len, beta)')
    else:
        w = eval('np.' + window + '(window_len)')

    # 进行卷积操作
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

# 用互相关衡量两个序列分段之间的相似性：单个分段类型
withNoise = False
cross_correlations = []
M = 4
N = 8
fileName = ["../data/data_mobile_indoor_1.mat"]
# fileName = ["../csi/csi_mobile_indoor_1_r"]
rawData = loadmat(fileName[0])

data1 = rawData['A'][:, 0][0:M * N]
data2 = rawData['A'][:, 1][0:M * N]

data1 = data1 - np.mean(data1)
data2 = data2 - np.mean(data2)

if withNoise:
    np.random.seed(1)
    noise = np.random.normal(0, 1, (M * N, M * N))
    data1 = data1 @ noise
    data2 = data2 @ noise

from test_seg import segment_data, generate_segment_lengths
segment_lengths = generate_segment_lengths(M, M * N)
print(segment_lengths)
data1 = segment_data(data1, M, segment_lengths)
data2 = segment_data(data2, M, segment_lengths)

np.random.seed(0)
# index = np.random.permutation(M)
# index = list(range(M - 1, -1, -1))
# data2 = data2[index]
permutation = list(range(M))
combineMetric = list(zip(data2, permutation))
np.random.shuffle(combineMetric)
data2, permutation = zip(*combineMetric)

# reshape成向量
# data1_flat = np.array(data1).flatten()
# data2_flat = np.array(data2).flatten()
data1_flat = np.hstack((data1))
data2_flat = np.hstack((data2))

# 初始化一个列表用于存储互相关结果
cross_correlation = np.correlate(data1_flat, data2_flat, 'full')

cross_correlation = np.array(cross_correlation, dtype=float)
max_len = len(data1_flat)
j = 1.0
intervals = np.arange(1, max_len + 1)
intervals = np.append(intervals, np.arange(max_len - 1, 0, -1))
cross_correlation /= intervals

# 对互相关进行滤波
# cross_correlation = np.convolve(cross_correlation, np.ones(N - 1) / N - 1, mode='same')
cross_correlation = smooth(cross_correlation, window_len=N, window='hanning')

plt.figure()
plt.plot(cross_correlation)
plt.show()
