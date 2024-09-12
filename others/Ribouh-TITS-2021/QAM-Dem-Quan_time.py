import math
import time

import numpy as np
from scipy.io import loadmat


# 根据样本数据估计概率分布
def frequency(samples):
    samples = np.array(samples)
    total_samples = len(samples)

    # 使用字典来记录每个数值出现的次数
    frequency_count = {}
    for sample in samples:
        if sample in frequency_count:
            frequency_count[sample] += 1
        else:
            frequency_count[sample] = 1

    # 计算每个数值的频率，即概率分布
    frequency = []
    for sample in frequency_count:
        frequency.append(frequency_count[sample] / total_samples)

    return frequency


def minEntropy(probabilities):
    return -np.log2(np.max(probabilities) + 1e-12)


def normalize(data):
    if np.max(data) == np.min(data):
        return (data - np.min(data)) / np.max(data)
    else:
        return (np.array(data) - np.min(data)) / (np.max(data) - np.min(data))


def qam_quantizer(data, m):
    quantized_data = np.zeros(len(data) // 2, dtype=int)

    # 分别对i和q平面进行量化
    for i in range(len(data) // 2):
        # 量化i平面数据
        i_quantized = int(data[i] * m)
        # 量化q平面数据
        q_quantized = int(data[len(data) // 2 + i] * m)
        # 将量化后的数据存入结果数组
        quantization = i_quantized * m + q_quantized
        quantization = max(0, min(m * m - 1, quantization))
        quantized_data[i] = quantization

    return quantized_data


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
    y = np.convolve(w / w.sum(), s, mode='valid')  # 6759
    return y


rawData = loadmat("../../data/data_mobile_indoor_1.mat")
CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
# stalking attack
CSIe2Orig = loadmat("../../data/data_static_indoor_1.mat")['A'][:, 0]
dataLen = min(len(CSIe2Orig), len(CSIa1Orig))

CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=3, window='flat')
CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=3, window='flat')

CSIa1Orig = normalize(CSIa1Orig)
CSIb1Orig = normalize(CSIb1Orig)

segLen = 2  # 实部和虚部
keyLen = 32 * segLen
m = 4
bit_len = math.floor(math.log2(int(m * m)))

times = 0
overhead = 0

for staInd in range(0, dataLen, keyLen):
    endInd = staInd + keyLen
    print("range:", staInd, endInd)
    if endInd >= len(CSIa1Orig):
        break
    times += 1

    origInd = np.array([xx for xx in range(staInd, endInd, 1)])

    CSIa1Epi = CSIa1Orig[origInd]
    CSIb1Epi = CSIb1Orig[origInd]

    CSIa1Orig[origInd] = CSIa1Epi
    CSIb1Orig[origInd] = CSIb1Epi

    np.random.seed(0)

    # imitation attack
    CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=len(CSIa1Orig))

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
    tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
    tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]

    # inference attack
    tmpCSIn1 = np.random.random(keyLen)

    start_time = time.time()
    a_list_number = qam_quantizer(tmpCSIa1, m)

    a_list = []

    # 转成二进制，0填充成000000
    for i in range(len(a_list_number)):
        number = bin(a_list_number[i])[2:].zfill(bit_len)
        a_list += number

    end_time = time.time() - start_time
    print("time", round(end_time, 9))
    overhead += end_time

print("times", times)
print("overhead", round(overhead / times, 9) * 1000, "ms")