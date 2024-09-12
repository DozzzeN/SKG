import math

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

# 最小熵最大的情况
# fileNames = ["../../data/data_mobile_outdoor_1.mat"]

# fileNames = ["../../data/data_mobile_indoor_1.mat",
#              "../../data/data_static_indoor_1.mat",
#              "../../data/data_mobile_outdoor_1.mat",
#              "../../data/data_static_outdoor_1.mat"
#              ]

fileNames = ["../../csi/csi_mobile_indoor_1_r",
            "../../csi/csi_static_indoor_1_r",
            "../../csi/csi_mobile_outdoor_r",
            "../../csi/csi_static_outdoor_r"]

# RSS security strength
# mi1 0.2948891754082981
# si1 0.1563417395227201
# mo1 0.4456248117436185
# so1 0.1762537678272253

# CSI security strength add_perturbation
# mi1 0.22286715118637715   0.5627624976387179
# si1 0.40839814128488006   0.5717308085615779
# mo1 0.11644200029394658   0.6132128379159766
# so1 0.08320265745891424   0.6033459531566389

for f in fileNames:
    rawData = loadmat(f)
    print(f)

    if f.find("csi") != -1:
        CSIa1Orig = rawData['testdata'][:, 0]
    else:
        CSIa1Orig = rawData['A'][:, 0]

    CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=3, window='flat')

    perturb = np.random.normal(0, 1, (len(CSIa1Orig), len(CSIa1Orig)))
    # 如果注释了下面的则安全强度变低
    CSIa1Orig = CSIa1Orig @ perturb

    CSIa1Orig = normalize(CSIa1Orig)

    segLen = 2  # 实部和虚部，两个CSI产生一个数据
    keyLen = 1 * segLen
    m = 16
    number_len = math.floor(math.log2(m * m))
    bit_len = int(keyLen / segLen) * number_len

    times = 0

    keys = []

    # for staInd in range(0, dataLen, keyLen):
    # 至少测试100*bit_len次，保证每次结果均出现
    for staInd in range(0, 2 ** bit_len * 100, keyLen):
        endInd = staInd + keyLen
        # print("range:", staInd, endInd)
        if endInd >= len(CSIa1Orig):
            print("too long")
            break
        times += 1

        origInd = np.array([xx for xx in range(staInd, endInd, 1)])

        CSIa1Epi = CSIa1Orig[origInd]

        CSIa1Orig[origInd] = CSIa1Epi

        tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]

        a_list_number = qam_quantizer(tmpCSIa1, m)

        a_list = []

        # 转成二进制，0填充成number_len长的0
        for i in range(len(a_list_number)):
            number = bin(a_list_number[i])[2:].zfill(number_len)
            a_list += number

        keys.append("".join(a_list))

    distribution = frequency(keys)
    print("minEntropy", minEntropy(distribution) / bit_len, "bit_len", bit_len, "keyLen", len(keys))
    print()
