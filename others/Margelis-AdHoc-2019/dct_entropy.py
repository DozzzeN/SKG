import time

import numpy as np
from scipy.fft import dct
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
# mi1 0.3705470112822523
# si1 0.3777255462883289
# mo1 0.37910392405694565
# so1 0.3073726058573136

# CSI security strength
# mi1 0.3343986745712877
# si1 0.4178104142568602
# mo1 0.355311439639592
# so1 0.39652262192120946

for f in fileNames:
    rawData = loadmat(f)
    print(f)

    if f.find("csi") != -1:
        CSIa1Orig = rawData['testdata'][:, 0]
    else:
        CSIa1Orig = rawData['A'][:, 0]

    # 扩展数据
    CSIa1Orig = np.tile(CSIa1Orig, 5)
    CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=3, window='flat')
    dataLen = len(CSIa1Orig)

    segLen = 1
    keyLen = 8 * segLen

    bit_len = int(keyLen / segLen)
    times = 0

    keys = []
    for staInd in range(0, 2 ** bit_len * 100, keyLen):
        endInd = staInd + keyLen
        if endInd >= len(CSIa1Orig):
            print("too long")
            break
        times += 1

        origInd = np.array([xx for xx in range(staInd, endInd, 1)])

        CSIa1Epi = CSIa1Orig[origInd]

        CSIa1Orig[origInd] = CSIa1Epi

        np.random.seed(0)

        tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
        tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)

        dctCSIa1 = dct(tmpCSIa1, n=int(len(tmpCSIa1) / 2))
        mean_a = np.mean(dctCSIa1)
        std_a = np.std(dctCSIa1)
        a_list = []
        a_list_number = []

        for i in range(len(dctCSIa1)):
            if dctCSIa1[i] > mean_a + std_a:
                a_list_number.append(3)
            elif dctCSIa1[i] <= mean_a + std_a and dctCSIa1[i] > mean_a:
                a_list_number.append(2)
            elif dctCSIa1[i] <= mean_a and dctCSIa1[i] > mean_a - std_a:
                a_list_number.append(1)
            elif dctCSIa1[i] <= mean_a - std_a:
                a_list_number.append(0)

        # 转成二进制，0填充成00
        for i in range(len(a_list_number)):
            number = bin(a_list_number[i])[2:].zfill(2)
            a_list += number

        keys.append("".join(map(str, a_list)))

    distribution = frequency(keys)
    print("minEntropy", minEntropy(distribution) / bit_len, "bit_len", bit_len, "keyLen", len(keys))

    print("times", times)

    print()
