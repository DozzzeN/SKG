import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct
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
    y = np.convolve(w / w.sum(), s, mode='valid')  # 6759
    return y


fileName = "../data/data_static_indoor_1.mat"
rawData = loadmat(fileName)
CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
dataLen = len(CSIa1Orig)

# keyBit 160 384 896 2048 4608 10240
# keyLen 32 64 128 256 512 1024
# segLen 5 6 7 8 9 10

segLen = 1
keyLen = 11 * 2048 * segLen
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

    CSIa1Orig[origInd] = CSIa1Epi

    CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]

    start = time.time()

    dctCSIa1 = dct(tmpCSIa1, n=int(len(tmpCSIa1) / 2))
    # dctCSIa1 = dct(tmpCSIa1)

    mean_a = np.mean(dctCSIa1)
    std_a = np.std(dctCSIa1)

    a_list = []

    for i in range(len(dctCSIa1)):
        if dctCSIa1[i] > mean_a + std_a:
            a_list.append("11")
        elif dctCSIa1[i] <= mean_a + std_a and dctCSIa1[i] > mean_a:
            a_list.append("10")
        elif dctCSIa1[i] <= mean_a and dctCSIa1[i] > mean_a - std_a:
            a_list.append("01")
        elif dctCSIa1[i] <= mean_a - std_a:
            a_list.append("00")

    a_bits = "".join(a_list)
    print("bit length", len(a_bits))
    end = time.time()
    overhead += end - start
    print("time:", end - start)

print(overhead / times)
