import time

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


rawData = loadmat("../../data/data_mobile_indoor_1.mat")
CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
# stalking attack
CSIe2Orig = loadmat("../../data/data_static_indoor_1.mat")['A'][:, 0]
dataLen = min(len(CSIe2Orig), len(CSIa1Orig))

CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')

segLen = 6
keyLen = 256 * segLen

originSum = 0
correctSum = 0
randomSum1 = 0
randomSum2 = 0
noiseSum1 = 0

originDecSum = 0
correctDecSum = 0
randomDecSum1 = 0
randomDecSum2 = 0
noiseDecSum1 = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum1 = 0
randomWholeSum2 = 0
noiseWholeSum1 = 0

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
    # tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
    # tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
    # tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
    # tmpCSIe2 = tmpCSIe2 - np.mean(tmpCSIe2)
    # tmpCSIn1 = tmpCSIn1 - np.mean(tmpCSIn1)

    start_time = time.time()

    dctCSIa1 = dct(tmpCSIa1, n=int(len(tmpCSIa1) / 2))

    # dctCSIa1 = dct(tmpCSIa1)
    # dctCSIb1 = dct(tmpCSIb1)
    # dctCSIe1 = dct(tmpCSIe1)
    # dctCSIe2 = dct(tmpCSIe2)
    # dctCSIn1 = dct(tmpCSIn1)

    mean_a = np.mean(dctCSIa1)

    std_a = np.std(dctCSIa1)

    a_list = []
    b_list = []
    e1_list = []
    e2_list = []
    n1_list = []

    a_list_number = []
    b_list_number = []
    e1_list_number = []
    e2_list_number = []
    n1_list_number = []

    for i in range(len(dctCSIa1)):
        if dctCSIa1[i] > mean_a + std_a:
            a_list_number.append(3)
        elif dctCSIa1[i] <= mean_a + std_a and dctCSIa1[i] > mean_a:
            a_list_number.append(2)
        elif dctCSIa1[i] <= mean_a and dctCSIa1[i] > mean_a - std_a:
            a_list_number.append(1)
        elif dctCSIa1[i] <= mean_a - std_a:
            a_list_number.append(0)

    end_time = time.time() - start_time
    print("time", round(end_time, 9))
    overhead += end_time

print("times", times)
print("overhead", round(overhead / times, 9))