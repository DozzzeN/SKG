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


fileName = "../data/data_mobile_indoor_1.mat"
rawData = loadmat(fileName)
CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
dataLen = len(CSIa1Orig)
CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)
CSIn1Orig = np.random.normal(loc=-1, scale=1, size=dataLen)

# CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
# CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')

segLen = 7
keyLen = 256 * segLen
times = 0
overhead = 0

originSum = 0
correctSum = 0
randomSum = 0
noiseSum = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum = 0
noiseWholeSum = 0

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

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
    tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
    tmpNoise = CSIn1Orig[range(staInd, endInd, 1)]

    # tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
    # tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
    # tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
    # tmpNoise = tmpNoise - np.mean(tmpNoise)

    tmpCSIa1 = smooth(np.array(tmpCSIa1), window_len=3, window='flat')
    tmpCSIb1 = smooth(np.array(tmpCSIb1), window_len=3, window='flat')

    start = time.time()

    dctCSIa1 = dct(tmpCSIa1)
    dctCSIb1 = dct(tmpCSIb1)
    dctCSIe1 = dct(tmpCSIe1)
    dctCSIn1 = dct(tmpNoise)

    mean_a = np.mean(dctCSIa1)
    mean_b = np.mean(dctCSIb1)
    mean_e = np.mean(dctCSIe1)
    mean_n = np.mean(dctCSIn1)

    std_a = np.std(dctCSIa1)
    std_b = np.std(dctCSIb1)
    std_e = np.std(dctCSIe1)
    std_n = np.std(dctCSIn1)

    a_list = []
    b_list = []
    e_list = []
    n_list = []

    for i in range(len(dctCSIe1)):
        if dctCSIa1[i] > mean_a + std_a:
            a_list.append("11")
        elif dctCSIa1[i] <= mean_a + std_a and dctCSIa1[i] > mean_a:
            a_list.append("10")
        elif dctCSIa1[i] <= mean_a and dctCSIa1[i] > mean_a - std_a:
            a_list.append("01")
        elif dctCSIa1[i] <= mean_a - std_a:
            a_list.append("00")

    for i in range(len(dctCSIb1)):
        if dctCSIb1[i] > mean_b + std_b:
            b_list.append("11")
        elif dctCSIb1[i] <= mean_b + std_b and dctCSIb1[i] > mean_b:
            b_list.append("10")
        elif dctCSIb1[i] <= mean_b and dctCSIb1[i] > mean_b - std_b:
            b_list.append("01")
        elif dctCSIb1[i] <= mean_b - std_b:
            b_list.append("00")

    for i in range(len(dctCSIe1)):
        if dctCSIe1[i] > mean_e + std_e:
            e_list.append("11")
        elif dctCSIe1[i] <= mean_e + std_e and dctCSIe1[i] > mean_e:
            e_list.append("10")
        elif dctCSIe1[i] <= mean_e and dctCSIe1[i] > mean_e - std_e:
            e_list.append("01")
        elif dctCSIe1[i] <= mean_e - std_e:
            e_list.append("00")

    for i in range(len(dctCSIn1)):
        if dctCSIn1[i] >= mean_n + std_n:
            n_list.append("11")
        elif dctCSIn1[i] < mean_n + std_n and dctCSIn1[i] >= mean_n:
            n_list.append("10")
        elif dctCSIn1[i] < mean_n and dctCSIn1[i] > mean_n - std_n:
            n_list.append("01")
        elif dctCSIn1[i] <= mean_n - std_n:
            n_list.append("00")

    end = time.time()
    overhead += end - start
    print("time:", end - start)

    # print(a_list)
    # print(b_list)
    # print(e_list)
    # print(n_list)

    sum1 = min(len(a_list), len(b_list))
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for i in range(0, sum1):
        sum2 += (a_list[i] == b_list[i])
    for i in range(min(len(a_list), len(e_list))):
        sum3 += (a_list[i] == e_list[i])
    for i in range(min(len(a_list), len(n_list))):
        sum4 += (a_list[i] == n_list[i])

    if sum2 == sum1:
        print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
    else:
        print("\033[0;31;40ma-b", sum2, sum2 / sum1, "\033[0m")
    print("a-e", sum3, sum3 / sum1)
    print("----------------------")
    originSum += sum1
    correctSum += sum2
    randomSum += sum3
    noiseSum += sum4

    originWholeSum += 1
    correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
    randomWholeSum = randomWholeSum + 1 if sum3 == sum1 else randomWholeSum
    noiseWholeSum = noiseWholeSum + 1 if sum4 == sum1 else noiseWholeSum

print("a-b all", correctSum, "/", originSum, "=", correctSum / originSum)
print("a-e all", randomSum, "/", originSum, "=", randomSum / originSum)
print("a-n all", noiseSum, "/", originSum, "=", noiseSum / originSum)
print("a-b whole match", correctWholeSum, "/", originWholeSum, "=", correctWholeSum / originWholeSum)
print("a-e whole match", randomWholeSum, "/", originWholeSum, "=", randomWholeSum / originWholeSum)
print("a-n whole match", noiseWholeSum, "/", originWholeSum, "=", noiseWholeSum / originWholeSum)
print("times", times)

print(overhead / times)
print(round(correctSum / originSum, 10), round(correctWholeSum / originWholeSum, 10), originSum / len(CSIa1Orig))