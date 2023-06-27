import csv
import math
import time
from tkinter import messagebox

import numpy as np
import scipy
from fastdtw import dtw
from matplotlib import pyplot as plt
from scipy.fft import dct
from scipy.io import loadmat
from scipy.stats import pearsonr, boxcox
from statsmodels.distributions.empirical_distribution import ECDF

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


def addNoise(origin, SNR):
    dataLen = len(origin)
    # np.random.seed(dataLen)
    noise = np.random.normal(0, 1, size=dataLen)
    signal_power = np.sum(origin ** 2) / dataLen
    noise_power = np.sum(noise ** 2) / dataLen
    noise_variance = signal_power / (10 ** (SNR / 10))
    noise = noise * np.sqrt(noise_variance / noise_power)
    return origin + noise

# SNR = 10，KeyLen=5*256
# 有噪音，排序后的索引
# 0.8517717634 0.0 1.6 1.3628348214285713
# 有噪音，原始值
# 0.8572823661 0.0 1.6 1.3716517857142858
# 无噪音，排序后的索引
# 0.6342773438 0.0 1.6 1.01484375
# 无噪音，原始值
# 0.6405552455 0.0 1.6 1.0248883928571428

# SNR = 15，KeyLen=5*256
# 有噪音，排序后的索引
# 0.9710518973 0.0 1.6 1.5536830357142857
# 有噪音，原始值
# 0.9778878348 0.0 1.6 1.5646205357142857
# 无噪音，排序后的索引
# 0.7479073661 0.0 1.6 1.1966517857142858
# 无噪音，原始值
# 0.7540457589 0.0 1.6 1.2064732142857142

# SNR = 20，KeyLen=5*256
# 有噪音，排序后的索引
# 有噪音，原始值
# 无噪音，排序后的索引
# 无噪音，原始值

# SNR = 10，KeyLen=5*1024
# 有噪音，排序后的索引
# 0.7323242187 0.0 2.0 1.4646484375
# 有噪音，原始值
# 0.751171875 0.0 2.0 1.50234375
# 无噪音，排序后的索引
# 0.59375 0.0 2.0 1.1875
# 无噪音，原始值
# 0.6026367188 0.0 2.0 1.2052734375

dataLen = 50000
SNR = 3
# np.random.seed(0)
CSIa1Orig = np.random.normal(0, 1, size=dataLen)
CSIb1Orig = addNoise(CSIa1Orig, SNR)
# imitation attack
CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=len(CSIa1Orig))

# CSIa1ECDF = ECDF(CSIa1Orig)
withoutSort = False
addNoise = "mul"

dataLen = len(CSIa1Orig)
print("dataLen", dataLen)

print(pearsonr(CSIa1Orig, CSIb1Orig)[0])
print(pearsonr(np.array(CSIa1Orig).argsort().argsort(), np.array(CSIb1Orig).argsort().argsort())[0])
print(pearsonr(CSIa1Orig, np.array(CSIa1Orig).argsort().argsort())[0])

CSIa1Orig = (CSIa1Orig - min(CSIa1Orig)) / (max(CSIa1Orig) - min(CSIa1Orig))
CSIb1Orig = (CSIb1Orig - min(CSIb1Orig)) / (max(CSIb1Orig) - min(CSIb1Orig))
CSIa1Ind = np.array(CSIa1Orig).argsort().argsort()
CSIb1Ind = np.array(CSIb1Orig).argsort().argsort()
CSIa1Ind = (CSIa1Ind - min(CSIa1Ind)) / (max(CSIa1Ind) - min(CSIa1Ind))
CSIb1Ind = (CSIb1Ind - min(CSIb1Ind)) / (max(CSIb1Ind) - min(CSIb1Ind))

print(sum(abs(CSIa1Orig - CSIb1Orig)))
print(sum(abs(CSIa1Ind - CSIb1Ind)))
print(sum(abs(CSIa1Orig - CSIa1Ind)))

segLen = 5
keyLen = 256 * segLen

print("segLen", segLen)
print("keyLen", keyLen / segLen)

originSum = 0
correctSum = 0
randomSum1 = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum1 = 0

times = 0

for staInd in range(0, dataLen, keyLen):
    endInd = staInd + keyLen
    print("range:", staInd, endInd)
    if endInd >= len(CSIa1Orig):
        break

    times += 1

    seed = np.random.randint(100000)
    np.random.seed(seed)

    CSIa1OrigBack = CSIa1Orig.copy()
    CSIb1OrigBack = CSIb1Orig.copy()
    CSIe1OrigBack = CSIe1Orig.copy()

    CSIa1OrigBack = smooth(np.array(CSIa1OrigBack), window_len=30, window='flat')
    CSIb1OrigBack = smooth(np.array(CSIb1OrigBack), window_len=30, window='flat')
    CSIe1OrigBack = smooth(np.array(CSIe1OrigBack), window_len=30, window='flat')

    tmpCSIa1 = CSIa1OrigBack[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1OrigBack[range(staInd, endInd, 1)]
    tmpCSIe1 = CSIe1OrigBack[range(staInd, endInd, 1)]

    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
    tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)

    if addNoise == "mul":
        np.random.seed(0)
        randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
        tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
        tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)
        tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix)

    # 最后各自的密钥
    a_list = []
    b_list = []
    e1_list = []

    # without sorting
    if withoutSort:
        tmpCSIa1Ind = np.array(tmpCSIa1)
        tmpCSIb1Ind = np.array(tmpCSIb1)
        tmpCSIe1Ind = np.array(tmpCSIe1)
    else:
        tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
        tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()
        tmpCSIe1Ind = np.array(tmpCSIe1).argsort().argsort()

    print(pearsonr(tmpCSIa1, tmpCSIb1)[0])
    print(pearsonr(tmpCSIa1.argsort().argsort(), tmpCSIb1.argsort().argsort())[0])

    minEpiIndClosenessLsb = np.zeros(int(keyLen / segLen), dtype=int)
    minEpiIndClosenessLse1 = np.zeros(int(keyLen / segLen), dtype=int)

    tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)
    permutation = list(range(int(keyLen / segLen)))
    combineMetric = list(zip(tmpCSIa1IndReshape, permutation))
    np.random.seed(staInd)
    np.random.shuffle(combineMetric)
    tmpCSIa1IndReshape, permutation = zip(*combineMetric)
    tmpCSIa1Ind = np.hstack((tmpCSIa1IndReshape))

    for i in range(int(keyLen / segLen)):
        epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]

        epiIndClosenessLsb = np.zeros(int(keyLen / segLen))
        epiIndClosenessLse1 = np.zeros(int(keyLen / segLen))

        for j in range(int(keyLen / segLen)):
            epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
            epiInde1 = tmpCSIe1Ind[j * segLen: (j + 1) * segLen]

            epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))
            epiIndClosenessLse1[j] = sum(abs(epiInde1 - np.array(epiInda1)))

        minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)
        minEpiIndClosenessLse1[i] = np.argmin(epiIndClosenessLse1)

    a_list_number = list(permutation)
    b_list_number = list(minEpiIndClosenessLsb)
    e1_list_number = list(minEpiIndClosenessLse1)

    # 转成二进制，0填充成0000
    for i in range(len(a_list_number)):
        number = bin(a_list_number[i])[2:].zfill(int(np.log2(len(a_list_number))))
        a_list += number
    for i in range(len(b_list_number)):
        number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
        b_list += number
    for i in range(len(e1_list_number)):
        number = bin(e1_list_number[i])[2:].zfill(int(np.log2(len(e1_list_number))))
        e1_list += number

    # 对齐密钥，随机补全
    for i in range(len(a_list) - len(e1_list)):
        e1_list += str(np.random.randint(0, 2))

    sum1 = min(len(a_list), len(b_list))
    sum2 = 0
    sum31 = 0
    for i in range(0, sum1):
        sum2 += (a_list[i] == b_list[i])
    for i in range(min(len(a_list), len(e1_list))):
        sum31 += (a_list[i] == e1_list[i])

    print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
    print("a-e1", sum31, sum31 / sum1)
    originSum += sum1
    correctSum += sum2
    randomSum1 += sum31

    originWholeSum += 1
    correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
    randomWholeSum1 = randomWholeSum1 + 1 if sum31 == sum1 else randomWholeSum1

print("\033[0;34;40ma-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10), "\033[0m")
print("a-e1 all", randomSum1, "/", originSum, "=", round(randomSum1 / originSum, 10))
print("\033[0;34;40ma-b whole match", correctWholeSum, "/", originWholeSum, "=",
      round(correctWholeSum / originWholeSum, 10), "\033[0m")
print("a-e1 whole match", randomWholeSum1, "/", originWholeSum, "=", round(randomWholeSum1 / originWholeSum, 10))
print("times", times)

print(round(correctSum / originSum, 10), round(correctWholeSum / originWholeSum, 10), originSum / times / keyLen,
      correctSum / times / keyLen)
if withoutSort:
    print("without sort")
else:
    print("with sort")
# messagebox.showinfo("提示", "测试结束")
