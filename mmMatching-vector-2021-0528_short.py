# from mwmatching import maxWeightMatching
# import cv2
import sys
import time

import numpy as np
from numpy.random import exponential as Exp
from scipy import signal
from scipy import sparse
from scipy.io import loadmat
from tsfresh.feature_extraction.feature_calculators import mean_second_derivative_central as msdc


def hp_filter(x, lamb=5000):
    w = len(x)
    b = [[1] * w, [-2] * w, [1] * w]
    D = sparse.spdiags(b, [0, 1, 2], w - 2, w)
    I = sparse.eye(w)
    B = (I + lamb * (D.transpose() * D))
    return sparse.linalg.dsolve.spsolve(B, x)


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


def sumSeries(CSITmp):
    if len(CSITmp) > 1:
        sumCSI = sum(CSITmp) + sumSeries(CSITmp[0:-1])
        return sumCSI
    else:
        return CSITmp[0]


rawData = loadmat('data/data_static_indoor_1_r_m.mat')

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
dataLen = len(CSIa1Orig)

CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)
CSIn1Orig = np.random.normal(loc=-1, scale=1, size=dataLen)  ## Multiplication item normal distribution
# CSIn1Orig = np.random.uniform(0, np.std(CSIa1Orig, ddof=1), size=dataLen)

# ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser']
CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')
CSIb1Orig = smooth(CSIb1Orig, window_len=15, window='flat')
CSIe1Orig = smooth(CSIe1Orig, window_len=15, window='flat')
CSIn1Orig = smooth(CSIn1Orig, window_len=15, window='flat')

# CSIa1Orig = hp_filter(CSIa1Orig, lamb=500)
# CSIb1Orig = hp_filter(CSIb1Orig, lamb=500)

# CSIa1Orig = savgol_filter(CSIa1Orig, 11, 1)
# CSIb1Orig = savgol_filter(CSIb1Orig, 11, 1)

CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
CSIe1OrigBack = CSIe1Orig.copy()
CSIn1OrigBack = CSIn1Orig.copy()

# 固定随机置换的种子
# np.random.seed(1)  # 8 1024 8; 4 128 4
# combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig, CSIe1Orig, CSIn1Orig))
# np.random.shuffle(combineCSIx1Orig)
# CSIa1Orig, CSIb1Orig, CSIe1Orig, CSIn1Orig = zip(*combineCSIx1Orig)

intvl = 7
keyLen = 128 * intvl

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

    CSIa1Orig = CSIa1OrigBack.copy()
    CSIb1Orig = CSIb1OrigBack.copy()
    CSIe1Orig = CSIe1OrigBack.copy()
    CSIn1Orig = CSIn1OrigBack.copy()

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
    tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
    tmpNoise = CSIn1Orig[range(staInd, endInd, 1)]

    tmpCSIa1 = tmpCSIa1 - (np.mean(tmpCSIa1) - np.mean(tmpCSIb1))

    # 去除直流分量
    # tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
    # tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
    # tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
    # tmpNoise = tmpNoise - np.mean(tmpNoise)

    tmpPulse = signal.square(
        2 * np.pi * 1 / intvl * np.linspace(0, np.pi * 0.5 * keyLen / intvl, keyLen))
    tmpCSIa1 = tmpPulse * np.float_power(np.abs(tmpCSIa1), tmpNoise)
    tmpCSIb1 = tmpPulse * np.float_power(np.abs(tmpCSIb1), tmpNoise)
    tmpCSIe1 = tmpPulse * np.float_power(np.abs(tmpCSIe1), tmpNoise)
    tmpNoise = tmpPulse * np.float_power(np.abs(tmpNoise), tmpNoise)

    tmpCSIa1 = tmpPulse * (tmpCSIa1 + tmpNoise / (max(tmpNoise) - min(tmpNoise)))
    tmpCSIb1 = tmpPulse * (tmpCSIb1 + tmpNoise / (max(tmpNoise) - min(tmpNoise)))
    tmpCSIe1 = tmpPulse * (tmpCSIe1 + tmpNoise / (max(tmpNoise) - min(tmpNoise)))
    tmpNoise = tmpPulse * (tmpNoise + tmpNoise / (max(tmpNoise) - min(tmpNoise)))

    CSIa1Orig[range(staInd, endInd, 1)] = tmpCSIa1
    CSIb1Orig[range(staInd, endInd, 1)] = tmpCSIb1
    CSIe1Orig[range(staInd, endInd, 1)] = tmpCSIe1
    CSIn1Orig[range(staInd, endInd, 1)] = tmpNoise

    permLen = len(range(staInd, endInd, intvl))
    origInd = np.array([xx for xx in range(staInd, endInd, intvl)])

    sortCSIa1 = np.zeros(permLen)
    sortCSIb1 = np.zeros(permLen)
    sortCSIe1 = np.zeros(permLen)
    sortNoise = np.zeros(permLen)

    for ii in range(permLen):
        aIndVec = np.array([aa for aa in range(origInd[ii], origInd[ii] + intvl, 1)])

        for jj in range(permLen, permLen * 2):
            bIndVec = np.array([bb for bb in range(origInd[jj - permLen], origInd[jj - permLen] + intvl, 1)])

            CSIa1Tmp = CSIa1Orig[aIndVec]
            CSIb1Tmp = CSIb1Orig[bIndVec]
            CSIe1Tmp = CSIe1Orig[bIndVec]
            NoiseTmp = CSIn1Orig[bIndVec]

            sortCSIa1[ii] = msdc(CSIa1Tmp)
            sortCSIb1[jj - permLen] = msdc(CSIb1Tmp)
            sortCSIe1[jj - permLen] = msdc(CSIe1Tmp)
            sortNoise[ii] = msdc(NoiseTmp)

    a_list_number = np.argsort(sortCSIa1)
    b_list_number = np.argsort(sortCSIb1)
    e_list_number = np.argsort(sortCSIe1)
    n_list_number = np.argsort(sortNoise)

    # 最后各自的密钥
    a_list = []
    b_list = []
    e_list = []
    n_list = []

    # 转成二进制，0填充成0000
    for i in range(len(a_list_number)):
        number = bin(a_list_number[i])[2:].zfill(int(np.log2(len(a_list_number))))
        a_list += number
    for i in range(len(b_list_number)):
        number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
        b_list += number
    for i in range(len(e_list_number)):
        number = bin(e_list_number[i])[2:].zfill(int(np.log2(len(e_list_number))))
        e_list += number
    for i in range(len(n_list_number)):
        number = bin(n_list_number[i])[2:].zfill(int(np.log2(len(n_list_number))))
        n_list += number

    # 对齐密钥，随机补全
    for i in range(len(a_list) - len(e_list)):
        e_list += str(np.random.randint(0, 2))
    for i in range(len(a_list) - len(n_list)):
        n_list += str(np.random.randint(0, 2))

    # print("keys of a:", len(a_list), a_list)
    print("keys of a:", len(a_list_number), a_list_number)
    # print("keys of b:", len(b_list), b_list)
    print("keys of b:", len(b_list_number), b_list_number)
    # print("keys of e:", len(e_list), e_list)
    print("keys of e:", len(e_list_number), e_list_number)
    # print("keys of n:", len(n_list), n_list)
    print("keys of n:", len(n_list_number), n_list_number)

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

    if sum1 == 0:
        continue
    if sum2 == sum1:
        print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
    else:
        print("\033[0;31;40ma-b", "bad", sum2, sum2 / sum1, "\033[0m")
    print("a-e", sum3, sum3 / sum1)
    print("a-n", sum4, sum4 / sum1)
    print("----------------------")
    originSum += sum1
    correctSum += sum2
    randomSum += sum3
    noiseSum += sum4

    originWholeSum += 1
    correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
    randomWholeSum = randomWholeSum + 1 if sum3 == sum1 else randomWholeSum
    noiseWholeSum = noiseWholeSum + 1 if sum4 == sum1 else noiseWholeSum

print("a-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10))
print("a-e all", randomSum, "/", originSum, "=", round(randomSum / originSum, 10))
print("a-n all", noiseSum, "/", originSum, "=", round(noiseSum / originSum, 10))
print("a-b whole match", correctWholeSum, "/", originWholeSum, "=", round(correctWholeSum / originWholeSum, 10))
print("a-e whole match", randomWholeSum, "/", originWholeSum, "=", round(randomWholeSum / originWholeSum, 10))
print("a-n whole match", noiseWholeSum, "/", originWholeSum, "=", round(noiseWholeSum / originWholeSum, 10))