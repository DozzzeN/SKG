import csv
import math
import sys
import time

import numpy as np
from pyentrp import entropy as ent
import scipy.signal
from dtw import dtw
from dtw import accelerated_dtw
from matplotlib import pyplot as plt
from scipy.fft import dct
from scipy.io import loadmat
from scipy.spatial import distance
from scipy.stats import pearsonr, boxcox


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
# data BMR BGR BGR-with-no-error
# mi1 (extended) 0.9978215144 0.0 1.2343069077422346 1.2316179879462217
# si1 (extended) 0.998581063  0.0 1.1921194029850746 1.1904278606965175
# mo1 (extended) 0.9922626202 0.0 1.1841660489251298 1.1750037064492216
# so1 (extended) 0.9963566707 0.0 1.1868229839967905 1.1824989970133286
# average BMR: 1-(0.9978215144+0.998581063 +0.9922626202+0.9963566707)/4=0.003744532924999988
# average BGR: (1.2316179879462217+1.1904278606965175+1.1750037064492216+1.1824989970133286)/4=1.1948871380263224

# extended
CSIa1Orig = np.tile(rawData['A'][:, 0], 5)
CSIb1Orig = np.tile(rawData['A'][:, 1], 5)

# stalking attack
CSIe2Orig = loadmat("../../data/data_static_indoor_1.mat")['A'][:, 0]
dataLen = min(len(CSIe2Orig), len(CSIa1Orig))

segLen = 10  # l=10, lb=13
# segLen = 5  # l=5, lb=6
# segLen = 4  # l=4, lb=5
keyLen = 1024 * segLen

lossySum = 0
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
for staInd in range(0, int(dataLen), keyLen):
    CSIa1Orig = np.tile(rawData['A'][:, 0], 5)[0: dataLen]
    CSIb1Orig = np.tile(rawData['A'][:, 1], 5)[0: dataLen]

    endInd = staInd + keyLen
    print("range:", staInd, endInd)
    if endInd >= len(CSIa1Orig):
        break
    times += 1

    np.random.seed(0)

    # imitation attack
    CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=len(CSIa1Orig))

    # CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
    # CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')
    # CSIe1Orig = smooth(np.array(CSIe1Orig), window_len=30, window='flat')
    # CSIe2Orig = smooth(np.array(CSIe2Orig), window_len=30, window='flat')

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
    tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
    tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]

    # inference attack
    tmpCSIn1 = np.random.random(keyLen)

    # 最后各自的密钥
    o_list = []  # lossy quantization
    a_list = []
    b_list = []
    e1_list = []
    e2_list = []
    n1_list = []

    o_list_number = []
    a_list_number = []
    b_list_number = []
    e1_list_number = []
    e2_list_number = []
    n1_list_number = []

    tmpCSIn1Reshape = tmpCSIn1[:keyLen] * tmpCSIn1[:keyLen]
    tmpCSIa1Reshape = tmpCSIa1[:keyLen] * tmpCSIn1[:keyLen]
    tmpCSIb1Reshape = tmpCSIb1[:keyLen] * tmpCSIn1[:keyLen]
    tmpCSIe1Reshape = tmpCSIe1[:keyLen] * tmpCSIn1[:keyLen]
    tmpCSIe2Reshape = tmpCSIe2[:keyLen] * tmpCSIn1[:keyLen]

    tmpCSIa1Reshape = np.array(tmpCSIa1Reshape).reshape(int(keyLen / segLen), segLen)
    tmpCSIb1Reshape = np.array(tmpCSIb1Reshape).reshape(int(keyLen / segLen), segLen)
    tmpCSIe1Reshape = np.array(tmpCSIe1Reshape).reshape(int(keyLen / segLen), segLen)
    tmpCSIe2Reshape = np.array(tmpCSIe2Reshape).reshape(int(keyLen / segLen), segLen)
    tmpCSIn1Reshape = np.array(tmpCSIn1Reshape).reshape(int(keyLen / segLen), segLen)

    min_q = sys.maxsize

    # key retrieval
    for i in range(int(keyLen / segLen)):
        for j in range(1, segLen):
            min_q = min(min_q, abs(tmpCSIa1Reshape[i][j] - tmpCSIa1Reshape[i][0]))
            deltaA = 1 if tmpCSIa1Reshape[i][j] - tmpCSIa1Reshape[i][0] > 0 else 0
            a_list_number.append(deltaA)
            deltaB = 1 if tmpCSIa1Reshape[i][j] - tmpCSIa1Reshape[i][0] > 0 else 0
            b_list_number.append(deltaB)
            deltaE1 = 1 if tmpCSIe1Reshape[i][j] - tmpCSIe1Reshape[i][0] > 0 else 0
            e1_list_number.append(deltaE1)
            deltaE2 = 1 if tmpCSIe2Reshape[i][j] - tmpCSIe2Reshape[i][0] > 0 else 0
            e2_list_number.append(deltaE2)
            deltaN1 = 1 if tmpCSIn1Reshape[i][j] - tmpCSIn1Reshape[i][0] > 0 else 0
            n1_list_number.append(deltaN1)
        for j in range(int((segLen - 1) / 2)):
            min_q = min(min_q, abs(abs(tmpCSIa1Reshape[i][j] - tmpCSIa1Reshape[i][0]) - abs(
                tmpCSIa1Reshape[i][j + 1] - tmpCSIa1Reshape[i][0])))
            deltaA = 1 if abs(tmpCSIa1Reshape[i][j] - tmpCSIa1Reshape[i][0]) - abs(
                tmpCSIa1Reshape[i][j + 1] - tmpCSIa1Reshape[i][0]) > 0 else 0
            a_list_number.append(deltaA)
            deltaB = 1 if abs(tmpCSIb1Reshape[i][j] - tmpCSIb1Reshape[i][0]) - abs(
                tmpCSIb1Reshape[i][j + 1] - tmpCSIb1Reshape[i][0]) > 0 else 0
            b_list_number.append(deltaB)
            deltaE1 = 1 if abs(tmpCSIe1Reshape[i][j] - tmpCSIe1Reshape[i][0]) - abs(
                tmpCSIe1Reshape[i][j + 1] - tmpCSIe1Reshape[i][0]) > 0 else 0
            e1_list_number.append(deltaE1)
            deltaE2 = 1 if abs(tmpCSIe2Reshape[i][j] - tmpCSIe2Reshape[i][0]) - abs(
                tmpCSIe2Reshape[i][j + 1] - tmpCSIe2Reshape[i][0]) > 0 else 0
            e2_list_number.append(deltaE2)
            deltaN1 = 1 if abs(tmpCSIn1Reshape[i][j] - tmpCSIn1Reshape[i][0]) - abs(
                tmpCSIn1Reshape[i][j + 1] - tmpCSIn1Reshape[i][0]) > 0 else 0
            n1_list_number.append(deltaN1)

    print("min_q", min_q)

    # 转成二进制，0填充成00
    for i in range(len(o_list_number)):
        number = bin(o_list_number[i])[2:].zfill(1)
        o_list += number
    for i in range(len(a_list_number)):
        number = bin(a_list_number[i])[2:].zfill(1)
        a_list += number
    for i in range(len(b_list_number)):
        number = bin(b_list_number[i])[2:].zfill(1)
        b_list += number
    for i in range(len(e1_list_number)):
        number = bin(e1_list_number[i])[2:].zfill(1)
        e1_list += number
    for i in range(len(e2_list_number)):
        number = bin(e2_list_number[i])[2:].zfill(1)
        e2_list += number
    for i in range(len(n1_list_number)):
        number = bin(n1_list_number[i])[2:].zfill(1)
        n1_list += number

    # 对齐密钥，随机补全
    for i in range(len(a_list) - len(e1_list)):
        e1_list += str(np.random.randint(0, 2))
    for i in range(len(a_list) - len(e2_list)):
        e2_list += str(np.random.randint(0, 2))
    for i in range(len(a_list) - len(n1_list)):
        n1_list += str(np.random.randint(0, 2))

    # print("keys of a:", len(a_list), a_list)
    print("keys of a:", len(a_list_number), a_list_number)
    # print("keys of b:", len(b_list), b_list)
    print("keys of b:", len(b_list_number), b_list_number)
    # print("keys of e:", len(e_list), e_list)
    print("keys of e1:", len(e1_list_number), e1_list_number)
    # print("keys of e:", len(e_list), e_list)
    print("keys of e2:", len(e2_list_number), e2_list_number)
    # print("keys of n1:", len(n1_list), n1_list)
    print("keys of n1:", len(n1_list_number), n1_list_number)

    print(ent.multiscale_entropy(a_list_number / np.max(a_list_number), 3, maxscale=1))

    lossySum += len(o_list)

    sum1 = min(len(a_list), len(b_list))
    sum2 = 0
    sum31 = 0
    sum32 = 0
    sum41 = 0
    for i in range(0, sum1):
        sum2 += (a_list[i] == b_list[i])
    for i in range(min(len(a_list), len(e1_list))):
        sum31 += (a_list[i] == e1_list[i])
    for i in range(min(len(a_list), len(e2_list))):
        sum32 += (a_list[i] == e2_list[i])
    for i in range(min(len(a_list), len(n1_list))):
        sum41 += (a_list[i] == n1_list[i])

    print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
    print("a-e1", sum31, sum31 / sum1)
    print("a-e2", sum32, sum32 / sum1)
    print("a-n1", sum41, sum41 / sum1)
    originSum += sum1
    correctSum += sum2
    randomSum1 += sum31
    randomSum2 += sum32
    noiseSum1 += sum41

    decSum1 = min(len(a_list_number), len(b_list_number))
    decSum2 = 0
    decSum31 = 0
    decSum32 = 0
    decSum41 = 0
    for i in range(0, decSum1):
        decSum2 += (a_list_number[i] == b_list_number[i])
    for i in range(min(len(a_list_number), len(e1_list_number))):
        decSum31 += (a_list_number[i] == e1_list_number[i])
    for i in range(min(len(a_list_number), len(e2_list_number))):
        decSum32 += (a_list_number[i] == e2_list_number[i])
    for i in range(min(len(a_list_number), len(n1_list_number))):
        decSum41 += (a_list_number[i] == n1_list_number[i])
    if decSum1 == 0:
        continue
    if decSum2 == decSum1:
        print("\033[0;32;40ma-b dec", decSum2, decSum2 / decSum1, "\033[0m")
    else:
        print("\033[0;31;40ma-b dec", "bad", decSum2, decSum2 / decSum1, "\033[0m")
    print("a-e1", decSum31, decSum31 / decSum1)
    print("a-e2", decSum32, decSum32 / decSum1)
    print("a-n1", decSum41, decSum41 / decSum1)
    print("----------------------")
    originDecSum += decSum1
    correctDecSum += decSum2
    randomDecSum1 += decSum31
    randomDecSum2 += decSum32
    noiseDecSum1 += decSum41

    originWholeSum += 1
    correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
    randomWholeSum1 = randomWholeSum1 + 1 if sum31 == sum1 else randomWholeSum1
    randomWholeSum2 = randomWholeSum2 + 1 if sum32 == sum1 else randomWholeSum2
    noiseWholeSum1 = noiseWholeSum1 + 1 if sum41 == sum1 else noiseWholeSum1

print("\033[0;32;40ma-b bit agreement rate", correctSum, "/", originSum, "=", round(correctSum / originSum, 10),
      "\033[0m")
print("a-e1 bit agreement rate", randomSum1, "/", originSum, "=", round(randomSum1 / originSum, 10))
print("a-e2 bit agreement rate", randomSum2, "/", originSum, "=", round(randomSum2 / originSum, 10))
print("a-n1 bit agreement rate", noiseSum1, "/", originSum, "=", round(noiseSum1 / originSum, 10))
print("\033[0;32;40ma-b dec agreement rate", correctDecSum, "/", originDecSum, "=",
      round(correctDecSum / originDecSum, 10), "\033[0m")
print("a-e1 dec agreement rate", randomDecSum1, "/", originDecSum, "=", round(randomDecSum1 / originDecSum, 10))
print("a-e2 dec agreement rate", randomDecSum2, "/", originDecSum, "=", round(randomDecSum2 / originDecSum, 10))
print("a-n1 dec agreement rate", noiseDecSum1, "/", originDecSum, "=", round(noiseDecSum1 / originDecSum, 10))
print("\033[0;32;40ma-b key agreement rate", correctWholeSum, "/", originWholeSum, "=",
      round(correctWholeSum / originWholeSum, 10), "\033[0m")
print("a-e1 key agreement rate", randomWholeSum1, "/", originWholeSum, "=", round(randomWholeSum1 / originWholeSum, 10))
print("a-e2 key agreement rate", randomWholeSum2, "/", originWholeSum, "=", round(randomWholeSum2 / originWholeSum, 10))
print("a-n1 key agreement rate", noiseWholeSum1, "/", originWholeSum, "=", round(noiseWholeSum1 / originWholeSum, 10))
print("times", times)
print("all bits", lossySum)
print(originSum / len(CSIa1Orig))
print(correctSum / len(CSIa1Orig))

print(round(correctSum / originSum, 10), round(correctWholeSum / originWholeSum, 10), originSum / len(CSIa1Orig),
      correctSum / len(CSIa1Orig))
