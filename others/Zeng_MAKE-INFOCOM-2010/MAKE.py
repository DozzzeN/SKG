import csv
import math
import time

import numpy as np
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


# 若m=10, alpha=0.5, n=100
# 将[0,100]分为[0,5],[10,15],[20,25],...,[90,95]
# q=0,10,20,...,100
# g=5,5,5,...,5
def quantizer(number, q):
    for i in range(0, len(q) - 1, 2):
        if number > q[i] and number < q[i + 1]:
            return int((i + 1) / 2)
    return -1


rawData = loadmat("../../data/data_mobile_indoor_1.mat")
# so1 0.7581350729 1.1616814514331566 0.8807114518789284
# si1 0.6901652145 1.121412935323383 0.7739601990049751
# mo2 0.7542655786 1.161318113288822 0.875942278699117
# mi2 0.8129475518 1.161515117455322 0.9442508710801394

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
# stalking attack
CSIe2Orig = loadmat("../../data/data_static_indoor_1.mat")['A'][:, 0]
dataLen = min(len(CSIe2Orig), len(CSIa1Orig))

keyLen = 128

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
    CSIa1Orig = rawData['A'][:, 0][0: dataLen]
    CSIb1Orig = rawData['A'][:, 1][0: dataLen]

    endInd = staInd + keyLen
    print("range:", staInd, endInd)
    if endInd >= len(CSIa1Orig):
        break
    times += 1

    np.random.seed(0)

    # imitation attack
    CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=len(CSIa1Orig))

    CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
    CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')
    CSIe1Orig = smooth(np.array(CSIe1Orig), window_len=30, window='flat')
    CSIe2Orig = smooth(np.array(CSIe2Orig), window_len=30, window='flat')

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

    # extracting randomness
    extCSIn1 = []
    extCSIa1 = []
    extCSIb1 = []
    extCSIe1 = []
    extCSIe2 = []
    w = 30
    for i in range(len(tmpCSIa1)):
        extCSIa1.append(tmpCSIa1[i] - sum(tmpCSIa1[i - int((w - 1) / 2):i + int(w / 2)]) / w)
        extCSIb1.append(tmpCSIb1[i] - sum(tmpCSIb1[i - int((w - 1) / 2):i + int(w / 2)]) / w)
        extCSIe1.append(tmpCSIe1[i] - sum(tmpCSIe1[i - int((w - 1) / 2):i + int(w / 2)]) / w)
        extCSIe2.append(tmpCSIe2[i] - sum(tmpCSIe2[i - int((w - 1) / 2):i + int(w / 2)]) / w)
        extCSIn1.append(tmpCSIn1[i] - sum(tmpCSIn1[i - int((w - 1) / 2):i + int(w / 2)]) / w)

    # deciding quantization intervals
    m = 4
    alpha = 0.3

    sortCSIa1 = extCSIa1.copy()
    sortCSIb1 = extCSIb1.copy()
    sortCSIe1 = extCSIe1.copy()
    sortCSIe2 = extCSIe2.copy()
    sortCSIn1 = extCSIn1.copy()

    sortCSIa1.sort()
    sortCSIb1.sort()
    sortCSIe1.sort()
    sortCSIe2.sort()
    sortCSIn1.sort()

    qa = []
    qb = []
    qe1 = []
    qe2 = []
    qn1 = []

    interval = int(keyLen * (1 - alpha) / m)
    for i in range(0, len(sortCSIa1), interval):
        qa.append(sortCSIa1[i])
        qb.append(sortCSIb1[i])
        qe1.append(sortCSIe1[i])
        qe2.append(sortCSIe2[i])
        qn1.append(sortCSIn1[i])

    # keyLen=30，alpha=0.2，m=4时
    # fallIn=30*(1-0.2)/4=6, fallOut=30*0.2/(4-1)=2
    # 0-6 8-14 16-22 24-30
    # fallIn = int(keyLen * (1 - alpha) / m)
    # fallOut = int(keyLen * alpha / (m - 1))
    # i = 0
    # while i + fallIn < len(sortCSIa1):
    #     qa.append(sortCSIa1[i])
    #     qa.append(sortCSIa1[i + fallIn])
    #     qb.append(sortCSIb1[i])
    #     qb.append(sortCSIb1[i + fallIn])
    #     qe1.append(sortCSIe1[i])
    #     qe1.append(sortCSIe1[i + fallIn])
    #     qe2.append(sortCSIe2[i])
    #     qe2.append(sortCSIe2[i + fallIn])
    #     qn1.append(sortCSIn1[i])
    #     qn1.append(sortCSIn1[i + fallIn])
    #     i += fallIn + fallOut

    for i in range(keyLen):
        o_list_number.append(0)
        if quantizer(extCSIa1[i], qa) != -1:
            a_list_number.append(quantizer(extCSIa1[i], qa))
        if quantizer(extCSIb1[i], qb) != -1:
            b_list_number.append(quantizer(extCSIb1[i], qb))
        if quantizer(extCSIe1[i], qe1) != -1:
            e1_list_number.append(quantizer(extCSIe1[i], qe1))
        if quantizer(extCSIe2[i], qe2) != -1:
            e2_list_number.append(quantizer(extCSIe2[i], qe2))
        if quantizer(extCSIn1[i], qn1) != -1:
            n1_list_number.append(quantizer(extCSIn1[i], qn1))

    # 转成二进制，0填充成00
    for i in range(len(o_list_number)):
        number = bin(o_list_number[i])[2:].zfill(int(np.log2(m)))
        o_list += number
    for i in range(len(a_list_number)):
        number = bin(a_list_number[i])[2:].zfill(int(np.log2(m)))
        a_list += number
    for i in range(len(b_list_number)):
        number = bin(b_list_number[i])[2:].zfill(int(np.log2(m)))
        b_list += number
    for i in range(len(e1_list_number)):
        number = bin(e1_list_number[i])[2:].zfill(int(np.log2(m)))
        e1_list += number
    for i in range(len(e2_list_number)):
        number = bin(e2_list_number[i])[2:].zfill(int(np.log2(m)))
        e2_list += number
    for i in range(len(n1_list_number)):
        number = bin(n1_list_number[i])[2:].zfill(int(np.log2(m)))
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

    # decSum1 = min(len(a_list_number), len(b_list_number))
    # decSum2 = 0
    # decSum31 = 0
    # decSum32 = 0
    # decSum41 = 0
    # for i in range(0, decSum1):
    #     decSum2 += (a_list_number[i] == b_list_number[i])
    # for i in range(min(len(a_list_number), len(e1_list_number))):
    #     decSum31 += (a_list_number[i] == e1_list_number[i])
    # for i in range(min(len(a_list_number), len(e2_list_number))):
    #     decSum32 += (a_list_number[i] == e2_list_number[i])
    # for i in range(min(len(a_list_number), len(n1_list_number))):
    #     decSum41 += (a_list_number[i] == n1_list_number[i])
    # if decSum1 == 0:
    #     continue
    # if decSum2 == decSum1:
    #     print("\033[0;32;40ma-b dec", decSum2, decSum2 / decSum1, "\033[0m")
    # else:
    #     print("\033[0;31;40ma-b dec", "bad", decSum2, decSum2 / decSum1, "\033[0m")
    # print("a-e1", decSum31, decSum31 / decSum1)
    # print("a-e2", decSum32, decSum32 / decSum1)
    # print("a-n1", decSum41, decSum41 / decSum1)
    # print("----------------------")
    # originDecSum += decSum1
    # correctDecSum += decSum2
    # randomDecSum1 += decSum31
    # randomDecSum2 += decSum32
    # noiseDecSum1 += decSum41

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
# print("\033[0;32;40ma-b dec agreement rate", correctDecSum, "/", originDecSum, "=",
#       round(correctDecSum / originDecSum, 10), "\033[0m")
# print("a-e1 dec agreement rate", randomDecSum1, "/", originDecSum, "=", round(randomDecSum1 / originDecSum, 10))
# print("a-e2 dec agreement rate", randomDecSum2, "/", originDecSum, "=", round(randomDecSum2 / originDecSum, 10))
# print("a-n1 dec agreement rate", noiseDecSum1, "/", originDecSum, "=", round(noiseDecSum1 / originDecSum, 10))
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
