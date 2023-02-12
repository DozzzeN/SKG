import hashlib
import math
import signal
import sys
import time

from scipy import signal
from scipy import sparse
from scipy.io import loadmat
from tsfresh.feature_extraction.feature_calculators import mean_second_derivative_central as msdc

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from scipy.io import loadmat
from sklearn import preprocessing

from algorithm import smooth, genSample, sortSegPermOfA, sortSegPermOfB

start_time = time.time()
fileName = "../data/data_static_indoor_1.mat"
rawData = loadmat(fileName)

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

dataLen = len(CSIa1Orig)

CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)
CSIn1Orig = np.random.normal(loc=0, scale=1, size=dataLen)
# CSIn1Orig = np.random.normal(loc=-100, scale=1000, size=dataLen)

np.random.seed(0)
noiseOrig = np.random.uniform(0, 1, size=dataLen)
CSIa1Orig = CSIa1Orig * noiseOrig
CSIb1Orig = CSIb1Orig * noiseOrig
CSIe1Orig = CSIe1Orig * noiseOrig

# 固定随机置换的种子
# np.random.seed(1)  # 8 1024 8; 4 128 4
# combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig, CSIe1Orig, CSIn1Orig))
# np.random.shuffle(combineCSIx1Orig)
# CSIa1Orig, CSIb1Orig, CSIe1Orig, CSIn1Orig = zip(*combineCSIx1Orig)

CSIa1Orig = np.array(CSIa1Orig)
CSIb1Orig = np.array(CSIb1Orig)
CSIe1Orig = np.array(CSIe1Orig)
CSIn1Orig = np.array(CSIn1Orig)

# entropyTime = time.time()
# CSIa1Orig, CSIb1Orig, CSIe1Orig = splitEntropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, 5, dataLen)
# print("--- entropyTime %s seconds ---" % (time.time() - entropyTime))

# CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')
# CSIb1Orig = smooth(CSIb1Orig, window_len=15, window='flat')
# CSIe1Orig = smooth(CSIe1Orig, window_len=15, window="flat")
# CSIn1Orig = smooth(CSIn1Orig, window_len=15, window="flat")

# CSIa1Orig = CSIa1Orig * CSIn1Orig
# CSIb1Orig = CSIb1Orig * CSIn1Orig
# CSIe1Orig = CSIe1Orig * CSIn1Orig
# CSIn1Orig = CSIn1Orig * CSIn1Orig

CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
CSIe1OrigBack = CSIe1Orig.copy()
CSIn1OrigBack = CSIn1Orig.copy()

segLen = 7
keyLen = 128 * segLen

ratio = 1
isShow = False

originSum = 0
correctSum = 0
randomSum = 0
noiseSum = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum = 0
noiseWholeSum = 0

times = 0

for staInd in range(0, dataLen, keyLen):
    endInd = staInd + keyLen
    print("range:", staInd, endInd)
    if endInd >= len(CSIa1Orig):
        break
    times += 1

    CSIa1Orig = CSIa1OrigBack.copy()
    CSIb1Orig = CSIb1OrigBack.copy()
    CSIe1Orig = CSIe1OrigBack.copy()
    CSIn1Orig = CSIn1OrigBack.copy()

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
    tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
    tmpNoise = CSIn1Orig[range(staInd, endInd, 1)]

    # tmpCSIa1 = tmpCSIa1 - (np.mean(tmpCSIa1) - np.mean(tmpCSIb1))

    # 去除直流分量
    # tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
    # tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
    # tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
    # tmpNoise = tmpNoise - np.mean(tmpNoise)

    # 加噪音
    tmpPulse = signal.square(
        2 * np.pi * 1 / segLen * np.linspace(0, np.pi * 0.5 * keyLen / segLen, keyLen))
    # tmpCSIa1 = tmpPulse * np.float_power(np.abs(tmpCSIa1), tmpNoise / (max(tmpNoise) - min(tmpNoise)))
    # tmpCSIb1 = tmpPulse * np.float_power(np.abs(tmpCSIb1), tmpNoise / (max(tmpNoise) - min(tmpNoise)))
    # tmpCSIe1 = tmpPulse * np.float_power(np.abs(tmpCSIe1), tmpNoise / (max(tmpNoise) - min(tmpNoise)))
    # tmpNoise = tmpPulse * np.float_power(np.abs(tmpNoise), tmpNoise / (max(tmpNoise) - min(tmpNoise)))

    # tmpCSIa1 = tmpPulse * (tmpCSIa1 * tmpNoise / (max(tmpNoise) - min(tmpNoise)))
    # tmpCSIb1 = tmpPulse * (tmpCSIb1 * tmpNoise / (max(tmpNoise) - min(tmpNoise)))
    # tmpCSIe1 = tmpPulse * (tmpCSIe1 * tmpNoise / (max(tmpNoise) - min(tmpNoise)))
    # tmpNoise = tmpPulse * (tmpNoise * tmpNoise / (max(tmpNoise) - min(tmpNoise)))

    # tmpCSIa1 = tmpPulse * (tmpCSIa1 + tmpNoise / (max(tmpNoise) - min(tmpNoise)))
    # tmpCSIb1 = tmpPulse * (tmpCSIb1 + tmpNoise / (max(tmpNoise) - min(tmpNoise)))
    # tmpCSIe1 = tmpPulse * (tmpCSIe1 + tmpNoise / (max(tmpNoise) - min(tmpNoise)))
    # tmpNoise = tmpPulse * (tmpNoise + tmpNoise / (max(tmpNoise) - min(tmpNoise)))

    sortCSIa1 = tmpCSIa1
    sortCSIb1 = tmpCSIb1
    sortCSIe1 = tmpCSIe1
    sortNoise = tmpNoise

    # 分段
    # CSIa1Orig[range(staInd, endInd, 1)] = tmpCSIa1
    # CSIb1Orig[range(staInd, endInd, 1)] = tmpCSIb1
    # CSIe1Orig[range(staInd, endInd, 1)] = tmpCSIe1
    # CSIn1Orig[range(staInd, endInd, 1)] = tmpNoise
    #
    # permLen = len(range(staInd, endInd, segLen))
    # origInd = np.array([xx for xx in range(staInd, endInd, segLen)])
    #
    # sortCSIa1 = np.zeros(permLen)
    # sortCSIb1 = np.zeros(permLen)
    # sortCSIe1 = np.zeros(permLen)
    # sortNoise = np.zeros(permLen)
    #
    # for ii in range(permLen):
    #     aIndVec = np.array([aa for aa in range(origInd[ii], origInd[ii] + segLen, 1)])
    #
    #     for jj in range(permLen, permLen * 2):
    #         bIndVec = np.array([bb for bb in range(origInd[jj - permLen], origInd[jj - permLen] + segLen, 1)])
    #
    #         CSIa1Tmp = CSIa1Orig[aIndVec]
    #         CSIb1Tmp = CSIb1Orig[bIndVec]
    #         CSIe1Tmp = CSIe1Orig[bIndVec]
    #         CSIn1Tmp = CSIn1Orig[bIndVec]
    #
    #         # sortCSIa1[ii] = np.mean(CSIa1Tmp)
    #         # sortCSIb1[jj - permLen] = np.mean(CSIb1Tmp)
    #         # sortCSIe1[jj - permLen] = np.mean(CSIe1Tmp)
    #         # sortNoise[ii - permLen] = np.mean(CSIn1Tmp)
    #
    #         sortCSIa1[ii] = msdc(CSIa1Tmp)
    #         sortCSIb1[jj - permLen] = msdc(CSIb1Tmp)
    #         sortCSIe1[jj - permLen] = msdc(CSIe1Tmp)
    #         sortNoise[ii] = msdc(CSIn1Tmp)

    # sortCSIa1 = smooth(np.array(sortCSIa1), window_len=segLen, window='flat')
    # sortCSIb1 = smooth(np.array(sortCSIb1), window_len=segLen, window='flat')
    # sortCSIe1 = smooth(np.array(sortCSIe1), window_len=segLen, window='flat')
    # sortNoise = smooth(np.array(sortNoise), window_len=segLen, window='flat')

    # sortCSIa1 = preprocessing.MinMaxScaler().fit_transform(np.array(sortCSIa1).reshape(-1, 1)).reshape(1, -1).tolist()[
    #     0]
    # sortCSIb1 = preprocessing.MinMaxScaler().fit_transform(np.array(sortCSIb1).reshape(-1, 1)).reshape(1, -1).tolist()[
    #     0]
    # sortCSIe1 = preprocessing.MinMaxScaler().fit_transform(np.array(sortCSIe1).reshape(-1, 1)).reshape(1, -1).tolist()[
    #     0]
    # sortNoise = preprocessing.MinMaxScaler().fit_transform(np.array(sortNoise).reshape(-1, 1)).reshape(1, -1).tolist()[
    #     0]

    # 取原数据的一部分来reshape
    # sortCSIa1 = sortCSIa1[0:segLen * int(len(sortCSIa1) / segLen)]
    # sortCSIb1 = sortCSIb1[0:segLen * int(len(sortCSIb1) / segLen)]
    # sortCSIe1 = sortCSIe1[0:segLen * int(len(sortCSIe1) / segLen)]
    # sortNoise = sortNoise[0:segLen * int(len(sortNoise) / segLen)]

    # 分段求和
    # sortCSIa1Reshape = np.array(sortCSIa1).reshape(int(len(sortCSIa1) / segLen), segLen)
    # sortCSIb1Reshape = np.array(sortCSIb1).reshape(int(len(sortCSIb1) / segLen), segLen)
    # sortCSIe1Reshape = np.array(sortCSIe1).reshape(int(len(sortCSIe1) / segLen), segLen)
    # sortNoiseReshape = np.array(sortNoise).reshape(int(len(sortNoise) / segLen), segLen)
    # sortCSIa1 = np.array(genSample(sortCSIa1Reshape, ratio))
    # sortCSIb1 = np.array(genSample(sortCSIb1Reshape, ratio))
    # sortCSIe1 = np.array(genSample(sortCSIe1Reshape, ratio))
    # sortNoise = np.array(genSample(sortNoiseReshape, ratio))

    # 差分产生segment
    # sortCSIa1 = []
    # sortCSIb1 = []
    # sortCSIe1 = []
    # sortNoise = []
    # for i in range(len(sortCSIa1Reshape)):
    #     # plt.figure()
    #     # plt.plot(sortCSIa1Reshape[i:i + segLen], "r")
    #     # plt.plot(sortCSIb1Reshape[i:i + segLen], "k")
    #     # plt.show()
    #     sortCSIa1.append(diffFilter(sortCSIa1Reshape[i:i + segLen]))
    #     sortCSIb1.append(diffFilter(sortCSIb1Reshape[i:i + segLen]))
    #     sortCSIe1.append(diffFilter(sortCSIe1Reshape[i:i + segLen]))
    #     sortNoise.append(diffFilter(sortNoiseReshape[i:i + segLen]))

    # 中值滤波产生segment
    # sortCSIa1 = []
    # sortCSIb1 = []
    # sortCSIe1 = []
    # sortNoise = []
    # for i in range(len(sortCSIa1Reshape)):
    #     sortCSIa1.append(medianFilter(sortCSIa1Reshape[i:i + segLen]))
    #     sortCSIb1.append(medianFilter(sortCSIb1Reshape[i:i + segLen]))
    #     sortCSIe1.append(medianFilter(sortCSIe1Reshape[i:i + segLen]))
    #     sortNoise.append(medianFilter(sortNoiseReshape[i:i + segLen]))

    # 求和产生segment
    # sortCSIa1 = np.array(genSample(sortCSIa1Reshape, ratio))
    # sortCSIb1 = np.array(genSample(sortCSIb1Reshape, ratio))
    # sortCSIe1 = np.array(genSample(sortCSIe1Reshape, ratio))
    # sortNoise = np.array(genSample(sortNoiseReshape, ratio))

    # np.random.seed(1)
    # combineSortCSIx1 = list(zip(sortCSIa1, sortCSIb1))
    # np.random.shuffle(combineSortCSIx1)
    # sortCSIa1, sortCSIb1 = zip(*combineSortCSIx1)
    # sortCSIa1 = np.array(sortCSIa1)
    # sortCSIb1 = np.array(sortCSIb1)

    # 最后各自的密钥
    a_list = []
    b_list = []
    e_list = []
    n_list = []

    if isShow:
        plt.figure()
        plt.hist(tmpCSIa1)
        plt.show()

        plt.figure()
        plt.hist(sortCSIa1)
        # plt.hist(sortCSIb1)
        plt.show()
        a_show = sortCSIa1.copy()
        b_show = sortCSIb1.copy()
        a_show.sort()
        b_show.sort()
        # a_show = a_show.reshape(int(len(a_show) / segLen), segLen)
        # b_show = b_show.reshape(int(len(b_show) / segLen), segLen)
        # a_show_com = []
        # b_show_com = []
        # for i in range(len(a_show)):
        #     # a_show_com.append(medianFilter(a_show[i]))
        #     # b_show_com.append(medianFilter(b_show[i]))
        #     a_show_com.append(sum(a_show[i]) / len(a_show[i]))
        #     b_show_com.append(sum(b_show[i]) / len(a_show[i]))
        # print(euclidean_metric(a_show_com, b_show_com))
        # print(euclidean_metric(a_show_com, b_show_com[1:] + b_show_com[:1]))
        # print(euclidean_metric(b_show_com, a_show_com[1:] + a_show_com[:1]))
        plt.figure()
        plt.scatter(list(range(len(a_show))), a_show, s=1.5, c="r")
        plt.scatter(list(range(len(b_show))), b_show, s=1.5, c="k")
        # plt.scatter(list(range(len(a_show_com))), a_show_com, s=1.5, c="r")
        # plt.scatter(list(range(len(b_show_com))), b_show_com, s=1.5, c="k")
        plt.show()
        exit()

    # np.random.seed(1)
    # combineMetric = list(zip(sortCSIa1, sortCSIb1, sortCSIe1, sortNoise))
    # np.random.shuffle(combineMetric)
    # sortCSIa2, sortCSIb2, sortCSIe2, sortNois2 = zip(*combineMetric)

    # np.random.seed(1)
    # combineMetric = list(zip(sortCSIa2, sortCSIb2, sortCSIe2, sortNois2))
    # np.random.shuffle(combineMetric)
    # sortCSIa3, sortCSIb3, sortCSIe3, sortNois3 = zip(*combineMetric)

    combineCSIa1 = []
    for i in range(len(sortCSIa1)):
        combineCSIa1.append(pow(sortCSIa1[i], 2) + pow(i / dataLen, 2))
    combineCSIb1 = []
    for i in range(len(sortCSIb1)):
        combineCSIb1.append(pow(sortCSIb1[i], 2) + pow(i / dataLen, 2))
    combineCSIe1 = []
    for i in range(len(sortCSIe1)):
        combineCSIe1.append(pow(sortCSIe1[i], 2) + pow(i / dataLen, 2))
    combineNoise = []
    for i in range(len(sortNoise)):
        combineNoise.append(pow(sortNoise[i], 2) + pow(i / dataLen, 2))

    # combineCSIa1 = []
    # for i in range(len(sortCSIa1)):
    #     combineCSIa1.append(pow(sortCSIa1[i], 2) + pow(sortCSIa2[i], 2))
    # combineCSIb1 = []
    # for i in range(len(sortCSIb1)):
    #     combineCSIb1.append(pow(sortCSIb1[i], 2) + pow(sortCSIb2[i], 2))
    # combineCSIe1 = []
    # for i in range(len(sortCSIe1)):
    #     combineCSIe1.append(pow(sortCSIe1[i], 2) + pow(sortCSIe2[i], 2))
    # combineNoise = []
    # for i in range(len(sortNoise)):
    #     combineNoise.append(pow(sortNoise[i], 2) + pow(sortNois2[i], 2))

    # a_metric, publish, a_list_number = sortSegPermOfA(list(combineCSIa1), segLen)
    # b_metric, b_list_number = sortSegPermOfB(publish, list(combineCSIb1), segLen)
    # e_metric, e_list_number = sortSegPermOfB(publish, list(combineCSIe1), segLen)
    # n_metric, n_list_number = sortSegPermOfB(publish, list(combineNoise), segLen)
    a_metric, publish, a_list_number = sortSegPermOfA(list(sortCSIa1), segLen)
    b_metric, b_list_number = sortSegPermOfB(publish, list(sortCSIb1), segLen)
    e_metric, e_list_number = sortSegPermOfB(publish, list(sortCSIe1), segLen)
    n_metric, n_list_number = sortSegPermOfB(publish, list(sortNoise), segLen)
    # publishOfB, b_metric, b_list_number = refSortSegPermOfB(publish, list(sortCSIb1), segLen)
    # _, e_metric, e_list_number = refSortSegPermOfB(publish, list(sortCSIe1), segLen)
    # _, n_metric, n_list_number = refSortSegPermOfB(publish, list(sortNoise), segLen)
    # 无论B是否发送给A，A的密钥都不变，因为是严格递增
    # a_metric, a_list_number = refSortSegPermOfA(publishOfB, list(sortCSIa1))

    # a_list_number = list(np.argsort(sortCSIa1))
    # b_list_number = list(np.argsort(sortCSIb1))
    # e_list_number = list(np.argsort(sortCSIe1))
    # n_list_number = list(np.argsort(sortNoise))

    # a_list_number = []
    # b_list_number = []
    # e_list_number = []
    # n_list_number = []
    #
    # for i in range(len(a_metric) -1):
    #     if a_metric[i + 1] - a_metric[i] > 0:
    #         a_list_number.append(1)
    #     else:
    #         a_list_number.append(0)
    #     if b_metric[i + 1] - b_metric[i] > 0:
    #         b_list_number.append(1)
    #     else:
    #         b_list_number.append(0)
    #     if e_metric[i + 1] - e_metric[i] > 0:
    #         e_list_number.append(1)
    #     else:
    #         e_list_number.append(0)
    #     if n_metric[i + 1] - n_metric[i] > 0:
    #         n_list_number.append(1)
    #     else:
    #         n_list_number.append(0)

    # a_metric, publish, a_list_number = adaSortSegPermOfA(list(sortCSIa1), segLen)
    # b_metric, b_list_number = adaSortSegPermOfB(publish, list(sortCSIb1))
    # _, e_list_number = adaSortSegPermOfB(publish, list(sortCSIe1))
    # _, n_list_number = adaSortSegPermOfB(publish, list(sortNoise))

    # print("a", a_metric)
    # print("b", b_metric)

    # plt.figure()
    # plt.scatter(list(range(len(sortCSIa1))), sortCSIa1, s=1.5, c="r")
    # plt.scatter(list(range(len(sortCSIb1))), sortCSIb1, s=1.5, c="k")
    # plt.show()

    # sortCSIa1Reshape = sortCSIa1Reshape.reshape(int(len(sortCSIa1Reshape) / segLen), segLen)
    # sortCSIb1Reshape = sortCSIb1Reshape.reshape(int(len(sortCSIb1Reshape) / segLen), segLen)

    # plt.figure()
    # plt.scatter(list(range(len(sortCSIa1Reshape))),
    #             genSample(sortCSIa1Reshape, ratio), s=1.5, c="r")
    # plt.scatter(list(range(len(sortCSIb1Reshape))),
    #             genSample(sortCSIb1Reshape, ratio), s=1.5, c="k")
    # plt.show()

    # a_metric.sort()
    # b_metric.sort()
    # print("a_sort", a_metric)
    # print("b_sort", b_metric)

    a_mis_metric = []
    b_mis_metric = []

    # for i in range(1, len(a_list_number) -1):
    #     if a_list_number[i] != b_list_number[i]:
    #         a_mis_metric.append(a_metric[i + 1] - a_metric[i])
    #         b_mis_metric.append(b_metric[i + 1] - b_metric[i])
    #         print("mis", a_metric[i + 1] - a_metric[i], a_metric[i] - a_metric[i - 1])
    #         print("mis", b_metric[i + 1] - b_metric[i], b_metric[i] - b_metric[i - 1])
    #     if a_list_number[i] == b_list_number[i]:
    #         print("match", a_metric[i + 1] - a_metric[i], a_metric[i] - a_metric[i - 1])
    #         print("match", b_metric[i + 1] - b_metric[i], b_metric[i] - b_metric[i - 1])
    # print("a_mis", len(a_mis_metric), max(a_mis_metric), min(a_mis_metric), np.mean(a_mis_metric))
    # print("b_mis", len(b_mis_metric), max(b_mis_metric), min(b_mis_metric), np.mean(b_mis_metric))

    a_mis_diff = 0
    b_mis_diff = 0

    # a_diff = sys.maxsize
    # b_diff = sys.maxsize
    # a_index = -1
    # b_index = -1
    # a_metric.sort()
    # b_metric.sort()
    # for i in range(len(a_metric) - 1):
    #     a_diff = min(a_diff, abs(a_metric[i + 1] - a_metric[i]))
    #     if a_diff == abs(a_metric[i + 1] - a_metric[i]):
    #         a_index = i
    # for i in range(len(b_metric) - 1):
    #     b_diff = min(b_diff, abs(b_metric[i + 1] - b_metric[i]))
    #     if b_diff == abs(b_metric[i + 1] - b_metric[i]):
    #         b_index = i
    # print("a_diff", a_index, a_metric[a_index], a_metric[a_index + 1], a_diff)
    # print("b_diff", b_index, b_metric[b_index], b_metric[b_index + 1], b_diff)

    # min_diff = sys.maxsize
    # min_index = 0
    # for i in range(len(a_metric)):
    #     min_diff = min(min_diff, abs(a_metric[i] - b_metric[i]))
    #     if min_diff == abs(a_metric[i] - b_metric[i]):
    #         min_index = i
    # print("min_diff", min_index, a_metric[min_index], b_metric[min_index], min_diff)

    # it = 5
    # wi = 1
    # a_list_number = overlap_moving_sum(a_list_number, it, wi)
    # b_list_number = overlap_moving_sum(b_list_number, it, wi)
    # e_list_number = overlap_moving_sum(e_list_number, it, wi)
    # n_list_number = overlap_moving_sum(n_list_number, it, wi)

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

    # 画出metric和对应的索引
    # if sum2 / sum1 < 0.95:
    #     plt.figure()
    #     plt.scatter(list(range(len(a_metric))), a_metric, s=1.5, c="r")
    #     plt.scatter(list(range(len(b_metric))), b_metric, s=1.5, c="k")
    #     for i in range(len(a_metric)):
    #         plt.annotate(a_list_number[i], (i, a_metric[i] + 0.1), c="r")
    #         plt.annotate(b_list_number[i], (i, a_metric[i] - 0.1), c="k")
    #     plt.show()
    #
    #     a_metric.sort()
    #     b_metric.sort()
    #
    #     plt.figure()
    #     plt.scatter(list(range(len(a_metric))), a_metric, s=1.5, c="r")
    #     plt.scatter(list(range(len(b_metric))), b_metric, s=1.5, c="k")
    #     for i in range(len(a_metric)):
    #         plt.annotate(list(np.array(a_metric).argsort().argsort())[i], (i, a_metric[i] + 0.1), c="r")
    #         plt.annotate(list(np.array(b_metric).argsort().argsort())[i], (i, a_metric[i] - 0.1), c="k")
    #     plt.show()
    #     exit()

    if sum1 == 0:
        continue
    if sum2 == sum1:
        print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
    else:
        print("\033[0;31;40ma-b", "bad", sum2, sum2 / sum1, "\033[0m")
        # a_pair = []
        # b_pair = []
        # for i in range(len(a_list_number)):
        #     if a_list_number[i] != b_list_number[i]:
        #         a_pair.append(a_metric[i])
        #         b_pair.append(b_metric[i])
        #         print(a_list_number[i], b_list_number[i])
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
print("times", times)
print("测试结束，耗时" + str(round(time.time() - start_time, 3)), "s")
# csv.write(fileName + ',' + str(times) + ',' + str(keyLen) + ',' + str(segLen) + ',' + str(interval)
#           + ',' + str(correctSum) + " / " + str(originSum) + " = " + str(round(correctSum / originSum, 10))
#           + ',' + str(randomSum) + " / " + str(originSum) + " = " + str(round(randomSum / originSum, 10))
#           + ',' + str(noiseSum) + " / " + str(originSum) + " = " + str(round(noiseSum / originSum, 10))
#           + ',' + str(correctWholeSum) + " / " + str(originWholeSum) + " = " + str(
#     round(correctWholeSum / originWholeSum, 10))
#           + ',' + str(randomWholeSum) + " / " + str(originWholeSum) + " = " + str(
#     round(randomWholeSum / originWholeSum, 10))
#           + ',' + str(noiseWholeSum) + " / " + str(originWholeSum) + " = " + str(
#     round(noiseWholeSum / originWholeSum, 10)) + '\n')
