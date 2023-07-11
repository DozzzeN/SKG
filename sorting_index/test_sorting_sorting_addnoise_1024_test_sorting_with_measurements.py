import csv
import math
import time
from tkinter import messagebox

import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import dct
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr, boxcox

from zca import ZCA


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


# fileName = ["../data/data_mobile_indoor_1.mat",
#             "../data/data_mobile_outdoor_1.mat",
#             "../data/data_static_outdoor_1.mat",
#             "../data/data_static_indoor_1.mat"
#             ]

fileName = ["../data/data_static_indoor_1.mat"]

# ../data/data_static_indoor_1.mat
# dataLen 100500
# segLen 5
# keyLen 1024.0
# normal
# a-b all 184253 / 184320 = 0.9996365017
# a-b whole match 9 / 18 = 0.5
# 0.9996365017 0.5 2.0 1.9992730034722221
#
#
# ../data/data_static_indoor_1.mat
# dataLen 100500
# segLen 5
# keyLen 1024.0
# no perturbation
# a-b all 121614 / 184320 = 0.6597981771
# a-b whole match 0 / 18 = 0.0
# 0.6597981771 0.0 2.0 1.3195963541666667

# 是否纠错
rec = False

# 是否添加噪声
addNoises = ["mul", ""]

for f in fileName:
    for addNoise in addNoises:
        print(f)
        rawData = loadmat(f)

        CSIa1Orig = rawData['A'][:, 0]
        CSIb1Orig = rawData['A'][:, 1]
        dataLen = len(CSIa1Orig)
        print("dataLen", dataLen)

        segLen = 5
        keyLen = 1024 * segLen
        tell = True

        print("segLen", segLen)
        print("keyLen", keyLen / segLen)

        originSum = 0
        correctSum = 0

        originDecSum = 0
        correctDecSum = 0

        originWholeSum = 0
        correctWholeSum = 0

        times = 0
        overhead = 0

        if addNoise == "":
            print("no perturbation")
        if addNoise == "mul":
            print("normal")

        dataLenLoop = dataLen
        keyLenLoop = keyLen
        if f == "../data/data_static_indoor_1.mat":
            dataLenLoop = int(dataLen / 5.5)
            keyLenLoop = int(keyLen / 5)
        for staInd in range(0, dataLenLoop, keyLenLoop):
            start = time.time()
            endInd = staInd + keyLen
            # print("range:", staInd, endInd)
            if endInd >= len(CSIa1Orig):
                break

            times += 1

            CSIa1Orig = rawData['A'][:, 0]
            CSIb1Orig = rawData['A'][:, 1]

            seed = np.random.randint(100000)
            np.random.seed(seed)

            # 固定随机置换的种子
            # np.random.seed(0)
            # combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig))
            # np.random.shuffle(combineCSIx1Orig)
            # CSIa1Orig, CSIb1Orig = zip(*combineCSIx1Orig)
            # CSIa1Orig = np.array(CSIa1Orig)
            # CSIb1Orig = np.array(CSIb1Orig)

            CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
            CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')

            tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
            tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]

            # 目的是把加噪音+无排序的结果降下来
            if addNoise == "mul":
                # randomMatrix = np.random.randint(0, 2, size=(keyLen, keyLen))
                # randomMatrix = np.random.uniform(0, 1, size=(keyLen, keyLen))
                randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
                tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
                tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)

                # 相当于乘了一个置换矩阵 permutation matrix
                # np.random.seed(0)
                # combineCSIx1Orig = list(zip(tmpCSIa1, tmpCSIb1))
                # np.random.shuffle(combineCSIx1Orig)
                # tmpCSIa1, tmpCSIb1 = zip(*combineCSIx1Orig)
                # tmpCSIa1 = np.array(tmpCSIa1)
                # tmpCSIb1 = np.array(tmpCSIb1)
            else:
                tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)

            # 最后各自的密钥
            a_list = []
            b_list = []

            tmpCSIa1Ind = np.array(tmpCSIa1)
            tmpCSIb1Ind = np.array(tmpCSIb1)

            tmpCSIa1Sort = np.array(tmpCSIa1).argsort().argsort()
            tmpCSIb1Sort = np.array(tmpCSIb1).argsort().argsort()

            for i in range(int(keyLen / segLen)):
                epiInda1 = tmpCSIa1Sort[i * segLen:(i + 1) * segLen]
                epiIndb1 = tmpCSIb1Sort[i * segLen:(i + 1) * segLen]

                np.random.seed(i)
                combineEpiIndx1 = list(zip(epiInda1, epiIndb1))
                np.random.shuffle(combineEpiIndx1)
                epiInda1, epiIndb1 = zip(*combineEpiIndx1)

                tmpCSIa1Sort[i * segLen:(i + 1) * segLen] = epiInda1
                tmpCSIb1Sort[i * segLen:(i + 1) * segLen] = epiIndb1

            tmpCSIa1Ind = np.append(tmpCSIa1Ind, tmpCSIa1Sort)
            tmpCSIb1Ind = np.append(tmpCSIb1Ind, tmpCSIb1Sort)

            minEpiIndClosenessLsb = np.zeros(int(keyLen / segLen), dtype=int)

            tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen) * 2, segLen)
            permutation = list(range(int(keyLen / segLen)))
            combineMetric = list(zip(tmpCSIa1IndReshape, permutation))
            np.random.seed(staInd)
            np.random.shuffle(combineMetric)
            tmpCSIa1IndReshape, permutation = zip(*combineMetric)
            tmpCSIa1Ind = np.hstack((tmpCSIa1IndReshape))

            for i in range(int(keyLen / segLen)):
                epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]

                epiIndClosenessLsb = np.zeros(int(keyLen / segLen))

                for j in range(int(keyLen / segLen)):
                    epiIndb1 = tmpCSIb1Ind[j * segLen:(j + 1) * segLen]

                    epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))

                minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)

            a_list_number = list(permutation)
            b_list_number = list(minEpiIndClosenessLsb)

            # 转成二进制，0填充成0000
            for i in range(len(a_list_number)):
                number = bin(a_list_number[i])[2:].zfill(int(np.log2(len(a_list_number))))
                a_list += number
            for i in range(len(b_list_number)):
                number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
                b_list += number

            sum1 = min(len(a_list), len(b_list))
            sum2 = 0

            for i in range(0, sum1):
                sum2 += (a_list[i] == b_list[i])

            end = time.time()
            overhead += end - start
            # print("time:", end - start)

            # 自适应纠错
            if sum1 != sum2 and rec:
                if tell:
                    # print("correction")
                    # a告诉b哪些位置出错，b对其纠错
                    for i in range(len(a_list_number)):
                        if a_list_number[i] != b_list_number[i]:
                            epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]

                            epiIndClosenessLsb = np.zeros(int(keyLen / segLen))

                            for j in range(int(keyLen / segLen)):
                                epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
                                epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))

                            min_b = np.argmin(epiIndClosenessLsb)
                            epiIndClosenessLsb[min_b] = keyLen * segLen
                            b_list_number[i] = np.argmin(epiIndClosenessLsb)

                            b_list = []

                            for i in range(len(b_list_number)):
                                number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
                                b_list += number

                            sum2 = 0
                            for i in range(0, min(len(a_list), len(b_list))):
                                sum2 += (a_list[i] == b_list[i])

            # print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
            originSum += sum1
            correctSum += sum2

            originWholeSum += 1
            correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum

        print("\033[0;34;40ma-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10), "\033[0m")
        print("\033[0;34;40ma-b whole match", correctWholeSum, "/", originWholeSum, "=",
              round(correctWholeSum / originWholeSum, 10), "\033[0m")
        # print("times", times)
        # print(overhead / originWholeSum)

        print(round(correctSum / originSum, 10), round(correctWholeSum / originWholeSum, 10),
              originSum / times / keyLen,
              correctSum / times / keyLen)
        print("\n")
