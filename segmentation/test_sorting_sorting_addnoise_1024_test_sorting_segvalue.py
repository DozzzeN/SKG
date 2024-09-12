import csv
import math
import time
from tkinter import messagebox

import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import dct
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr, boxcox
from scipy.spatial import distance

def segAndSum(data, segLen):
    res = []
    for i in range(len(data)):
        res.append(sum(data[i * segLen:(i + 1) * segLen]))
    return res


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

# 最原始的基于排序索引分段的密钥生成+排序前后是否分段
fileName = ["../data/data_mobile_indoor_1.mat"]


# segLen = 4    keyLen = 1024 * segLen  无纠错
# mi1 normal                            0.999674479 0.666666667 2.0 1.999348958
# mi1 no sorting with perturbation      0.999772135 0.666666667 2.0 1.999544271

# segLen = 5    keyLen = 1024 * segLen  无纠错
# mi1 normal                            0.998168945 0.0         2.5 2.495422363
# mi1 no sorting with perturbation      0.99921875  0.25        2.5 2.498046875

# 分段内索引值洗牌
# mi1 with segSort & perturbation       0.999772135 0.666666667 2.0 1.999544271
# mi1 normal with segSort               0.999804687 0.666666667 2.0 1.999609375

# 全体索引值洗牌
# mi1 no sorting with shuffle           0.999772135 0.666666667 2.0 1.999544271
# mi1 normal with shuffle               1.0         1.0         2.0 2.0

# 全体索引值洗牌及分段内索引值洗牌
# mi1 no sorting with shuffle segSort   0.999772135 0.666666667 2.0 1.999544271
# mi1 normal with shuffle segSort       1.0         1.0         2.0 2.0

# 不同的距离度量（之前的测试都是基于city距离）
# mi1 normal city distance              0.999674479 0.666666667 2.0 1.999348958
# mi1 normal Euclidean distance         0.999902344 0.666666667 2.0 1.999804687
# mi1 normal cosine distance            0.998307292 0.0         2.0 1.996614583
# mi1 normal sum distance               0.524967448 0.0         2.0 1.049934896

# segLen = 2    keyLen = 4 * segLen     无纠错
# mi1 normal Euclidean distance         0.928873094 0.77750309  1.0 0.928873094
# mi1 normal with shuffle               0.936701689 0.769262464 1.0 0.936701689
# mi1 normal with shuffle segSort       0.930984755 0.756489493 1.0 0.930984755
# mi1 normal with segSort               0.93644417  0.790688092 1.0 0.93644417


isShuffle = False
isSegSort = False
isValueShuffle = False

# 是否纠错
rec = False

# 是否排序
# withoutSorts = [True, False]
withoutSorts = [False]
# 是否添加噪声
# addNoises = ["mul", ""]
addNoises = ["mul"]

for f in fileName:
    for addNoise in addNoises:
        for withoutSort in withoutSorts:
            print(f)
            rawData = loadmat(f)

            if f.find("data_alignment") != -1:
                CSIa1Orig = rawData['csi'][:, 0]
                CSIb1Orig = rawData['csi'][:, 1]
                CSIe1Orig = rawData['csi'][:, 2]
            elif f.find("csi") != -1:
                CSIa1Orig = rawData['testdata'][:, 0]
                CSIb1Orig = rawData['testdata'][:, 1]
            else:
                CSIa1Orig = rawData['A'][:, 0]
                CSIb1Orig = rawData['A'][:, 1]

            dataLen = len(CSIa1Orig)
            print("dataLen", dataLen)

            # 不如将索引分段长度置为4，排序前进行分段的效果不明显
            segLen1 = 2     # 测量值进行排序前的分段长度
            segLen2 = 2     # 测量值进行排序后的（索引）分段长度
            segLen = segLen1 * segLen2
            keyLen = 16 * segLen
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

            if withoutSort:
                if addNoise == "mul":
                    print("no sorting")
            if withoutSort:
                if addNoise == "":
                    print("no sorting and no perturbation")
            if withoutSort is False:
                if addNoise == "":
                    print("no perturbation")
                if addNoise == "mul":
                    print("normal")
            if isShuffle:
                print("with shuffle")
            if isSegSort:
                print("with segSort")

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

                if f.find("data_alignment") != -1:
                    CSIa1Orig = rawData['csi'][:, 0]
                    CSIb1Orig = rawData['csi'][:, 1]
                    CSIe1Orig = rawData['csi'][:, 2]
                elif f.find("csi") != -1:
                    CSIa1Orig = rawData['testdata'][:, 0]
                    CSIb1Orig = rawData['testdata'][:, 1]
                else:
                    CSIa1Orig = rawData['A'][:, 0]
                    CSIb1Orig = rawData['A'][:, 1]

                if f == "../data/data_NLOS.mat":
                    # 先整体shuffle一次
                    shuffleInd = np.random.permutation(dataLen)
                    CSIa1Orig = CSIa1Orig[shuffleInd]
                    CSIb1Orig = CSIb1Orig[shuffleInd]
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
                    np.random.seed(10000)
                    randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
                    # 均值化
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    # tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1))
                    # tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1))
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
                    # tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1))
                    # tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1))

                # 最后各自的密钥
                a_list = []
                b_list = []

                # with value shuffling
                if isValueShuffle:
                    np.random.seed(10000)
                    combineCSIx1Orig = list(zip(tmpCSIa1, tmpCSIb1))
                    np.random.shuffle(combineCSIx1Orig)
                    tmpCSIa1, tmpCSIb1 = zip(*combineCSIx1Orig)
                    tmpCSIa1 = list(tmpCSIa1)
                    tmpCSIb1 = list(tmpCSIb1)

                # segment value
                tmpCSIa1 = segAndSum(tmpCSIa1, segLen1)
                tmpCSIb1 = segAndSum(tmpCSIb1, segLen1)

                # without sorting
                # print(pearsonr(tmpCSIa1, tmpCSIb1)[0])
                if withoutSort:
                    tmpCSIa1Ind = np.array(tmpCSIa1)
                    tmpCSIb1Ind = np.array(tmpCSIb1)
                else:
                    tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
                    tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()
                    # print(pearsonr(tmpCSIa1Ind, tmpCSIb1Ind)[0])

                    # with shuffling
                    if isShuffle:
                        np.random.seed(0)
                        combineCSIx1Orig = list(zip(tmpCSIa1Ind, tmpCSIb1Ind))
                        np.random.shuffle(combineCSIx1Orig)
                        tmpCSIa1Ind, tmpCSIb1Ind = zip(*combineCSIx1Orig)
                        tmpCSIa1Ind = list(tmpCSIa1Ind)
                        tmpCSIb1Ind = list(tmpCSIb1Ind)
                        # print(pearsonr(tmpCSIa1Ind, tmpCSIb1Ind)[0])

                minEpiIndClosenessLsb = np.zeros(int(keyLen / segLen2), dtype=int)

                # with segSort
                if isSegSort:
                    if withoutSort is False:
                        for i in range(int(keyLen / segLen2)):
                            epiInda1 = tmpCSIa1Ind[i * segLen2:(i + 1) * segLen2]
                            epiIndb1 = tmpCSIb1Ind[i * segLen2:(i + 1) * segLen2]

                            np.random.seed(i)
                            combineEpiIndx1 = list(zip(epiInda1, epiIndb1))
                            np.random.shuffle(combineEpiIndx1)
                            epiInda1, epiIndb1 = zip(*combineEpiIndx1)

                            tmpCSIa1Ind[i * segLen2:(i + 1) * segLen2] = epiInda1
                            tmpCSIb1Ind[i * segLen2:(i + 1) * segLen2] = epiIndb1
                        # print(pearsonr(tmpCSIa1Ind, tmpCSIb1Ind)[0])

                tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen2), segLen2)
                tmpCSIa1IndBack = tmpCSIa1Ind.copy()
                permutation = list(range(int(keyLen / segLen2)))
                combineMetric = list(zip(tmpCSIa1IndReshape, permutation))
                np.random.seed(staInd)
                np.random.shuffle(combineMetric)
                tmpCSIa1IndReshape, permutation = zip(*combineMetric)
                tmpCSIa1Ind = np.hstack((tmpCSIa1IndReshape))

                for i in range(int(keyLen / segLen2)):
                    epiInda1 = tmpCSIa1Ind[i * segLen2:(i + 1) * segLen2]

                    epiIndClosenessLsb = np.zeros(int(keyLen / segLen2))

                    for j in range(int(keyLen / segLen2)):
                        epiIndb1 = tmpCSIb1Ind[j * segLen2:(j + 1) * segLen2]

                        # 欧式距离度量更好
                        epiIndClosenessLsb[j] = sum(np.square(epiIndb1 - np.array(epiInda1)))

                        # epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))
                        # epiIndClosenessLsb[j] = distance.cosine(epiIndb1, np.array(epiInda1))
                        # epiIndClosenessLsb[j] = abs(sum(epiIndb1) - sum(np.array(epiInda1)))

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

                # if sum1 != sum2:
                #     print()

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

                                # 第一个找到的错误的，将其距离置为最大，下次找到的就是第二个，作为正确结果
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

            print("\033[0;34;40ma-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 9), "\033[0m")
            print("\033[0;34;40ma-b whole match", correctWholeSum, "/", originWholeSum, "=",
                  round(correctWholeSum / originWholeSum, 9), "\033[0m")

            print(round(correctSum / originSum, 9), round(correctWholeSum / originWholeSum, 9),
                  round(originSum / times / keyLen, 9),
                  round(correctSum / times / keyLen, 9))
            if withoutSort:
                print("withoutSort")
            else:
                print("withSort")
            print("\n")