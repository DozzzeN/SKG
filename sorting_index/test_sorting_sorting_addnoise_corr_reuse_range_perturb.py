import csv
import math
import random
import time
from itertools import chain
from tkinter import messagebox

import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import dct
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr, boxcox
from pyentrp import entropy as ent
import entropy_estimators as ee

from zca import ZCA
import warnings

# 忽略常数的相关系数不存在的警告
warnings.filterwarnings("ignore")


# 定义计算离散点积分的函数
def integral_from_start(x, y):
    import scipy
    from scipy.integrate import simps  # 用于计算积分
    integrals = []
    for i in range(len(y)):  # 计算梯形的面积，由于是累加，所以是切片"i+1"
        integrals.append(scipy.integrate.trapz(y[:i + 1], x[:i + 1]))
    return integrals


# 定义计算离散点导数的函数
def derivative(x, y):  # x, y的类型均为列表
    diff_x = []  # 用来存储x列表中的两数之差
    for i, j in zip(x[0::], x[1::]):
        diff_x.append(j - i)

    diff_y = []  # 用来存储y列表中的两数之差
    for i, j in zip(y[0::], y[1::]):
        diff_y.append(j - i)

    slopes = []  # 用来存储斜率
    for i in range(len(diff_y)):
        slopes.append(diff_y[i] / diff_x[i])

    deriv = []  # 用来存储一阶导数
    for i, j in zip(slopes[0::], slopes[1::]):
        deriv.append((0.5 * (i + j)))  # 根据离散点导数的定义，计算并存储结果
    deriv.insert(0, slopes[0])  # (左)端点的导数即为与其最近点的斜率
    deriv.append(slopes[-1])  # (右)端点的导数即为与其最近点的斜率
    return deriv  # 返回存储一阶导数结果的列表


def integral_sq_derivative_increment(data, noise):
    index = list(range(len(data)))
    intgrl = integral_from_start(index, data)
    # square = np.power(intgrl, 2)
    square = intgrl + noise
    diff = derivative(index, square)
    return diff


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


def normal2uniform(data):
    data1 = data[:int(len(data) / 2)]
    data2 = data[int(len(data) / 2):]
    data_reshape = np.array(data[0: 2 * int(len(data) / 2)])
    data_reshape = data_reshape.reshape(int(len(data_reshape) / 2), 2)
    x_list = []
    for i in range(len(data_reshape)):
        # r = np.sum(np.square(data_reshape[i]))
        r = np.sum(data1[i] * data1[i] + data2[i] * data2[i])
        x_list.append(np.exp(-0.5 * r))
        x_list.append(np.exp(-0.5 * r))
        # r = data2[i] / data1[i]
        # x_list.append(math.atan(r) / math.pi + 0.5)
        # x_list.append(math.atan(r) / math.pi + 0.5)

    return x_list


# rawData = loadmat("../data/data_mobile_outdoor_1.mat")
rawData = loadmat("../skyglow/Scenario2-Office-LoS/data3_upto5.mat")
corrMatName = "corr_si.mat"
rssMatName = "corr_si_rss.mat"

# stalking attack
CSIe2Orig = loadmat("../skyglow/Scenario2-Office-LoS/data3_eave_upto5.mat")['A'][:, 0]
csi_csv = open("./perturb.csv", "w")

# data BMR BGR BGR-with-no-error
# si1 - for staInd in range(0, int(dataLen / 5.5), int(keyLen / 10)):
# mi1 1.0 1.0 2.0 2.0
# si1 1.0 1.0 2.0 2.0
# mo1 1.0 1.0 2.0 2.0
# 因为so的数据长度没有补齐
# so1 1.0 1.0 1.8461538461538463 1.8461538461538463

# CSIa1Orig = rawData['A'][:, 0][0: 20000]
# CSIb1Orig = rawData['A'][:, 1][0: 20000]
CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
dataLen = len(CSIa1Orig)

segLen = 7
keyLen = 256 * segLen
rec = False
tell = True
singular = True
attackerOrUser = "user"

originSum = 0
correctSum = 0
randomSum1 = 0
randomSum2 = 0
noiseSum1 = 0
noiseSum2 = 0

originDecSum = 0
correctDecSum = 0
randomDecSum1 = 0
randomDecSum2 = 0
noiseDecSum1 = 0
noiseDecSum2 = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum1 = 0
randomWholeSum2 = 0
noiseWholeSum1 = 0
noiseWholeSum2 = 0

trueKey = []
legiKey = []
randomKey = []
inferKey = []
imitKey = []
stalkKey = []

times = 0
overhead = 0

addNoise = "mul"
codings = ""

reuseTimes = 2

for erange in range(3, 6):
    keyECorr = []
    keyBCorr = []
    # static indoor
    for staInd in range(0, int(dataLen / 5.5), int(keyLen / 10)):
        # for staInd in range(0, dataLen, int(keyLen / 10)):
        # for staInd in range(0, dataLen, keyLen):

        ke1 = []
        ke2 = []

        kb1 = []
        kb2 = []
        for reuse in range(reuseTimes):
            start = time.time()
            endInd = staInd + keyLen
            # print("range:", staInd, endInd)
            if endInd >= len(CSIa1Orig) or endInd >= len(CSIe2Orig):
                break
            times += 1

            # np.random.seed(1)
            CSIa1Orig = rawData['A'][:, 0]
            CSIb1Orig = rawData['A'][:, 1]

            # imitation attack
            CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=len(CSIa1Orig))

            # noiseOrig = np.random.normal(np.mean(CSIa1Orig), np.std(CSIa1Orig), size=len(CSIa1Orig))
            # noiseOrig = np.random.normal(0, np.std(CSIa1Orig), size=len(CSIa1Orig))
            # np.random.seed(int(seeds[times - 1][0]))
            seed = np.random.randint(100000)
            np.random.seed(seed)

            tmpCSIa1b = []
            tmpCSIb1b = []
            tmpCSIe1b = []
            tmpCSIe2b = []
            tmpNoiseb = []

            if addNoise == "mul":
                # 静态数据需要置换
                # 固定随机置换的种子
                # np.random.seed(0)
                # combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig))
                # np.random.shuffle(combineCSIx1Orig)
                # CSIa1Orig, CSIb1Orig = zip(*combineCSIx1Orig)

                CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
                CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')

                tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
                tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
                tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
                tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]

                perturbation = np.ones(keyLen)
                for i in range(keyLen):
                    if random.random() < 0.5:
                        perturbation[i] = perturbation[i] * erange
                    else:
                        perturbation[i] = perturbation[i] * -erange
                if reuse > 0:
                    # tmpCSIe2 = tmpCSIe2 + np.random.uniform(-erange, erange, keyLen)
                    # tmpCSIe2 = tmpCSIe2 + perturbation
                    # tmpCSIb1 = tmpCSIb1 + np.random.uniform(-erange, erange, keyLen)
                    tmpCSIb1 = tmpCSIb1 + perturbation

                # 重新生成randomMatrix
                if singular:
                    # 使用奇异矩阵
                    seed = staInd
                    np.random.seed(seed)
                    randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen - 1))
                    linearElement = []
                    np.random.seed(seed)
                    randomCoff = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=keyLen - 1)
                    for i in range(keyLen):
                        # linearElement.append([np.sum(randomMatrix[i]) / len(randomMatrix[i])])
                        linearElement.append(np.sum(np.multiply(randomCoff, randomMatrix[i])))
                    # 随机选一列插入
                    np.random.seed(seed)
                    randomIndex = np.random.randint(0, keyLen)
                    randomMatrix = np.insert(randomMatrix, randomIndex, linearElement, axis=1)
                    # print("round", reuse)
                    # print(np.array(randomMatrix))
                else:
                    randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))

                tmpCSIa1b = tmpCSIa1 - np.mean(tmpCSIa1)
                tmpCSIb1b = tmpCSIb1 - np.mean(tmpCSIb1)
                tmpCSIe1b = tmpCSIe1 - np.mean(tmpCSIe1)
                tmpCSIe2b = tmpCSIe2 - np.mean(tmpCSIe2)
                tmpNoiseb = np.ones(keyLen)
                tmpCSIa1 = np.matmul(tmpCSIa1b, randomMatrix)
                tmpCSIb1 = np.matmul(tmpCSIb1b, randomMatrix)
                tmpCSIe1 = np.matmul(tmpCSIe1b, randomMatrix)
                tmpCSIe2 = np.matmul(tmpCSIe2b, randomMatrix)

                # inference attack
                tmpNoise = np.matmul(tmpNoiseb, randomMatrix)

                # 相关系数降低
                # for i in range(len(tmpCSIa1) - 1):
                #     tmpCSIa1[i] = tmpCSIa1[i] * 10 + tmpCSIa1[i + 1]
                #     tmpCSIb1[i] = tmpCSIb1[i] * 10 + tmpCSIb1[i + 1]
                #     tmpCSIe1[i] = tmpCSIe1[i] * 10 + tmpCSIe1[i + 1]
                #     tmpCSIe2[i] = tmpCSIe2[i] * 10 + tmpCSIe2[i + 1]
                #     tmpNoise[i] = tmpNoise[i] * 10 + tmpNoise[i + 1]
            else:
                tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
                tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
                tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
                tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]
                tmpNoise = np.random.normal(0, np.std(CSIa1Orig), size=keyLen)

            # 最后各自的密钥
            a_list = []
            b_list = []
            e1_list = []
            e2_list = []
            n1_list = []
            n2_list = []

            tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
            tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()
            tmpCSIe1Ind = np.array(tmpCSIe1).argsort().argsort()
            tmpCSIe2Ind = np.array(tmpCSIe2).argsort().argsort()
            tmpCSIn1Ind = np.array(tmpNoise).argsort().argsort()

            minEpiIndClosenessLsb = np.zeros(int(keyLen / segLen), dtype=int)
            minEpiIndClosenessLse1 = np.zeros(int(keyLen / segLen), dtype=int)
            minEpiIndClosenessLse2 = np.zeros(int(keyLen / segLen), dtype=int)
            minEpiIndClosenessLsn = np.zeros(int(keyLen / segLen), dtype=int)

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
                epiIndClosenessLse2 = np.zeros(int(keyLen / segLen))
                epiIndClosenessLsn = np.zeros(int(keyLen / segLen))

                for j in range(int(keyLen / segLen)):
                    epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
                    epiInde1 = tmpCSIe1Ind[j * segLen: (j + 1) * segLen]
                    epiInde2 = tmpCSIe2Ind[j * segLen: (j + 1) * segLen]
                    epiIndn1 = tmpCSIn1Ind[j * segLen: (j + 1) * segLen]

                    epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))
                    epiIndClosenessLse1[j] = sum(abs(epiInde1 - np.array(epiInda1)))
                    epiIndClosenessLse2[j] = sum(abs(epiInde2 - np.array(epiInda1)))
                    epiIndClosenessLsn[j] = sum(abs(epiIndn1 - np.array(epiInda1)))

                minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)
                minEpiIndClosenessLse1[i] = np.argmin(epiIndClosenessLse1)
                minEpiIndClosenessLse2[i] = np.argmin(epiIndClosenessLse2)
                minEpiIndClosenessLsn[i] = np.argmin(epiIndClosenessLsn)

            # a_list_number = list(range(int(keyLen / segLen)))
            a_list_number = list(permutation)
            b_list_number = list(minEpiIndClosenessLsb)
            e1_list_number = list(minEpiIndClosenessLse1)
            e2_list_number = list(minEpiIndClosenessLse2)
            n1_list_number = list(minEpiIndClosenessLsn)
            n2_list_number = list(np.random.permutation(len(a_list_number)))

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
            for i in range(len(e2_list_number)):
                number = bin(e2_list_number[i])[2:].zfill(int(np.log2(len(e2_list_number))))
                e2_list += number
            for i in range(len(n1_list_number)):
                number = bin(n1_list_number[i])[2:].zfill(int(np.log2(len(n1_list_number))))
                n1_list += number
            for i in range(len(n2_list_number)):
                number = bin(n2_list_number[i])[2:].zfill(int(np.log2(len(n2_list_number))))
                n2_list += number

            # 对齐密钥，随机补全
            for i in range(len(a_list) - len(e1_list)):
                e1_list += str(np.random.randint(0, 2))
            for i in range(len(a_list) - len(e2_list)):
                e2_list += str(np.random.randint(0, 2))
            for i in range(len(a_list) - len(n1_list)):
                n1_list += str(np.random.randint(0, 2))
            for i in range(len(a_list) - len(n2_list)):
                n2_list += str(np.random.randint(0, 2))

            # print("keys of a:", len(a_list), a_list)
            # print(str(reuse), " keys of a:", len(a_list_number), a_list_number)
            # print("keys of b:", len(b_list), b_list)
            # print("keys of b:", len(b_list_number), b_list_number)
            # print("keys of e:", len(e_list), e_list)
            # print("keys of e:", len(e_list_number), e_list_number)
            # print("keys of e2:", len(e2_list_number), e2_list_number)

            # if reuse == 0:
            #     ke1 = e2_list_number
            # elif reuse == 1:
            #     ke2 = e2_list_number

            if reuse == 0:
                kb1 = b_list_number
            elif reuse == 1:
                kb2 = b_list_number

            # print("keys of n:", len(n_list), n_list)
            # print("keys of n:", len(n_list_number), n_list_number)

            sum1 = min(len(a_list), len(b_list))
            sum2 = 0
            sum31 = 0
            sum32 = 0
            sum41 = 0
            sum42 = 0
            for i in range(0, sum1):
                sum2 += (a_list[i] == b_list[i])
            for i in range(min(len(a_list), len(e1_list))):
                sum31 += (a_list[i] == e1_list[i])
            for i in range(min(len(a_list), len(e2_list))):
                sum32 += (a_list[i] == e2_list[i])
            for i in range(min(len(a_list), len(n1_list))):
                sum41 += (a_list[i] == n1_list[i])
            for i in range(min(len(a_list), len(n2_list))):
                sum42 += (a_list[i] == n2_list[i])

            end = time.time()
            overhead += end - start
            # print("time:", end - start)

            # 只计算重用后的猜测成功率
            if reuse > 0:
                # print("\033[0;33;40mround", reuse, "\033[0m")
                # print("\033[0;32;40ma-b", maxSum2, maxSum2 / sum1, "\033[0m")
                # print("a-e1", sum31, sum31 / sum1)
                # print("a-e2", sum32, sum32 / sum1)
                # print("a-n1", sum41, sum41 / sum1)
                # print("a-n2", sum42, sum42 / sum1)
                originSum += sum1
                correctSum += sum2
                randomSum1 += sum31
                randomSum2 += sum32
                noiseSum1 += sum41
                noiseSum2 += sum42

                originWholeSum += 1
                correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
                randomWholeSum1 = randomWholeSum1 + 1 if sum31 == sum1 else randomWholeSum1
                randomWholeSum2 = randomWholeSum2 + 1 if sum32 == sum1 else randomWholeSum2
                noiseWholeSum1 = noiseWholeSum1 + 1 if sum41 == sum1 else noiseWholeSum1
                noiseWholeSum2 = noiseWholeSum2 + 1 if sum42 == sum1 else noiseWholeSum2
        # print("\033[0;34;40m", abs(pearsonr(ke1, ke2)[0]), "\033[0m")
        # keyECorr.append(abs(pearsonr(ke1, ke2)[0]))
        keyBCorr.append(abs(pearsonr(kb1, kb2)[0]))

    # print("\033[0;34;40ma-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10), "\033[0m")
    # print("a-e1 all", randomSum1, "/", originSum, "=", round(randomSum1 / originSum, 10))
    # print("\033[0;34;40ma-e2 all", randomSum2, "/", originSum, "=", round(randomSum2 / originSum, 10), "\033[0m")
    # print("a-n1 all", noiseSum1, "/", originSum, "=", round(noiseSum1 / originSum, 10))
    # print("a-n2 all", noiseSum2, "/", originSum, "=", round(noiseSum2 / originSum, 10))
    # print("\033[0;34;40ma-b whole match", correctWholeSum, "/", originWholeSum, "=",
    #       round(correctWholeSum / originWholeSum, 10), "\033[0m")
    # print("a-e1 whole match", randomWholeSum1, "/", originWholeSum, "=", round(randomWholeSum1 / originWholeSum, 10))
    # print("\033[0;34;40ma-e2 whole match", randomWholeSum2, "/", originWholeSum, "=",
    #       round(randomWholeSum2 / originWholeSum, 10), "\033[0m")
    # print("a-n1 whole match", noiseWholeSum1, "/", originWholeSum, "=", round(noiseWholeSum1 / originWholeSum, 10))
    # print("a-n2 whole match", noiseWholeSum2, "/", originWholeSum, "=", round(noiseWholeSum2 / originWholeSum, 10))
    # print("times", times)

    # print("key e corr", erange, np.min(keyECorr), np.max(keyECorr), np.mean(keyECorr))
    print("key b corr", erange, np.min(keyBCorr), np.max(keyBCorr), np.mean(keyBCorr))
    print(singular, attackerOrUser)
    csi_csv.write(str(np.mean(keyBCorr)) + ',' + str(np.min(keyBCorr)) + ',' + str(np.max(keyBCorr)) + '\n')
csi_csv.close()
messagebox.showinfo("提示", "测试结束")
