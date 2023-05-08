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

for solutionRange in range(1, 6):
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

    legiCorr = []
    randomCorr = []
    inferCorr = []
    imitCorr = []
    stalkCorr = []

    reuseTimes = 2

    # static indoor
    for staInd in range(0, int(dataLen / 5.5), int(keyLen / 10)):
        # for staInd in range(0, dataLen, int(keyLen / 10)):
        # for staInd in range(0, dataLen, keyLen):

        # 攻击者猜测出来的原始测量值
        beforeMulCSIe2 = []

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
            CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1),
                                         size=len(CSIa1Orig))

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

                if reuse > 0:
                    tmpCSIa1b = tmpCSIa1 - np.mean(tmpCSIa1)
                    # 在给定范围内寻找合适的Cn
                    # CSIa1Origb = CSIa1Orig - np.mean(CSIa1Orig)
                    partial = np.identity(keyLen) - np.matmul(publicMatrix, np.linalg.pinv(publicMatrix))
                    # step = 100
                    # beforeMulCSIe2 = particularSolution + np.matmul(np.random.uniform(
                    #     (min(particularSolution) - min(CSIa1Origb)) / step,
                    #     (max(CSIa1Origb) - max(particularSolution)) / step, keyLen), partial)
                    # print(min(beforeMulCSIe2), max(beforeMulCSIe2), np.mean(beforeMulCSIe2))
                    # print(min(CSIa1Origb), max(CSIa1Origb), np.mean(CSIa1Origb))
                    # while min(beforeMulCSIe2) < min(CSIa1Origb) or max(beforeMulCSIe2) > max(CSIa1Origb):
                    #     step = step / 10
                    #     beforeMulCSIe2 = particularSolution + np.matmul(np.random.uniform(
                    #         (min(particularSolution) - min(CSIa1Origb)) / step,
                    #         (max(CSIa1Origb) - max(particularSolution)) / step, keyLen), partial)

                    # 根据想要生成的解倒推cn
                    # y = np.random.uniform(min(tmpCSIa1b), max(tmpCSIa1b), keyLen)
                    # y = np.random.normal(np.mean(tmpCSIa1b), np.std(tmpCSIa1b), keyLen)
                    # y = tmpCSIa1b + np.random.normal(np.mean(tmpCSIa1b), np.std(tmpCSIa1b), keyLen)
                    # y = particularSolution + np.random.uniform(-solutionRange, solutionRange, keyLen)

                    perturbation = np.ones(keyLen)
                    for i in range(keyLen):
                        if random.random() < 0.5:
                            perturbation[i] = perturbation[i] * solutionRange
                        else:
                            perturbation[i] = perturbation[i] * -solutionRange
                    y = particularSolution + perturbation
                    # y = tmpCSIa1b + np.random.uniform(min(tmpCSIa1b), max(tmpCSIa1b), keyLen)
                    # y = [random.randint(int(min(tmpCSIa1b)), int(max(tmpCSIa1b))) for _ in range(keyLen)]
                    cn = np.matmul(y - particularSolution, np.linalg.pinv(partial))
                    beforeMulCSIe2 = particularSolution + np.matmul(cn, partial)

                    # print("round", reuse)
                    # print(min(tmpCSIa1b), max(tmpCSIa1b), np.mean(tmpCSIa1b))
                    # print(min(beforeMulCSIe2), max(beforeMulCSIe2), np.mean(beforeMulCSIe2))
                    # print("the real solution", tmpCSIa1b)
                    # print("particular solution", particularSolution)
                    # print("general solution", beforeMulCSIe2)

                    # print("multiplication")
                    # print("the real product", np.matmul(tmpCSIa1b, randomMatrix))
                    # # print("particular solution", np.matmul(particularSolution, randomMatrix))
                    # print("general solution", np.matmul(beforeMulCSIe2, randomMatrix))
                    tmpCSIe2 = beforeMulCSIe2

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

                if reuse == 0:
                    publicMatrix = randomMatrix

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

            if reuse == 0:
                # 需要排序一次的结果才能恢复
                tmpCSIa1IndReshape2 = np.array(np.array(tmpCSIa1).argsort()).reshape(int(keyLen / segLen), segLen)
                combineMetric2 = list(zip(tmpCSIa1IndReshape2, list(range(int(keyLen / segLen)))))
                np.random.seed(staInd + reuse)
                np.random.shuffle(combineMetric2)
                tmpCSIa1IndReshape2, permutation2 = zip(*combineMetric2)
                tmpCSIa1Ind2 = np.hstack((tmpCSIa1IndReshape2))

                # 攻击者已知的密钥
                combineMetric2 = sorted(combineMetric2, key=lambda combineMetric2: combineMetric2[1])
                tmpCSIe2IndReshape, _ = zip(*combineMetric2)
                afterSortCSIe2Ind = np.hstack((tmpCSIe2IndReshape))
                if attackerOrUser == "attacker":
                    # 攻击者根据自己的测量值和密钥（排序索引）恢复shuffle前的结果
                    beforeSortCSIe2 = list(zip(sorted(tmpCSIe1), afterSortCSIe2Ind))
                elif attackerOrUser == "user":
                    # 把tmpCSIe1换成tmpCSIb1即是攻击者利用B的数据进行推导的结果
                    beforeSortCSIe2 = list(zip(sorted(tmpCSIb1), afterSortCSIe2Ind))
                else:
                    raise Exception("invalid parameter of attackerOrUser: " + attackerOrUser)
                beforeSortCSIe2 = sorted(beforeSortCSIe2, key=lambda beforeSortCSIe2: beforeSortCSIe2[1])
                # 攻击者根据公开的矩阵求逆，恢复原始的测量值
                beforeSortCSIe2, index = zip(*beforeSortCSIe2)
                # 将tmpCSIe1换成tmpCSIa1可以看到结果是一样的
                # print(tmpCSIe1)
                # print(beforeSortCSIe2)
                if singular:
                    # 广义{1,2,3,4}逆pinv
                    # 随机矩阵采用高斯分布比均匀分布效果好，范围越大效果也越好
                    # beforeMulCSIe2 = np.matmul(np.array(beforeSortCSIe2), np.linalg.pinv(publicMatrix)) + \
                    #                  np.matmul(np.random.normal(0, np.std(tmpCSIa1b), keyLen), np.identity(keyLen) -
                    #                            np.matmul(publicMatrix, np.linalg.pinv(publicMatrix)))

                    particularSolution = np.matmul(np.array(beforeSortCSIe2), np.linalg.pinv(publicMatrix))
                    # partial = np.identity(keyLen) - np.matmul(publicMatrix, np.linalg.pinv(publicMatrix))
                    # beforeMulCSIe2 = particularSolution + np.matmul(np.random.uniform(
                    #     1 / keyLen / max(abs(np.min(partial)), abs(np.max(partial))) * -abs(min(min(CSIa1Orig), min(CSIb1Orig)) - min(particularSolution)),
                    #     1 / keyLen / max(abs(np.min(partial)), abs(np.max(partial))) * abs(max(max(CSIa1Orig), max(CSIb1Orig)) - max(particularSolution)), keyLen), partial)
                else:
                    beforeMulCSIe2 = np.matmul(np.array(beforeSortCSIe2), np.linalg.inv(publicMatrix))
                # print(tmpCSIe1b)
                # print(beforeMulCSIe2)

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
            # print(str(reuse), " keys of b:", len(b_list_number), b_list_number)
            # print("keys of e:", len(e_list), e_list)
            # print("keys of e:", len(e_list_number), e_list_number)
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

            # 自适应纠错
            if sum1 != sum2 and rec:
                if tell:
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

                            # print("keys of b:", len(b_list_number), b_list_number)

                            sum2 = 0
                            for i in range(0, min(len(a_list), len(b_list))):
                                sum2 += (a_list[i] == b_list[i])
                else:
                    # 正式纠错
                    trueError = []
                    for i in range(len(a_list_number)):
                        if a_list_number[i] != b_list_number[i]:
                            trueError.append(i)
                    # print("true error", trueError)
                    # print("a-b", sum2, sum2 / sum1)
                    reconciliation = b_list_number.copy()
                    reconciliation.sort()

                    repeatInd = []
                    # 检查两个候选
                    closeness = []
                    for i in range(len(reconciliation) - 1):
                        # 相等的索引就是密钥出错的地方
                        if reconciliation[i] == reconciliation[i + 1]:
                            repeatInd.append(reconciliation[i])
                    repeatNumber = []
                    for i in range(len(repeatInd)):
                        tmp = []
                        for j in range(len(b_list_number)):
                            if repeatInd[i] == b_list_number[j]:
                                tmp.append(j)
                        repeatNumber.append(tmp)
                    for i in range(len(repeatNumber)):
                        tmp = []
                        for j in range(len(repeatNumber[i])):
                            epiInda1 = tmpCSIa1Ind[repeatNumber[i][j] * segLen:(repeatNumber[i][j] + 1) * segLen]

                            epiIndClosenessLsb = np.zeros(int(keyLen / segLen))

                            for k in range(int(keyLen / segLen)):
                                epiIndb1 = tmpCSIb1Ind[k * segLen: (k + 1) * segLen]

                                epiIndClosenessLsb[k] = sum(abs(epiIndb1 - np.array(epiInda1)))

                            min_b = np.argmin(epiIndClosenessLsb)
                            tmp.append(epiIndClosenessLsb[min_b])

                        closeness.append(tmp)

                    errorInd = []

                    for i in range(len(closeness)):
                        for j in range(len(closeness[i]) - 1):
                            if closeness[i][j] < closeness[i][j + 1]:
                                errorInd.append(repeatNumber[i][j + 1])
                            else:
                                errorInd.append(repeatNumber[i][j])
                    # print(errorInd)
                    b_list_number1 = b_list_number.copy()
                    for i in errorInd:
                        epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]
                        epiIndClosenessLsb = np.zeros(int(keyLen / segLen))

                        for j in range(int(keyLen / segLen)):
                            epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
                            epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))

                        min_b = np.argmin(epiIndClosenessLsb)
                        while min_b in b_list_number:
                            epiIndClosenessLsb[min_b] = keyLen * segLen
                            min_b = np.argmin(epiIndClosenessLsb)
                        b_list_number1[i] = min_b

                    b_list = []

                    for i in range(len(b_list_number1)):
                        number = bin(b_list_number1[i])[2:].zfill(int(np.log2(len(b_list_number1))))
                        b_list += number

                    # print("keys of b:", len(b_list_number1), b_list_number1)

                    sum2 = 0
                    for i in range(0, min(len(a_list), len(b_list))):
                        sum2 += (a_list[i] == b_list[i])

                    if sum1 == sum2:
                        b_list_number = b_list_number1

                    # 二次纠错
                    if sum1 != sum2:
                        for r in range(len(repeatNumber)):
                            tmp = list(set(repeatNumber[r]) - set(errorInd))
                            errorInd = tmp
                            # print(errorInd)
                            b_list_number2 = b_list_number.copy()
                            for i in errorInd:
                                epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]
                                epiIndClosenessLsb = np.zeros(int(keyLen / segLen))

                                for j in range(int(keyLen / segLen)):
                                    epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
                                    epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))

                                min_b = np.argmin(epiIndClosenessLsb)
                                while min_b in b_list_number:
                                    epiIndClosenessLsb[min_b] = keyLen * segLen
                                    min_b = np.argmin(epiIndClosenessLsb)
                                b_list_number2[i] = min_b

                            b_list = []

                            for i in range(len(b_list_number2)):
                                number = bin(b_list_number2[i])[2:].zfill(int(np.log2(len(b_list_number2))))
                                b_list += number

                            # print("keys of b:", len(b_list_number2), b_list_number2)

                            sum2 = 0
                            for i in range(0, min(len(a_list), len(b_list))):
                                sum2 += (a_list[i] == b_list[i])

                            if sum1 == sum2:
                                b_list_number = b_list_number2
                    # 正式纠错 end

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

                legiCorr.append(abs(pearsonr(a_list_number, b_list_number)[0]))
                imitCorr.append(abs(pearsonr(a_list_number, e1_list_number)[0]))
                stalkCorr.append(abs(pearsonr(a_list_number, e2_list_number)[0]))
                inferCorr.append(abs(pearsonr(a_list_number, n1_list_number)[0]))
                randomCorr.append(abs(pearsonr(a_list_number, n2_list_number)[0]))

                originWholeSum += 1
                correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
                randomWholeSum1 = randomWholeSum1 + 1 if sum31 == sum1 else randomWholeSum1
                randomWholeSum2 = randomWholeSum2 + 1 if sum32 == sum1 else randomWholeSum2
                noiseWholeSum1 = noiseWholeSum1 + 1 if sum41 == sum1 else noiseWholeSum1
                noiseWholeSum2 = noiseWholeSum2 + 1 if sum42 == sum1 else noiseWholeSum2

    print("\033[0;34;40ma-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10), "\033[0m")
    print("a-e1 all", randomSum1, "/", originSum, "=", round(randomSum1 / originSum, 10))
    print("\033[0;34;40ma-e2 all", randomSum2, "/", originSum, "=", round(randomSum2 / originSum, 10), "\033[0m")
    print("a-n1 all", noiseSum1, "/", originSum, "=", round(noiseSum1 / originSum, 10))
    print("a-n2 all", noiseSum2, "/", originSum, "=", round(noiseSum2 / originSum, 10))
    print("\033[0;34;40ma-b whole match", correctWholeSum, "/", originWholeSum, "=",
          round(correctWholeSum / originWholeSum, 10), "\033[0m")
    print("a-e1 whole match", randomWholeSum1, "/", originWholeSum, "=", round(randomWholeSum1 / originWholeSum, 10))
    print("\033[0;34;40ma-e2 whole match", randomWholeSum2, "/", originWholeSum, "=",
          round(randomWholeSum2 / originWholeSum, 10), "\033[0m")
    print("a-n1 whole match", noiseWholeSum1, "/", originWholeSum, "=", round(noiseWholeSum1 / originWholeSum, 10))
    print("a-n2 whole match", noiseWholeSum2, "/", originWholeSum, "=", round(noiseWholeSum2 / originWholeSum, 10))
    print("times", times)

    # 最终推测密钥和合法密钥的相关系数
    print("correlation analysis")
    print("\033[0;34;40mlegi", np.mean(legiCorr), np.min(legiCorr), np.max(legiCorr), "\033[0m")
    print("random", np.mean(randomCorr), np.min(randomCorr), np.max(randomCorr))
    print("infer", np.mean(inferCorr), np.min(inferCorr), np.max(inferCorr))
    print("imit", np.mean(imitCorr), np.min(imitCorr), np.max(imitCorr))
    print("\033[0;34;40mstalk", np.mean(stalkCorr), np.min(stalkCorr), np.max(stalkCorr), "\033[0m")
    print(solutionRange, singular, attackerOrUser)
messagebox.showinfo("提示", "测试结束")
