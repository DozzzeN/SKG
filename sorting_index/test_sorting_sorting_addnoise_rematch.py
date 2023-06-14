import csv
import math
import time
from itertools import chain

import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import dct
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr, boxcox
from pyentrp import entropy as ent

from zca import ZCA


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


rawData = loadmat("../data/data_static_outdoor_1.mat")
# rawData = loadmat("../skyglow/Scenario2-Office-LoS/data3_upto5.mat")
corrMatName = "corr_so.mat"
rssMatName = "corr_so_rss.mat"

# stalking attack
# CSIe2Orig = loadmat("../skyglow/Scenario3-Mobile2/data_eave_mobile_2.mat")['A'][:, 0]
CSIe2Orig = loadmat("../skyglow/Scenario2-Office-LoS-eve_NLoS/data_eave_LOS_EVE_NLOS.mat")['A'][:, 0]

# 7 256
# data BMR BGR BGR-with-no-error
# si1 - for staInd in range(0, int(dataLen / 5.5), int(keyLen / 10)):
# mi1 1.0 1.0 2.0 2.0
# si1 1.0 1.0 2.0 2.0
# mo1 1.0 1.0 2.0 2.0
# 因为so的数据长度没有补齐
# so1 1.0 1.0 1.8461538461538463 1.8461538461538463

# 5 1024
# # data BMR BGR BGR-with-no-error
# # si1 - for staInd in range(0, int(dataLen / 5.5), int(keyLen / 10)):
# # mi1 1.0 1.0 2.9 2.9
# # si1 1.0 1.0 2.8216216216216217 2.8216216216216217
# # mo1 1.0 1.0 2.9 2.9
# # 因为so的数据长度没有补齐
# # so1 1.0 1.0 2.32 2.32

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
dataLen = len(CSIa1Orig)
print("dataLen: ", dataLen)

originSum = 0
correctSum = 0
randomSum1 = 0
randomSum2 = 0
noiseSum1 = 0
noiseSum2 = 0

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

legiCorr = []
randomCorr = []
inferCorr = []
imitCorr = []
stalkCorr = []

trueRSS = []
legiRSS = []
inferRSS = []
imitRSS = []
stalkRSS = []

legiCorrRSS = []
inferCorrRSS = []
imitCorrRSS = []
stalkCorrRSS = []

times = 0

roBits = 128
episodeLen = 5

epiLen = roBits
segLen = 1
# 只重用一次
while segLen < episodeLen * 2:
    if segLen == 1:
        segLen = episodeLen
        epiLen = roBits
    else:
        segLen = segLen * 2
        epiLen = int(epiLen / 2)
    print("\033[0;34;40msegLen", segLen, "epiLen", epiLen, "\033[0m")
    keyLen = epiLen * segLen

    trueCurrKey = []
    legiCurrKey = []
    randomCurrKey = []
    inferCurrKey = []
    imitCurrKey = []
    stalkCurrKey = []

    trueCurrRSS = []
    legiCurrRSS = []
    randomCurrRSS = []
    inferCurrRSS = []
    imitCurrRSS = []
    stalkCurrRSS = []

    # static indoor
    # for staInd in range(0, int(dataLen / 5.5), keyLen):
    for staInd in range(0, dataLen, keyLen):
        endInd = staInd + keyLen
        print("range:", staInd, endInd)
        if endInd >= len(CSIa1Orig) or endInd >= len(CSIe2Orig):
            break

        # 只计算最底层RO的长度作为样本值个数
        if epiLen == roBits:
            times += 1

        # np.random.seed(1)
        CSIa1Orig = rawData['A'][:, 0]
        CSIb1Orig = rawData['A'][:, 1]

        # imitation attack
        CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=len(CSIa1Orig))

        seed = np.random.randint(100000)
        np.random.seed(seed)

        CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
        CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')

        tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
        tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
        tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
        tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]

        randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
        tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
        tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
        tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
        tmpCSIe2 = tmpCSIe2 - np.mean(tmpCSIe2)
        tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
        tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)
        tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix)
        tmpCSIe2 = np.matmul(tmpCSIe2, randomMatrix)
        # inference attack
        tmpNoise = np.matmul(np.ones(keyLen), randomMatrix)

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
        # np.random.seed(staInd)
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
        # print("keys of a:", len(a_list_number), a_list_number)
        # print("keys of b:", len(b_list), b_list)
        # print("keys of b:", len(b_list_number), b_list_number)
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

        # 自适应纠错
        if sum1 != sum2:
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

        trueCurrKey.append(a_list_number)
        legiCurrKey.append(b_list_number)
        randomCurrKey.append(n2_list_number)
        inferCurrKey.append(n1_list_number)
        imitCurrKey.append(e1_list_number)
        stalkCurrKey.append(e2_list_number)

        trueCurrRSS.append(tmpCSIa1)
        legiCurrRSS.append(tmpCSIb1)
        inferCurrRSS.append(tmpNoise)
        imitCurrRSS.append(tmpCSIe1)
        stalkCurrRSS.append(tmpCSIe2)

        if sum2 == sum1:
            print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
            print("a-e1", sum31, sum31 / sum1)
            print("a-e2", sum32, sum32 / sum1)
            print("a-n1", sum41, sum41 / sum1)
            print("a-n2", sum42, sum42 / sum1)
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

    trueKey.append(trueCurrKey)
    legiKey.append(legiCurrKey)
    randomKey.append(randomCurrKey)
    inferKey.append(inferCurrKey)
    imitKey.append(imitCurrKey)
    stalkKey.append(stalkCurrKey)

    trueRSS.append(trueCurrRSS)
    legiRSS.append(legiCurrRSS)
    inferRSS.append(inferCurrRSS)
    imitRSS.append(imitCurrRSS)
    stalkRSS.append(stalkCurrRSS)

trueRO = []
legiRO = []
randomRO = []
inferRO = []
imitRO = []
stalkRO = []

trueSample = []
legiSample = []
inferSample = []
imitSample = []
stalkSample = []

# 组装成完整的密钥
for i in range(len(trueKey[0])):
    trueTmp = []
    legiTmp = []
    randomTmp = []
    inferTmp = []
    imitTmp = []
    stalkTmp = []
    for j in range(len(trueKey)):
        if i >= len(trueKey[j]):
            break
        trueTmp.append(trueKey[j][i])
        legiTmp.append(legiKey[j][i])
        randomTmp.append(randomKey[j][i])
        inferTmp.append(inferKey[j][i])
        imitTmp.append(imitKey[j][i])
        stalkTmp.append(stalkKey[j][i])
    trueRO.append(list(chain.from_iterable(trueTmp)))
    legiRO.append(list(chain.from_iterable(legiTmp)))
    randomRO.append(list(chain.from_iterable(randomTmp)))
    inferRO.append(list(chain.from_iterable(inferTmp)))
    imitRO.append(list(chain.from_iterable(imitTmp)))
    stalkRO.append(list(chain.from_iterable(stalkTmp)))

for i in range(len(trueRSS[0])):
    trueTmp = []
    legiTmp = []
    inferTmp = []
    imitTmp = []
    stalkTmp = []
    for j in range(len(trueRSS)):
        if i >= len(trueKey[j]):
            break
        trueTmp.append(trueRSS[j][i])
        legiTmp.append(legiRSS[j][i])
        inferTmp.append(inferRSS[j][i])
        imitTmp.append(imitRSS[j][i])
        stalkTmp.append(stalkRSS[j][i])
    trueSample.append(list(chain.from_iterable(trueTmp)))
    legiSample.append(list(chain.from_iterable(legiTmp)))
    inferSample.append(list(chain.from_iterable(inferTmp)))
    imitSample.append(list(chain.from_iterable(imitTmp)))
    stalkSample.append(list(chain.from_iterable(stalkTmp)))

print("\033[0;34;40ma-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10), "\033[0m")
print("a-e1 all", randomSum1, "/", originSum, "=", round(randomSum1 / originSum, 10))
print("a-e2 all", randomSum2, "/", originSum, "=", round(randomSum2 / originSum, 10))
print("a-n1 all", noiseSum1, "/", originSum, "=", round(noiseSum1 / originSum, 10))
print("a-n2 all", noiseSum2, "/", originSum, "=", round(noiseSum2 / originSum, 10))
print("\033[0;34;40ma-b whole match", correctWholeSum, "/", originWholeSum, "=",
      round(correctWholeSum / originWholeSum, 10), "\033[0m")
print("a-e1 whole match", randomWholeSum1, "/", originWholeSum, "=", round(randomWholeSum1 / originWholeSum, 10))
print("a-e2 whole match", randomWholeSum2, "/", originWholeSum, "=", round(randomWholeSum2 / originWholeSum, 10))
print("a-n1 whole match", noiseWholeSum1, "/", originWholeSum, "=", round(noiseWholeSum1 / originWholeSum, 10))
print("a-n2 whole match", noiseWholeSum2, "/", originWholeSum, "=", round(noiseWholeSum2 / originWholeSum, 10))
print("times", times)

print(round(correctSum / originSum, 10), round(correctWholeSum / originWholeSum, 10),
      originSum / times / roBits / episodeLen,
      correctSum / times / roBits / episodeLen)

# 每次生成的RO之间的相关系数
for i in range(len(trueRO)):
    legiCorr.append(abs(pearsonr(trueRO[i], legiRO[i])[0]))
    randomCorr.append(abs(pearsonr(trueRO[i], randomRO[i])[0]))
    inferCorr.append(abs(pearsonr(trueRO[i], inferRO[i])[0]))
    imitCorr.append(abs(pearsonr(trueRO[i], imitRO[i])[0]))
    stalkCorr.append(abs(pearsonr(trueRO[i], stalkRO[i])[0]))

for i in range(len(trueRO)):
    legiCorrRSS.append(abs(pearsonr(trueSample[i], legiSample[i])[0]))
    inferCorrRSS.append(abs(pearsonr(trueSample[i], inferSample[i])[0]))
    imitCorrRSS.append(abs(pearsonr(trueSample[i], imitSample[i])[0]))
    stalkCorrRSS.append(abs(pearsonr(trueSample[i], stalkSample[i])[0]))

savemat(corrMatName, {"legiCorr": legiCorr, "randomCorr": randomCorr, "inferCorr": inferCorr,
                      "imitCorr": imitCorr, "stalkCorr": stalkCorr})

savemat(rssMatName, {"legiCorrRSS": legiCorrRSS, "inferCorrRSS": inferCorrRSS,
                     "imitCorrRSS": imitCorrRSS, "stalkCorrRSS": stalkCorrRSS})

print("correlation between the inferred and actual RO")
print(np.mean(legiCorr), np.min(legiCorr), np.max(legiCorr))
print(np.mean(randomCorr), np.min(randomCorr), np.max(randomCorr))
print(np.mean(inferCorr), np.min(inferCorr), np.max(inferCorr))
print(np.mean(imitCorr), np.min(imitCorr), np.max(imitCorr))
print(np.mean(stalkCorr), np.min(stalkCorr), np.max(stalkCorr))

legiAdjCorr = []
randomAdjCorr = []
inferAdjCorr = []
imitAdjCorr = []
stalkAdjCorr = []

# 相邻两次生成的RO之间的相关系数
for i in range(len(trueRO) - 1):
    legiAdjCorr.append(abs(pearsonr(legiRO[i], legiRO[i + 1])[0]))
    randomAdjCorr.append(abs(pearsonr(randomRO[i], randomRO[i + 1])[0]))
    inferAdjCorr.append(abs(pearsonr(inferRO[i], inferRO[i + 1])[0]))
    imitAdjCorr.append(abs(pearsonr(imitRO[i], imitRO[i + 1])[0]))
    stalkAdjCorr.append(abs(pearsonr(stalkRO[i], stalkRO[i + 1])[0]))

print("correlation between adjacent ROs")
print(np.mean(legiAdjCorr), np.min(legiAdjCorr), np.max(legiAdjCorr))
print(np.mean(randomAdjCorr), np.min(randomAdjCorr), np.max(randomAdjCorr))
print(np.mean(inferAdjCorr), np.min(inferAdjCorr), np.max(inferAdjCorr))
print(np.mean(imitAdjCorr), np.min(imitAdjCorr), np.max(imitAdjCorr))
print(np.mean(stalkAdjCorr), np.min(stalkAdjCorr), np.max(stalkAdjCorr))

# 某次生成的RO与所有其他RO之间的相关系数
# for i in range(len(trueRO)):
#     for j in range(len(trueRO)):
#         if i == j:
#             continue
#         legiAdjCorr.append(pearsonr(legiRO[i], legiRO[j])[0])
#         randomAdjCorr.append(pearsonr(randomRO[i], randomRO[j])[0])
#         inferAdjCorr.append(pearsonr(inferRO[i], inferRO[j])[0])
#         imitAdjCorr.append(pearsonr(imitRO[i], imitRO[j])[0])
#         stalkAdjCorr.append(pearsonr(stalkRO[i], stalkRO[j])[0])

legiAdjCorr = []
randomAdjCorr = []
inferAdjCorr = []
imitAdjCorr = []
stalkAdjCorr = []

csv = open("adjCorr.csv", "a+")
# 相邻RO之间的相关系数
for i in range(len(trueKey)):
    legiAdjCorr = []
    randomAdjCorr = []
    inferAdjCorr = []
    imitAdjCorr = []
    stalkAdjCorr = []
    for j in range(len(trueKey[i]) - 1):
        legiAdjCorr.append(abs(pearsonr(trueKey[i][j], trueKey[i][j + 1])[0]))
        legiAdjCorr[len(legiAdjCorr) - 1] = 0 if math.isnan(legiAdjCorr[len(legiAdjCorr) - 1]) \
                                                 is True else legiAdjCorr[len(legiAdjCorr) - 1]
        randomAdjCorr.append(abs(pearsonr(randomKey[i][j], randomKey[i][j + 1])[0]))
        randomAdjCorr[len(randomAdjCorr) - 1] = 0 if math.isnan(randomAdjCorr[len(randomAdjCorr) - 1]) \
                                                     is True else randomAdjCorr[len(randomAdjCorr) - 1]
        inferAdjCorr.append(abs(pearsonr(inferKey[i][j], inferKey[i][j + 1])[0]))
        inferAdjCorr[len(inferAdjCorr) - 1] = 0 if math.isnan(inferAdjCorr[len(inferAdjCorr) - 1]) \
                                                   is True else inferAdjCorr[len(inferAdjCorr) - 1]
        imitAdjCorr.append(abs(pearsonr(imitKey[i][j], imitKey[i][j + 1])[0]))
        imitAdjCorr[len(imitAdjCorr) - 1] = 0 if math.isnan(imitAdjCorr[len(imitAdjCorr) - 1]) \
                                                 is True else imitAdjCorr[len(imitAdjCorr) - 1]
        stalkAdjCorr.append(abs(pearsonr(stalkKey[i][j], stalkKey[i][j + 1])[0]))
        stalkAdjCorr[len(stalkAdjCorr) - 1] = 0 if math.isnan(stalkAdjCorr[len(stalkAdjCorr) - 1]) \
                                                   is True else stalkAdjCorr[len(stalkAdjCorr) - 1]

    print("correlation between layers of RO in ROs with length=", len(trueKey[i][0]))
    print(np.mean(legiAdjCorr), np.min(legiAdjCorr), np.max(legiAdjCorr))
    print(np.mean(randomAdjCorr), np.min(randomAdjCorr), np.max(randomAdjCorr))
    print(np.mean(inferAdjCorr), np.min(inferAdjCorr), np.max(inferAdjCorr))
    print(np.mean(imitAdjCorr), np.min(imitAdjCorr), np.max(imitAdjCorr))
    print(np.mean(stalkAdjCorr), np.min(stalkAdjCorr), np.max(stalkAdjCorr))
    csv.write(str(len(trueKey[i][0])) + '\n')
    csv.write(str(np.mean(legiAdjCorr)) + ',' + str(np.min(legiAdjCorr)) + ',' + str(np.max(legiAdjCorr)) + '\n' +
              str(np.mean(randomAdjCorr)) + ',' + str(np.min(randomAdjCorr)) + ',' + str(np.max(randomAdjCorr)) + '\n' +
              str(np.mean(inferAdjCorr)) + ',' + str(np.min(inferAdjCorr)) + ',' + str(np.max(inferAdjCorr)) + '\n' +
              str(np.mean(imitAdjCorr)) + ',' + str(np.min(imitAdjCorr)) + ',' + str(np.max(imitAdjCorr)) + '\n' +
              str(np.mean(stalkAdjCorr)) + ',' + str(np.min(stalkAdjCorr)) + ',' + str(np.max(stalkAdjCorr)) + '\n')
    csv.write('\n')
csv.close()

# 相邻两次生成的RO之间的排列熵
for o in range(3, 8):
    legiAdjCorr = []
    randomAdjCorr = []
    inferAdjCorr = []
    imitAdjCorr = []
    stalkAdjCorr = []
    for i in range(len(trueRO)):
        legiAdjCorr.append(ent.permutation_entropy(legiRO[i], order=o))
        randomAdjCorr.append(ent.permutation_entropy(randomRO[i], order=o))
        inferAdjCorr.append(ent.permutation_entropy(inferRO[i], order=o))
        imitAdjCorr.append(ent.permutation_entropy(imitRO[i], order=o))
        stalkAdjCorr.append(ent.permutation_entropy(stalkRO[i], order=o))

    # print("permutation entropy of all ROs with order = " + str(o))
    # print(np.mean(legiAdjCorr), np.min(legiAdjCorr), np.max(legiAdjCorr))
    # print(np.mean(randomAdjCorr), np.min(randomAdjCorr), np.max(randomAdjCorr))
    # print(np.mean(inferAdjCorr), np.min(inferAdjCorr), np.max(inferAdjCorr))
    # print(np.mean(imitAdjCorr), np.min(imitAdjCorr), np.max(imitAdjCorr))
    # print(np.mean(stalkAdjCorr), np.min(stalkAdjCorr), np.max(stalkAdjCorr))

# 相邻两次生成的RO之间的排列熵
for i in range(len(trueKey)):
    for o in range(3, 8):
        legiAdjCorr = []
        randomAdjCorr = []
        inferAdjCorr = []
        imitAdjCorr = []
        stalkAdjCorr = []
        if o >= len(trueKey[i][0]):
            continue
        d = 1 if int(len(trueKey[i][0]) / (o - 1)) == 0 else int(len(trueKey[i][0]) / (o - 1))
        d = int(o / 2) if int(len(trueKey[i][0]) / (o - 1)) >= o else int(len(trueKey[i][0]) / (o - 1))
        for j in range(len(trueKey[i])):
            # permutation_entropy在运算中对order和delay有制约关系：N-(order-1)*delay>=0
            legiAdjCorr.append(ent.permutation_entropy(legiKey[i][j], order=o, delay=d))
            randomAdjCorr.append(ent.permutation_entropy(randomKey[i][j], order=o, delay=d))
            inferAdjCorr.append(ent.permutation_entropy(inferKey[i][j], order=o, delay=d))
            imitAdjCorr.append(ent.permutation_entropy(imitKey[i][j], order=o, delay=d))
            stalkAdjCorr.append(ent.permutation_entropy(stalkKey[i][j], order=o, delay=d))

        # print("permutation entropy of each layer of RO in ROs with order = "
        #       + str(o) + ", delay = " + str(d) + ", key length = " + str(
        #           len(trueKey[i][0])))
        # print(np.mean(legiAdjCorr), np.min(legiAdjCorr), np.max(legiAdjCorr))
        # print(np.mean(randomAdjCorr), np.min(randomAdjCorr), np.max(randomAdjCorr))
        # print(np.mean(inferAdjCorr), np.min(inferAdjCorr), np.max(inferAdjCorr))
        # print(np.mean(imitAdjCorr), np.min(imitAdjCorr), np.max(imitAdjCorr))
        # print(np.mean(stalkAdjCorr), np.min(stalkAdjCorr), np.max(stalkAdjCorr))
