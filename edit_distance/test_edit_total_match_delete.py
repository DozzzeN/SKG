import random

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import loadmat

from alignment import genAlign, alignFloat, genLongestContinuous


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


def genRandomStep(len, lowBound, highBound):
    length = 0
    randomStep = []
    # 少于三则无法分，因为至少要划分出一个三角形
    while len - length >= lowBound:
        step = random.randint(lowBound, highBound)
        randomStep.append(step)
        length += step
    return randomStep


rawData = loadmat('../data/data_static_indoor_1.mat')

CSIa1OrigRaw = rawData['A'][:, 0]
CSIb1OrigRaw = rawData['A'][:, 1]

CSIa1Orig = []
CSIb1Orig = []
for i in range(5):
    CSIa1Orig.append(CSIa1OrigRaw[i])
    CSIb1Orig.append(CSIb1OrigRaw[i])
for i in range(7000):
    CSIa1Orig.append(CSIa1OrigRaw[i + 20000])
    CSIb1Orig.append(CSIb1OrigRaw[i + 20000])

# CSIa1Orig = rawData['A'][:, 0]
# CSIb1Orig = rawData['A'][:, 1]

CSIa1Orig = np.array(CSIa1Orig)
CSIb1Orig = np.array(CSIb1Orig)

dataLen = len(CSIa1Orig)  # 6745

CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)

# plt.close()
# plt.figure()
# plt.plot(range(len(CSIa1Orig)), CSIa1Orig, color="blue", linewidth=.5, label="CSIa1Orig raw")
# plt.legend(loc='upper left')
# plt.show()

CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')
CSIb1Orig = smooth(CSIb1Orig, window_len=15, window='flat')
CSIe1Orig = smooth(CSIe1Orig, window_len=15, window="flat")

CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
CSIe1OrigBack = CSIe1Orig.copy()

noise = np.random.normal(loc=-1, scale=1, size=dataLen)  ## Multiplication item normal distribution
noiseAdd = np.random.normal(loc=0, scale=10, size=dataLen)  ## Addition item normal distribution

sft = 2
intvl = 2 * sft + 1
keyLen = 128
addNoise = True
opNums = int(keyLen / 2)
# 只有插入的编辑操作时将-的操作权重调大
# rule = {"=": 2, "+": 1, "-": 0, "~": 0, "^": 0}
rule = {"=": 0, "+": 1, "-": 2, "~": 2, "^": 2}

originSum = 0
correctSum = 0
randomSum = 0
noiseSum = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum = 0
noiseWholeSum = 0

codings = ""
times = 19
maxDiffAB = 0
for staInd in range(0, times * intvl + 1, intvl):
    endInd = staInd + keyLen * intvl
    print("range:", staInd, endInd)
    if endInd > len(CSIa1Orig):
        break

    CSIa1Orig = CSIa1OrigBack.copy()
    CSIb1Orig = CSIb1OrigBack.copy()
    CSIe1Orig = CSIe1OrigBack.copy()

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
    tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]

    tmpNoise = noise[range(staInd, endInd, 1)]
    tmpNoiseAdd = noiseAdd[range(staInd, endInd, 1)]

    tmpCSIa1 = tmpCSIa1 - (np.mean(tmpCSIa1) - np.mean(tmpCSIb1))  # Mean value consistency

    # linspace函数生成元素为50的等间隔数列，可以指定第三个参数为元素个数
    # signal.square返回周期性的方波波形
    tmpPulse = signal.square(
        2 * np.pi * 1 / intvl * np.linspace(0, np.pi * 0.5 * keyLen, keyLen * intvl))  ## Rectangular pulse

    if addNoise:
        # tmpCSIa1 = tmpPulse * (np.float_power(np.abs(tmpCSIa1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIa1))
        # tmpCSIb1 = tmpPulse * (np.float_power(np.abs(tmpCSIb1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIb1))
        # tmpCSIe1 = tmpPulse * (np.float_power(np.abs(tmpCSIe1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIe1))
        tmpCSIa1 = tmpPulse * np.float_power(np.abs(tmpCSIa1), tmpNoise)
        tmpCSIb1 = tmpPulse * np.float_power(np.abs(tmpCSIb1), tmpNoise)
        tmpCSIe1 = tmpPulse * np.float_power(np.abs(tmpCSIe1), tmpNoise)
    else:
        tmpCSIa1 = tmpPulse * tmpCSIa1
        tmpCSIb1 = tmpPulse * tmpCSIb1
        tmpCSIe1 = tmpPulse * tmpCSIe1

    CSIa1Orig[range(staInd, endInd, 1)] = tmpCSIa1
    CSIb1Orig[range(staInd, endInd, 1)] = tmpCSIb1
    CSIe1Orig[range(staInd, endInd, 1)] = tmpCSIe1

    permLen = len(range(staInd, endInd, intvl))
    origInd = np.array([xx for xx in range(staInd, endInd, intvl)])

    sortCSIa1 = np.zeros(permLen)
    sortCSIb1 = np.zeros(permLen)
    sortCSIe1 = np.zeros(permLen)
    sortNoise = np.zeros(permLen)

    for ii in range(permLen):
        aIndVec = np.array([aa for aa in range(origInd[ii], origInd[ii] + intvl, 1)])  ## for non-permuted CSIa1

        for jj in range(permLen, permLen * 2):
            bIndVec = np.array([bb for bb in range(origInd[jj - permLen], origInd[jj - permLen] + intvl, 1)])

            CSIa1Tmp = CSIa1Orig[aIndVec]
            CSIb1Tmp = CSIb1Orig[bIndVec]
            CSIe1Tmp = CSIe1Orig[bIndVec]
            noiseTmp = noise[aIndVec]

            sortCSIa1[ii] = np.mean(CSIa1Tmp)  ## Metric 1: Mean
            sortCSIb1[jj - permLen] = np.mean(CSIb1Tmp)  # 只赋值一次
            sortCSIe1[jj - permLen] = np.mean(CSIe1Tmp)
            sortNoise[ii] = np.mean(noiseTmp)

    # sortCSIa1是原始算法中排序前的数据
    sortCSIa1 = np.log10(np.abs(sortCSIa1))
    sortCSIb1 = np.log10(np.abs(sortCSIb1))
    sortCSIe1 = np.log10(np.abs(sortCSIe1))
    sortNoise = np.log10(np.abs(sortNoise))

    # 最后各自的密钥
    a_list = []
    b_list = []
    e_list = []
    n_list = []

    # print("sortCSIa1", sortCSIa1)
    # print("sortCSIb1", sortCSIb1)
    # print("sortCSIe1", sortCSIe1)
    # print("sortNoise", sortNoise)

    diffAB = 0
    for i in range(len(sortCSIa1)):
        diffAB = max(diffAB, abs(sortCSIa1[i] - sortCSIb1[i]))
    print("最大差距", diffAB)
    maxDiffAB = max(maxDiffAB, diffAB)
    arrayIndexA = random.sample(range(len(sortCSIa1)), opNums)
    arrayIndexA.sort()
    sortCSIa1P = list(sortCSIa1)
    # 从后往前删除
    for i in range(opNums - 1, -1, -1):
        sortCSIa1P.remove(sortCSIa1P[arrayIndexA[i]])

    print("sortCSIa1P", len(sortCSIa1P), sortCSIa1P)
    print("sortCSIa1", len(sortCSIa1), list(sortCSIa1))
    print("sortCSIb1", len(sortCSIb1), list(sortCSIb1))
    print("sortCSIe1", len(sortCSIe1), list(sortCSIe1))

    # 用a1P匹配ai，得到rule，再用rule对其a1P
    # 密钥和sortCSIa1P相同
    threshold = 0.005
    # threshold = diffAB
    ruleStr1 = alignFloat(rule, sortCSIa1P, sortCSIa1, threshold)
    alignStr1 = genAlign(ruleStr1)
    print("ruleStr1", ruleStr1)
    ruleStr2 = alignFloat(rule, sortCSIa1P, sortCSIb1, threshold)
    alignStr2 = genAlign(ruleStr2)
    print("ruleStr2", ruleStr2)
    ruleStr3 = alignFloat(rule, sortCSIa1P, sortCSIe1, threshold)
    alignStr3 = genAlign(ruleStr3)
    print("ruleStr3", ruleStr3)
    ruleStr4 = alignFloat(rule, sortCSIa1P, sortNoise, threshold)
    alignStr4 = genAlign(ruleStr4)

    # 检错
    for i in range(min(len(ruleStr1), len(ruleStr2))):
        if ruleStr1[i] != ruleStr2[i]:
            if i >= len(sortCSIa1P):
                continue
            print("\033[0;30;41m", i, sortCSIa1P[i], "\033[0m")
            print("\033[0;30;41m", i, sortCSIa1[i], abs(sortCSIa1P[i] - sortCSIa1[i]), "\033[0m")
            print("\033[0;30;41m", i, sortCSIb1[i], abs(sortCSIa1P[i] - sortCSIb1[i]), "\033[0m")
            print("\033[0;30;41m", i, ruleStr1[i], "\033[0m")
            print("\033[0;30;41m", i, ruleStr2[i], "\033[0m")

    a_list = alignStr1
    b_list = alignStr2
    e_list = alignStr3
    n_list = alignStr4

    # range(len(sortCSIa1))中有而arrayIndexA没有的就是最后相等的位置（密钥）
    editOps = list(set(range(len(sortCSIa1))).difference(set(arrayIndexA)))
    print("editOps", len(editOps), editOps)
    print("keys of a:", len(a_list), a_list)
    print("keys of b:", len(b_list), b_list)
    print("keys of e:", len(e_list), e_list)
    print("keys of n:", len(n_list), n_list)

    # a和aP进行匹配
    for i in range(min(len(a_list), len(editOps))):
        if a_list[i] != editOps[i]:
            if i >= len(sortCSIa1P):
                continue
            print("\033[0;30;42m", i, sortCSIa1P[i], "\033[0m")
            print("\033[0;30;42m", i, sortCSIa1[i], "\033[0m")
            print("\033[0;30;42m", i, a_list[i], "\033[0m")
            print("\033[0;30;42m", i, editOps[i], "\033[0m")

    print("longest numbers of a:", genLongestContinuous(a_list))
    print("longest numbers of b:", genLongestContinuous(b_list))
    print("longest numbers of e:", genLongestContinuous(e_list))
    print("longest numbers of n:", genLongestContinuous(n_list))

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

    if sum2 == sum1:
        print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
    else:
        print("\033[0;31;40ma-b", sum2, sum2 / sum1, "\033[0m")
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
print(maxDiffAB)
print("a-b all", correctSum, "/", originSum, "=", correctSum / originSum)
print("a-e all", randomSum, "/", originSum, "=", randomSum / originSum)
print("a-n all", noiseSum, "/", originSum, "=", noiseSum / originSum)
print("a-b whole match", correctWholeSum, "/", originWholeSum, "=", correctWholeSum / originWholeSum)
print("a-e whole match", randomWholeSum, "/", originWholeSum, "=", randomWholeSum / originWholeSum)
print("a-n whole match", noiseWholeSum, "/", originWholeSum, "=", noiseWholeSum / originWholeSum)
