import csv
import random
import time

import numpy as np
from pyentrp import entropy as ent
from scipy import signal
from scipy.io import loadmat
from scipy.spatial.distance import euclidean, chebyshev
from sklearn import preprocessing

from alignment import genAlign, alignFloatInsDelWithMetrics, absolute, cosine, dtw, manhattan, correlation, absolute2


def entropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, dataLen, entropyThres):
    ts = CSIa1Orig / np.max(CSIa1Orig)
    # shanon_entropy = ent.shannon_entropy(ts)
    # perm_entropy = ent.permutation_entropy(ts, order=3, delay=1, normalize=True)
    # mulperm_entropy = ent.multiscale_permutation_entropy(ts, 3, 1, 1)
    mul_entropy = ent.multiscale_entropy(ts, 3, maxscale=1)
    # print(mul_entropy)

    cnts = 0
    while mul_entropy < entropyThres and cnts < 10:
        # while mul_entropy < 2.510
        shuffleInd = np.random.permutation(dataLen)
        CSIa1Orig = CSIa1Orig[shuffleInd]
        CSIb1Orig = CSIb1Orig[shuffleInd]
        CSIe1Orig = CSIe1Orig[shuffleInd]
        # CSIa2Orig = CSIa2Orig[shuffleInd]
        # CSIb2Orig = CSIb2Orig[shuffleInd]

        ts = CSIa1Orig / np.max(CSIa1Orig)
        mul_entropy = ent.multiscale_entropy(ts, 4, maxscale=1)
        cnts += 1
        # print(mul_entropy[0])

    return CSIa1Orig, CSIb1Orig, CSIe1Orig


def incEntropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, dataLen, entropyThres):
    CSIa1OrigBack = []
    CSIb1OrigBack = []
    CSIe1OrigBack = []
    indices = []
    for i in range(int(dataLen / 2)):
        CSIa1OrigBack.append(CSIa1Orig[i])
        CSIb1OrigBack.append(CSIb1Orig[i])
        CSIe1OrigBack.append(CSIe1Orig[i])

    for i in range(int(dataLen / 2), dataLen):
        ins = random.randint(0, len(CSIa1OrigBack))
        CSIa1OrigBack.insert(ins, CSIa1Orig[i])
        ts = CSIa1OrigBack / np.max(CSIa1OrigBack)
        mul_entropy = ent.multiscale_entropy(ts, 3, maxscale=1)
        cnts = 0
        while cnts < 2 and mul_entropy < entropyThres:
            CSIa1OrigBack.remove(CSIa1OrigBack[ins])
            ins = random.randint(0, len(CSIa1OrigBack))
            CSIa1OrigBack.insert(ins, CSIa1Orig[i])
            ts = CSIa1OrigBack / np.max(CSIa1OrigBack)
            mul_entropy = ent.multiscale_entropy(ts, 4, maxscale=1)
            cnts += 1
        indices.append(ins)

    for i in range(int(dataLen / 2), dataLen):
        CSIb1OrigBack.append(CSIb1Orig[indices[i - int(dataLen / 2)]])

    for i in range(int(dataLen / 2), dataLen):
        CSIe1OrigBack.append(CSIe1Orig[indices[i - int(dataLen / 2)]])

    return CSIa1Orig, CSIb1Orig, CSIe1Orig


def splitEntropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, segLen, dataLen, entropyThres):
    # 先整体shuffle一次
    shuffleInd = np.random.permutation(dataLen)
    CSIa1Orig = CSIa1Orig[shuffleInd]
    CSIb1Orig = CSIb1Orig[shuffleInd]
    CSIe1Orig = CSIe1Orig[shuffleInd]

    sortCSIa1Reshape = CSIa1Orig[0:segLen * int(dataLen / segLen)]
    sortCSIb1Reshape = CSIb1Orig[0:segLen * int(dataLen / segLen)]
    sortCSIe1Reshape = CSIe1Orig[0:segLen * int(dataLen / segLen)]

    sortCSIa1Reshape = sortCSIa1Reshape.reshape(int(len(sortCSIa1Reshape) / segLen), segLen)
    sortCSIb1Reshape = sortCSIb1Reshape.reshape(int(len(sortCSIb1Reshape) / segLen), segLen)
    sortCSIe1Reshape = sortCSIe1Reshape.reshape(int(len(sortCSIe1Reshape) / segLen), segLen)
    n = len(sortCSIa1Reshape)

    for i in range(n):
        a_mul_entropy = ent.multiscale_entropy(sortCSIa1Reshape[i], 3, maxscale=1)

        cnts = 0
        while a_mul_entropy < entropyThres and cnts < 10:
            shuffleInd = np.random.permutation(len(sortCSIa1Reshape[i]))
            sortCSIa1Reshape[i] = sortCSIa1Reshape[i][shuffleInd]
            sortCSIb1Reshape[i] = sortCSIb1Reshape[i][shuffleInd]
            sortCSIe1Reshape[i] = sortCSIe1Reshape[i][shuffleInd]

            a_mul_entropy = ent.multiscale_entropy(sortCSIa1Reshape[i], 3, maxscale=1)
            cnts += 1

    _CSIa1Orig = []
    _CSIb1Orig = []
    _CSIe1Orig = []

    for i in range(len(sortCSIa1Reshape)):
        for j in range(len(sortCSIa1Reshape[i])):
            _CSIa1Orig.append(sortCSIa1Reshape[i][j])
            _CSIb1Orig.append(sortCSIb1Reshape[i][j])
            _CSIe1Orig.append(sortCSIe1Reshape[i][j])

    return np.array(_CSIa1Orig), np.array(_CSIb1Orig), np.array(_CSIe1Orig)


isShow = False
fileNameA = "../data/CSIa_r.mat"
fileNameB = "../data/CSIb_r.mat"
rawDataA = loadmat(fileNameA)
rawDataB = loadmat(fileNameB)
csi_csv = open("evaluations/CSI.csv", "a+")

CSIa1OrigRaw1 = rawDataA['CSIa'][0]
CSIb1OrigRaw1 = rawDataB['CSIb'][0]

# csv1 = open("../correlation/csi1.csv", "r")
# reader1 = csv.reader(csv1)
# CSIa1OrigRaw2 = []
# for item in reader1:
#     CSIa1OrigRaw2.append(float(item[0]))
#
# csv2 = open("../correlation/csi2.csv", "r")
# reader2 = csv.reader(csv2)
# CSIb1OrigRaw2 = []
# for item in reader2:
#     CSIb1OrigRaw2.append(float(item[0]))

CSIi1OrigRaw1 = loadmat('../data/CSIa_r.mat')['CSIa'][0]
minLen1 = min(len(CSIa1OrigRaw1), len(CSIi1OrigRaw1))
CSIa1Orig = CSIa1OrigRaw1[:minLen1]
CSIb1Orig = CSIb1OrigRaw1[:minLen1]
CSIi1Orig = CSIi1OrigRaw1[:minLen1]

dataLen = len(CSIa1Orig)
CSIb1Orig = CSIb1Orig - (np.mean(CSIb1Orig) - np.mean(CSIa1Orig))
CSIn1Orig = np.random.normal(loc=-1, scale=1, size=dataLen)  ## Multiplication item normal distribution
CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)

entropyTime = time.time()
entropyThres = 2
CSIa1Orig, CSIb1Orig, CSIe1Orig = splitEntropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, 32, dataLen, entropyThres)
# CSIa1Orig, CSIb1Orig, CSIe1Orig = entropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, dataLen, entropyThres)
print("--- entropyTime %s seconds ---" % (time.time() - entropyTime))

CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
CSIe1OrigBack = CSIe1Orig.copy()
CSIi1OrigBack = CSIi1Orig.copy()
CSIn1OrigBack = CSIn1Orig.copy()

intvl = 5
keyLen = 64
segLen = 5
addNoise = False
metrics = [absolute, absolute2, euclidean, manhattan, chebyshev, cosine, dtw, correlation]
metric = metrics[0]
rule = {'=': 0, '+': 1, '-': 2}

originSum = 0
correctSum = 0
randomSum = 0
noiseSum = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum = 0
noiseWholeSum = 0

codings = ""
times = 0
maxDiffAB = 0

# 不同的距离函数对应着不同的阈值
# WITHOUT NOISE
threshold = 0.2  # absolute
for staInd in range(0, len(CSIa1Orig), intvl * keyLen):
    processTime = time.time()

    endInd = staInd + keyLen * intvl
    print("range:", staInd, endInd)
    if endInd >= len(CSIa1Orig):
        break
    times += 1

    CSIa1Orig = CSIa1OrigBack.copy()
    CSIb1Orig = CSIb1OrigBack.copy()
    CSIe1Orig = CSIe1OrigBack.copy()
    CSIi1Orig = CSIi1OrigBack.copy()
    CSIn1Orig = CSIn1OrigBack.copy()

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
    tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
    tmpCSIi1 = CSIi1Orig[range(staInd, endInd, 1)]
    tmpNoise = CSIn1Orig[range(staInd, endInd, 1)]

    tmpCSIa1 = tmpCSIa1 - (np.mean(tmpCSIa1) - np.mean(tmpCSIb1))  # Mean value consistency

    # linspace函数生成元素为50的等间隔数列，可以指定第三个参数为元素个数
    # signal.square返回周期性的方波波形
    tmpPulse = signal.square(
        2 * np.pi * 1 / intvl * np.linspace(0, np.pi * 0.5 * keyLen, keyLen * intvl))  ## Rectangular pulse

    if addNoise:
        # tmpCSIa1 = tmpPulse * (np.float_power(np.abs(tmpCSIa1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIa1))
        # tmpCSIb1 = tmpPulse * (np.float_power(np.abs(tmpCSIb1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIb1))
        # tmpCSIe1 = tmpPulse * (np.float_power(np.abs(tmpCSIe1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIe1))
        # tmpCSIi1 = tmpPulse * (np.float_power(np.abs(tmpCSIi1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIi1))
        tmpCSIa1 = tmpPulse * np.float_power(np.abs(tmpCSIa1), tmpNoise)
        tmpCSIb1 = tmpPulse * np.float_power(np.abs(tmpCSIb1), tmpNoise)
        tmpCSIe1 = tmpPulse * np.float_power(np.abs(tmpCSIe1), tmpNoise)
        tmpCSIi1 = tmpPulse * np.float_power(np.abs(tmpCSIi1), tmpNoise)
    else:
        tmpCSIa1 = tmpPulse * tmpCSIa1
        tmpCSIb1 = tmpPulse * tmpCSIb1
        tmpCSIe1 = tmpPulse * tmpCSIe1
        tmpCSIi1 = tmpPulse * tmpCSIi1

    CSIa1Orig[range(staInd, endInd, 1)] = tmpCSIa1
    CSIb1Orig[range(staInd, endInd, 1)] = tmpCSIb1
    CSIe1Orig[range(staInd, endInd, 1)] = tmpCSIe1
    CSIi1Orig[range(staInd, endInd, 1)] = tmpCSIi1
    CSIn1Orig[range(staInd, endInd, 1)] = tmpNoise

    permLen = len(range(staInd, endInd, intvl))
    origInd = np.array([xx for xx in range(staInd, endInd, intvl)])

    sortCSIa1 = np.zeros(permLen)
    sortCSIb1 = np.zeros(permLen)
    sortCSIe1 = np.zeros(permLen)
    sortNoise = np.zeros(permLen)
    sortCSIi1 = np.zeros(permLen)

    # for ii in range(permLen):
    #     aIndVec = np.array([aa for aa in range(origInd[ii], origInd[ii] + intvl, 1)])
    #     sortCSIa1[ii] = np.mean(CSIa1Orig[aIndVec])
    #     sortCSIb1[ii] = np.mean(CSIb1Orig[aIndVec])
    #     sortCSIe1[ii] = np.mean(CSIe1Orig[aIndVec])
    #     sortCSIi1[ii] = np.mean(CSIi1Orig[aIndVec])
    #     sortNoise[ii] = np.mean(CSIn1Orig[aIndVec])

    # 求平均的好处的不同分段长度的测试可以都用同一个threshold
    # for ii in range(permLen):
    #     aIndVec = np.array([aa for aa in range(origInd[ii], origInd[ii] + intvl, 1)])  ## for non-permuted CSIa1
    #
    #     CSIa1Tmp = CSIa1Orig[aIndVec]
    #     CSIb1Tmp = CSIb1Orig[aIndVec]
    #     CSIe1Tmp = CSIe1Orig[aIndVec]
    #     CSIi1Tmp = CSIi1Orig[aIndVec]
    #     CSIn1Tmp = CSIn1Orig[aIndVec]
    #
    #     sortCSIa1[ii] = np.mean(CSIa1Tmp)  ## Metric 1: Mean
    #     sortCSIb1[ii] = np.mean(CSIb1Tmp)  # 只赋值一次
    #     sortCSIe1[ii] = np.mean(CSIe1Tmp)
    #     sortCSIi1[ii] = np.mean(CSIi1Tmp)
    #     sortNoise[ii] = np.mean(CSIn1Tmp)

    # for jj in range(permLen, permLen * 2):
    #     bIndVec = np.array([bb for bb in range(origInd[jj - permLen], origInd[jj - permLen] + intvl, 1)])
    #
    #     CSIa1Tmp = CSIa1Orig[bIndVec]
    #     CSIb1Tmp = CSIb1Orig[bIndVec]
    #     CSIe1Tmp = CSIe1Orig[bIndVec]
    #     CSIi1Tmp = CSIi1Orig[bIndVec]
    #     CSIn1Tmp = CSIn1Orig[bIndVec]
    #
    #     sortCSIa1[jj - permLen] = np.mean(CSIa1Tmp)  ## Metric 1: Mean
    #     sortCSIb1[jj - permLen] = np.mean(CSIb1Tmp)  # 只赋值一次
    #     sortCSIe1[jj - permLen] = np.mean(CSIe1Tmp)
    #     sortCSIi1[jj - permLen] = np.mean(CSIi1Tmp)
    #     sortNoise[jj - permLen] = np.mean(CSIn1Tmp)

    for ii in range(permLen):
        aIndVec = np.array([aa for aa in range(origInd[ii], origInd[ii] + intvl, 1)])  ## for non-permuted CSIa1

        for jj in range(permLen, permLen * 2):
            bIndVec = np.array([bb for bb in range(origInd[jj - permLen], origInd[jj - permLen] + intvl, 1)])

            CSIa1Tmp = CSIa1Orig[aIndVec]
            CSIb1Tmp = CSIb1Orig[bIndVec]
            CSIe1Tmp = CSIe1Orig[bIndVec]
            CSIi1Tmp = CSIi1Orig[bIndVec]
            CSIn1Tmp = CSIn1Orig[aIndVec]

            sortCSIa1[ii] = np.mean(CSIa1Tmp)  ## Metric 1: Mean
            sortCSIb1[jj - permLen] = np.mean(CSIb1Tmp)  # 只赋值一次
            sortCSIe1[jj - permLen] = np.mean(CSIe1Tmp)
            sortCSIi1[jj - permLen] = np.mean(CSIi1Tmp)
            sortNoise[ii - permLen] = np.mean(CSIn1Tmp)

    # sortCSIa1是原始算法中排序前的数据
    # 防止对数的真数为0导致计算错误（不平滑的话没有这个问题）
    sortCSIa1 = np.log10(np.abs(sortCSIa1) + 0.1)
    sortCSIb1 = np.log10(np.abs(sortCSIb1) + 0.1)
    sortCSIe1 = np.log10(np.abs(sortCSIe1) + 0.1)
    # sortNoise = np.log10(np.abs(sortNoise) + 0.1)
    sortCSIi1 = np.log10(np.abs(sortCSIi1) + 0.1)

    # 取原数据的一部分来reshape
    sortCSIa1Reshape = sortCSIa1[0:segLen * int(len(sortCSIa1) / segLen)]
    sortCSIb1Reshape = sortCSIb1[0:segLen * int(len(sortCSIb1) / segLen)]
    sortCSIe1Reshape = sortCSIe1[0:segLen * int(len(sortCSIe1) / segLen)]
    sortNoiseReshape = sortNoise[0:segLen * int(len(sortNoise) / segLen)]
    sortCSIi1Reshape = sortCSIi1[0:segLen * int(len(sortCSIi1) / segLen)]

    sortCSIa1Reshape = sortCSIa1Reshape.reshape(int(len(sortCSIa1Reshape) / segLen), segLen)
    sortCSIb1Reshape = sortCSIb1Reshape.reshape(int(len(sortCSIb1Reshape) / segLen), segLen)
    sortCSIe1Reshape = sortCSIe1Reshape.reshape(int(len(sortCSIe1Reshape) / segLen), segLen)
    sortNoiseReshape = sortNoiseReshape.reshape(int(len(sortNoiseReshape) / segLen), segLen)
    sortCSIi1Reshape = sortCSIi1Reshape.reshape(int(len(sortCSIi1Reshape) / segLen), segLen)

    sortCSIa1 = []
    sortCSIb1 = []
    sortCSIe1 = []
    sortNoise = []
    sortCSIi1 = []

    # 归一化
    for i in range(len(sortCSIa1Reshape)):
        # sklearn的归一化是按列转换，因此需要先转为列向量
        sortCSIa1.append(preprocessing.MinMaxScaler().fit_transform(
            np.array(sortCSIa1Reshape[i]).reshape(-1, 1)).reshape(1, -1).tolist()[0])
        sortCSIb1.append(preprocessing.MinMaxScaler().fit_transform(
            np.array(sortCSIb1Reshape[i]).reshape(-1, 1)).reshape(1, -1).tolist()[0])
        sortCSIe1.append(preprocessing.MinMaxScaler().fit_transform(
            np.array(sortCSIe1Reshape[i]).reshape(-1, 1)).reshape(1, -1).tolist()[0])
        sortNoise.append(preprocessing.MinMaxScaler().fit_transform(
            np.array(sortNoiseReshape[i]).reshape(-1, 1)).reshape(1, -1).tolist()[0])
        sortCSIi1.append(preprocessing.MinMaxScaler().fit_transform(
            np.array(sortCSIi1Reshape[i]).reshape(-1, 1)).reshape(1, -1).tolist()[0])

    shuffleArray = list(range(len(sortCSIa1)))
    CSIa1Back = []
    CSIb1Back = []
    CSIe1Back = []
    CSIn1Back = []
    random.shuffle(shuffleArray)
    for i in range(len(sortCSIa1)):
        CSIa1Back.append(sortCSIa1[shuffleArray[i]])
    for i in range(len(sortCSIb1)):
        CSIb1Back.append(sortCSIb1[shuffleArray[i]])
    for i in range(len(sortCSIe1)):
        CSIe1Back.append(sortCSIe1[shuffleArray[i]])
    for i in range(len(sortNoise)):
        CSIn1Back.append(sortNoise[shuffleArray[i]])
    sortCSIa1 = CSIa1Back
    sortCSIb1 = CSIb1Back
    sortCSIe1 = CSIe1Back
    sortNoise = CSIn1Back

    # 最后各自的密钥
    a_list = []
    b_list = []
    e_list = []
    n_list = []

    diffAB = 0
    for i in range(len(sortCSIa1)):
        diffAB = max(diffAB, metric(sortCSIa1[i], sortCSIb1[i]))
    print("\033[0;32;40mAB对应位置最大差距", diffAB, "\033[0m")
    maxDiffAB = max(maxDiffAB, diffAB)
    allDiffAB = 0
    for i in range(len(sortCSIa1)):
        for j in range(len(sortCSIb1)):
            allDiffAB = max(allDiffAB, metric(sortCSIa1[i], sortCSIb1[j]))
    print("\033[0;32;40mAB所有的对应位置最大差距", allDiffAB, "\033[0m")

    opNums = int(len(sortCSIa1) / 2)
    index = random.sample(range(opNums), opNums)

    sortCSIa1P = list(sortCSIa1)
    insertNum = 0
    deleteNum = 0

    opIndex = []
    editOps = random.sample(range(int(len(sortCSIa1))), opNums)
    editOps.sort()
    # sortCSIi1 = np.random.normal(loc=np.mean(sortCSIa1), scale=np.std(sortCSIa1, ddof=1), size=len(sortCSIa1))
    # sortCSIi1 = np.random.uniform(min(sortCSIa1), max(sortCSIa1), len(sortCSIa1))
    for i in range(opNums - 1, -1, -1):
        flag = random.randint(0, 1)
        # 不重复编辑同一个元素
        if flag == 0:
            insertIndex = random.randint(0, len(sortCSIi1) - 1)
            sortCSIa1P.insert(editOps[i], sortCSIi1[insertIndex])
            insertNum += 1
        elif flag == 1:
            sortCSIa1P.remove(sortCSIa1P[editOps[i]])
            deleteNum += 1

    # print("numbers of insert:", insertNum)
    # print("numbers of delete:", deleteNum)

    # print("sortCSIa1P", len(sortCSIa1P), list(sortCSIa1P))
    # print("sortCSIa1", len(sortCSIa1), list(sortCSIa1))
    # print("sortCSIb1", len(sortCSIb1), list(sortCSIb1))
    # print("sortCSIe1", len(sortCSIe1), list(sortCSIe1))
    # print("sortNoise", len(sortNoise), list(sortNoise))
    print("--- processTime: %s seconds ---" % (time.time() - processTime))

    editTime = time.time()

    # 用a1P匹配ai，得到rule，再用rule对其a1P
    # 匹配不相等的是在敌手成功率和加密强度间的折中
    # threshold = diffAB * 1.5
    ruleStr1 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortCSIa1, threshold, metric)
    alignStr1 = genAlign(ruleStr1)
    print("--- editTime %s seconds ---" % (time.time() - editTime))

    # print("ruleStr1", len(ruleStr1), ruleStr1)
    ruleStr2 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortCSIb1, threshold, metric)
    alignStr2 = genAlign(ruleStr2)
    # print("ruleStr2", len(ruleStr2), ruleStr2)
    ruleStr3 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortCSIe1, threshold, metric)
    alignStr3 = genAlign(ruleStr3)
    # print("ruleStr3", len(ruleStr3), ruleStr3)
    ruleStr4 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortNoise, threshold, metric)
    alignStr4 = genAlign(ruleStr4)

    a_list = ""
    b_list = ""
    e_list = ""
    n_list = ""

    # 转为二进制
    for i in range(len(alignStr1)):
        a_list += bin(alignStr1[i])[2:]
    for i in range(len(alignStr2)):
        b_list += bin(alignStr2[i])[2:]
    for i in range(len(alignStr3)):
        e_list += bin(alignStr3[i])[2:]
    for i in range(len(alignStr4)):
        n_list += bin(alignStr4[i])[2:]

    # 对齐密钥，随机补全
    for i in range(len(a_list) - len(e_list)):
        e_list += str(np.random.randint(0, 2))
    for i in range(len(a_list) - len(n_list)):
        n_list += str(np.random.randint(0, 2))

    # 转化为keyLen长的bits
    a_bits = ""
    b_bits = ""
    e_bits = ""
    n_bits = ""

    rounda = int(len(a_list) / keyLen)
    remaina = len(a_list) - rounda * keyLen
    if remaina >= 0:
        rounda += 1

    for i in range(keyLen):
        tmp = 0
        for j in range(rounda):
            if j * keyLen + i >= len(a_list):
                continue
            tmp += int(a_list[j * keyLen + i])
        a_bits += str(tmp % 2)

    roundb = int(len(b_list) / keyLen)
    remainb = len(b_list) - roundb * keyLen
    if remainb >= 0:
        roundb += 1

    for i in range(keyLen):
        tmp = 0
        for j in range(roundb):
            if j * keyLen + i >= len(b_list):
                continue
            tmp += int(b_list[j * keyLen + i])
        b_bits += str(tmp % 2)

    rounde = int(len(e_list) / keyLen)
    remaine = len(e_list) - rounde * keyLen
    if remaine >= 0:
        rounde += 1

    for i in range(keyLen):
        tmp = 0
        for j in range(rounde):
            if j * keyLen + i >= len(e_list):
                continue
            tmp += int(e_list[j * keyLen + i])
        e_bits += str(tmp % 2)

    roundn = int(len(n_list) / keyLen)
    remainn = len(n_list) - roundn * keyLen
    if remainn >= 0:
        roundn += 1

    for i in range(keyLen):
        tmp = 0
        for j in range(roundn):
            if j * keyLen + i >= len(n_list):
                continue
            tmp += int(n_list[j * keyLen + i])
        n_bits += str(tmp % 2)

    editOps = list(set(range(len(sortCSIa1))).difference(set(editOps)))
    # print("editOps", len(editOps), editOps)
    print("keys of a:", len(a_list), a_list)
    print("keys of a:", len(a_bits), a_bits)
    print("keys of b:", len(b_list), b_list)
    print("keys of b:", len(b_bits), b_bits)
    print("keys of e:", len(e_list), e_list)
    print("keys of e:", len(e_bits), e_bits)
    print("keys of n:", len(n_list), n_list)
    print("keys of n:", len(n_bits), n_bits)

    # print("longest numbers of a:", genLongestContinuous(a_list))
    # print("longest numbers of b:", genLongestContinuous(b_list))
    # print("longest numbers of e:", genLongestContinuous(e_list))
    # print("longest numbers of n:", genLongestContinuous(n_list))

    sum1 = min(len(a_bits), len(b_bits))
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for i in range(0, sum1):
        sum2 += (a_bits[i] == b_bits[i])
    for i in range(min(len(a_bits), len(e_bits))):
        sum3 += (a_bits[i] == e_bits[i])
    for i in range(min(len(a_bits), len(n_bits))):
        sum4 += (a_bits[i] == n_bits[i])

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

    # 编码密钥
    # char_weights = []
    # for i in range(len(a_list)):
    #     char_weights.append((a_list[i], a_list[i]))
    # tree = HuffmanTree(char_weights)
    # tree.get_code()
    # HuffmanTree.codings += "\n"

    # coding = 0
    # for i in range(len(a_list)):
    #     coding += a_list[i]
    #     for j in range(len(sortCSIa1P[i])):
    #         coding += sortCSIa1P[i][j]
    #     coding = int(coding)
    #     codings += bin(coding)[2:] + "\n"

#     for i in range(len(a_list)):
#         codings += bin(a_list[i])[2:] + "\n"
#
# with open('./evaluations/key_data_static_indoor_1.txt', 'a', ) as f:
#     f.write(codings)
print(maxDiffAB)
print("a-b all", correctSum, "/", originSum, "=", correctSum / originSum)
print("a-e all", randomSum, "/", originSum, "=", randomSum / originSum)
print("a-n all", noiseSum, "/", originSum, "=", noiseSum / originSum)
print("a-b whole match", correctWholeSum, "/", originWholeSum, "=", correctWholeSum / originWholeSum)
print("a-e whole match", randomWholeSum, "/", originWholeSum, "=", randomWholeSum / originWholeSum)
print("a-n whole match", noiseWholeSum, "/", originWholeSum, "=", noiseWholeSum / originWholeSum)
print("times", times)
# csi_csv.write(fileNameA + ',' + str(times) + ',' + str(threshold) + ',' + str(segLen) + ',' + str(maxDiffAB)
#               + ',' + str(correctSum / originSum) + ',' + str(randomSum / originSum)
#               + ',' + str(noiseSum / originSum) + ',' + str(correctWholeSum / originWholeSum)
#               + ',' + str(randomWholeSum / originWholeSum) + ',' + str(noiseWholeSum / originWholeSum) + '\n')
# csi_csv.close()
