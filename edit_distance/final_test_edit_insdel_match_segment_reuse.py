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


def splitEntropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, segLen, dataLen):
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
        # entropyThres = 0.2 * np.std(sortCSIa1Reshape[i])
        entropyThres = 2.5
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
fileName = "../data/data_static_outdoor_1.mat"
rawData = loadmat(fileName)

for rep in range(50):
    print("rep", rep)
    CSIa1OrigRaw = rawData['A'][:, 0]
    CSIb1OrigRaw = rawData['A'][:, 1]
    CSIi1OrigRaw = loadmat('../data/data_static_indoor_1_r.mat')['A'][:, 0]
    minLen = min(len(CSIa1OrigRaw), len(CSIb1OrigRaw))
    CSIa1Orig = CSIa1OrigRaw[:minLen]
    CSIb1Orig = CSIb1OrigRaw[:minLen]
    CSIi1Orig = CSIi1OrigRaw[:minLen]
    dataLen = len(CSIa1Orig)
    CSIb1Orig = CSIb1Orig - (np.mean(CSIb1Orig) - np.mean(CSIa1Orig))

    CSIa1Orig = np.array(CSIa1Orig)
    CSIb1Orig = np.array(CSIb1Orig)
    CSIi1Orig = np.array(CSIi1Orig)
    CSIn1Orig = np.random.normal(loc=-1, scale=1, size=dataLen)  ## Multiplication item normal distribution
    CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)
    CSIa1Orig, CSIb1Orig, CSIe1Orig = splitEntropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, 100, dataLen)
    # CSIa1Orig, CSIb1Orig, CSIe1Orig = entropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, dataLen, entropyThres=2)

    CSIa1OrigBack = CSIa1Orig.copy()
    CSIb1OrigBack = CSIb1Orig.copy()
    CSIe1OrigBack = CSIe1Orig.copy()
    CSIi1OrigBack = CSIi1Orig.copy()
    CSIn1OrigBack = CSIn1Orig.copy()

    intvl = 5
    keyLen = 64
    segLen = 6
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
    threshold = 0.05  # absolute
    for staInd in range(0, len(CSIa1Orig), intvl * keyLen):
        endInd = staInd + keyLen * intvl
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

        tmpCSIa1 = tmpPulse * tmpCSIa1
        tmpCSIb1 = tmpPulse * tmpCSIb1
        tmpCSIe1 = tmpPulse * tmpCSIe1
        tmpCSIi1 = tmpPulse * tmpCSIi1

        sortCSIa1 = tmpCSIa1
        sortCSIb1 = tmpCSIb1
        sortCSIe1 = tmpCSIe1
        sortCSIi1 = tmpCSIi1
        sortNoise = tmpNoise

        # sortCSIa1是原始算法中排序前的数据
        # 防止对数的真数为0导致计算错误（不平滑的话没有这个问题）
        sortCSIa1 = np.log10(np.abs(sortCSIa1) + 0.1)
        sortCSIb1 = np.log10(np.abs(sortCSIb1) + 0.1)
        sortCSIe1 = np.log10(np.abs(sortCSIe1) + 0.1)
        sortNoise = np.log10(np.abs(sortNoise) + 0.1)
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

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 第一次生成密钥 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
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

        # 用a1P匹配ai，得到rule，再用rule对其a1P
        # 匹配不相等的是在敌手成功率和加密强度间的折中
        # threshold = diffAB * 1.5
        ruleStr1 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortCSIa1, threshold, metric)
        alignStr1 = genAlign(ruleStr1)

        ruleStr2 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortCSIb1, threshold, metric)
        alignStr2 = genAlign(ruleStr2)
        ruleStr3 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortCSIe1, threshold, metric)
        alignStr3 = genAlign(ruleStr3)
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

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 第二次生成密钥 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
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

        # 用a1P匹配ai，得到rule，再用rule对其a1P
        # 匹配不相等的是在敌手成功率和加密强度间的折中
        # threshold = diffAB * 1.5
        ruleStr1 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortCSIa1, threshold, metric)
        alignStr1 = genAlign(ruleStr1)

        ruleStr2 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortCSIb1, threshold, metric)
        alignStr2 = genAlign(ruleStr2)
        ruleStr3 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortCSIe1, threshold, metric)
        alignStr3 = genAlign(ruleStr3)
        ruleStr4 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortNoise, threshold, metric)
        alignStr4 = genAlign(ruleStr4)

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

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 第三次生成密钥 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
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

        # 用a1P匹配ai，得到rule，再用rule对其a1P
        # 匹配不相等的是在敌手成功率和加密强度间的折中
        # threshold = diffAB * 1.5
        ruleStr1 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortCSIa1, threshold, metric)
        alignStr1 = genAlign(ruleStr1)

        ruleStr2 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortCSIb1, threshold, metric)
        alignStr2 = genAlign(ruleStr2)
        ruleStr3 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortCSIe1, threshold, metric)
        alignStr3 = genAlign(ruleStr3)
        ruleStr4 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortNoise, threshold, metric)
        alignStr4 = genAlign(ruleStr4)

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

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 第四次生成密钥 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
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

        # 用a1P匹配ai，得到rule，再用rule对其a1P
        # 匹配不相等的是在敌手成功率和加密强度间的折中
        # threshold = diffAB * 1.5
        ruleStr1 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortCSIa1, threshold, metric)
        alignStr1 = genAlign(ruleStr1)

        ruleStr2 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortCSIb1, threshold, metric)
        alignStr2 = genAlign(ruleStr2)
        ruleStr3 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortCSIe1, threshold, metric)
        alignStr3 = genAlign(ruleStr3)
        ruleStr4 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortNoise, threshold, metric)
        alignStr4 = genAlign(ruleStr4)

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

        # 编码密钥
        # char_weights = []
        # for i in range(len(a_list)):
        #     char_weights.append((a_list[i], a_list[i]))
        # tree = HuffmanTree(char_weights)
        # tree.get_code()
        # HuffmanTree.codings += "\n"

        for i in range(0, len(a_list), 25):
            codings += a_list[i:i + 25] + "\n"

    with open('./evaluations/key_data_static_outdoor_1_r.txt', 'a', ) as f:
        f.write(codings)
