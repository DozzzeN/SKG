import random
from collections import Counter

import numpy as np
from scipy import signal
from scipy.io import loadmat
from scipy.spatial.distance import euclidean, chebyshev
from sklearn import preprocessing
import graycode

from alignment import genAlign, alignFloatInsDelWithMetrics, absolute, cosine, dtw, manhattan, correlation, absolute2
from huffman import HuffmanTree


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


isShow = False
fileName = "../skyglow/Scenario2-Office-NLoS/data_NLOS.mat"
rawData = loadmat(fileName)
csv = open("evaluations/res.csv", "a+")

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
CSIe1Orig = loadmat('../skyglow/Scenario2-Office-NLoS/data_eave_NLOS.mat')['A'][:, 0]
CSIi10rig = loadmat('../skyglow/Scenario2-Office-LoS-eve_NLoS/data_LOS_Eve_NLOS.mat')['A'][:, 0]

dataLen = len(CSIe1Orig)  # 以eve的数据为准计算

CSIa1Orig = np.array(CSIa1Orig)
CSIb1Orig = np.array(CSIb1Orig)
CSIi10rig = np.array(CSIi10rig)
CSIn10rig = np.random.normal(loc=-1, scale=1, size=dataLen)  ## Multiplication item normal distribution

CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
CSIe1OrigBack = CSIe1Orig.copy()
CSIi1OrigBack = CSIi10rig.copy()
CSIn10rigBack = CSIn10rig.copy()

sft = 2
intvl = 2 * sft + 1
keyLen = 128
csv.write("\n")
csv.write("filename," + "times," + "threshold," + "segLen," + "maxDiffAB," +
          "correctBitRate," + "randomBitRate," + "noiseBitRate," +
          "correctWholeRate," + "randomWholeRate," + "noiseWholeRate\n")

for segLen in range(2, 11):
    print("segLen", segLen)
    addNoise = False
    metrics = [absolute, absolute2, euclidean, manhattan, chebyshev, cosine, dtw, correlation]
    metric = metrics[0]
    rule = {'=': 0, '+': 1, '-': 1}

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
    threshold = 0.01  # absolute
    # threshold = diffAB
    for staInd in range(0, len(CSIe1Orig), keyLen * intvl):
        endInd = staInd + keyLen * intvl
        print("range:", staInd, endInd)
        if endInd >= len(CSIe1Orig):
            break
        times += 1

        CSIa1Orig = CSIa1OrigBack.copy()
        CSIb1Orig = CSIb1OrigBack.copy()
        CSIe1Orig = CSIe1OrigBack.copy()
        CSIi1Orig = CSIi1OrigBack.copy()
        CSIn10rig = CSIn10rigBack.copy()

        tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
        tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
        tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
        tmpCSIi1 = CSIi1Orig[range(staInd, endInd, 1)]
        tmpNoise = CSIn10rig[range(staInd, endInd, 1)]

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
        CSIn10rig[range(staInd, endInd, 1)] = tmpNoise

        permLen = len(range(staInd, endInd, intvl))
        origInd = np.array([xx for xx in range(staInd, endInd, intvl)])

        sortCSIa1 = np.zeros(permLen)
        sortCSIb1 = np.zeros(permLen)
        sortCSIe1 = np.zeros(permLen)
        sortNoise = np.zeros(permLen)
        sortCSIi1 = np.zeros(permLen)

        # 求平均的好处的不同分段长度的测试可以都用同一个threshold
        for ii in range(permLen):
            aIndVec = np.array([aa for aa in range(origInd[ii], origInd[ii] + intvl, 1)])  ## for non-permuted CSIa1

            for jj in range(permLen, permLen * 2):
                bIndVec = np.array([bb for bb in range(origInd[jj - permLen], origInd[jj - permLen] + intvl, 1)])

                CSIa1Tmp = CSIa1Orig[aIndVec]
                CSIb1Tmp = CSIb1Orig[bIndVec]
                CSIe1Tmp = CSIe1Orig[bIndVec]
                CSIi1Tmp = CSIi1Orig[bIndVec]
                CSIn1Tmp = CSIn10rig[aIndVec]

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

        # shuffleArray = list(range(len(sortCSIa1)))
        # CSIa1Back = []
        # CSIb1Back = []
        # CSIe1Back = []
        # CSIn1Back = []
        # random.shuffle(shuffleArray)
        # for i in range(len(sortCSIa1)):
        #     CSIa1Back.append(sortCSIa1[shuffleArray[i]])
        # for i in range(len(sortCSIb1)):
        #     CSIb1Back.append(sortCSIb1[shuffleArray[i]])
        # for i in range(len(sortCSIe1)):
        #     CSIe1Back.append(sortCSIe1[shuffleArray[i]])
        # for i in range(len(sortNoise)):
        #     CSIn1Back.append(sortNoise[shuffleArray[i]])
        # sortCSIa1 = CSIa1Back
        # sortCSIb1 = CSIb1Back
        # sortCSIe1 = CSIe1Back
        # sortNoise = CSIn1Back

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

        # 用a1P匹配ai，得到rule，再用rule对其a1P
        # 匹配不相等的是在敌手成功率和加密强度间的折中
        ruleStr1 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortCSIa1, threshold, metric)
        alignStr1 = genAlign(ruleStr1)
        ruleStr2 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortCSIb1, threshold, metric)
        alignStr2 = genAlign(ruleStr2)
        ruleStr3 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortCSIe1, threshold, metric)
        alignStr3 = genAlign(ruleStr3)
        ruleStr4 = alignFloatInsDelWithMetrics(rule, sortCSIa1P, sortNoise, threshold, metric)
        alignStr4 = genAlign(ruleStr4)

        a_list = alignStr1
        b_list = alignStr2
        e_list = alignStr3
        n_list = alignStr4

        editOps = list(set(range(len(sortCSIa1))).difference(set(editOps)))

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

        # 编码密钥
        # char_weights = []
        # weights = Counter(a_list)  # 得到list中元素出现次数
        # for i in range(len(a_list)):
        #     char_weights.append((a_list[i], weights[a_list[i]]))
        # tree = HuffmanTree(char_weights)
        # tree.get_code()
        # HuffmanTree.codings += "\n"
        #
        # for i in range(len(a_list)):
        #     codings += bin(a_list[i])[2:] + "\n"
        for i in range(len(a_list)):
            codings += '{:03b}'.format(graycode.tc_to_gray_code(a_list[i])) + "\n"

    with open('./evaluations/key.txt', 'a', ) as f:
        f.write(codings)

    print("a-b all", correctSum, "/", originSum, "=", correctSum / originSum)
    print("a-e all", randomSum, "/", originSum, "=", randomSum / originSum)
    print("a-n all", noiseSum, "/", originSum, "=", noiseSum / originSum)
    print("a-b whole match", correctWholeSum, "/", originWholeSum, "=", correctWholeSum / originWholeSum)
    print("a-e whole match", randomWholeSum, "/", originWholeSum, "=", randomWholeSum / originWholeSum)
    print("a-n whole match", noiseWholeSum, "/", originWholeSum, "=", noiseWholeSum / originWholeSum)
    print("times", times)
    csv.write(fileName + ',' + str(times) + ',' + str(threshold) + ',' + str(segLen) + ',' + str(maxDiffAB)
              + ',' + str(correctSum / originSum) + ',' + str(randomSum / originSum)
              + ',' + str(noiseSum / originSum) + ',' + str(correctWholeSum / originWholeSum)
              + ',' + str(randomWholeSum / originWholeSum) + ',' + str(noiseWholeSum / originWholeSum) + '\n')
csv.close()
