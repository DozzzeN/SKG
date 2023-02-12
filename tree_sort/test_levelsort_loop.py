import math
import time

import numpy as np
from scipy import signal
from scipy.io import loadmat

from algorithm import smooth, binaryTreeSort, quadTreeSort, genSample, insertNoise, binaryTreeSortPerm, binaryLevelSort, \
    binaryLevelSortPerm

fileName = "../data/data_mobile_indoor_1_r.mat"
rawData = loadmat(fileName)

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

dataLen = len(CSIa1Orig)

CSIa1Orig = np.array(CSIa1Orig)
CSIb1Orig = np.array(CSIb1Orig)
CSIn1Orig = np.random.normal(loc=-100, scale=1000, size=dataLen)
# CSIa1OrigMean = np.mean(CSIa1Orig)
# CSIa1OrigStd = np.std(CSIa1Orig, ddof=1)
# randomLoc = np.random.randint(CSIa1OrigMean - CSIa1OrigMean / 2, - CSIa1OrigMean / 2) if CSIa1OrigMean < 0 \
#     else np.random.randint(- CSIa1OrigMean / 2, CSIa1OrigMean - CSIa1OrigMean / 2)
# randomScale = np.random.randint(CSIa1OrigStd / 2, CSIa1OrigStd + CSIa1OrigStd / 2)
# CSIn1Orig = np.random.normal(loc=randomLoc, scale=randomScale, size=dataLen)
# CSIn1Orig = np.random.normal(loc=0, scale=np.std(CSIa1Orig, ddof=1), size=dataLen)
# CSIn1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)
CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)

CSIi1Orig = loadmat('../data/data_mobile_indoor_1_r.mat')['A'][:, 0]
# 保护原始数据
# CSIi1Orig = CSIi1Orig + np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)
# CSIi1Orig = CSIi1Orig + np.random.normal(loc=0, scale=np.std(CSIi1Orig, ddof=1), size=dataLen)
np.random.shuffle(CSIi1Orig)  # 不打乱则无意义，容易出错的位置插入容易出错的点还是容易出错

# entropyTime = time.time()
# CSIa1Orig, CSIb1Orig, CSIe1Orig = splitEntropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, 5, dataLen)
# print("--- entropyTime %s seconds ---" % (time.time() - entropyTime))

CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')
CSIb1Orig = smooth(CSIb1Orig, window_len=15, window='flat')
CSIe1Orig = smooth(CSIe1Orig, window_len=15, window="flat")
CSIn1Orig = smooth(CSIn1Orig, window_len=15, window="flat")
CSIi1Orig = smooth(CSIi1Orig, window_len=15, window="flat")

CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
CSIe1OrigBack = CSIe1Orig.copy()
CSIn1OrigBack = CSIn1Orig.copy()
CSIi1OrigBack = CSIi1Orig.copy()

sortMethods = [[binaryTreeSortPerm, binaryTreeSort], quadTreeSort, [binaryLevelSortPerm, binaryLevelSort]]
sortMethod = sortMethods[2]

intvl = 5
keyLen = 128
interval_length = 4
addNoise = False
insertRatio = 1
ratio = 1
closeness = 0.2

for segLen in range(0, 5):
    segLen = int(math.pow(2, segLen))
    print("segLen", segLen)
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

    for staInd in range(0, len(CSIa1Orig), intvl * keyLen):
        processTime = time.time()

        endInd = staInd + keyLen * intvl
        # print("range:", staInd, endInd)
        if endInd >= len(CSIa1Orig):
            break
        times += 1

        CSIa1Orig = CSIa1OrigBack.copy()
        CSIb1Orig = CSIb1OrigBack.copy()
        CSIe1Orig = CSIe1OrigBack.copy()
        CSIn1Orig = CSIn1OrigBack.copy()
        CSIi1Orig = CSIi1OrigBack.copy()

        # tmpCSIa1 = np.fft.fft(CSIa1Orig[range(staInd, endInd, 1)])
        # tmpCSIb1 = np.fft.fft(CSIb1Orig[range(staInd, endInd, 1)])
        # tmpCSIe1 = np.fft.fft(CSIe1Orig[range(staInd, endInd, 1)])
        # tmpNoise = np.fft.fft(CSIn1Orig[range(staInd, endInd, 1)])

        tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
        tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
        tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
        tmpNoise = CSIn1Orig[range(staInd, endInd, 1)]
        tmpCSIi1 = CSIi1Orig[range(staInd, endInd, 1)]

        # print(np.mean(tmpCSIa1), np.std(tmpCSIa1, ddof=1))
        # print(np.mean(tmpCSIb1), np.std(tmpCSIb1, ddof=1))
        # print(np.mean(tmpCSIe1), np.std(tmpCSIe1, ddof=1))
        # print(np.mean(tmpNoise), np.std(tmpNoise, ddof=1))
        # print(np.mean(tmpCSIi1), np.std(tmpCSIi1, ddof=1))

        # 去除直流分量
        # tmpCSIa1 = tmpCSIa1 - (np.mean(tmpCSIa1) - np.mean(tmpCSIb1))  # Mean value consistency
        tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
        tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
        # tmpCSIi1 = tmpCSIi1 - np.mean(tmpCSIi1)
        # tmpNoise = tmpNoise - np.mean(tmpNoise)

        # linspace函数生成元素为50的等间隔数列，可以指定第三个参数为元素个数
        # signal.square返回周期性的方波波形
        tmpPulse = signal.square(
            2 * np.pi * 1 / intvl * np.linspace(0, np.pi * 0.5 * keyLen, keyLen * intvl))  ## Rectangular pulse

        # if staInd >= 40000:
        #     plt.plot(CSIa1Orig, 'r')
        #     # plt.plot(CSIa1Orig,'b')
        #     plt.show()
        #
        #     plt.plot(tmpCSIa1, 'r')
        #     # plt.plot(tmpCSIb1,'b')
        #     plt.show()
        if addNoise:
            # tmpCSIa1 = (np.float_power(np.abs(tmpCSIa1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIa1))
            # tmpCSIb1 = (np.float_power(np.abs(tmpCSIb1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIb1))
            # tmpCSIe1 = (np.float_power(np.abs(tmpCSIe1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIe1))
            # tmpCSIa1 = tmpPulse * np.float_power(np.abs(tmpCSIa1), tmpNoise)
            # tmpCSIb1 = tmpPulse * np.float_power(np.abs(tmpCSIb1), tmpNoise)
            # tmpCSIe1 = tmpPulse * np.float_power(np.abs(tmpCSIe1), tmpNoise)
            tmpCSIa1 = tmpCSIa1 * tmpNoise
            tmpCSIb1 = tmpCSIb1 * tmpNoise
            tmpCSIe1 = tmpCSIe1 * tmpNoise
            tmpNoise = tmpNoise * tmpNoise
            # tmpCSIa1 = tmpCSIa1 + tmpNoise
            # tmpCSIb1 = tmpCSIb1 + tmpNoise
            # tmpCSIe1 = tmpCSIe1 + tmpNoise
            # tmpNoise = tmpNoise + tmpNoise
        else:
            # tmpCSIa1 = tmpPulse * tmpCSIa1
            # tmpCSIb1 = tmpPulse * tmpCSIb1
            # tmpCSIe1 = tmpPulse * tmpCSIe1
            tmpCSIa1 = tmpCSIa1
            tmpCSIb1 = tmpCSIb1
            tmpCSIe1 = tmpCSIe1
            tmpNoise = tmpNoise
        # if staInd >= 40000:
        #     plt.plot(tmpCSIa1, 'g')
        #     # plt.plot(tmpCSIb1,'y')
        #     plt.show()

        CSIa1Orig[range(staInd, endInd, 1)] = tmpCSIa1
        CSIb1Orig[range(staInd, endInd, 1)] = tmpCSIb1
        CSIe1Orig[range(staInd, endInd, 1)] = tmpCSIe1
        CSIn1Orig[range(staInd, endInd, 1)] = tmpNoise
        CSIi1Orig[range(staInd, endInd, 1)] = tmpCSIi1

        permLen = len(range(staInd, endInd, intvl))
        origInd = np.array([xx for xx in range(staInd, endInd, intvl)])

        sortCSIa1 = np.zeros(permLen)
        sortCSIb1 = np.zeros(permLen)
        sortCSIe1 = np.zeros(permLen)
        sortNoise = np.zeros(permLen)
        sortCSIi1 = np.zeros(permLen)

        for ii in range(permLen):
            aIndVec = np.array([aa for aa in range(origInd[ii], origInd[ii] + intvl, 1)])  ## for non-permuted CSIa1

            for jj in range(permLen, permLen * 2):
                bIndVec = np.array([bb for bb in range(origInd[jj - permLen], origInd[jj - permLen] + intvl, 1)])

                CSIa1Tmp = CSIa1Orig[aIndVec]
                CSIb1Tmp = CSIb1Orig[bIndVec]
                CSIe1Tmp = CSIe1Orig[bIndVec]
                CSIn1Tmp = CSIn1Orig[aIndVec]
                CSIi1Tmp = CSIi1Orig[aIndVec]

                sortCSIa1[ii] = np.mean(CSIa1Tmp)  ## Metric 1: Mean
                sortCSIb1[jj - permLen] = np.mean(CSIb1Tmp)  # 只赋值一次
                sortCSIe1[jj - permLen] = np.mean(CSIe1Tmp)
                sortNoise[ii - permLen] = np.mean(CSIn1Tmp)
                sortCSIi1[ii - permLen] = np.mean(CSIi1Tmp)

        # sortCSIa1是原始算法中排序前的数据
        # sortCSIa1 = np.log10(np.abs(sortCSIa1))
        # sortCSIb1 = np.log10(np.abs(sortCSIb1))
        # sortCSIe1 = np.log10(np.abs(sortCSIe1))
        # sortNoise = np.log10(np.abs(sortNoise))

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

        # 归一化
        # sortCSIa1 = []
        # sortCSIb1 = []
        # sortCSIe1 = []
        # sortNoise = []
        # sortCSIi1 = []
        #
        # for i in range(len(sortCSIa1Reshape)):
        #     # sklearn的归一化是按列转换，因此需要先转为列向量
        #     sortCSIa1.append(preprocessing.MinMaxScaler().fit_transform(
        #         np.array(sortCSIa1Reshape[i]).reshape(-1, 1)).reshape(1, -1).tolist()[0])
        #     sortCSIb1.append(preprocessing.MinMaxScaler().fit_transform(
        #         np.array(sortCSIb1Reshape[i]).reshape(-1, 1)).reshape(1, -1).tolist()[0])
        #     sortCSIe1.append(preprocessing.MinMaxScaler().fit_transform(
        #         np.array(sortCSIe1Reshape[i]).reshape(-1, 1)).reshape(1, -1).tolist()[0])
        #     sortNoise.append(preprocessing.MinMaxScaler().fit_transform(
        #         np.array(sortNoiseReshape[i]).reshape(-1, 1)).reshape(1, -1).tolist()[0])

        # sortCSIa1 = np.array(genArray(sortCSIa1Reshape))
        # sortCSIb1 = np.array(genArray(sortCSIb1Reshape))
        # sortCSIe1 = np.array(genArray(sortCSIe1Reshape))
        # sortNoise = np.array(genArray(sortNoiseReshape))

        sortCSIa1 = np.array(genSample(sortCSIa1Reshape, ratio))
        sortCSIb1 = np.array(genSample(sortCSIb1Reshape, ratio))
        sortCSIe1 = np.array(genSample(sortCSIe1Reshape, ratio))
        sortNoise = np.array(genSample(sortNoiseReshape, ratio))
        sortCSIi1 = np.array(genSample(sortCSIi1Reshape, ratio))

        # plt.plot(sortCSIa1, "b")
        # plt.show()
        # plt.plot(sortCSIb1, "y")
        # plt.show()
        # plt.plot(sortCSIe1, "r")
        # plt.show()

        # 这样插入噪音点和sortCSIa1数据过于接近，效果不好，不如在一个大范围内产生的噪音点好
        # sortCSIi1 = np.random.normal(loc=np.mean(sortCSIa1), scale=np.std(sortCSIa1, ddof=1), size=len(sortCSIa1))

        np.random.shuffle(sortCSIi1)
        insertIndex = np.random.permutation(len(sortCSIa1))
        sortCSIa1 = insertNoise(sortCSIa1, sortCSIi1, insertIndex, insertRatio)
        sortCSIb1 = insertNoise(sortCSIb1, sortCSIi1, insertIndex, insertRatio)
        sortCSIe1 = insertNoise(sortCSIe1, sortCSIi1, insertIndex, insertRatio)
        sortNoise = insertNoise(sortNoise, sortCSIi1, insertIndex, insertRatio)

        # plt.plot(sortCSIa1, "b")
        # plt.plot(sortCSIb1, "y")
        # plt.show()
        # plt.plot(sortCSIb1, "y")
        # plt.plot(sortCSIe1, "r")
        # plt.plot(sortNoise, "r")
        # plt.show()

        # 最后各自的密钥
        a_list = []
        b_list = []
        e_list = []
        n_list = []

        sortCSIa1 = np.array(sortCSIa1)
        sortCSIb1 = np.array(sortCSIb1)
        sortCSIe1 = np.array(sortCSIe1)
        sortNoise = np.array(sortNoise)

        a_list_number, perm = sortMethod[0](list(sortCSIa1), interval_length, closeness)
        b_list_number = sortMethod[1](list(sortCSIb1), interval_length, perm)
        e_list_number = sortMethod[1](list(sortCSIe1), interval_length, perm)
        n_list_number = sortMethod[1](list(sortNoise), interval_length, perm)

        # a_list_number = sortMethod(list(sortCSIa1), interval_length)
        # b_list_number = sortMethod(list(sortCSIb1), interval_length)
        # e_list_number = sortMethod(list(sortCSIe1), interval_length)
        # n_list_number = sortMethod(list(sortNoise), interval_length)

        # scale = 1
        # for i in range(len(a_list_number)):
        #     a_list.append(int(a_list_number[i] / scale))
        # for i in range(len(b_list_number)):
        #     b_list.append(int(b_list_number[i] / scale))
        # for i in range(len(e_list_number)):
        #     e_list.append(int(e_list_number[i] / scale))
        # for i in range(len(n_list_number)):
        #     n_list.append(int(n_list_number[i] / scale))

        # 转为十进制
        # for i in range(len(a_list_number)):
        #     a_list.append(int(a_list_number[i], 2))
        # for i in range(len(b_list_number)):
        #     b_list.append(int(b_list_number[i], 2))
        # for i in range(len(e_list_number)):
        #     e_list.append(int(e_list_number[i], 2))
        # for i in range(len(n_list_number)):
        #     n_list.append(int(n_list_number[i], 2))

        coding_bits = len(a_list_number[0])  # 用于二进制codes左补齐
        # 转为二进制
        for i in range(len(a_list_number)):
            a_list += bin(int(a_list_number[i]))[2:]
        for i in range(len(b_list_number)):
            b_list += bin(int(b_list_number[i]))[2:]
        for i in range(len(e_list_number)):
            e_list += bin(int(e_list_number[i]))[2:]
        for i in range(len(n_list_number)):
            n_list += bin(int(n_list_number[i]))[2:]

        # 对齐密钥，随机补全
        for i in range(len(a_list) - len(e_list)):
            e_list += str(np.random.randint(0, 2))
        for i in range(len(a_list) - len(n_list)):
            n_list += str(np.random.randint(0, 2))

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
        sum3 = 0
        sum4 = 0
        for i in range(0, sum1):
            sum2 += (a_list[i] == b_list[i])
        for i in range(min(len(a_list), len(e_list))):
            sum3 += (a_list[i] == e_list[i])
        for i in range(min(len(a_list), len(n_list))):
            sum4 += (a_list[i] == n_list[i])

        # if sum2 == sum1:
        #     print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
        # else:
        #     print("\033[0;31;40ma-b", sum2, sum2 / sum1, "\033[0m")
        # print("a-e", sum3, sum3 / sum1)
        # print("a-n", sum4, sum4 / sum1)
        # print("----------------------")
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

        # for i in range(len(a_list)):
        #     codings += bin(a_list[i]) + "\n"

    # with open('../edit_distance/evaluations/key.txt', 'a', ) as f:
    #     f.write(codings)

    print("a-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10))
    print("a-e all", randomSum, "/", originSum, "=", round(randomSum / originSum, 10))
    print("a-n all", noiseSum, "/", originSum, "=", round(noiseSum / originSum, 10))
    print("a-b whole match", correctWholeSum, "/", originWholeSum, "=", round(correctWholeSum / originWholeSum, 10))
    print("a-e whole match", randomWholeSum, "/", originWholeSum, "=", round(randomWholeSum / originWholeSum, 10))
    print("a-n whole match", noiseWholeSum, "/", originWholeSum, "=", round(noiseWholeSum / originWholeSum, 10))
    print("times", times)
