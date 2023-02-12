import math
import sys
import time
from collections import deque

import numpy as np
from scipy.io import loadmat

from algorithm import smooth, genSample


def round5(data):
    for i in range(len(data)):
        data[i] = round(data[i], 5)
    return data

def genDiff(data):
    diff = []
    for i in range(len(data) - 1):
        diff.append(abs(data[i + 1] - data[i]))
    return diff


def recoverData(start, diff, ref, threshold):
    rec = []
    rec.append(start)
    for i in range(len(diff)):
        # if abs(ref[i + 1] - ref[i]) < threshold:
        #     rec.append(rec[i])
        # elif ref[i + 1] - ref[i] > 0:
        if ref[i + 1] - ref[i] > 0:
            rec.append(rec[i] + diff[i])
        else:
            rec.append(rec[i] - diff[i])
    return rec


start_time = time.time()
fileName = "../data/data_static_indoor_1_r_m.mat"
rawData = loadmat(fileName)

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

dataLen = len(CSIa1Orig)

# combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig))
# np.random.shuffle(combineCSIx1Orig)
# CSIa1Orig, CSIb1Orig = zip(*combineCSIx1Orig)

CSIa1Orig = np.array(CSIa1Orig)
CSIb1Orig = np.array(CSIb1Orig)
CSIn1Orig = np.random.normal(loc=-100, scale=1000, size=dataLen)

CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)

CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')
CSIb1Orig = smooth(CSIb1Orig, window_len=15, window='flat')
CSIe1Orig = smooth(CSIe1Orig, window_len=15, window="flat")
CSIn1Orig = smooth(CSIn1Orig, window_len=15, window="flat")

segLen = 8
keyLen = 128 * segLen
ratio = 1

originSum = 0
correctSum = 0
randomSum = 0
noiseSum = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum = 0
noiseWholeSum = 0

times = 0

for staInd in range(0, len(CSIa1Orig), keyLen):
    endInd = staInd + keyLen
    print("range:", staInd, endInd)
    if endInd >= len(CSIa1Orig):
        break
    times += 1

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
    tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
    tmpNoise = CSIn1Orig[range(staInd, endInd, 1)]

    # 去除直流分量
    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
    tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
    tmpNoise = tmpNoise - np.mean(tmpNoise)

    sortCSIa1 = tmpCSIa1
    sortCSIb1 = tmpCSIb1
    sortCSIe1 = tmpCSIe1
    sortNoise = tmpNoise

    # 取原数据的一部分来reshape
    sortCSIa1Reshape = sortCSIa1[0:segLen * int(len(sortCSIa1) / segLen)]
    sortCSIb1Reshape = sortCSIb1[0:segLen * int(len(sortCSIb1) / segLen)]
    sortCSIe1Reshape = sortCSIe1[0:segLen * int(len(sortCSIe1) / segLen)]
    sortNoiseReshape = sortNoise[0:segLen * int(len(sortNoise) / segLen)]

    sortCSIa1Reshape = sortCSIa1Reshape.reshape(int(len(sortCSIa1Reshape) / segLen), segLen)
    sortCSIb1Reshape = sortCSIb1Reshape.reshape(int(len(sortCSIb1Reshape) / segLen), segLen)
    sortCSIe1Reshape = sortCSIe1Reshape.reshape(int(len(sortCSIe1Reshape) / segLen), segLen)
    sortNoiseReshape = sortNoiseReshape.reshape(int(len(sortNoiseReshape) / segLen), segLen)

    sortCSIa1 = genSample(sortCSIa1Reshape, ratio)
    sortCSIb1 = genSample(sortCSIb1Reshape, ratio)
    sortCSIe1 = genSample(sortCSIe1Reshape, ratio)
    sortNoise = genSample(sortNoiseReshape, ratio)

    # 最后各自的密钥
    a_list = []
    b_list = []
    e_list = []
    n_list = []

    threshold = 2
    # a-b all 88229 / 102758 = 0.8586095486
    # a-e all 53959 / 102758 = 0.5251075342
    # a-n all 54059 / 102758 = 0.5260806944
    # a-b whole match 307 / 741 = 0.4143049933
    # a-e whole match 0 / 741 = 0.0
    # a-n whole match 0 / 741 = 0.0

    print("b", round5(sortCSIb1))
    print("a", round5(sortCSIa1))
    print("rec a", round5(recoverData(sortCSIa1[0], genDiff(sortCSIa1), sortCSIa1, threshold)))
    print("rec b", round5(recoverData(sortCSIa1[0], genDiff(sortCSIa1), sortCSIb1, threshold)))
    print("diff a", round5(genDiff(sortCSIa1)))
    print("diff b", round5(genDiff(sortCSIb1)))

    # threshold = 2
    # a-b all 100317 / 104370 = 0.961167002
    # a-e all 54357 / 104370 = 0.5208105778
    # a-n all 53214 / 104370 = 0.5098591549
    # a-b whole match 648 / 741 = 0.8744939271
    # a-e whole match 10 / 741 = 0.0134952767
    # a-n whole match 0 / 741 = 0.0
    a_list_number = list(np.argsort(recoverData(sortCSIa1[0], genDiff(sortCSIa1), sortCSIa1, threshold)))
    b_list_number = list(np.argsort(recoverData(sortCSIa1[0], genDiff(sortCSIa1), sortCSIb1, threshold)))
    e_list_number = list(np.argsort(recoverData(sortCSIa1[0], genDiff(sortCSIa1), sortCSIe1, threshold)))
    n_list_number = list(np.argsort(recoverData(sortCSIa1[0], genDiff(sortCSIa1), sortNoise, threshold)))

    # a-b all 168378 / 191178 = 0.8807394156
    # a-e all 149452 / 191178 = 0.7817426691
    # a-n all 149406 / 191178 = 0.7815020557
    # a-b whole match 265 / 741 = 0.3576248313
    # a-e whole match 258 / 741 = 0.3481781377
    # a-n whole match 260 / 741 = 0.350877193

    # threshold = 2
    # a-b all 185340 / 191178 = 0.9694630135
    # a-e all 106266 / 191178 = 0.5558484763
    # a-n all 96334 / 191178 = 0.5038968919
    # a-b whole match 541 / 741 = 0.7300944669
    # a-e whole match 2 / 741 = 0.0026990553
    # a-n whole match 0 / 741 = 0.0
    # a_list_number = levelSort("a", sortCSIa1, sortCSIa1, interval_length, threshold)
    # b_list_number = levelSort("b", sortCSIb1, sortCSIa1, interval_length, threshold)
    # e_list_number = levelSort("e", sortCSIe1, sortCSIa1, interval_length, threshold)
    # n_list_number = levelSort("n", sortNoise, sortCSIa1, interval_length, threshold)

    # 转成层序密钥
    a_level_number = []
    b_level_number = []
    e_level_number = []
    n_level_number = []
    i = 0
    step = 1
    while i < len(a_list_number):
        a_level_number.append(list(a_list_number[i: i + 2 ** step]))
        i = i + 2 ** step
        step += 1
    i = 0
    step = 1
    while i < len(b_list_number):
        b_level_number.append(list(b_list_number[i: i + 2 ** step]))
        i = i + 2 ** step
        step += 1
    i = 0
    step = 1
    while i < len(e_list_number):
        e_level_number.append(list(e_list_number[i: i + 2 ** step]))
        i = i + 2 ** step
        step += 1
    i = 0
    step = 1
    while i < len(n_list_number):
        n_level_number.append(list(n_list_number[i: i + 2 ** step]))
        i = i + 2 ** step
        step += 1

    # 转成二进制
    for i in range(len(a_level_number)):
        for j in range(len(a_level_number[i])):
            number = bin(int(a_level_number[i][j]))[2:].zfill(i + 1)
            a_list += number
    for i in range(len(b_level_number)):
        for j in range(len(b_level_number[i])):
            number = bin(int(b_level_number[i][j]))[2:].zfill(i + 1)
            b_list += number
    for i in range(len(e_level_number)):
        for j in range(len(e_level_number[i])):
            number = bin(int(e_level_number[i][j]))[2:].zfill(i + 1)
            e_list += number
    for i in range(len(n_level_number)):
        for j in range(len(n_level_number[i])):
            number = bin(int(n_level_number[i][j]))[2:].zfill(i + 1)
            n_list += number

    # 对齐密钥，随机补全
    for i in range(len(a_list) - len(e_list)):
        e_list += str(np.random.randint(0, 2))
    for i in range(len(a_list) - len(n_list)):
        n_list += str(np.random.randint(0, 2))

    # print("keys of a:", len(a_list), a_list)
    print("keys of a:", len(a_list_number), a_list_number)
    # print("keys of b:", len(b_list), b_list)
    print("keys of b:", len(b_list_number), b_list_number)
    # print("keys of e:", len(e_list), e_list)
    print("keys of e:", len(e_list_number), e_list_number)
    # print("keys of n:", len(n_list), n_list)
    print("keys of n:", len(n_list_number), n_list_number)

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

    if sum1 == 0:
        continue
    if sum2 == sum1:
        print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
    else:
        print("\033[0;31;40ma-b", "bad", sum2, sum2 / sum1, "\033[0m")
        # a_list_number_test = sortMethod[3](list(sortCSIa1), interval_length, list(sortCSIi1), perm, blots, metric)
        # b_list_number_test = sortMethod[3](list(sortCSIb1), interval_length, list(sortCSIi2), perm, blots, metric)
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


print("a-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10))
print("a-e all", randomSum, "/", originSum, "=", round(randomSum / originSum, 10))
print("a-n all", noiseSum, "/", originSum, "=", round(noiseSum / originSum, 10))
print("a-b whole match", correctWholeSum, "/", originWholeSum, "=", round(correctWholeSum / originWholeSum, 10))
print("a-e whole match", randomWholeSum, "/", originWholeSum, "=", round(randomWholeSum / originWholeSum, 10))
print("a-n whole match", noiseWholeSum, "/", originWholeSum, "=", round(noiseWholeSum / originWholeSum, 10))
print("times", times)

print("测试结束，耗时" + str(time.time() - start_time))
# messagebox.showinfo("提示", "测试结束，耗时" + str(time.time() - start_time))
