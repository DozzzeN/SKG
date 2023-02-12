import hashlib
import sys
import time

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from scipy.fft import dct
from scipy.io import loadmat
from scipy.signal import medfilt
from sklearn import preprocessing

from algorithm import smooth, genSample, sortSegPermOfB, sortSegPermOfA, difference, integral_sq_derivative, \
    diff_sq_integral_rough, overlap_moving_sum, medianFilter, splitEntropyPerm, diffFilter, euclidean_metric, \
    adaSortSegPermOfA, adaSortSegPermOfB
from zca import ZCA

def normal2uniform(data):
    data_reshape = np.array(data[0: 2 * int(len(data) / 2)])
    data_reshape = data_reshape.reshape(int(len(data_reshape) / 2), 2)
    x_list = []
    for i in range(len(data_reshape)):
        r = np.sum(np.square(data_reshape[i]))
        x_list.append(np.exp(-0.5 * r))

    # plt.figure()
    # plt.hist(x_list)
    # plt.show()

    return x_list

fileName = "../data/data_static_indoor_1_r.mat"
rawData = loadmat(fileName)

csv = open("./sorting.csv", "a+")

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

dataLen = len(CSIa1Orig)  # 94873

CSIn1Orig = np.random.normal(loc=-100, scale=1000, size=dataLen)

CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)

# 固定随机置换的种子 5
for s in range(100):
    print("seed", s)
    np.random.seed(s)
    start_time = time.time()

    combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig, CSIe1Orig, CSIn1Orig))
    np.random.shuffle(combineCSIx1Orig)
    CSIa1Orig, CSIb1Orig, CSIe1Orig, CSIn1Orig = zip(*combineCSIx1Orig)

    CSIa1Orig = np.array(CSIa1Orig)
    CSIb1Orig = np.array(CSIb1Orig)
    CSIe1Orig = np.array(CSIe1Orig)
    CSIn1Orig = np.array(CSIn1Orig)

    CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')
    CSIb1Orig = smooth(CSIb1Orig, window_len=15, window='flat')
    CSIe1Orig = smooth(CSIe1Orig, window_len=15, window="flat")
    CSIn1Orig = smooth(CSIn1Orig, window_len=15, window="flat")

    segLen = 8
    keyLen = 128 * segLen

    ratio = 1
    rawOp = "uniform"

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

        scale = 10
        offset = -200
        if rawOp == "fft":
            sortCSIa1 = np.abs(np.fft.fft(tmpCSIa1))
            sortCSIb1 = np.abs(np.fft.fft(tmpCSIb1))
            sortCSIe1 = np.abs(np.fft.fft(tmpCSIe1))
            sortNoise = np.abs(np.fft.fft(tmpNoise))

            sortCSIa1 = sortCSIa1 - np.mean(sortCSIa1)
            sortCSIb1 = sortCSIb1 - np.mean(sortCSIb1)
            sortCSIe1 = sortCSIe1 - np.mean(sortCSIe1)
            sortNoise = sortNoise - np.mean(sortNoise)
        elif rawOp == "dct":
            sortCSIa1 = np.abs(dct(tmpCSIa1))
            sortCSIb1 = np.abs(dct(tmpCSIb1))
            sortCSIe1 = np.abs(dct(tmpCSIe1))
            sortNoise = np.abs(dct(tmpNoise))

            sortCSIa1 = sortCSIa1 - np.mean(sortCSIa1)
            sortCSIb1 = sortCSIb1 - np.mean(sortCSIb1)
            sortCSIe1 = sortCSIe1 - np.mean(sortCSIe1)
            sortNoise = sortNoise - np.mean(sortNoise)
        elif rawOp == "zca":
            step = 1
            tmpCSIa12D = tmpCSIa1.reshape(int(len(tmpCSIa1) / step), step)
            trf = ZCA().fit(tmpCSIa12D)
            sortCSIa1 = trf.transform(np.abs(tmpCSIa12D)).reshape(1, -1)[0]
            tmpCSIb12D = tmpCSIb1.reshape(int(len(tmpCSIb1) / step), step)
            sortCSIb1 = trf.transform(np.abs(tmpCSIb12D)).reshape(1, -1)[0]
            tmpCSIe12D = tmpCSIe1.reshape(int(len(tmpCSIe1) / step), step)
            sortCSIe1 = trf.transform(np.abs(tmpCSIe12D)).reshape(1, -1)[0]
            tmpNoise2D = tmpNoise.reshape(int(len(tmpNoise) / step), step)
            sortNoise = trf.transform(np.abs(tmpNoise2D)).reshape(1, -1)[0]
        elif rawOp == "uniform":
            sortCSIa1 = normal2uniform(tmpCSIa1)
            sortCSIb1 = normal2uniform(tmpCSIb1)
            sortCSIe1 = normal2uniform(tmpCSIe1)
            sortNoise = normal2uniform(tmpNoise)
        elif rawOp == "coxbox":
            sortCSIa1 = scipy.stats.boxcox(np.abs(tmpCSIa1))[0]
            sortCSIb1 = scipy.stats.boxcox(np.abs(tmpCSIb1))[0]
            sortCSIe1 = scipy.stats.boxcox(np.abs(tmpCSIe1))[0]
            sortNoise = scipy.stats.boxcox(np.abs(tmpNoise))[0]
        elif rawOp == "coxbox-uniform":
            sortCSIa1 = scipy.stats.boxcox(np.abs(tmpCSIa1))[0]
            sortCSIb1 = scipy.stats.boxcox(np.abs(tmpCSIb1))[0]
            sortCSIe1 = scipy.stats.boxcox(np.abs(tmpCSIe1))[0]
            sortNoise = scipy.stats.boxcox(np.abs(tmpNoise))[0]
            sortCSIa1 = normal2uniform(sortCSIa1)
            sortCSIb1 = normal2uniform(sortCSIb1)
            sortCSIe1 = normal2uniform(sortCSIe1)
            sortNoise = normal2uniform(sortNoise)
        else:
            sortCSIa1 = tmpCSIa1
            sortCSIb1 = tmpCSIb1
            sortCSIe1 = tmpCSIe1
            sortNoise = tmpNoise

        def reverse(number):
            s = str(number)
            return s[::-1]


        # 尝试非线性变换
        transform = ""
        if transform == "linear":
            scale = 10
            offset = -200
            sortCSIa1 = sortCSIa1 * scale + offset
            sortCSIb1 = sortCSIb1 * scale + offset
            sortCSIe1 = sortCSIe1 * scale + offset
            sortNoise = sortNoise * scale + offset
        elif transform == "square":
            sortCSIa1 = sortCSIa1 * sortCSIa1
            sortCSIb1 = sortCSIb1 * sortCSIb1
            sortCSIe1 = sortCSIe1 * sortCSIe1
            sortNoise = sortNoise * sortNoise
        elif transform == "cosine":
            sortCSIa1 = np.cos(sortCSIa1)
            sortCSIb1 = np.cos(sortCSIb1)
            sortCSIe1 = np.cos(sortCSIe1)
            sortNoise = np.cos(sortNoise)
        elif transform == "exponent":
            sortCSIa1 = np.power(1.1, np.abs(sortCSIa1))
            sortCSIb1 = np.power(1.1, np.abs(sortCSIb1))
            sortCSIe1 = np.power(1.1, np.abs(sortCSIe1))
            sortNoise = np.power(1.1, np.abs(sortNoise))
        elif transform == "base":
            sortCSIa1 = np.power(np.abs(sortCSIa1), 1.5)
            sortCSIb1 = np.power(np.abs(sortCSIb1), 1.5)
            sortCSIe1 = np.power(np.abs(sortCSIe1), 1.5)
            sortNoise = np.power(np.abs(sortNoise), 1.5)
        elif transform == "logarithm":
            sortCSIa1 = np.log2(np.abs(sortCSIa1)) * 10
            sortCSIb1 = np.log2(np.abs(sortCSIb1)) * 10
            sortCSIe1 = np.log2(np.abs(sortCSIe1)) * 10
            sortNoise = np.log2(np.abs(sortNoise)) * 10
        elif transform == "box-cox":
            pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
            sortCSIa12D = sortCSIa1.reshape(2, int(len(sortCSIa1) / 2))
            sortCSIa1 = pt.fit_transform(np.abs(sortCSIa12D)).reshape(1, -1)[0]
            sortCSIb12D = sortCSIb1.reshape(2, int(len(sortCSIb1) / 2))
            sortCSIb1 = pt.fit_transform(np.abs(sortCSIb12D)).reshape(1, -1)[0]
            sortCSIe12D = sortCSIe1.reshape(2, int(len(sortCSIe1) / 2))
            sortCSIe1 = pt.fit_transform(np.abs(sortCSIe12D)).reshape(1, -1)[0]
            sortNoise2D = sortNoise.reshape(2, int(len(sortNoise) / 2))
            sortNoise = pt.fit_transform(np.abs(sortNoise2D)).reshape(1, -1)[0]
            sortCSIa1 = sortCSIa1 * 10
            sortCSIb1 = sortCSIb1 * 10
            sortCSIe1 = sortCSIe1 * 10
            sortNoise = sortNoise * 10
        elif transform == "reciprocal":
            sortCSIa1 = 5000 / sortCSIa1 + sortCSIa1
            sortCSIb1 = 5000 / sortCSIb1 + sortCSIb1
            sortCSIe1 = 5000 / sortCSIe1 + sortCSIe1
            sortNoise = 5000 / sortNoise + sortNoise
        elif transform == "tangent":
            sortCSIa1 = np.tan(np.abs(sortCSIa1) / np.pi)
            sortCSIb1 = np.tan(np.abs(sortCSIb1) / np.pi)
            sortCSIe1 = np.tan(np.abs(sortCSIe1) / np.pi)
            sortNoise = np.tan(np.abs(sortNoise) / np.pi)
        elif transform == "remainder":
            sortCSIa1 = sortCSIa1 - np.mean(sortCSIa1) * (np.round(sortCSIa1 / np.mean(sortCSIa1)) - 1)
            sortCSIb1 = sortCSIb1 - np.mean(sortCSIb1) * (np.round(sortCSIb1 / np.mean(sortCSIb1)) - 1)
            sortCSIe1 = sortCSIe1 - np.mean(sortCSIe1) * (np.round(sortCSIe1 / np.mean(sortCSIe1)) - 1)
            sortNoise = sortNoise - np.mean(sortNoise) * (np.round(sortNoise / np.mean(sortNoise)) - 1)
        elif transform == "quotient":
            sortCSIa1 = sortCSIa1 / np.mean(sortCSIa1) * 100
            sortCSIb1 = sortCSIb1 / np.mean(sortCSIb1) * 100
            sortCSIe1 = sortCSIe1 / np.mean(sortCSIe1) * 100
            sortNoise = sortNoise / np.mean(sortNoise) * 100
        elif transform == "difference":
            sortCSIa1 = difference(sortCSIa1)
            sortCSIb1 = difference(sortCSIb1)
            sortCSIe1 = difference(sortCSIe1)
            sortNoise = difference(sortNoise)
        elif transform == "integral_square_derivative":
            tmpCSIa1 = np.array(integral_sq_derivative(tmpCSIa1))
            tmpCSIb1 = np.array(integral_sq_derivative(tmpCSIb1))
            tmpCSIe1 = np.array(integral_sq_derivative(tmpCSIe1))
            tmpNoise = np.array(integral_sq_derivative(tmpNoise))
        elif transform == "diff_sq_integral_rough":
            tmpCSIa1 = np.array(diff_sq_integral_rough(tmpCSIa1))
            tmpCSIb1 = np.array(diff_sq_integral_rough(tmpCSIb1))
            tmpCSIe1 = np.array(diff_sq_integral_rough(tmpCSIe1))
            tmpNoise = np.array(diff_sq_integral_rough(tmpNoise))
        elif transform == "reverse_number":
            tmpCSIa1 = np.array(tmpCSIa1) - min(tmpCSIa1)
            tmpCSIb1 = np.array(tmpCSIb1) - min(tmpCSIb1)
            tmpCSIe1 = np.array(tmpCSIe1) - min(tmpCSIe1)
            tmpNoise = np.array(tmpNoise) - min(tmpNoise)
            for i in range(len(tmpCSIa1)):
                tmpCSIa1[i] = reverse(int(tmpCSIa1[i] * 100))
                tmpCSIb1[i] = reverse(int(tmpCSIb1[i] * 100))
                tmpCSIe1[i] = reverse(int(tmpCSIe1[i] * 100))
                tmpNoise[i] = reverse(int(tmpNoise[i] * 100))
        elif transform == "seg_fft":
            for i in range(0, len(tmpCSIa1), segLen):
                tmpCSIa1[i:i + segLen] = tmpCSIa1[i:i + segLen] - np.mean(tmpCSIa1[i:i + segLen])
                tmpCSIb1[i:i + segLen] = tmpCSIb1[i:i + segLen] - np.mean(tmpCSIb1[i:i + segLen])
                tmpCSIe1[i:i + segLen] = tmpCSIe1[i:i + segLen] - np.mean(tmpCSIe1[i:i + segLen])
                tmpNoise[i:i + segLen] = tmpNoise[i:i + segLen] - np.mean(tmpNoise[i:i + segLen])
                tmpCSIa1[i:i + segLen] = np.abs(np.fft.fft(tmpCSIa1[i:i + segLen]))
                tmpCSIb1[i:i + segLen] = np.abs(np.fft.fft(tmpCSIb1[i:i + segLen]))
                tmpCSIe1[i:i + segLen] = np.abs(np.fft.fft(tmpCSIe1[i:i + segLen]))
                tmpNoise[i:i + segLen] = np.abs(np.fft.fft(tmpNoise[i:i + segLen]))

        # 取原数据的一部分来reshape
        sortCSIa1Reshape = sortCSIa1[0:segLen * int(len(sortCSIa1) / segLen)]
        sortCSIb1Reshape = sortCSIb1[0:segLen * int(len(sortCSIb1) / segLen)]
        sortCSIe1Reshape = sortCSIe1[0:segLen * int(len(sortCSIe1) / segLen)]
        sortNoiseReshape = sortNoise[0:segLen * int(len(sortNoise) / segLen)]

        sortCSIa1 = sortCSIa1Reshape
        sortCSIb1 = sortCSIb1Reshape
        sortCSIe1 = sortCSIe1Reshape
        sortNoise = sortNoiseReshape

        # 最后各自的密钥
        a_list = []
        b_list = []
        e_list = []
        n_list = []

        a_metric, publish, a_list_number = sortSegPermOfA(list(sortCSIa1), segLen)
        b_metric, b_list_number = sortSegPermOfB(publish, list(sortCSIb1), segLen)
        e_metric, e_list_number = sortSegPermOfB(publish, list(sortCSIe1), segLen)
        n_metric, n_list_number = sortSegPermOfB(publish, list(sortNoise), segLen)

        # 转成二进制，0填充成0000
        for i in range(len(a_list_number)):
            number = bin(a_list_number[i])[2:].zfill(int(np.log2(len(a_list_number))))
            a_list += number
        for i in range(len(b_list_number)):
            number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
            b_list += number
        for i in range(len(e_list_number)):
            number = bin(e_list_number[i])[2:].zfill(int(np.log2(len(e_list_number))))
            e_list += number
        for i in range(len(n_list_number)):
            number = bin(n_list_number[i])[2:].zfill(int(np.log2(len(n_list_number))))
            n_list += number

        # 对齐密钥，随机补全
        for i in range(len(a_list) - len(e_list)):
            e_list += str(np.random.randint(0, 2))
        for i in range(len(a_list) - len(n_list)):
            n_list += str(np.random.randint(0, 2))

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
    print("测试结束，耗时" + str(round(time.time() - start_time, 3)), "s")
    if correctWholeSum == originWholeSum:
        break

