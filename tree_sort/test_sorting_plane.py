import hashlib
import math
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
    adaSortSegPermOfA, adaSortSegPermOfB, refSortSegPermOfB, refSortSegPermOfA
from zca import ZCA


# https://blog.csdn.net/gfuugff/article/details/84020509
def projection(p1, p2, p3):
    v1 = [p3[0] - p1[0], p3[1] - p1[1]]
    v2 = [p2[0] - p1[0], p2[1] - p1[1]]
    v1v2 = v1[0] * v2[0] + v1[1] * v2[1]
    k = v1v2 / (math.pow(p2[0] - p1[0], 2) + math.pow(p2[1] - p1[1], 2))
    p0 = [p1[0] + k * (p2[0] - p1[0]), p1[1] + k * (p2[1] - p1[1])]
    return p0


def normal2uniform(data):
    # data1 = data[:int(len(data) / 2)]
    # data2 = data[int(len(data) / 2):]
    data_reshape = np.array(data[0: 2 * int(len(data) / 2)])
    data_reshape = data_reshape.reshape(int(len(data_reshape) / 2), 2)
    x_list = []
    for i in range(len(data_reshape)):
        r = np.sum(np.square(data_reshape[i]))
        # r = np.sum(data1[i] * data1[i] + data2[i] * data2[i])
        x_list.append(np.exp(-0.5 * r))
        # r = data2[i] / data1[i]
        # x_list.append(math.atan(r) / math.pi + 0.5)

    # plt.figure()
    # plt.hist(x_list)
    # plt.show()

    return x_list


def normal2uniform2(data, shuffle):
    data_back = data.copy()
    data = np.array(data_back)[shuffle]
    x_list = []
    for i in range(len(data)):
        r = data[i] * data[i] + data_back[i] * data_back[i]
        x_list.append(np.exp(-r / 2))
        # r = data[i] / data_back[i]
        # x_list.append(math.atan(r) / math.pi + 0.5)
    return x_list


def rayleigh2uniform(data):
    return 1 - np.exp(-0.5 * np.square(data))


# a = np.random.normal(loc=0, scale=0.05, size=1000)
# plt.figure()
# plt.hist(a)
# plt.show()
# a = normal2uniform(a, np.random.permutation(len(a)))
# plt.figure()
# plt.hist(a)
# plt.show()
# exit()

l1 = [1, 0]
l2 = [0, 1]

start_time = time.time()
fileName = "../data/data_static_indoor_1_r_m.mat"
rawData = loadmat(fileName)

csv = open("./sorting.csv", "a+")

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

dataLen = len(CSIa1Orig)  # 94873

CSIn1Orig = np.random.normal(loc=-100, scale=1000, size=dataLen)

CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)
# CSIe1Orig = loadmat("../data/data_static_indoor_1_e_r.mat")['A'][:, 0]

# CSIa1Orig = np.abs(np.fft.fft(CSIa1Orig))
# CSIb1Orig = np.abs(np.fft.fft(CSIb1Orig))
# CSIe1Orig = np.abs(np.fft.fft(CSIe1Orig))
# CSIn1Orig = np.abs(np.fft.fft(CSIn1Orig))

# plt.figure()
# plt.subplot(2, 3, 1)
# plt.plot(CSIa1Orig)
# plt.subplot(2, 3, 2)
# plt.plot(CSIb1Orig)
# plt.subplot(2, 3, 3)
# plt.plot(CSIe1Orig)
# plt.show()

# 固定随机置换的种子
np.random.seed(1)  # 8 1024 8; 4 128 4
combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig, CSIe1Orig, CSIn1Orig))
np.random.shuffle(combineCSIx1Orig)
CSIa1Orig, CSIb1Orig, CSIe1Orig, CSIn1Orig = zip(*combineCSIx1Orig)

CSIa1Orig = np.array(CSIa1Orig)
CSIb1Orig = np.array(CSIb1Orig)
CSIe1Orig = np.array(CSIe1Orig)
CSIn1Orig = np.array(CSIn1Orig)

# CSIa1Orig.sort()
# CSIb1Orig.sort()

# entropyTime = time.time()
# CSIa1Orig, CSIb1Orig, CSIe1Orig = splitEntropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, 5, dataLen)
# print("--- entropyTime %s seconds ---" % (time.time() - entropyTime))

CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')
CSIb1Orig = smooth(CSIb1Orig, window_len=15, window='flat')
CSIe1Orig = smooth(CSIe1Orig, window_len=15, window="flat")
CSIn1Orig = smooth(CSIn1Orig, window_len=15, window="flat")

segLen = 7
keyLen = 64 * segLen

ratio = 1
rawOp = ""
# rawOp = "uniform"
isShow = False

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

    # perm = np.argsort(tmpCSIa1)
    # tmpCSIa1 = np.array(tmpCSIa1)[perm]
    # tmpCSIb1 = np.array(tmpCSIb1)[perm]
    # tmpCSIe1 = np.array(tmpCSIe1)[perm]
    # tmpNoise = np.array(tmpNoise)[perm]
    # np.random.seed(1)  # 8 1024 8; 4 128 4
    # combineCSIx1Orig = list(zip(tmpCSIa1, tmpCSIb1, tmpCSIe1, tmpNoise))
    # np.random.shuffle(combineCSIx1Orig)
    # tmpCSIa1, tmpCSIb1, tmpCSIe1, tmpNoise = zip(*combineCSIx1Orig)

    # tmpCSIa1 = tmpCSIa1 - (np.mean(tmpCSIa1) - np.mean(tmpCSIb1))

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
        shuffle = np.random.permutation(len(tmpCSIa1))
        sortCSIa1 = normal2uniform2(tmpCSIa1, shuffle)
        sortCSIb1 = normal2uniform2(tmpCSIb1, shuffle)
        sortCSIe1 = normal2uniform2(tmpCSIe1, shuffle)
        sortNoise = normal2uniform2(tmpNoise, shuffle)
        # sortCSIa1 = normal2uniform(tmpCSIa1)
        # sortCSIb1 = normal2uniform(tmpCSIb1)
        # sortCSIe1 = normal2uniform(tmpCSIe1)
        # sortNoise = normal2uniform(tmpNoise)
    elif rawOp == "coxbox":
        sortCSIa1 = scipy.stats.boxcox(tmpCSIa1 - np.min(tmpCSIa1) + 0.1)[0]
        sortCSIb1 = scipy.stats.boxcox(tmpCSIb1 - np.min(tmpCSIb1) + 0.1)[0]
        sortCSIe1 = scipy.stats.boxcox(tmpCSIe1 - np.min(tmpCSIe1) + 0.1)[0]
        sortNoise = scipy.stats.boxcox(tmpNoise - np.min(tmpNoise) + 0.1)[0]
    elif rawOp == "coxbox-uniform":
        shuffle = np.random.permutation(len(tmpCSIa1))
        sortCSIa1 = scipy.stats.boxcox(tmpCSIa1 - np.min(tmpCSIa1) + 0.1)[0]
        sortCSIb1 = scipy.stats.boxcox(tmpCSIb1 - np.min(tmpCSIb1) + 0.1)[0]
        sortCSIe1 = scipy.stats.boxcox(tmpCSIe1 - np.min(tmpCSIe1) + 0.1)[0]
        sortNoise = scipy.stats.boxcox(tmpNoise - np.min(tmpNoise) + 0.1)[0]
        sortCSIa1 = normal2uniform2(sortCSIa1, shuffle)
        sortCSIb1 = normal2uniform2(sortCSIb1, shuffle)
        sortCSIe1 = normal2uniform2(sortCSIe1, shuffle)
        sortNoise = normal2uniform2(sortNoise, shuffle)
        # sortCSIa1 = normal2uniform(sortCSIa1)
        # sortCSIb1 = normal2uniform(sortCSIb1)
        # sortCSIe1 = normal2uniform(sortCSIe1)
        # sortNoise = normal2uniform(sortNoise)
    elif rawOp == "rayleigh-uniform":
        sortCSIa1 = rayleigh2uniform(tmpCSIa1)
        sortCSIb1 = rayleigh2uniform(tmpCSIb1)
        sortCSIe1 = rayleigh2uniform(tmpCSIe1)
        sortNoise = rayleigh2uniform(tmpNoise)
    else:
        if rawOp is not None and rawOp != "":
            raise Exception("error rawOp")
        sortCSIa1 = tmpCSIa1
        sortCSIb1 = tmpCSIb1
        sortCSIe1 = tmpCSIe1
        sortNoise = tmpNoise

    sortCSIa1 = smooth(np.array(sortCSIa1), window_len=segLen, window='flat')
    sortCSIb1 = smooth(np.array(sortCSIb1), window_len=segLen, window='flat')
    sortCSIe1 = smooth(np.array(sortCSIe1), window_len=segLen, window='flat')
    sortNoise = smooth(np.array(sortNoise), window_len=segLen, window='flat')

    # np.random.seed(1)
    # combineCSIx1Orig = list(zip(sortCSIa1, sortCSIb1, sortCSIe1, sortNoise))
    # np.random.shuffle(combineCSIx1Orig)
    # sortCSIa1, sortCSIb1, sortCSIe1, sortNoise = zip(*combineCSIx1Orig)
    # sortCSIa1 = np.array(sortCSIa1)
    # sortCSIb1 = np.array(sortCSIb1)
    # sortCSIe1 = np.array(sortCSIe1)
    # sortNoise = np.array(sortNoise)

    # plt.figure()
    # plt.plot(sortCSIa1, "r")
    # plt.plot(sortCSIb1, "k")
    # plt.show()

    def reverse(number):
        s = str(number)
        return s[::-1]


    isShow = False
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
    elif transform == "exp":
        sortCSIa1 = np.exp(-sortCSIa1 / 2)
        sortCSIb1 = np.exp(-sortCSIb1 / 2)
        sortCSIe1 = np.exp(-sortCSIe1 / 2)
        sortNoise = np.exp(-sortNoise / 2)
        # sortCSIa1 = np.exp(-(sortCSIa1 * sortCSIa1) / 2)
        # sortCSIb1 = np.exp(-(sortCSIb1 * sortCSIb1) / 2)
        # sortCSIe1 = np.exp(-(sortCSIe1 * sortCSIe1) / 2)
        # sortNoise = np.exp(-(sortNoise * sortNoise) / 2)
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
    elif transform == "arctangent":
        sortCSIa1 = np.arctan(sortCSIa1)
        sortCSIb1 = np.arctan(sortCSIb1)
        sortCSIe1 = np.arctan(sortCSIe1)
        sortNoise = np.arctan(sortNoise)
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
        sortCSIa1 = np.array(integral_sq_derivative(sortCSIa1))
        sortCSIb1 = np.array(integral_sq_derivative(sortCSIb1))
        sortCSIe1 = np.array(integral_sq_derivative(sortCSIe1))
        sortNoise = np.array(integral_sq_derivative(sortNoise))
    elif transform == "diff_sq_integral_rough":
        sortCSIa1 = np.array(diff_sq_integral_rough(sortCSIa1))
        sortCSIb1 = np.array(diff_sq_integral_rough(sortCSIb1))
        sortCSIe1 = np.array(diff_sq_integral_rough(sortCSIe1))
        sortNoise = np.array(diff_sq_integral_rough(sortNoise))
    elif transform == "reverse_number":
        sortCSIa1 = np.array(sortCSIa1) - min(sortCSIa1)
        sortCSIb1 = np.array(sortCSIb1) - min(sortCSIb1)
        sortCSIe1 = np.array(sortCSIe1) - min(sortCSIe1)
        sortNoise = np.array(sortNoise) - min(sortNoise)
        for i in range(len(sortCSIa1)):
            sortCSIa1[i] = reverse(int(sortCSIa1[i] * 100))
            sortCSIb1[i] = reverse(int(sortCSIb1[i] * 100))
            sortCSIe1[i] = reverse(int(sortCSIe1[i] * 100))
            sortNoise[i] = reverse(int(sortNoise[i] * 100))
    elif transform == "seg_fft":
        for i in range(0, len(tmpCSIa1), segLen):
            sortCSIa1[i:i + segLen] = sortCSIa1[i:i + segLen] - np.mean(sortCSIa1[i:i + segLen])
            sortCSIb1[i:i + segLen] = sortCSIb1[i:i + segLen] - np.mean(sortCSIb1[i:i + segLen])
            sortCSIe1[i:i + segLen] = sortCSIe1[i:i + segLen] - np.mean(sortCSIe1[i:i + segLen])
            sortNoise[i:i + segLen] = sortNoise[i:i + segLen] - np.mean(sortNoise[i:i + segLen])
            sortCSIa1[i:i + segLen] = np.abs(np.fft.fft(sortCSIa1[i:i + segLen]))
            sortCSIb1[i:i + segLen] = np.abs(np.fft.fft(sortCSIb1[i:i + segLen]))
            sortCSIe1[i:i + segLen] = np.abs(np.fft.fft(sortCSIe1[i:i + segLen]))
            sortNoise[i:i + segLen] = np.abs(np.fft.fft(sortNoise[i:i + segLen]))

    sortCSIa1 = preprocessing.MinMaxScaler().fit_transform(np.array(sortCSIa1).reshape(-1, 1)).reshape(1, -1).tolist()[
        0]
    sortCSIb1 = preprocessing.MinMaxScaler().fit_transform(np.array(sortCSIb1).reshape(-1, 1)).reshape(1, -1).tolist()[
        0]
    sortCSIe1 = preprocessing.MinMaxScaler().fit_transform(np.array(sortCSIe1).reshape(-1, 1)).reshape(1, -1).tolist()[
        0]
    sortNoise = preprocessing.MinMaxScaler().fit_transform(np.array(sortNoise).reshape(-1, 1)).reshape(1, -1).tolist()[
        0]

    # disperse
    # sortCSIb1 = np.array(level_disperse(sortCSIb1, perm_threshold, 4))
    # sortCSIe1 = np.array(level_disperse(sortCSIe1, perm_threshold, 4))
    # sortCSIa1 = disperse(sortCSIa1, perm_threshold)
    # sortCSIb1 = disperse(sortCSIb1, perm_threshold)

    # combineSortCSIiX = list(zip(sortCSIb1, sortCSIe1))
    # np.random.shuffle(combineSortCSIiX)
    # sortCSIb1, sortCSIe1 = zip(*combineSortCSIiX)
    # sortCSIb1 = np.array(sortCSIb1)
    # sortCSIe1 = np.array(sortCSIe1)

    # 取原数据的一部分来reshape
    sortCSIa1Reshape = sortCSIa1[0:segLen * int(len(sortCSIa1) / segLen)]
    sortCSIb1Reshape = sortCSIb1[0:segLen * int(len(sortCSIb1) / segLen)]
    sortCSIe1Reshape = sortCSIe1[0:segLen * int(len(sortCSIe1) / segLen)]
    sortNoiseReshape = sortNoise[0:segLen * int(len(sortNoise) / segLen)]

    # 分段求和
    # sortCSIa1Reshape = sortCSIa1Reshape.reshape(int(len(sortCSIa1Reshape) / segLen), segLen)
    # sortCSIb1Reshape = sortCSIb1Reshape.reshape(int(len(sortCSIb1Reshape) / segLen), segLen)
    # sortCSIe1Reshape = sortCSIe1Reshape.reshape(int(len(sortCSIe1Reshape) / segLen), segLen)
    # sortNoiseReshape = sortNoiseReshape.reshape(int(len(sortNoiseReshape) / segLen), segLen)
    # sortCSIa1 = np.array(genSample(sortCSIa1Reshape, ratio))
    # sortCSIb1 = np.array(genSample(sortCSIb1Reshape, ratio))
    # sortCSIe1 = np.array(genSample(sortCSIe1Reshape, ratio))
    # sortNoise = np.array(genSample(sortNoiseReshape, ratio))

    sortCSIa1 = sortCSIa1Reshape
    sortCSIb1 = sortCSIb1Reshape
    sortCSIe1 = sortCSIe1Reshape
    sortNoise = sortNoiseReshape

    # 差分产生segment
    # sortCSIa1 = []
    # sortCSIb1 = []
    # sortCSIe1 = []
    # sortNoise = []
    # for i in range(len(sortCSIa1Reshape)):
    #     # plt.figure()
    #     # plt.plot(sortCSIa1Reshape[i:i + segLen], "r")
    #     # plt.plot(sortCSIb1Reshape[i:i + segLen], "k")
    #     # plt.show()
    #     sortCSIa1.append(diffFilter(sortCSIa1Reshape[i:i + segLen]))
    #     sortCSIb1.append(diffFilter(sortCSIb1Reshape[i:i + segLen]))
    #     sortCSIe1.append(diffFilter(sortCSIe1Reshape[i:i + segLen]))
    #     sortNoise.append(diffFilter(sortNoiseReshape[i:i + segLen]))

    # 中值滤波产生segment
    # sortCSIa1 = []
    # sortCSIb1 = []
    # sortCSIe1 = []
    # sortNoise = []
    # for i in range(len(sortCSIa1Reshape)):
    #     sortCSIa1.append(medianFilter(sortCSIa1Reshape[i:i + segLen]))
    #     sortCSIb1.append(medianFilter(sortCSIb1Reshape[i:i + segLen]))
    #     sortCSIe1.append(medianFilter(sortCSIe1Reshape[i:i + segLen]))
    #     sortNoise.append(medianFilter(sortNoiseReshape[i:i + segLen]))

    # 求和产生segment
    # sortCSIa1 = np.array(genSample(sortCSIa1Reshape, ratio))
    # sortCSIb1 = np.array(genSample(sortCSIb1Reshape, ratio))
    # sortCSIe1 = np.array(genSample(sortCSIe1Reshape, ratio))
    # sortNoise = np.array(genSample(sortNoiseReshape, ratio))

    # np.random.seed(1)
    # combineSortCSIx1 = list(zip(sortCSIa1, sortCSIb1))
    # np.random.shuffle(combineSortCSIx1)
    # sortCSIa1, sortCSIb1 = zip(*combineSortCSIx1)
    # sortCSIa1 = np.array(sortCSIa1)
    # sortCSIb1 = np.array(sortCSIb1)

    # 最后各自的密钥
    a_list = []
    b_list = []
    e_list = []
    n_list = []

    if isShow:
        plt.figure()
        plt.hist(tmpCSIa1)
        plt.show()

        plt.figure()
        plt.hist(sortCSIa1)
        # plt.hist(sortCSIb1)
        plt.show()
        a_show = sortCSIa1.copy()
        b_show = sortCSIb1.copy()
        a_show.sort()
        b_show.sort()
        # a_show = a_show.reshape(int(len(a_show) / segLen), segLen)
        # b_show = b_show.reshape(int(len(b_show) / segLen), segLen)
        # a_show_com = []
        # b_show_com = []
        # for i in range(len(a_show)):
        #     # a_show_com.append(medianFilter(a_show[i]))
        #     # b_show_com.append(medianFilter(b_show[i]))
        #     a_show_com.append(sum(a_show[i]) / len(a_show[i]))
        #     b_show_com.append(sum(b_show[i]) / len(a_show[i]))
        # print(euclidean_metric(a_show_com, b_show_com))
        # print(euclidean_metric(a_show_com, b_show_com[1:] + b_show_com[:1]))
        # print(euclidean_metric(b_show_com, a_show_com[1:] + a_show_com[:1]))
        plt.figure()
        plt.scatter(list(range(len(a_show))), a_show, s=1.5, c="r")
        plt.scatter(list(range(len(b_show))), b_show, s=1.5, c="k")
        # plt.scatter(list(range(len(a_show_com))), a_show_com, s=1.5, c="r")
        # plt.scatter(list(range(len(b_show_com))), b_show_com, s=1.5, c="k")
        plt.show()
        exit()

    # np.random.seed(1)
    # combineMetric = list(zip(sortCSIa1, sortCSIb1, sortCSIe1, sortNoise))
    # np.random.shuffle(combineMetric)
    # sortCSIa2, sortCSIb2, sortCSIe2, sortNois2 = zip(*combineMetric)

    # np.random.seed(1)
    # combineMetric = list(zip(sortCSIa2, sortCSIb2, sortCSIe2, sortNois2))
    # np.random.shuffle(combineMetric)
    # sortCSIa3, sortCSIb3, sortCSIe3, sortNois3 = zip(*combineMetric)

    combineCSIa1 = []
    for i in range(len(sortCSIa1)):
        combineCSIa1.append(pow(sortCSIa1[i], 2) + pow(i / len(sortCSIa1), 2))
    combineCSIb1 = []
    for i in range(len(sortCSIb1)):
        combineCSIb1.append(pow(sortCSIb1[i], 2) + pow(i / len(sortCSIa1), 2))
    combineCSIe1 = []
    for i in range(len(sortCSIe1)):
        combineCSIe1.append(pow(sortCSIe1[i], 2) + pow(i / len(sortCSIa1), 2))
    combineNoise = []
    for i in range(len(sortNoise)):
        combineNoise.append(pow(sortNoise[i], 2) + pow(i / len(sortCSIa1), 2))

    # combineCSIa1 = []
    # for i in range(len(sortCSIa1)):
    #     combineCSIa1.append(pow(sortCSIa1[i], 2) + pow(sortCSIa2[i], 2))
    # combineCSIb1 = []
    # for i in range(len(sortCSIb1)):
    #     combineCSIb1.append(pow(sortCSIb1[i], 2) + pow(sortCSIb2[i], 2))
    # combineCSIe1 = []
    # for i in range(len(sortCSIe1)):
    #     combineCSIe1.append(pow(sortCSIe1[i], 2) + pow(sortCSIe2[i], 2))
    # combineNoise = []
    # for i in range(len(sortNoise)):
    #     combineNoise.append(pow(sortNoise[i], 2) + pow(sortNois2[i], 2))

    a_metric, publish, a_list_number = sortSegPermOfA(list(combineCSIa1), segLen)
    b_metric, b_list_number = sortSegPermOfB(publish, list(combineCSIb1), segLen)
    e_metric, e_list_number = sortSegPermOfB(publish, list(combineCSIe1), segLen)
    n_metric, n_list_number = sortSegPermOfB(publish, list(combineNoise), segLen)
    # a_metric, publish, a_list_number = sortSegPermOfA(list(sortCSIa1), segLen)
    # b_metric, b_list_number = sortSegPermOfB(publish, list(sortCSIb1), segLen)
    # e_metric, e_list_number = sortSegPermOfB(publish, list(sortCSIe1), segLen)
    # n_metric, n_list_number = sortSegPermOfB(publish, list(sortNoise), segLen)
    # publishOfB, b_metric, b_list_number = refSortSegPermOfB(publish, list(sortCSIb1), segLen)
    # _, e_metric, e_list_number = refSortSegPermOfB(publish, list(sortCSIe1), segLen)
    # _, n_metric, n_list_number = refSortSegPermOfB(publish, list(sortNoise), segLen)
    # 无论B是否发送给A，A的密钥都不变，因为是严格递增
    # a_metric, a_list_number = refSortSegPermOfA(publishOfB, list(sortCSIa1))

    # np.random.seed(1)
    # combineMetric = list(zip(a_metric, b_metric, e_metric, n_metric))
    # np.random.shuffle(combineMetric)
    # a_metric1, b_metric1, e_metric1, n_metric1 = zip(*combineMetric)
    #
    # projCSIa1XY = []
    # projCSIb1XY = []
    # projCSIe1XY = []
    # projCSIn1XY = []
    #
    # projCSIa1X = []
    # projCSIb1X = []
    # projCSIe1X = []
    # projCSIn1X = []
    #
    # for i in range(len(a_metric)):
    #     projCSIa1XY.append(projection(l1, l2, [a_metric[i], a_metric1[i]]))
    #     projCSIb1XY.append(projection(l1, l2, [b_metric[i], b_metric1[i]]))
    #     projCSIe1XY.append(projection(l1, l2, [e_metric[i], e_metric1[i]]))
    #     projCSIn1XY.append(projection(l1, l2, [n_metric[i], n_metric1[i]]))
    #
    #     projCSIa1X.append(projection(l1, l2, [a_metric[i], a_metric1[i]])[0])
    #     projCSIb1X.append(projection(l1, l2, [b_metric[i], b_metric1[i]])[0])
    #     projCSIe1X.append(projection(l1, l2, [e_metric[i], e_metric1[i]])[0])
    #     projCSIn1X.append(projection(l1, l2, [n_metric[i], n_metric1[i]])[0])
    #
    # a_list_number = list(np.argsort(projCSIa1X))
    # b_list_number = list(np.argsort(projCSIb1X))
    # e_list_number = list(np.argsort(projCSIe1X))
    # n_list_number = list(np.argsort(projCSIn1X))

    # a_list_number = list(np.argsort(sortCSIa1))
    # b_list_number = list(np.argsort(sortCSIb1))
    # e_list_number = list(np.argsort(sortCSIe1))
    # n_list_number = list(np.argsort(sortNoise))

    # a_list_number = []
    # b_list_number = []
    # e_list_number = []
    # n_list_number = []
    #
    # for i in range(len(a_metric) -1):
    #     if a_metric[i + 1] - a_metric[i] > 0:
    #         a_list_number.append(1)
    #     else:
    #         a_list_number.append(0)
    #     if b_metric[i + 1] - b_metric[i] > 0:
    #         b_list_number.append(1)
    #     else:
    #         b_list_number.append(0)
    #     if e_metric[i + 1] - e_metric[i] > 0:
    #         e_list_number.append(1)
    #     else:
    #         e_list_number.append(0)
    #     if n_metric[i + 1] - n_metric[i] > 0:
    #         n_list_number.append(1)
    #     else:
    #         n_list_number.append(0)

    # a_metric, publish, a_list_number = adaSortSegPermOfA(list(sortCSIa1), segLen)
    # b_metric, b_list_number = adaSortSegPermOfB(publish, list(sortCSIb1))
    # _, e_list_number = adaSortSegPermOfB(publish, list(sortCSIe1))
    # _, n_list_number = adaSortSegPermOfB(publish, list(sortNoise))

    # print("a", a_metric)
    # print("b", b_metric)

    # plt.figure()
    # plt.scatter(list(range(len(sortCSIa1))), sortCSIa1, s=1.5, c="r")
    # plt.scatter(list(range(len(sortCSIb1))), sortCSIb1, s=1.5, c="k")
    # plt.show()

    # sortCSIa1Reshape = sortCSIa1Reshape.reshape(int(len(sortCSIa1Reshape) / segLen), segLen)
    # sortCSIb1Reshape = sortCSIb1Reshape.reshape(int(len(sortCSIb1Reshape) / segLen), segLen)

    # plt.figure()
    # plt.scatter(list(range(len(sortCSIa1Reshape))),
    #             genSample(sortCSIa1Reshape, ratio), s=1.5, c="r")
    # plt.scatter(list(range(len(sortCSIb1Reshape))),
    #             genSample(sortCSIb1Reshape, ratio), s=1.5, c="k")
    # plt.show()

    # a_metric.sort()
    # b_metric.sort()
    # print("a_sort", a_metric)
    # print("b_sort", b_metric)

    a_mis_metric = []
    b_mis_metric = []

    # for i in range(1, len(a_list_number) -1):
    #     if a_list_number[i] != b_list_number[i]:
    #         a_mis_metric.append(a_metric[i + 1] - a_metric[i])
    #         b_mis_metric.append(b_metric[i + 1] - b_metric[i])
    #         print("mis", a_metric[i + 1] - a_metric[i], a_metric[i] - a_metric[i - 1])
    #         print("mis", b_metric[i + 1] - b_metric[i], b_metric[i] - b_metric[i - 1])
    #     if a_list_number[i] == b_list_number[i]:
    #         print("match", a_metric[i + 1] - a_metric[i], a_metric[i] - a_metric[i - 1])
    #         print("match", b_metric[i + 1] - b_metric[i], b_metric[i] - b_metric[i - 1])
    # print("a_mis", len(a_mis_metric), max(a_mis_metric), min(a_mis_metric), np.mean(a_mis_metric))
    # print("b_mis", len(b_mis_metric), max(b_mis_metric), min(b_mis_metric), np.mean(b_mis_metric))

    a_mis_diff = 0
    b_mis_diff = 0

    # a_diff = sys.maxsize
    # b_diff = sys.maxsize
    # a_index = -1
    # b_index = -1
    # a_metric.sort()
    # b_metric.sort()
    # for i in range(len(a_metric) - 1):
    #     a_diff = min(a_diff, abs(a_metric[i + 1] - a_metric[i]))
    #     if a_diff == abs(a_metric[i + 1] - a_metric[i]):
    #         a_index = i
    # for i in range(len(b_metric) - 1):
    #     b_diff = min(b_diff, abs(b_metric[i + 1] - b_metric[i]))
    #     if b_diff == abs(b_metric[i + 1] - b_metric[i]):
    #         b_index = i
    # print("a_diff", a_index, a_metric[a_index], a_metric[a_index + 1], a_diff)
    # print("b_diff", b_index, b_metric[b_index], b_metric[b_index + 1], b_diff)

    # min_diff = sys.maxsize
    # min_index = 0
    # for i in range(len(a_metric)):
    #     min_diff = min(min_diff, abs(a_metric[i] - b_metric[i]))
    #     if min_diff == abs(a_metric[i] - b_metric[i]):
    #         min_index = i
    # print("min_diff", min_index, a_metric[min_index], b_metric[min_index], min_diff)

    # it = 5
    # wi = 1
    # a_list_number = overlap_moving_sum(a_list_number, it, wi)
    # b_list_number = overlap_moving_sum(b_list_number, it, wi)
    # e_list_number = overlap_moving_sum(e_list_number, it, wi)
    # n_list_number = overlap_moving_sum(n_list_number, it, wi)

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

    # 画出metric和对应的索引
    # if sum2 / sum1 < 0.95:
    #     plt.figure()
    #     plt.scatter(list(range(len(a_metric))), a_metric, s=1.5, c="r")
    #     plt.scatter(list(range(len(b_metric))), b_metric, s=1.5, c="k")
    #     for i in range(len(a_metric)):
    #         plt.annotate(a_list_number[i], (i, a_metric[i] + 0.1), c="r")
    #         plt.annotate(b_list_number[i], (i, a_metric[i] - 0.1), c="k")
    #     plt.show()
    #
    #     a_metric.sort()
    #     b_metric.sort()
    #
    #     plt.figure()
    #     plt.scatter(list(range(len(a_metric))), a_metric, s=1.5, c="r")
    #     plt.scatter(list(range(len(b_metric))), b_metric, s=1.5, c="k")
    #     for i in range(len(a_metric)):
    #         plt.annotate(list(np.array(a_metric).argsort().argsort())[i], (i, a_metric[i] + 0.1), c="r")
    #         plt.annotate(list(np.array(b_metric).argsort().argsort())[i], (i, a_metric[i] - 0.1), c="k")
    #     plt.show()
    #     exit()

    if sum1 == 0:
        continue
    if sum2 == sum1:
        print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
    else:
        print("\033[0;31;40ma-b", "bad", sum2, sum2 / sum1, "\033[0m")
        # a_pair = []
        # b_pair = []
        # for i in range(len(a_list_number)):
        #     if a_list_number[i] != b_list_number[i]:
        #         a_pair.append(a_metric[i])
        #         b_pair.append(b_metric[i])
        #         print(a_list_number[i], b_list_number[i])
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
print("测试结束，耗时" + str(round(time.time() - start_time, 3)), "s")
# csv.write(fileName + ',' + str(times) + ',' + str(keyLen) + ',' + str(segLen) + ',' + str(interval)
#           + ',' + str(correctSum) + " / " + str(originSum) + " = " + str(round(correctSum / originSum, 10))
#           + ',' + str(randomSum) + " / " + str(originSum) + " = " + str(round(randomSum / originSum, 10))
#           + ',' + str(noiseSum) + " / " + str(originSum) + " = " + str(round(noiseSum / originSum, 10))
#           + ',' + str(correctWholeSum) + " / " + str(originWholeSum) + " = " + str(
#     round(correctWholeSum / originWholeSum, 10))
#           + ',' + str(randomWholeSum) + " / " + str(originWholeSum) + " = " + str(
#     round(randomWholeSum / originWholeSum, 10))
#           + ',' + str(noiseWholeSum) + " / " + str(originWholeSum) + " = " + str(
#     round(noiseWholeSum / originWholeSum, 10)) + '\n')
