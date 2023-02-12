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
from hilbertcurve.hilbertcurve import HilbertCurve

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


def findMinX(data):
    l = len(data)
    min_X = sys.maxsize
    for i in range(l):
        min_X = min(min_X, data[i][0])
    return min_X


def findMinY(data):
    l = len(data)
    min_Y = sys.maxsize
    for i in range(l):
        min_Y = min(min_Y, data[i][1])
    return min_Y


def findMinZ(data):
    l = len(data)
    min_Z = sys.maxsize
    for i in range(l):
        min_Z = min(min_Z, data[i][2])
    return min_Z


def findMaxX(data):
    l = len(data)
    max_X = 0
    for i in range(l):
        max_X = max(max_X, data[i][0])
    return max_X


def findMaxY(data):
    l = len(data)
    max_Y = 0
    for i in range(l):
        max_Y = max(max_Y, data[i][1])
    return max_Y


def findMaxZ(data):
    l = len(data)
    max_Z = 0
    for i in range(l):
        max_Z = max(max_Z, data[i][2])
    return max_Z


def genCoordinate(data, dimension):
    res = []
    for i in range(len(data)):
        coord = []
        if dimension == 2:
            tmp1 = 0
            tmp2 = 0
            halfLen = int(len(data[i]) / 2)
            for j in range(halfLen):
                tmp1 += data[i][j]
            for j in range(halfLen, len(data[i])):
                tmp2 += data[i][j]
            coord.append(tmp1 / halfLen)
            coord.append(tmp2 / halfLen)
            res.append(coord)
        elif dimension == 3:
            tmp1 = 0
            tmp2 = 0
            tmp3 = 0
            thirdLen = int(len(data[i]) / 3)
            for j in range(thirdLen):
                tmp1 += data[i][j]
            for j in range(thirdLen, thirdLen * 2):
                tmp2 += data[i][j]
            for j in range(thirdLen * 2, len(data[i])):
                tmp3 += data[i][j]
            coord.append(tmp1 / thirdLen)
            coord.append(tmp2 / thirdLen)
            coord.append(tmp3 / thirdLen)
            res.append(coord)
    return res


def listToGridIndexWithFixedParams(list, p, grid_size):
    minX = findMinX(list)
    minY = findMinY(list)
    for i in range(len(list)):
        list[i][0] -= minX
        list[i][1] -= minY

    points = []

    for i in range(len(list)):
        indexX = int(list[i][0] / grid_size)
        indexY = int(list[i][1] / grid_size)
        points.append([indexX, indexY])

    row_space = math.ceil(findMaxX(points) / p)
    col_space = math.ceil(findMaxY(points) / p)

    row_space = 1 if row_space == 0 else row_space
    col_space = 1 if col_space == 0 else col_space

    distances = []

    for i in range(len(points)):
        distances.append(int(points[i][0] / row_space) * p + int(points[i][1] / col_space) + 1)

    return distances


def listToHilbertCurveIndexWithFixedGrid(list, grid_size):
    minX = findMinX(list)
    minY = findMinY(list)
    for i in range(len(list)):
        list[i][0] -= minX
        list[i][1] -= minY

    points = []

    for i in range(len(list)):
        indexX = int(list[i][0] / grid_size)
        indexY = int(list[i][1] / grid_size)
        points.append([indexX, indexY])

    n = 2
    total_grids = findMaxX(points) * findMaxY(points)
    p = int(math.log10(total_grids) / math.log10(2))
    print(total_grids, p)

    hilbert_curve = HilbertCurve(p, n)
    distances = hilbert_curve.distances_from_points(points, match_type=True)
    return distances


min_of_a = [0, 0]


def listToHilbertCurveIndexWithFixedParams(data, p, grid_size, dimension):
    # plt.figure()
    # plt.scatter(np.array(data).T[0], np.array(data).T[1], s=1.5, c="r")
    # plt.show()

    # 分段归一化
    # data_norm = []
    # interval = 16
    # for i in range(0, len(data), interval):
    #     tmp = preprocessing.MinMaxScaler().fit_transform(data[i:i+interval])
    #     for j in range(len(tmp)):
    #         data_norm.append(tmp[j].tolist())
    # data = data_norm.copy()

    data = preprocessing.MinMaxScaler().fit_transform(data)  # 0.9651658682

    # plt.figure()
    # plt.scatter(np.array(data).T[0], np.array(data).T[1], s=1.5, c="r")
    # plt.show()

    # 坐标点平移  0.9538599739
    # if dimension == 2:
    #     min_of_a = [0, 0]
    #     minX = findMinX(data)
    #     minY = findMinY(data)
    #     min_of_a[0] = minX
    #     min_of_a[1] = minY
    #     for i in range(len(data)):
    #         data[i][0] -= min_of_a[0]
    #         data[i][1] -= min_of_a[1]
    # elif dimension == 3:
    #     min_of_a = [0, 0, 0]
    #     minX = findMinX(data)
    #     minY = findMinY(data)
    #     minZ = findMinZ(data)
    #     min_of_a[0] = minX
    #     min_of_a[1] = minY
    #     min_of_a[2] = minZ
    #     for i in range(len(data)):
    #         data[i][0] -= min_of_a[0]
    #         data[i][1] -= min_of_a[1]
    #         data[i][2] -= min_of_a[2]

    points = []

    # 网格化
    # for i in range(len(data)):
    #     if dimension == 2:
    #         indexX = round(data[i][0] / grid_size)
    #         indexY = round(data[i][1] / grid_size)
    #         points.append([indexX, indexY])
    #     elif dimension == 3:
    #         indexX = round(data[i][0] / grid_size)
    #         indexY = round(data[i][1] / grid_size)
    #         indexZ = round(data[i][2] / grid_size)
    #         points.append([indexX, indexY, indexZ])

    # 在九宫格里面找最近的网格点
    expand = 2 ** p - 1
    if dimension == 2:
        for i in range(len(data)):
            indexX = data[i][0] * expand
            indexY = data[i][1] * expand
            x = int(indexX)
            y = int(indexY)
            neighbor = [[x - 1, y - 1], [x, y - 1], [x + 1, y - 1],
                        [x - 1, y], [x, y], [x + 1, y],
                        [x - 1, y + 1], [x, y + 1], [x + 1, y + 1]]
            min_dis = sys.maxsize
            min_integer_point = None
            for point in neighbor:
                if np.square(indexX - point[0]) + np.square(indexY - point[1]) < min_dis:
                    min_dis = np.square(indexX - point[0]) + np.square(indexY - point[1])
                    min_integer_point = point
            points.append(min_integer_point)

        # plt.figure()
        # plt.scatter(np.array(points).T[0], np.array(points).T[1], s=1.5, c="r")
        # plt.show()

        maxX = 1 if findMaxX(points) == 0 else findMaxX(points)
        maxY = 1 if findMaxY(points) == 0 else findMaxY(points)

        # 压缩至指定维度的SFC曲线
        if maxX > expand:
            for i in range(len(points)):
                points[i][0] = round(points[i][0] / (maxX / expand))
        if maxY > expand:
            for i in range(len(points)):
                points[i][1] = round(points[i][1] / (maxY / expand))
    elif dimension == 3:
        for i in range(len(data)):
            indexX = data[i][0] * expand
            indexY = data[i][1] * expand
            indexZ = data[i][2] * expand
            x = int(indexX)
            y = int(indexY)
            z = int(indexZ)
            neighbor = [[x - 1, y - 1, z - 1], [x, y - 1, z - 1], [x + 1, y - 1, z - 1],
                        [x - 1, y, z - 1], [x, y, z - 1], [x + 1, y, z - 1],
                        [x - 1, y + 1, z - 1], [x, y + 1, z - 1], [x + 1, y + 1, z - 1],
                        [x - 1, y - 1, z], [x, y - 1, z], [x + 1, y - 1, z],
                        [x - 1, y, z], [x, y, z], [x + 1, y, z],
                        [x - 1, y + 1, z], [x, y + 1, z], [x + 1, y + 1, z],
                        [x - 1, y - 1, z + 1], [x, y - 1, z + 1], [x + 1, y - 1, z + 1],
                        [x - 1, y, z + 1], [x, y, z + 1], [x + 1, y, z + 1],
                        [x - 1, y + 1, z + 1], [x, y + 1, z + 1], [x + 1, y + 1, z + 1]
                        ]
            min_dis = sys.maxsize
            min_integer_point = None
            for point in neighbor:
                if np.square(indexX - point[0]) + np.square(indexY - point[1]) + np.square(indexY - point[2]) < min_dis:
                    min_dis = np.square(indexX - point[0]) + np.square(indexY - point[1]) + np.square(indexY - point[2])
                    min_integer_point = point
            points.append(min_integer_point)

        # plt.figure()
        # plt.scatter(np.array(points).T[0], np.array(points).T[1], s=1.5, c="r")
        # plt.show()

        maxX = 1 if findMaxX(points) == 0 else findMaxX(points)
        maxY = 1 if findMaxY(points) == 0 else findMaxY(points)
        maxZ = 1 if findMaxZ(points) == 0 else findMaxZ(points)

        # 压缩至指定维度的SFC曲线
        if maxX > expand:
            for i in range(len(points)):
                points[i][0] = round(points[i][0] / (maxX / expand))
        if maxY > expand:
            for i in range(len(points)):
                points[i][1] = round(points[i][1] / (maxY / expand))
        if maxZ > expand:
            for i in range(len(points)):
                points[i][2] = round(points[i][2] / (maxZ / expand))

    # plt.figure()
    # plt.scatter(np.array(points).T[0], np.array(points).T[1], s=1.5, c="k")
    # plt.show()

    # 映射
    print("points", len(points), points)
    hilbert_curve = HilbertCurve(p, dimension)
    distances = hilbert_curve.distances_from_points(points, match_type=True)
    return distances, points, data


# def listToHilbertCurveIndexWithFixedParams(data, p, grid_size):
#     data = preprocessing.MinMaxScaler().fit_transform(data)
#     # 坐标点平移
#     # minX = findMinX(data)
#     # minY = findMinY(data)
#     # for i in range(len(data)):
#     #     data[i][0] -= minX
#     #     data[i][1] -= minY
#
#     points = []
#
#     # 网格化
#     for i in range(len(data)):
#         indexX = int(data[i][0] / grid_size)
#         indexY = int(data[i][1] / grid_size)
#         points.append([indexX, indexY])
#
#     maxX = 1 if findMaxX(points) == 0 else findMaxX(points)
#     maxY = 1 if findMaxY(points) == 0 else findMaxY(points)
#
#     # 压缩至指定维度的SFC曲线
#     if maxX > 2 ** p - 1:
#         for i in range(len(points)):
#             points[i][0] = int(points[i][0] / (maxX / (2 ** p - 1)))
#     if maxY > 2 ** p - 1:
#         for i in range(len(points)):
#             points[i][1] = int(points[i][1] / (maxY / (2 ** p - 1)))
#
#     # 映射
#     n = 2
#     print("points", len(points), points)
#     hilbert_curve = HilbertCurve(p, n)
#     distances = hilbert_curve.distances_from_points(points, match_type=True)
#     return distances


start_time = time.time()
fileName = "../data/data_static_indoor_1_r.mat"
rawData = loadmat(fileName)

csv = open("./sorting.csv", "a+")

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

dataLen = len(CSIa1Orig)  # 94873

CSIn1Orig = np.random.normal(loc=-100, scale=1000, size=dataLen)

CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)

# CSIa1Orig = np.abs(np.fft.fft(CSIa1Orig))
# CSIb1Orig = np.abs(np.fft.fft(CSIb1Orig))
# CSIe1Orig = np.abs(np.fft.fft(CSIe1Orig))
# CSIn1Orig = np.abs(np.fft.fft(CSIn1Orig))

# 固定随机置换的种子
np.random.seed(1)  # 8 1024 8; 4 128 4
combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig, CSIe1Orig, CSIn1Orig))
np.random.shuffle(combineCSIx1Orig)
CSIa1Orig, CSIb1Orig, CSIe1Orig, CSIn1Orig = zip(*combineCSIx1Orig)

# CSIa1Orig = CSIa1Orig - np.mean(CSIa1Orig)
# CSIb1Orig = CSIb1Orig - np.mean(CSIb1Orig)
# CSIe1Orig = CSIe1Orig - np.mean(CSIe1Orig)
# CSIn1Orig = CSIn1Orig - np.mean(CSIn1Orig)

# CSIa1Orig = scipy.stats.boxcox(np.abs(CSIa1Orig))[0]
# CSIb1Orig = scipy.stats.boxcox(np.abs(CSIb1Orig))[0]
# CSIe1Orig = scipy.stats.boxcox(np.abs(CSIe1Orig))[0]
# CSIn1Orig = scipy.stats.boxcox(np.abs(CSIn1Orig))[0]
#
# CSIa1Orig = normal2uniform(CSIa1Orig)
# CSIb1Orig = normal2uniform(CSIb1Orig)
# CSIe1Orig = normal2uniform(CSIe1Orig)
# CSIn1Orig = normal2uniform(CSIn1Orig)

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

segLen = 8
keyLen = 128 * segLen

ratio = 1
rawOp = "coxbox-uniform"

originSum = 0
correctSum = 0
randomSum = 0
noiseSum = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum = 0
noiseWholeSum = 0

times = 0

grid_size = 0.1
hilbert_p = 2  # 迭代3次，密钥最大值为2 ** 3 * 2 ** 3 = 64
dimension = 3  # 迭代2次，三维中密钥最大值为2 ** 2 * 2 ** 2 * 2 ** 2 = 64

# hp=3, d=2 0.965165868
# hp=2, d=3 0.972756257
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
        if rawOp is not None and rawOp != "":
            raise Exception("error rawOp")
        sortCSIa1 = tmpCSIa1
        sortCSIb1 = tmpCSIb1
        sortCSIe1 = tmpCSIe1
        sortNoise = tmpNoise

    sortCSIa1 = smooth(np.array(sortCSIa1), window_len=15, window='flat')
    sortCSIb1 = smooth(np.array(sortCSIb1), window_len=15, window='flat')
    sortCSIe1 = smooth(np.array(sortCSIe1), window_len=15, window='flat')
    sortNoise = smooth(np.array(sortNoise), window_len=15, window='flat')

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

    # a_show = sortCSIa1.copy()
    # b_show = sortCSIb1.copy()
    # a_show.sort()
    # b_show.sort()
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
    # plt.figure()
    # plt.scatter(list(range(len(a_show_com))), a_show_com, s=1.5, c="r")
    # plt.scatter(list(range(len(b_show_com))), b_show_com, s=1.5, c="k")
    # plt.show()

    a_metric, publish, a_list_number = sortSegPermOfA(list(sortCSIa1), segLen)
    b_metric, b_list_number = sortSegPermOfB(publish, list(sortCSIb1), segLen)
    e_metric, e_list_number = sortSegPermOfB(publish, list(sortCSIe1), segLen)
    n_metric, n_list_number = sortSegPermOfB(publish, list(sortNoise), segLen)

    np.random.seed(1)  # 8 1024 8; 4 128 4
    combineMetric = list(zip(a_metric, b_metric, e_metric, n_metric))
    np.random.shuffle(combineMetric)
    a_metric1, b_metric1, e_metric1, n_metric1 = zip(*combineMetric)

    np.random.seed(1)  # 8 1024 8; 4 128 4
    combineMetric = list(zip(a_metric1, b_metric1, e_metric1, n_metric1))
    np.random.shuffle(combineMetric)
    a_metric2, b_metric2, e_metric2, n_metric2 = zip(*combineMetric)

    projCSIa1XY = []
    projCSIb1XY = []
    projCSIe1XY = []
    projCSIn1XY = []

    for i in range(len(a_metric)):
        projCSIa1XY.append([a_metric[i], a_metric1[i], a_metric2[i]])
        projCSIb1XY.append([b_metric[i], b_metric1[i], b_metric2[i]])
        projCSIe1XY.append([e_metric[i], e_metric1[i], e_metric2[i]])
        projCSIn1XY.append([n_metric[i], n_metric1[i], n_metric2[i]])

    # # 使用index作为纵坐标
    # useIndex = False
    # if useIndex:
    #     projCSIa1XY = []
    #     projCSIb1XY = []
    #     projCSIe1XY = []
    #     projCSIn1XY = []
    #
    #     for i in range(len(a_metric)):
    #         projCSIa1XY.append([a_metric[i], i])
    #         projCSIb1XY.append([b_metric[i], i])
    #         projCSIe1XY.append([e_metric[i], i])
    #         projCSIn1XY.append([n_metric[i], i])
    # else:
    #     a_metric = a_metric[0:dimension * int(len(a_metric) / dimension)]
    #     b_metric = b_metric[0:dimension * int(len(b_metric) / dimension)]
    #     e_metric = e_metric[0:dimension * int(len(e_metric) / dimension)]
    #     n_metric = n_metric[0:dimension * int(len(n_metric) / dimension)]
    #
    #     a_metric = np.array(a_metric)
    #     b_metric = np.array(b_metric)
    #     e_metric = np.array(e_metric)
    #     n_metric = np.array(n_metric)
    #
    #     a_metric = a_metric.reshape(int(len(a_metric) / dimension), dimension)
    #     b_metric = b_metric.reshape(int(len(b_metric) / dimension), dimension)
    #     e_metric = e_metric.reshape(int(len(e_metric) / dimension), dimension)
    #     n_metric = n_metric.reshape(int(len(n_metric) / dimension), dimension)
    #
    #     sortCSIa1 = genCoordinate(a_metric, dimension)
    #     sortCSIb1 = genCoordinate(b_metric, dimension)
    #     sortCSIe1 = genCoordinate(e_metric, dimension)
    #     sortNoise = genCoordinate(n_metric, dimension)
    #
    #     projCSIa1XY = []
    #     projCSIb1XY = []
    #     projCSIe1XY = []
    #     projCSIn1XY = []
    #
    #     for i in range(len(sortCSIa1)):
    #         if dimension == 2:
    #             projCSIa1XY.append([sortCSIa1[i][0], sortCSIa1[i][1]])
    #             projCSIb1XY.append([sortCSIb1[i][0], sortCSIb1[i][1]])
    #             projCSIe1XY.append([sortCSIe1[i][0], sortCSIe1[i][1]])
    #             projCSIn1XY.append([sortNoise[i][0], sortNoise[i][1]])
    #         elif dimension == 3:
    #             projCSIa1XY.append([sortCSIa1[i][0], sortCSIa1[i][1], sortCSIa1[i][2]])
    #             projCSIb1XY.append([sortCSIb1[i][0], sortCSIb1[i][1], sortCSIb1[i][2]])
    #             projCSIe1XY.append([sortCSIe1[i][0], sortCSIe1[i][1], sortCSIe1[i][2]])
    #             projCSIn1XY.append([sortNoise[i][0], sortNoise[i][1], sortNoise[i][2]])

    # min_of_a = [findMinX(projCSIa1XY), findMinY(projCSIa1XY)]
    a_list_number, a_point, a_data = list(
        listToHilbertCurveIndexWithFixedParams(projCSIa1XY, hilbert_p, grid_size, dimension))
    b_list_number, b_point, b_data = list(
        listToHilbertCurveIndexWithFixedParams(projCSIb1XY, hilbert_p, grid_size, dimension))
    # min_of_a = [findMinX(projCSIe1XY), findMinY(projCSIe1XY)]
    e_list_number, e_point, _ = list(
        listToHilbertCurveIndexWithFixedParams(projCSIe1XY, hilbert_p, grid_size, dimension))
    # min_of_a = [findMinX(projCSIn1XY), findMinY(projCSIn1XY)]
    n_list_number, n_point, _ = list(
        listToHilbertCurveIndexWithFixedParams(projCSIn1XY, hilbert_p, grid_size, dimension))

    # for i in range(len(a_list_number)):
    #     if a_list_number[i] != b_list_number[i]:
    #         print("\033[0;31;40m", projCSIa1XY[i], projCSIb1XY[i], a_point[i], b_point[i], a_data[i], b_data[i],
    #               "\033[0m")
    #     else:
    #         print("\033[0;32;40m", projCSIa1XY[i], projCSIb1XY[i], a_point[i], b_point[i], a_data[i], b_data[i],
    #               "\033[0m")

    # grid_p = 0.05
    # grid_p = 6
    #
    # a_list_number = list(listToGridIndexWithFixedParams(projCSIa1XY, grid_p, grid_size))
    # b_list_number = list(listToGridIndexWithFixedParams(projCSIb1XY, grid_p, grid_size))
    # e_list_number = list(listToGridIndexWithFixedParams(projCSIe1XY, grid_p, grid_size))
    # n_list_number = list(listToGridIndexWithFixedParams(projCSIn1XY, grid_p, grid_size))

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

    # sortCSIa1Reshape = np.array(sortCSIa1Reshape)
    # sortCSIb1Reshape = np.array(sortCSIb1Reshape)

    # sortCSIa1Reshape = sortCSIa1Reshape.reshape(int(len(sortCSIa1Reshape) / segLen), segLen)
    # sortCSIb1Reshape = sortCSIb1Reshape.reshape(int(len(sortCSIb1Reshape) / segLen), segLen)

    # plt.figure()
    # plt.scatter(list(range(len(sortCSIa1Reshape))),
    #             genSample(sortCSIa1Reshape, ratio), s=1.5, c="r")
    # plt.scatter(list(range(len(sortCSIb1Reshape))),
    #             genSample(sortCSIb1Reshape, ratio), s=1.5, c="k")
    # plt.show()

    # plt.figure()
    # plt.scatter(list(range(len(a_metric))), a_metric, s=1.5, c="r")
    # plt.scatter(list(range(len(b_metric))), b_metric, s=1.5, c="k")
    # plt.show()

    a_metric.sort()
    b_metric.sort()
    # print("a_sort", a_metric)
    # print("b_sort", b_metric)

    # plt.figure()
    # plt.scatter(list(range(len(a_metric))), a_metric, s=1.5, c="r")
    # plt.scatter(list(range(len(b_metric))), b_metric, s=1.5, c="k")
    # plt.show()

    # a_mis_metric = []
    # b_mis_metric = []

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

    # a_mis_diff = 0
    # b_mis_diff = 0

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
