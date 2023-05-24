import math
import sys
import time
from tkinter import messagebox

import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from pyentrp import entropy as ent
from scipy import signal
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr


def search(data, p):
    for i in range(len(data)):
        if p == data[i]:
            return i

def findMinInterval(list):
    l = len(list)
    min_interval = sys.maxsize
    for i in range(l):
        for j in range(i + 1, l):
            interval = math.sqrt(math.pow(list[i][0] - list[j][0], 2) + math.pow(list[i][1] - list[j][1], 2))
            min_interval = min(min_interval, interval)
    return min_interval


def findMaxInterval(list):
    l = len(list)
    max_interval = 0
    for i in range(l):
        for j in range(i + 1, l):
            interval = math.sqrt(math.pow(list[i][0] - list[j][0], 2) + math.pow(list[i][1] - list[j][1], 2))
            max_interval = max(max_interval, interval)
    return max_interval


def listToHilbertCurveIndex(list):
    grid_size = findMinInterval(list) / 2 * math.sqrt(2)
    list = np.array(list) - np.min(list, axis=0)
    points = np.array(list / grid_size).astype(int)

    column_number = (np.max(list, axis=0)[0] - np.min(list, axis=0)[0]) / grid_size
    row_number = (np.max(list, axis=0)[1] - np.min(list, axis=0)[1]) / grid_size

    total_grids = (int(column_number) + 1) * (int(row_number) + 1)
    p = int(math.log2(total_grids))
    print(p, grid_size)

    n = 2
    hilbert_curve = HilbertCurve(p, n)
    distances = hilbert_curve.distances_from_points(points, match_type=True)
    return distances


def cut(list):
    for i in range(len(list)):
        # list[i] = str(list[i])[0:int(len(str(list[i])) / 2)]
        list[i] = str(list[i])[0:2]
        list[i] = int(list[i])
    return list


def genCoordinate(list):
    res = []
    for i in range(len(list)):
        coord = []
        tmp1 = 0
        tmp2 = 0
        halfLen = int(len(list[i]) / 2)
        for j in range(halfLen):
            tmp1 += list[i][j]
        for j in range(halfLen, len(list[i])):
            tmp2 += list[i][j]
        coord.append(tmp1 / halfLen)
        coord.append(tmp2 / halfLen)
        res.append(coord)
    return res


def listToGridIndexWithFixedParams(list, p, grid_size):
    list = np.array(list) - np.min(list, axis=0)
    points = np.array(list / grid_size).astype(int)

    row_space = math.ceil(np.max(list, axis=0)[0] / grid_size)
    col_space = math.ceil(np.max(list, axis=0)[1] / grid_size)

    row_space = 1 if row_space == 0 else row_space
    col_space = 1 if col_space == 0 else col_space

    distances = []

    for i in range(len(points)):
        distances.append(int(points[i][0] / row_space) * p + int(points[i][1] / col_space) + 1)

    return distances


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


# 数组第二维的所有内容求和
def sumEachDim(list, index):
    res = 0
    for i in range(len(list[index])):
        res += (list[index][i][0] + list[index][i][1])
        # res += (list[index][i][0] * list[index][i][1])
    return round(res, 8)


def projection(l1, l2, p):
    v1 = [p[0] - l1[0], p[1] - l1[1]]
    v2 = [l2[0] - l1[0], l2[1] - l1[1]]
    v1v2 = v1[0] * v2[0] + v1[1] * v2[1]
    k = v1v2 / (math.pow(l2[0] - l1[0], 2) + math.pow(l2[1] - l1[1], 2))
    p0 = [l1[0] + k * (l2[0] - l1[0]), l1[1] + k * (l2[1] - l1[1])]
    return p0


def splitEntropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, segLen, dataLen):
    print("SE total_se", ent.multiscale_entropy(CSIa1Orig, 3, maxscale=1))
    min_se_f = 999
    for i in range(int(len(CSIa1Orig) / segLen)):
        tmp = CSIa1Orig[i * segLen: i * segLen + segLen]
        a_mul_entropy = ent.multiscale_entropy(tmp, 3, maxscale=1)
        min_se_f = min(min_se_f, a_mul_entropy)
    print("SE min_se", min_se_f)

    # 先整体shuffle一次
    shuffleInd = np.random.permutation(dataLen)
    CSIa1Orig = CSIa1Orig[shuffleInd]
    CSIb1Orig = CSIb1Orig[shuffleInd]
    CSIe1Orig = CSIe1Orig[shuffleInd]
    savemat('../data/data_NLOS_permh.mat', {'A': CSIa1Orig})

    sortCSIa1Reshape = CSIa1Orig[0:segLen * int(dataLen / segLen)]
    sortCSIb1Reshape = CSIb1Orig[0:segLen * int(dataLen / segLen)]
    sortCSIe1Reshape = CSIe1Orig[0:segLen * int(dataLen / segLen)]

    sortCSIa1Reshape = sortCSIa1Reshape.reshape(int(len(sortCSIa1Reshape) / segLen), segLen)
    sortCSIb1Reshape = sortCSIb1Reshape.reshape(int(len(sortCSIb1Reshape) / segLen), segLen)
    sortCSIe1Reshape = sortCSIe1Reshape.reshape(int(len(sortCSIe1Reshape) / segLen), segLen)
    n = len(sortCSIa1Reshape)
    print("SE_h total_se_h", ent.multiscale_entropy(CSIa1Orig, 3, maxscale=1))
    min_se_f = 999
    for i in range(n):
        a_mul_entropy = ent.multiscale_entropy(sortCSIa1Reshape[i], 3, maxscale=1)
        min_se_f = min(min_se_f, a_mul_entropy)
    print("SE_h min_se_f", min_se_f)

    min_se_f = 999
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

        min_se_f = min(min_se_f, a_mul_entropy)
    print("SE_f min_se_f", min_se_f)
    _CSIa1Orig = []
    _CSIb1Orig = []
    _CSIe1Orig = []

    for i in range(len(sortCSIa1Reshape)):
        for j in range(len(sortCSIa1Reshape[i])):
            _CSIa1Orig.append(sortCSIa1Reshape[i][j])
            _CSIb1Orig.append(sortCSIb1Reshape[i][j])
            _CSIe1Orig.append(sortCSIe1Reshape[i][j])

    print("SE_f total_se_h", ent.multiscale_entropy(_CSIa1Orig, 3, maxscale=1))

    return np.array(_CSIa1Orig), np.array(_CSIb1Orig), np.array(_CSIe1Orig)

fileName = ["../data/data_mobile_indoor_1.mat",
            "../data/data_mobile_outdoor_1.mat",
            "../data/data_static_outdoor_1.mat",
            "../data/data_static_indoor_1.mat"
            ]

isShow = False
print("file", "\t", "bit", "\t", "key", "\t", "KGR", "\t", "KGR with error free", "\t", "mode")
for f in fileName:
    rawData = loadmat(f)
    CSIa1Orig = rawData['A'][:, 0]
    CSIb1Orig = rawData['A'][:, 1]

    dataLen = len(CSIa1Orig)

    segLen = 5
    keyLen = 256 * segLen

    originSum = 0
    correctSum = 0
    randomSum1 = 0
    randomSum2 = 0
    noiseSum1 = 0
    noiseSum2 = 0
    noiseSum3 = 0

    originDecSum = 0
    correctDecSum = 0
    randomDecSum1 = 0
    randomDecSum2 = 0
    noiseDecSum1 = 0
    noiseDecSum2 = 0
    noiseDecSum3 = 0

    originWholeSum = 0
    correctWholeSum = 0
    randomWholeSum1 = 0
    randomWholeSum2 = 0
    noiseWholeSum1 = 0
    noiseWholeSum2 = 0
    noiseWholeSum3 = 0

    times = 0
    # no perturbation
    withoutSort = True
    addNoise = "mul"
    operationMode = ""
    if withoutSort:
        if addNoise == "mul":
            operationMode = "no sorting"
            print("no sorting")
    if withoutSort:
        if addNoise == "":
            operationMode = "no sorting and no perturbation"
            print("no sorting and no perturbation")
    if withoutSort is False:
        if addNoise == "":
            operationMode = "no perturbation"
            print("no perturbation")
        if addNoise == "mul":
            operationMode = "normal"
            print("normal")

    if withoutSort:
        grid_size = 0.01
        hilbert_p = 8
    else:
        grid_size = 10
        hilbert_p = 8

    ab_dist = 0
    ae1_dist = 0
    ae2_dist = 0
    an1_dist = 0
    an2_dist = 0
    an3_dist = 0

    ab_corr = 0
    ae1_corr = 0
    ae2_corr = 0
    an1_corr = 0
    an2_corr = 0
    an3_corr = 0

    dataLenLoop = dataLen
    keyLenLoop = keyLen
    if f == "../data/data_static_indoor_1.mat":
        dataLenLoop = int(dataLen / 5.5)
        keyLenLoop = int(keyLen / 5)
    for staInd in range(0, dataLenLoop, keyLenLoop):
        start = time.time()

        grid_size = 0.05
        grid_p = 6  # 2 ** 6 = 64 密钥points最大为64
        # for grid_p in range(1, 6):
        #     grid_p *= 2
        #     print("grid_p", grid_p)

        endInd = staInd + keyLen
        # print("range:", staInd, endInd)
        if endInd >= len(CSIa1Orig):
            break

        times += 1

        # np.random.seed(0)
        CSIa1Orig = rawData['A'][:, 0]
        CSIb1Orig = rawData['A'][:, 1]

        # imitation attack
        CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=len(CSIa1Orig))
        # stalking attack
        CSIe2Orig = loadmat("../skyglow/Scenario2-Office-LoS-eve_NLoS/data_eave_LOS_EVE_NLOS.mat")['A'][:, 0]

        tmpNoise1 = []
        tmpNoise2 = []
        tmpNoise3 = []

        seed = np.random.randint(100000)
        np.random.seed(seed)

        if addNoise == "add":
            CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
            CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')

            tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
            tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
            tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
            tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]
            noiseOrig = np.random.uniform(0, 0.5, size=keyLen)
            tmpCSIa1 = (tmpCSIa1 - np.mean(tmpCSIa1)) + noiseOrig
            tmpCSIb1 = (tmpCSIb1 - np.mean(tmpCSIb1)) + noiseOrig
            tmpCSIe1 = (tmpCSIe1 - np.mean(tmpCSIe1)) + noiseOrig
            tmpCSIe2 = (tmpCSIe2 - np.mean(tmpCSIe2)) + noiseOrig
            tmpNoise = noiseOrig
        elif addNoise == "mul":
            # 静态数据需要置换
            # 固定随机置换的种子
            # np.random.seed(0)
            # combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig))
            # np.random.shuffle(combineCSIx1Orig)
            # CSIa1Orig, CSIb1Orig = zip(*combineCSIx1Orig)

            CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
            CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')
            CSIe1Orig = smooth(np.array(CSIe1Orig), window_len=30, window='flat')
            CSIe2Orig = smooth(np.array(CSIe2Orig), window_len=30, window='flat')

            tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
            tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
            tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
            tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]

            randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
            tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
            tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
            tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
            tmpCSIe2 = tmpCSIe2 - np.mean(tmpCSIe2)

            # operationMode = "sort the value"
            # tmpCSIa1 = np.sort(tmpCSIa1)
            # tmpCSIb1 = np.sort(tmpCSIb1)
            # tmpCSIe1 = np.sort(tmpCSIe1)
            # tmpCSIe2 = np.sort(tmpCSIe2)

            # operationMode = "index"
            # tmpCSIa1 = np.array(tmpCSIa1).argsort().argsort()
            # tmpCSIb1 = np.array(tmpCSIb1).argsort().argsort()
            # tmpCSIe1 = np.array(tmpCSIe1).argsort().argsort()
            # tmpCSIe2 = np.array(tmpCSIe2).argsort().argsort()

            # inference attack
            tmpNoise1 = np.matmul(np.ones(keyLen), randomMatrix)  # 按列求均值
            tmpNoise2 = randomMatrix.mean(axis=1)  # 按行求均值
            tmpNoise3 = np.random.normal(loc=np.mean(tmpCSIa1), scale=np.std(tmpCSIa1, ddof=1), size=len(tmpCSIa1))
        else:
            tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
            tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
            tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
            tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]
            tmpNoise = np.random.normal(0, np.std(CSIa1Orig), size=keyLen)

        tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1))
        tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1))
        tmpCSIe1 = (tmpCSIe1 - np.min(tmpCSIe1)) / (np.max(tmpCSIe1) - np.min(tmpCSIe1))
        tmpCSIe2 = (tmpCSIe2 - np.min(tmpCSIe2)) / (np.max(tmpCSIe2) - np.min(tmpCSIe2))
        tmpNoise1 = (tmpNoise1 - np.min(tmpNoise1)) / (np.max(tmpNoise1) - np.min(tmpNoise1))
        tmpNoise2 = (tmpNoise2 - np.min(tmpNoise2)) / (np.max(tmpNoise2) - np.min(tmpNoise2))
        tmpNoise3 = (tmpNoise3 - np.min(tmpNoise3)) / (np.max(tmpNoise3) - np.min(tmpNoise3))

        # 最后各自的密钥
        a_list = []
        b_list = []
        e1_list = []
        e2_list = []
        n1_list = []
        n2_list = []
        n3_list = []

        if withoutSort:
            tmpCSIa1Ind = np.array(tmpCSIa1)
            tmpCSIb1Ind = np.array(tmpCSIb1)
            tmpCSIe1Ind = np.array(tmpCSIe1)
            tmpCSIe2Ind = np.array(tmpCSIe2)
            tmpCSIn1Ind = np.array(tmpNoise1)
            tmpCSIn2Ind = np.array(tmpNoise2)
            tmpCSIn3Ind = np.array(tmpNoise3)
        else:
            tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
            tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()
            tmpCSIe1Ind = np.array(tmpCSIe1).argsort().argsort()
            tmpCSIe2Ind = np.array(tmpCSIe2).argsort().argsort()
            tmpCSIn1Ind = np.array(tmpNoise1).argsort().argsort()
            tmpCSIn2Ind = np.array(tmpNoise2).argsort().argsort()
            tmpCSIn3Ind = np.array(tmpNoise3).argsort().argsort()

        # # 取原数据的一部分来reshape
        # shuffling = np.random.permutation(range(keyLen))
        # tmpCSIa1Ind = np.append(tmpCSIa1Ind, tmpCSIa1Ind[shuffling])
        # tmpCSIb1Ind = np.append(tmpCSIb1Ind, tmpCSIb1Ind[shuffling])
        # tmpCSIe1Ind = np.append(tmpCSIe1Ind, tmpCSIe1Ind[shuffling])
        # tmpCSIe2Ind = np.append(tmpCSIe2Ind, tmpCSIe2Ind[shuffling])
        # tmpCSIn1Ind = np.append(tmpCSIn1Ind, tmpCSIn1Ind[shuffling])
        # tmpCSIn2Ind = np.append(tmpCSIn2Ind, tmpCSIn2Ind[shuffling])
        # tmpCSIn3Ind = np.append(tmpCSIn3Ind, tmpCSIn3Ind[shuffling])

        projCSIa1XY = tmpCSIa1Ind.reshape(int(len(tmpCSIa1Ind) / 2), 2)
        projCSIb1XY = tmpCSIb1Ind.reshape(int(len(tmpCSIb1Ind) / 2), 2)
        projCSIe1XY = tmpCSIe1Ind.reshape(int(len(tmpCSIe1Ind) / 2), 2)
        projCSIe2XY = tmpCSIe2Ind.reshape(int(len(tmpCSIe2Ind) / 2), 2)
        projCSIn1XY = tmpCSIn1Ind.reshape(int(len(tmpCSIn1Ind) / 2), 2)
        projCSIn2XY = tmpCSIn2Ind.reshape(int(len(tmpCSIn2Ind) / 2), 2)
        projCSIn3XY = tmpCSIn3Ind.reshape(int(len(tmpCSIn3Ind) / 2), 2)

        tmpCSIa1Ind = list(listToGridIndexWithFixedParams(projCSIa1XY, grid_p, grid_size))
        tmpCSIb1Ind = list(listToGridIndexWithFixedParams(projCSIb1XY, grid_p, grid_size))
        tmpCSIe1Ind = list(listToGridIndexWithFixedParams(projCSIe1XY, grid_p, grid_size))
        tmpCSIe2Ind = list(listToGridIndexWithFixedParams(projCSIe2XY, grid_p, grid_size))
        tmpCSIn1Ind = list(listToGridIndexWithFixedParams(projCSIn1XY, grid_p, grid_size))
        tmpCSIn2Ind = list(listToGridIndexWithFixedParams(projCSIn2XY, grid_p, grid_size))
        tmpCSIn3Ind = list(listToGridIndexWithFixedParams(projCSIn3XY, grid_p, grid_size))

        # operationMode = "grid and index"
        # tmpCSIa1Ind = np.array(tmpCSIa1Ind).argsort().argsort()
        # tmpCSIb1Ind = np.array(tmpCSIb1Ind).argsort().argsort()
        # tmpCSIe1Ind = np.array(tmpCSIe1Ind).argsort().argsort()
        # tmpCSIe2Ind = np.array(tmpCSIe2Ind).argsort().argsort()
        # tmpCSIn1Ind = np.array(tmpCSIn1Ind).argsort().argsort()
        # tmpCSIn2Ind = np.array(tmpCSIn2Ind).argsort().argsort()
        # tmpCSIn3Ind = np.array(tmpCSIn3Ind).argsort().argsort()

        operationMode = "grid and value"
        minEpiIndClosenessLsb = np.zeros(int(len(tmpCSIa1Ind) / segLen), dtype=int)
        minEpiIndClosenessLse1 = np.zeros(int(len(tmpCSIa1Ind) / segLen), dtype=int)
        minEpiIndClosenessLse2 = np.zeros(int(len(tmpCSIa1Ind) / segLen), dtype=int)
        minEpiIndClosenessLsn1 = np.zeros(int(len(tmpCSIa1Ind) / segLen), dtype=int)
        minEpiIndClosenessLsn2 = np.zeros(int(len(tmpCSIa1Ind) / segLen), dtype=int)
        minEpiIndClosenessLsn3 = np.zeros(int(len(tmpCSIa1Ind) / segLen), dtype=int)

        tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(len(tmpCSIa1Ind) / segLen), segLen)

        permutation = list(range(int(len(tmpCSIa1Ind) / segLen)))
        combineMetric = list(zip(tmpCSIa1IndReshape, permutation))
        np.random.seed(staInd)
        np.random.shuffle(combineMetric)
        tmpCSIa1IndReshape, permutation = zip(*combineMetric)
        tmpCSIa1Ind = np.hstack((tmpCSIa1IndReshape))

        for i in range(int(len(tmpCSIa1Ind) / segLen)):
            epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]

            epiIndClosenessLsb = np.zeros(int(len(tmpCSIa1Ind) / segLen))
            epiIndClosenessLse1 = np.zeros(int(len(tmpCSIa1Ind) / segLen))
            epiIndClosenessLse2 = np.zeros(int(len(tmpCSIa1Ind) / segLen))
            epiIndClosenessLsn1 = np.zeros(int(len(tmpCSIa1Ind) / segLen))
            epiIndClosenessLsn2 = np.zeros(int(len(tmpCSIa1Ind) / segLen))
            epiIndClosenessLsn3 = np.zeros(int(len(tmpCSIa1Ind) / segLen))

            for j in range(int(len(tmpCSIa1Ind) / segLen)):
                epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
                epiInde1 = tmpCSIe1Ind[j * segLen: (j + 1) * segLen]
                epiInde2 = tmpCSIe2Ind[j * segLen: (j + 1) * segLen]
                epiIndn1 = tmpCSIn1Ind[j * segLen: (j + 1) * segLen]
                epiIndn2 = tmpCSIn2Ind[j * segLen: (j + 1) * segLen]
                epiIndn3 = tmpCSIn3Ind[j * segLen: (j + 1) * segLen]

                epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))
                epiIndClosenessLse1[j] = sum(abs(epiInde1 - np.array(epiInda1)))
                epiIndClosenessLse2[j] = sum(abs(epiInde2 - np.array(epiInda1)))
                epiIndClosenessLsn1[j] = sum(abs(epiIndn1 - np.array(epiInda1)))
                epiIndClosenessLsn2[j] = sum(abs(epiIndn2 - np.array(epiInda1)))
                epiIndClosenessLsn3[j] = sum(abs(epiIndn3 - np.array(epiInda1)))

            minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)
            minEpiIndClosenessLse1[i] = np.argmin(epiIndClosenessLse1)
            minEpiIndClosenessLse2[i] = np.argmin(epiIndClosenessLse2)
            minEpiIndClosenessLsn1[i] = np.argmin(epiIndClosenessLsn1)
            minEpiIndClosenessLsn2[i] = np.argmin(epiIndClosenessLsn2)
            minEpiIndClosenessLsn3[i] = np.argmin(epiIndClosenessLsn3)

        # a_list_number = list(range(int(len(tmpCSIa1Ind) / segLen)))
        a_list_number = list(permutation)
        b_list_number = list(minEpiIndClosenessLsb)
        e1_list_number = list(minEpiIndClosenessLse1)
        e2_list_number = list(minEpiIndClosenessLse2)
        n1_list_number = list(minEpiIndClosenessLsn1)
        n2_list_number = list(minEpiIndClosenessLsn2)
        n3_list_number = list(minEpiIndClosenessLsn3)

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
        for i in range(len(n3_list_number)):
            number = bin(n3_list_number[i])[2:].zfill(int(np.log2(len(n3_list_number))))
            n3_list += number

        # 对齐密钥，随机补全
        for i in range(len(a_list) - len(e1_list)):
            e1_list += str(np.random.randint(0, 2))
        for i in range(len(a_list) - len(e2_list)):
            e2_list += str(np.random.randint(0, 2))
        for i in range(len(a_list) - len(n1_list)):
            n1_list += str(np.random.randint(0, 2))
        for i in range(len(a_list) - len(n2_list)):
            n2_list += str(np.random.randint(0, 2))
        for i in range(len(a_list) - len(n3_list)):
            n3_list += str(np.random.randint(0, 2))

        # print("keys of a:", len(a_list), a_list)
        # print("keys of a:", len(a_list_number), a_list_number)
        # print("keys of b:", len(b_list), b_list)
        # print("keys of b:", len(b_list_number), b_list_number)
        # print("keys of e:", len(e_list), e_list)
        # print("keys of e1:", len(e1_list_number), e1_list_number)
        # print("keys of e:", len(e_list), e_list)
        # print("keys of e2:", len(e2_list_number), e2_list_number)
        # print("keys of n1:", len(n1_list), n1_list)
        # print("keys of n1:", len(n1_list_number), n1_list_number)
        # print("keys of n2:", len(n2_list), n2_list)
        # print("keys of n2:", len(n2_list_number), n2_list_number)
        # print("keys of n3:", len(n3_list), n3_list)
        # print("keys of n3:", len(n3_list_number), n3_list_number)

        sum1 = min(len(a_list), len(b_list))
        sum2 = 0
        sum31 = 0
        sum32 = 0
        sum41 = 0
        sum42 = 0
        sum43 = 0
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
        for i in range(min(len(a_list), len(n3_list))):
            sum43 += (a_list[i] == n3_list[i])

        # 计算平均错误距离
        tmp_dist = 0
        for i in range(len(b_list_number)):
            real_pos = search(a_list_number, b_list_number[i])
            guess_pos = i
            tmp_dist += abs(real_pos - guess_pos)
        ab_dist += (tmp_dist / int(len(tmpCSIa1Ind) / segLen))

        tmp_dist = 0
        for i in range(len(e1_list_number)):
            real_pos = search(a_list_number, e1_list_number[i])
            guess_pos = i
            tmp_dist += abs(real_pos - guess_pos)
        ae1_dist += (tmp_dist / int(len(tmpCSIa1Ind) / segLen))

        tmp_dist = 0
        for i in range(len(e2_list_number)):
            real_pos = search(a_list_number, e2_list_number[i])
            guess_pos = i
            tmp_dist += abs(real_pos - guess_pos)
        ae2_dist += (tmp_dist / int(len(tmpCSIa1Ind) / segLen))

        tmp_dist = 0
        for i in range(len(n1_list_number)):
            real_pos = search(a_list_number, n1_list_number[i])
            guess_pos = i
            tmp_dist += abs(real_pos - guess_pos)
        an1_dist += (tmp_dist / int(len(tmpCSIa1Ind) / segLen))

        tmp_dist = 0
        for i in range(len(n2_list_number)):
            real_pos = search(a_list_number, n2_list_number[i])
            guess_pos = i
            tmp_dist += abs(real_pos - guess_pos)
        an2_dist += (tmp_dist / int(len(tmpCSIa1Ind) / segLen))

        tmp_dist = 0
        for i in range(len(n3_list_number)):
            real_pos = search(a_list_number, n3_list_number[i])
            guess_pos = i
            tmp_dist += abs(real_pos - guess_pos)
        an3_dist += (tmp_dist / int(len(tmpCSIa1Ind) / segLen))

        # 计算相关系数
        ab_corr += 0 if np.isnan(pearsonr(a_list_number, b_list_number)[0]) else \
            pearsonr(a_list_number, b_list_number)[0]
        ae1_corr += 0 if np.isnan(pearsonr(a_list_number, e1_list_number)[0]) \
            else pearsonr(a_list_number, e1_list_number)[0]
        ae2_corr += 0 if np.isnan(pearsonr(a_list_number, e2_list_number)[0]) \
            else pearsonr(a_list_number, e2_list_number)[0]
        an1_corr += 0 if np.isnan(pearsonr(a_list_number, n1_list_number)[0]) \
            else pearsonr(a_list_number, n1_list_number)[0]
        an2_corr += 0 if np.isnan(pearsonr(a_list_number, n2_list_number)[0]) \
            else pearsonr(a_list_number, n2_list_number)[0]
        an3_corr += 0 if np.isnan(pearsonr(a_list_number, n3_list_number)[0]) \
            else pearsonr(a_list_number, n3_list_number)[0]

        # print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
        # print("a-e1", sum31, sum31 / sum1)
        # print("a-e2", sum32, sum32 / sum1)
        # print("a-n1", sum41, sum41 / sum1)
        # print("a-n2", sum42, sum42 / sum1)
        # print("a-n3", sum43, sum43 / sum1)
        originSum += sum1
        correctSum += sum2
        randomSum1 += sum31
        randomSum2 += sum32
        noiseSum1 += sum41
        noiseSum2 += sum42
        noiseSum3 += sum43

        originWholeSum += 1
        correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
        randomWholeSum1 = randomWholeSum1 + 1 if sum31 == sum1 else randomWholeSum1
        randomWholeSum2 = randomWholeSum2 + 1 if sum32 == sum1 else randomWholeSum2
        noiseWholeSum1 = noiseWholeSum1 + 1 if sum41 == sum1 else noiseWholeSum1
        noiseWholeSum2 = noiseWholeSum2 + 1 if sum42 == sum1 else noiseWholeSum2
        noiseWholeSum3 = noiseWholeSum3 + 1 if sum43 == sum1 else noiseWholeSum3

    if isShow:
        print("\033[0;34;40ma-b bit agreement rate", correctSum, "/", originSum, "=", round(correctSum / originSum, 10),
              "\033[0m")
        print("a-e1 bit agreement rate", randomSum1, "/", originSum, "=", round(randomSum1 / originSum, 10))
        print("a-e2 bit agreement rate", randomSum2, "/", originSum, "=", round(randomSum2 / originSum, 10))
        print("a-n1 bit agreement rate", noiseSum1, "/", originSum, "=", round(noiseSum1 / originSum, 10))
        print("a-n2 bit agreement rate", noiseSum2, "/", originSum, "=", round(noiseSum2 / originSum, 10))
        print("a-n3 bit agreement rate", noiseSum3, "/", originSum, "=", round(noiseSum3 / originSum, 10))
        print("\033[0;34;40ma-b key agreement rate", correctWholeSum, "/", originWholeSum, "=",
              round(correctWholeSum / originWholeSum, 10), "\033[0m")
        print("a-e1 key agreement rate", randomWholeSum1, "/", originWholeSum, "=",
              round(randomWholeSum1 / originWholeSum, 10))
        print("a-e2 key agreement rate", randomWholeSum2, "/", originWholeSum, "=",
              round(randomWholeSum2 / originWholeSum, 10))
        print("a-n1 key agreement rate", noiseWholeSum1, "/", originWholeSum, "=",
              round(noiseWholeSum1 / originWholeSum, 10))
        print("a-n2 key agreement rate", noiseWholeSum2, "/", originWholeSum, "=",
              round(noiseWholeSum2 / originWholeSum, 10))
        print("a-n3 key agreement rate", noiseWholeSum3, "/", originWholeSum, "=",
              round(noiseWholeSum3 / originWholeSum, 10))
        print("\033[0;34;40ma-b average distance", round(ab_dist / times, 8), "\033[0m")
        print("ae1 average distance", round(ae1_dist / times, 8))
        print("ae2 average distance", round(ae2_dist / times, 8))
        print("an1 average distance", round(an1_dist / times, 8))
        print("an2 average distance", round(an2_dist / times, 8))
        print("an3 average distance", round(an3_dist / times, 8))
        print("\033[0;34;40ma-b average correlation", round(ab_corr / times, 8), "\033[0m")
        print("ae1 average correlation", round(ae1_corr / times, 8))
        print("ae2 average correlation", round(ae2_corr / times, 8))
        print("an1 average correlation", round(an1_corr / times, 8))
        print("an2 average correlation", round(an2_corr / times, 8))
        print("an3 average correlation", round(an3_corr / times, 8))
        # print("times", times)
    spiltFileName = f.split("_")
    print(spiltFileName[1][0] + spiltFileName[2][0] + spiltFileName[3][0], "\t\t",
          round(correctSum / originSum, 4), "\t\t",
          round(correctWholeSum / originWholeSum, 4), "\t\t",
          round(originSum / times / keyLen, 4), "\t\t",
          round(correctSum / times / keyLen, 4), "\t\t", operationMode)
messagebox.showinfo("提示", "测试结束")
