import math
import random
import time
import sys

import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from matplotlib import pyplot as plt
from pyentrp import entropy as ent
from scipy import signal
from scipy.io import loadmat, savemat
from scipy.spatial.distance import euclidean, chebyshev
from sklearn import preprocessing


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
    column_number = (np.max(list, axis=0)[0] - np.min(list, axis=0)[0]) / grid_size
    row_number = (np.max(list, axis=0)[1] - np.min(list, axis=0)[1]) / grid_size
    points = np.array(list / grid_size).astype(int)

    total_grids = (int(column_number) + 1) * (int(row_number) + 1)
    p = int(math.log10(total_grids) / math.log10(2))

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
        for j in range(int(len(list[i]) / 2)):
            tmp1 += list[i][j]
        for j in range(int(len(list[i]) / 2), len(list[i])):
            tmp2 += list[i][j]
        coord.append(tmp1)
        coord.append(tmp2)
        res.append(coord)
    return res


def listToHilbertCurveIndexWithFixedParams(list, p, grid_size):
    list = np.array(list) - np.min(list, axis=0)
    points = np.array(list / grid_size).astype(int)

    n = 2
    hilbert_curve = HilbertCurve(p, n)
    distances = hilbert_curve.distances_from_points(points, match_type=True)
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


fileName = "../data/data_mobile_indoor_1.mat"
rawData = loadmat(fileName)
CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

dataLen = len(CSIa1Orig)

CSIa1Orig = np.array(CSIa1Orig)
CSIb1Orig = np.array(CSIb1Orig)
CSIn1Orig = np.random.normal(loc=-1, scale=1, size=dataLen)  ## Multiplication item normal distribution
CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)

# entropyTime = time.time()
# CSIa1Orig, CSIb1Orig, CSIe1Orig = splitEntropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, 5, dataLen)
# print("--- entropyTime %s seconds ---" % (time.time() - entropyTime))

# CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')
# CSIb1Orig = smooth(CSIb1Orig, window_len=15, window='flat')
# CSIe1Orig = smooth(CSIe1Orig, window_len=15, window="flat")

CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
CSIe1OrigBack = CSIe1Orig.copy()
CSIn1OrigBack = CSIn1Orig.copy()

noise = np.random.normal(loc=-1, scale=1, size=dataLen)  ## Multiplication item normal distribution
noiseAdd = np.random.normal(loc=0, scale=10, size=dataLen)  ## Addition item normal distribution

sft = 2
intvl = 2 * sft + 1
keyLen = 128
segLen = int(math.pow(2, 2))
addNoise = False

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

grid_size = 0.1
hilbert_p = 10
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

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
    tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
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
            CSIn1Tmp = CSIn1Orig[aIndVec]

            sortCSIa1[ii] = np.mean(CSIa1Tmp)  ## Metric 1: Mean
            sortCSIb1[jj - permLen] = np.mean(CSIb1Tmp)  # 只赋值一次
            sortCSIe1[jj - permLen] = np.mean(CSIe1Tmp)
            sortNoise[ii - permLen] = np.mean(CSIn1Tmp)

    # sortCSIa1是原始算法中排序前的数据
    sortCSIa1 = np.log10(np.abs(sortCSIa1) + 0.1)
    sortCSIb1 = np.log10(np.abs(sortCSIb1) + 0.1)
    sortCSIe1 = np.log10(np.abs(sortCSIe1) + 0.1)
    sortNoise = np.log10(np.abs(sortNoise) + 0.1)

    # 取原数据的一部分来reshape
    sortCSIa1Reshape = sortCSIa1[0:segLen * int(len(sortCSIa1) / segLen)]
    sortCSIb1Reshape = sortCSIb1[0:segLen * int(len(sortCSIb1) / segLen)]
    sortCSIe1Reshape = sortCSIe1[0:segLen * int(len(sortCSIe1) / segLen)]
    sortNoiseReshape = sortNoise[0:segLen * int(len(sortNoise) / segLen)]

    sortCSIa1Reshape = sortCSIa1Reshape.reshape(int(len(sortCSIa1Reshape) / segLen), segLen)
    sortCSIb1Reshape = sortCSIb1Reshape.reshape(int(len(sortCSIb1Reshape) / segLen), segLen)
    sortCSIe1Reshape = sortCSIe1Reshape.reshape(int(len(sortCSIe1Reshape) / segLen), segLen)
    sortNoiseReshape = sortNoiseReshape.reshape(int(len(sortNoiseReshape) / segLen), segLen)

    # sortCSIa1 = []
    # sortCSIb1 = []
    # sortCSIe1 = []
    # sortNoise = []

    # 归一化
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

    # sortCSIa1 = genCoordinate(sortCSIa1)
    # sortCSIb1 = genCoordinate(sortCSIb1)
    # sortCSIe1 = genCoordinate(sortCSIe1)
    # sortNoise = genCoordinate(sortNoise)

    sortCSIa1 = genCoordinate(sortCSIa1Reshape)
    sortCSIb1 = genCoordinate(sortCSIb1Reshape)
    sortCSIe1 = genCoordinate(sortCSIe1Reshape)
    sortNoise = genCoordinate(sortNoiseReshape)

    # 最后各自的密钥
    a_list = []
    b_list = []
    e_list = []
    n_list = []

    projCSIa1XY = []
    projCSIb1XY = []
    projCSIe1XY = []
    projCSIn1XY = []

    for i in range(len(sortCSIa1)):
        projCSIa1XY.append([sortCSIa1[i][0], sortCSIa1[i][1]])
        projCSIb1XY.append([sortCSIb1[i][0], sortCSIb1[i][1]])
        projCSIe1XY.append([sortCSIe1[i][0], sortCSIe1[i][1]])
        projCSIn1XY.append([sortNoise[i][0], sortNoise[i][1]])

    a_list = list(listToHilbertCurveIndexWithFixedParams(projCSIa1XY, hilbert_p, grid_size))
    b_list = list(listToHilbertCurveIndexWithFixedParams(projCSIb1XY, hilbert_p, grid_size))
    e_list = list(listToHilbertCurveIndexWithFixedParams(projCSIe1XY, hilbert_p, grid_size))
    n_list = list(listToHilbertCurveIndexWithFixedParams(projCSIn1XY, hilbert_p, grid_size))

    # 转为二进制
    # for i in range(len(alignStr1)):
    #     a_list += bin(alignStr1[i])[2:]
    # for i in range(len(alignStr2)):
    #     b_list += bin(alignStr2[i])[2:]
    # for i in range(len(alignStr3)):
    #     e_list += bin(alignStr3[i])[2:]
    # for i in range(len(alignStr4)):
    #     n_list += bin(alignStr4[i])[2:]
    #
    # # 对齐密钥，随机补全
    # for i in range(len(a_list) - len(e_list)):
    #     e_list += str(np.random.randint(0, 2))
    # for i in range(len(a_list) - len(n_list)):
    #     n_list += str(np.random.randint(0, 2))

    print("keys of a:", len(a_list), a_list)
    print("keys of b:", len(b_list), b_list)
    print("keys of e:", len(e_list), e_list)
    print("keys of n:", len(n_list), n_list)

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

    # for i in range(len(a_list)):
    #     codings += bin(a_list[i]) + "\n"

# with open('../edit_distance/evaluations/key.txt', 'a', ) as f:
#     f.write(codings)

print("a-b all", correctSum, "/", originSum, "=", correctSum / originSum)
print("a-e all", randomSum, "/", originSum, "=", randomSum / originSum)
print("a-n all", noiseSum, "/", originSum, "=", noiseSum / originSum)
print("a-b whole match", correctWholeSum, "/", originWholeSum, "=", correctWholeSum / originWholeSum)
print("a-e whole match", randomWholeSum, "/", originWholeSum, "=", randomWholeSum / originWholeSum)
print("a-n whole match", noiseWholeSum, "/", originWholeSum, "=", noiseWholeSum / originWholeSum)
print("times", times)
print(grid_size, hilbert_p)
