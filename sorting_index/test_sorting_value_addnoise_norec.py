import csv
import math
import time

import numpy as np
from dtw import dtw
from dtw import accelerated_dtw
from matplotlib import pyplot as plt
from scipy.fft import dct
from scipy.io import loadmat
from scipy.spatial import distance
from scipy.stats import pearsonr, boxcox

from zca import ZCA


def search(data, p):
    for i in range(len(data)):
        if p == data[i]:
            return i


# 定义计算离散点积分的函数
def integral_from_start(x, y):
    import scipy
    from scipy.integrate import simps  # 用于计算积分
    integrals = []
    for i in range(len(y)):  # 计算梯形的面积，由于是累加，所以是切片"i+1"
        integrals.append(scipy.integrate.trapz(y[:i + 1], x[:i + 1]))
    return integrals


# 定义计算离散点导数的函数
def derivative(x, y):  # x, y的类型均为列表
    diff_x = []  # 用来存储x列表中的两数之差
    for i, j in zip(x[0::], x[1::]):
        diff_x.append(j - i)

    diff_y = []  # 用来存储y列表中的两数之差
    for i, j in zip(y[0::], y[1::]):
        diff_y.append(j - i)

    slopes = []  # 用来存储斜率
    for i in range(len(diff_y)):
        slopes.append(diff_y[i] / diff_x[i])

    deriv = []  # 用来存储一阶导数
    for i, j in zip(slopes[0::], slopes[1::]):
        deriv.append((0.5 * (i + j)))  # 根据离散点导数的定义，计算并存储结果
    deriv.insert(0, slopes[0])  # (左)端点的导数即为与其最近点的斜率
    deriv.append(slopes[-1])  # (右)端点的导数即为与其最近点的斜率
    return deriv  # 返回存储一阶导数结果的列表


def integral_sq_derivative_increment(data, noise):
    index = list(range(len(data)))
    intgrl = integral_from_start(index, data)
    # square = np.power(intgrl, 2)
    square = intgrl + noise
    diff = derivative(index, square)
    return diff


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


def normal2uniform(data):
    data1 = data[:int(len(data) / 2)]
    data2 = data[int(len(data) / 2):]
    data_reshape = np.array(data[0: 2 * int(len(data) / 2)])
    data_reshape = data_reshape.reshape(int(len(data_reshape) / 2), 2)
    x_list = []
    for i in range(len(data_reshape)):
        # r = np.sum(np.square(data_reshape[i]))
        r = np.sum(data1[i] * data1[i] + data2[i] * data2[i])
        x_list.append(np.exp(-0.5 * r))
        x_list.append(np.exp(-0.5 * r))
        # r = data2[i] / data1[i]
        # x_list.append(math.atan(r) / math.pi + 0.5)
        # x_list.append(math.atan(r) / math.pi + 0.5)

    return x_list


rawData = loadmat("../data/data_static_outdoor_1.mat")

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
# CSIa1Orig = rawData['A'][:, 0][9000:20000]
# CSIb1Orig = rawData['A'][:, 1][9000:20000]
# stalking attack
CSIe2Orig = loadmat("../data/data_static_indoor_1.mat")['A'][:, 0]
dataLen = min(len(CSIe2Orig), len(CSIa1Orig))

segLen = 5
keyLen = 256 * segLen

originSum = 0
correctSum = 0
randomSum1 = 0
randomSum2 = 0
noiseSum1 = 0
noiseSum2 = 0
noiseSum3 = 0
noiseSum4 = 0

originDecSum = 0
correctDecSum = 0
randomDecSum1 = 0
randomDecSum2 = 0
noiseDecSum1 = 0
noiseDecSum2 = 0
noiseDecSum3 = 0
noiseDecSum4 = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum1 = 0
randomWholeSum2 = 0
noiseWholeSum1 = 0
noiseWholeSum2 = 0
noiseWholeSum3 = 0
noiseWholeSum4 = 0

times = 0
# multiplicative noise
addNoise = "mul"

ab_dist = 0
ae1_dist = 0
ae2_dist = 0
an1_dist = 0
an2_dist = 0
an3_dist = 0
an4_dist = 0

ab_corr = 0
ae1_corr = 0
ae2_corr = 0
an1_corr = 0
an2_corr = 0
an3_corr = 0
an4_corr = 0

# static indoor
# for staInd in range(0, int(dataLen / 3), keyLen):
for staInd in range(0, dataLen, keyLen):
    start = time.time()

    endInd = staInd + keyLen
    print("range:", staInd, endInd)
    if endInd >= len(CSIa1Orig):
        break
    times += 1

    np.random.seed(0)
    CSIa1Orig = rawData['A'][:, 0][0: dataLen]
    CSIb1Orig = rawData['A'][:, 1][0: dataLen]

    # imitation attack
    CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=len(CSIa1Orig))

    tmpNoise1 = []
    tmpNoise2 = []
    tmpNoise3 = []
    tmpNoise4 = []

    if addNoise == "add":
        # CSIa1Orig = CSIa1Orig - np.mean(CSIa1Orig)
        # CSIb1Orig = CSIb1Orig - np.mean(CSIb1Orig)
        # CSIe1Orig = CSIe1Orig - np.mean(CSIe1Orig)
        # CSIe2Orig = CSIe2Orig - np.mean(CSIe2Orig)

        # noiseOrig = np.random.normal(0, np.std(CSIa1Orig), size=len(CSIa1Orig))
        noiseOrig = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=len(CSIa1Orig))
        CSIa1Orig = CSIa1Orig + noiseOrig
        CSIb1Orig = CSIb1Orig + noiseOrig
        CSIe1Orig = CSIe1Orig + noiseOrig
        CSIe2Orig = CSIe2Orig + noiseOrig[0:len(CSIe2Orig)]
        CSIn1Orig = noiseOrig

        tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
        tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
        tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
        tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]
        tmpNoise = CSIn1Orig[range(staInd, endInd, 1)]
    elif addNoise == "mul":
        CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
        CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')
        CSIe1Orig = smooth(np.array(CSIe1Orig), window_len=30, window='flat')
        CSIe2Orig = smooth(np.array(CSIe2Orig), window_len=30, window='flat')

        tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
        tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
        tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
        tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]

        randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
        # randomMatrix = np.random.normal(0, 1, size=(keyLen, keyLen))
        tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
        tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
        tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
        tmpCSIe2 = tmpCSIe2 - np.mean(tmpCSIe2)
        tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
        tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)
        tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix)
        tmpCSIe2 = np.matmul(tmpCSIe2, randomMatrix)
        # inference attack
        tmpNoise1 = randomMatrix.mean(axis=0)  # 按列求均值
        tmpNoise2 = randomMatrix.mean(axis=1)  # 按行求均值
        tmpNoise3 = np.matmul(np.ones(keyLen), randomMatrix)
        tmpNoise4 = np.random.normal(loc=np.mean(tmpCSIa1), scale=np.std(tmpCSIa1, ddof=1), size=len(tmpCSIa1))
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
    tmpNoise4 = (tmpNoise4 - np.min(tmpNoise4)) / (np.max(tmpNoise4) - np.min(tmpNoise4))

    # other transformations
    # tmpCSIa1 = np.array(integral_sq_derivative_increment(tmpCSIa1, tmpNoise)) * tmpCSIa1
    # tmpCSIb1 = np.array(integral_sq_derivative_increment(tmpCSIb1, tmpNoise)) * tmpCSIb1
    # tmpCSIe1 = np.array(integral_sq_derivative_increment(tmpCSIe1, tmpNoise)) * tmpCSIe1
    # tmpCSIe2 = np.array(integral_sq_derivative_increment(tmpCSIe2, tmpNoise)) * tmpCSIe2
    print("correlation a-b", pearsonr(tmpCSIa1, tmpCSIb1)[0])
    print("correlation a-e1", pearsonr(tmpCSIa1, tmpCSIe1)[0])
    print("correlation a-e2", pearsonr(tmpCSIa1, tmpCSIe2)[0])
    print("correlation a-n1", pearsonr(tmpCSIa1, tmpNoise1)[0])
    print("correlation a-n2", pearsonr(tmpCSIa1, tmpNoise2)[0])
    print("correlation a-n3", pearsonr(tmpCSIa1, tmpNoise3)[0])
    print("correlation a-n4", pearsonr(tmpCSIa1, tmpNoise4)[0])
    # tmpNoise = np.array(integral_sq_derivative_increment(np.ones(keyLen), tmpNoise)) * np.ones(keyLen)

    # tmpCSIe1 = np.random.normal(loc=np.mean(tmpCSIe1), scale=np.std(tmpCSIe1, ddof=1), size=len(tmpCSIe1))

    # power
    # noise = np.random.uniform(0, 1, keyLen)
    # tmpCSIa1 = np.power(np.abs(tmpCSIa1), noise)
    # tmpCSIb1 = np.power(np.abs(tmpCSIb1), noise)
    # tmpCSIe1 = np.power(np.abs(tmpCSIe1), noise)
    # tmpNoise = np.power(np.abs(tmpNoise), noise)

    # box-muller
    # tmpCSIa1 = boxcox(np.abs(tmpCSIa1))[0]
    # tmpCSIb1 = boxcox(np.abs(tmpCSIb1))[0]
    # tmpCSIe1 = boxcox(np.abs(tmpCSIe1))[0]
    # tmpNoise = boxcox(np.abs(tmpNoise))[0]
    # tmpCSIa1 = normal2uniform(tmpCSIa1)
    # tmpCSIb1 = normal2uniform(tmpCSIb1)
    # tmpCSIe1 = normal2uniform(tmpCSIe1)
    # tmpNoise = normal2uniform(tmpNoise)

    # plt.figure()
    # plt.plot(np.sort(tmpCSIa1))
    # plt.show()

    # 最后各自的密钥
    a_list = []
    b_list = []
    e1_list = []
    e2_list = []
    n1_list = []
    n2_list = []
    n3_list = []
    n4_list = []

    tmpCSIa1Ind = np.array(tmpCSIa1)
    tmpCSIb1Ind = np.array(tmpCSIb1)
    tmpCSIe1Ind = np.array(tmpCSIe1)
    tmpCSIe2Ind = np.array(tmpCSIe2)
    tmpCSIn1Ind = np.array(tmpNoise1)
    tmpCSIn2Ind = np.array(tmpNoise2)
    tmpCSIn3Ind = np.array(tmpNoise3)
    tmpCSIn4Ind = np.array(tmpNoise4)

    minEpiIndClosenessLsb = np.zeros(int(keyLen / segLen), dtype=int)
    minEpiIndClosenessLse1 = np.zeros(int(keyLen / segLen), dtype=int)
    minEpiIndClosenessLse2 = np.zeros(int(keyLen / segLen), dtype=int)
    minEpiIndClosenessLsn1 = np.zeros(int(keyLen / segLen), dtype=int)
    minEpiIndClosenessLsn2 = np.zeros(int(keyLen / segLen), dtype=int)
    minEpiIndClosenessLsn3 = np.zeros(int(keyLen / segLen), dtype=int)
    minEpiIndClosenessLsn4 = np.zeros(int(keyLen / segLen), dtype=int)

    tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)

    # fig = plt.figure()
    # ax = fig.gca(projection="3d")
    # for i in range(len(tmpCSIa1IndReshape)):
    #     ax.plot(np.ones(len(tmpCSIa1IndReshape[i])) * i, list(range(len(tmpCSIa1IndReshape[i]))), tmpCSIa1IndReshape[i])
    # plt.show()

    # tmpCSIe1IndReshape = np.array(tmpCSIe1Ind).reshape(int(keyLen / segLen), segLen)
    # fig = plt.figure()
    # ax = fig.gca(projection="3d")
    # for i in range(len(tmpCSIe1IndReshape)):
    #     ax.plot(np.ones(len(tmpCSIe1IndReshape[i])) * i, list(range(len(tmpCSIe1IndReshape[i]))), tmpCSIe1IndReshape[i])
    # plt.show()

    permutation = list(range(int(keyLen / segLen)))
    combineMetric = list(zip(tmpCSIa1IndReshape, permutation))
    np.random.seed(staInd)
    np.random.shuffle(combineMetric)
    tmpCSIa1IndReshape, permutation = zip(*combineMetric)
    tmpCSIa1Ind = np.hstack((tmpCSIa1IndReshape))

    for i in range(int(keyLen / segLen)):
        epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]

        epiIndClosenessLsb = np.zeros(int(keyLen / segLen))
        epiIndClosenessLse1 = np.zeros(int(keyLen / segLen))
        epiIndClosenessLse2 = np.zeros(int(keyLen / segLen))
        epiIndClosenessLsn1 = np.zeros(int(keyLen / segLen))
        epiIndClosenessLsn2 = np.zeros(int(keyLen / segLen))
        epiIndClosenessLsn3 = np.zeros(int(keyLen / segLen))
        epiIndClosenessLsn4 = np.zeros(int(keyLen / segLen))

        for j in range(int(keyLen / segLen)):
            epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
            epiInde1 = tmpCSIe1Ind[j * segLen: (j + 1) * segLen]
            epiInde2 = tmpCSIe2Ind[j * segLen: (j + 1) * segLen]
            epiIndn1 = tmpCSIn1Ind[j * segLen: (j + 1) * segLen]
            epiIndn2 = tmpCSIn2Ind[j * segLen: (j + 1) * segLen]
            epiIndn3 = tmpCSIn3Ind[j * segLen: (j + 1) * segLen]
            epiIndn4 = tmpCSIn4Ind[j * segLen: (j + 1) * segLen]

            # epiIndClosenessLsb[j] = abs(sum(epiIndb1) - sum(epiInda1))
            # distance = lambda x, y: np.abs(x - y)
            # epiIndClosenessLsb[j] = accelerated_dtw(epiIndb1, epiInda1, dist=distance)[0]
            # epiIndClosenessLse1[j] = accelerated_dtw(epiInde1, epiInda1, dist=distance)[0]
            # epiIndClosenessLse2[j] = accelerated_dtw(epiInde2, epiInda1, dist=distance)[0]
            # epiIndClosenessLsn1[j] = accelerated_dtw(epiIndn1, epiInda1, dist=distance)[0]
            # epiIndClosenessLsn2[j] = accelerated_dtw(epiIndn2, epiInda1, dist=distance)[0]
            # epiIndClosenessLsn3[j] = accelerated_dtw(epiIndn3, epiInda1, dist=distance)[0]
            # epiIndClosenessLsn4[j] = accelerated_dtw(epiIndn4, epiInda1, dist=distance)[0]
            epiIndClosenessLsb[j] = distance.cityblock(epiIndb1, epiInda1)
            epiIndClosenessLse1[j] = distance.cityblock(epiInde1, epiInda1)
            epiIndClosenessLse2[j] = distance.cityblock(epiInde2, epiInda1)
            epiIndClosenessLsn1[j] = distance.cityblock(epiIndn1, epiInda1)
            epiIndClosenessLsn2[j] = distance.cityblock(epiIndn2, epiInda1)
            epiIndClosenessLsn3[j] = distance.cityblock(epiIndn3, epiInda1)
            epiIndClosenessLsn4[j] = distance.cityblock(epiIndn4, epiInda1)

        minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)
        minEpiIndClosenessLse1[i] = np.argmin(epiIndClosenessLse1)
        minEpiIndClosenessLse2[i] = np.argmin(epiIndClosenessLse2)
        minEpiIndClosenessLsn1[i] = np.argmin(epiIndClosenessLsn1)
        minEpiIndClosenessLsn2[i] = np.argmin(epiIndClosenessLsn2)
        minEpiIndClosenessLsn3[i] = np.argmin(epiIndClosenessLsn3)
        minEpiIndClosenessLsn4[i] = np.argmin(epiIndClosenessLsn4)

    # a_list_number = list(range(int(keyLen / segLen)))
    a_list_number = list(permutation)
    b_list_number = list(minEpiIndClosenessLsb)
    e1_list_number = list(minEpiIndClosenessLse1)
    e2_list_number = list(minEpiIndClosenessLse2)
    n1_list_number = list(minEpiIndClosenessLsn1)
    n2_list_number = list(minEpiIndClosenessLsn2)
    n3_list_number = list(minEpiIndClosenessLsn3)
    n4_list_number = list(minEpiIndClosenessLsn4)

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
    for i in range(len(n4_list_number)):
        number = bin(n4_list_number[i])[2:].zfill(int(np.log2(len(n4_list_number))))
        n4_list += number

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
    for i in range(len(a_list) - len(n4_list)):
        n4_list += str(np.random.randint(0, 2))

    # print("keys of a:", len(a_list), a_list)
    print("keys of a:", len(a_list_number), a_list_number)
    # print("keys of b:", len(b_list), b_list)
    print("keys of b:", len(b_list_number), b_list_number)
    # print("keys of e:", len(e_list), e_list)
    print("keys of e1:", len(e1_list_number), e1_list_number)
    # print("keys of e:", len(e_list), e_list)
    print("keys of e2:", len(e2_list_number), e2_list_number)
    # print("keys of n1:", len(n1_list), n1_list)
    print("keys of n1:", len(n1_list_number), n1_list_number)
    # print("keys of n2:", len(n2_list), n2_list)
    print("keys of n2:", len(n2_list_number), n2_list_number)
    # print("keys of n3:", len(n3_list), n3_list)
    print("keys of n3:", len(n3_list_number), n3_list_number)
    # print("keys of n4:", len(n4_list), n4_list)
    print("keys of n4:", len(n4_list_number), n4_list_number)

    sum1 = min(len(a_list), len(b_list))
    sum2 = 0
    sum31 = 0
    sum32 = 0
    sum41 = 0
    sum42 = 0
    sum43 = 0
    sum44 = 0
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
    for i in range(min(len(a_list), len(n4_list))):
        sum44 += (a_list[i] == n4_list[i])

    # 计算平均错误距离
    tmp_dist = 0
    for i in range(len(b_list_number)):
        real_pos = search(a_list_number, b_list_number[i])
        guess_pos = i
        tmp_dist += abs(real_pos - guess_pos)
    ab_dist += (tmp_dist / int(keyLen / segLen))

    tmp_dist = 0
    for i in range(len(e1_list_number)):
        real_pos = search(a_list_number, e1_list_number[i])
        guess_pos = i
        tmp_dist += abs(real_pos - guess_pos)
    ae1_dist += (tmp_dist / int(keyLen / segLen))

    tmp_dist = 0
    for i in range(len(e2_list_number)):
        real_pos = search(a_list_number, e2_list_number[i])
        guess_pos = i
        tmp_dist += abs(real_pos - guess_pos)
    ae2_dist += (tmp_dist / int(keyLen / segLen))

    tmp_dist = 0
    for i in range(len(n1_list_number)):
        real_pos = search(a_list_number, n1_list_number[i])
        guess_pos = i
        tmp_dist += abs(real_pos - guess_pos)
    an1_dist += (tmp_dist / int(keyLen / segLen))

    tmp_dist = 0
    for i in range(len(n2_list_number)):
        real_pos = search(a_list_number, n2_list_number[i])
        guess_pos = i
        tmp_dist += abs(real_pos - guess_pos)
    an2_dist += (tmp_dist / int(keyLen / segLen))

    tmp_dist = 0
    for i in range(len(n3_list_number)):
        real_pos = search(a_list_number, n3_list_number[i])
        guess_pos = i
        tmp_dist += abs(real_pos - guess_pos)
    an3_dist += (tmp_dist / int(keyLen / segLen))

    tmp_dist = 0
    for i in range(len(n4_list_number)):
        real_pos = search(a_list_number, n4_list_number[i])
        guess_pos = i
        tmp_dist += abs(real_pos - guess_pos)
    an4_dist += (tmp_dist / int(keyLen / segLen))

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
    an4_corr += 0 if np.isnan(pearsonr(a_list_number, n4_list_number)[0]) \
        else pearsonr(a_list_number, n4_list_number)[0]

    print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
    print("a-e1", sum31, sum31 / sum1)
    print("a-e2", sum32, sum32 / sum1)
    print("a-n1", sum41, sum41 / sum1)
    print("a-n2", sum42, sum42 / sum1)
    print("a-n3", sum43, sum43 / sum1)
    print("a-n4", sum44, sum44 / sum1)
    originSum += sum1
    correctSum += sum2
    randomSum1 += sum31
    randomSum2 += sum32
    noiseSum1 += sum41
    noiseSum2 += sum42
    noiseSum3 += sum43
    noiseSum4 += sum44

    decSum1 = min(len(a_list_number), len(b_list_number))
    decSum2 = 0
    decSum31 = 0
    decSum32 = 0
    decSum41 = 0
    decSum42 = 0
    decSum43 = 0
    decSum44 = 0
    for i in range(0, decSum1):
        decSum2 += (a_list_number[i] == b_list_number[i])
    for i in range(min(len(a_list_number), len(e1_list_number))):
        decSum31 += (a_list_number[i] == e1_list_number[i])
    for i in range(min(len(a_list_number), len(e2_list_number))):
        decSum32 += (a_list_number[i] == e2_list_number[i])
    for i in range(min(len(a_list_number), len(n1_list_number))):
        decSum41 += (a_list_number[i] == n1_list_number[i])
    for i in range(min(len(a_list_number), len(n2_list_number))):
        decSum42 += (a_list_number[i] == n2_list_number[i])
    for i in range(min(len(a_list_number), len(n3_list_number))):
        decSum43 += (a_list_number[i] == n3_list_number[i])
    for i in range(min(len(a_list_number), len(n4_list_number))):
        decSum44 += (a_list_number[i] == n4_list_number[i])
    if decSum1 == 0:
        continue
    if decSum2 == decSum1:
        print("\033[0;32;40ma-b dec", decSum2, decSum2 / decSum1, "\033[0m")
    else:
        print("\033[0;31;40ma-b dec", "bad", decSum2, decSum2 / decSum1, "\033[0m")
    print("a-e1", decSum31, decSum31 / decSum1)
    print("a-e2", decSum32, decSum32 / decSum1)
    print("a-n1", decSum41, decSum41 / decSum1)
    print("a-n2", decSum42, decSum42 / decSum1)
    print("a-n3", decSum43, decSum43 / decSum1)
    print("a-n4", decSum44, decSum44 / decSum1)
    print("----------------------")
    originDecSum += decSum1
    correctDecSum += decSum2
    randomDecSum1 += decSum31
    randomDecSum2 += decSum32
    noiseDecSum1 += decSum41
    noiseDecSum2 += decSum42
    noiseDecSum3 += decSum43
    noiseDecSum4 += decSum44

    originWholeSum += 1
    correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
    randomWholeSum1 = randomWholeSum1 + 1 if sum31 == sum1 else randomWholeSum1
    randomWholeSum2 = randomWholeSum2 + 1 if sum32 == sum1 else randomWholeSum2
    noiseWholeSum1 = noiseWholeSum1 + 1 if sum41 == sum1 else noiseWholeSum1
    noiseWholeSum2 = noiseWholeSum2 + 1 if sum42 == sum1 else noiseWholeSum2
    noiseWholeSum3 = noiseWholeSum3 + 1 if sum43 == sum1 else noiseWholeSum3
    noiseWholeSum4 = noiseWholeSum4 + 1 if sum44 == sum1 else noiseWholeSum4

    # coding = ""
    # for i in range(len(a_list)):
    #     coding += a_list[i]
    # codings += coding + "\n"
    #
    # with open('./key/' + fileName + ".txt", 'a', ) as f:
    #     f.write(codings)

print("\033[0;32;40ma-b bit agreement rate", correctSum, "/", originSum, "=", round(correctSum / originSum, 10),
      "\033[0m")
print("a-e1 bit agreement rate", randomSum1, "/", originSum, "=", round(randomSum1 / originSum, 10))
print("a-e2 bit agreement rate", randomSum2, "/", originSum, "=", round(randomSum2 / originSum, 10))
print("a-n1 bit agreement rate", noiseSum1, "/", originSum, "=", round(noiseSum1 / originSum, 10))
print("a-n2 bit agreement rate", noiseSum2, "/", originSum, "=", round(noiseSum2 / originSum, 10))
print("a-n3 bit agreement rate", noiseSum3, "/", originSum, "=", round(noiseSum3 / originSum, 10))
print("a-n4 bit agreement rate", noiseSum4, "/", originSum, "=", round(noiseSum4 / originSum, 10))
print("\033[0;32;40ma-b dec agreement rate", correctDecSum, "/", originDecSum, "=",
      round(correctDecSum / originDecSum, 10), "\033[0m")
print("a-e1 dec agreement rate", randomDecSum1, "/", originDecSum, "=", round(randomDecSum1 / originDecSum, 10))
print("a-e2 dec agreement rate", randomDecSum2, "/", originDecSum, "=", round(randomDecSum2 / originDecSum, 10))
print("a-n1 dec agreement rate", noiseDecSum1, "/", originDecSum, "=", round(noiseDecSum1 / originDecSum, 10))
print("a-n2 dec agreement rate", noiseDecSum2, "/", originDecSum, "=", round(noiseDecSum2 / originDecSum, 10))
print("a-n3 dec agreement rate", noiseDecSum3, "/", originDecSum, "=", round(noiseDecSum3 / originDecSum, 10))
print("a-n4 dec agreement rate", noiseDecSum4, "/", originDecSum, "=", round(noiseDecSum4 / originDecSum, 10))
print("\033[0;32;40ma-b key agreement rate", correctWholeSum, "/", originWholeSum, "=",
      round(correctWholeSum / originWholeSum, 10), "\033[0m")
print("a-e1 key agreement rate", randomWholeSum1, "/", originWholeSum, "=", round(randomWholeSum1 / originWholeSum, 10))
print("a-e2 key agreement rate", randomWholeSum2, "/", originWholeSum, "=", round(randomWholeSum2 / originWholeSum, 10))
print("a-n1 key agreement rate", noiseWholeSum1, "/", originWholeSum, "=", round(noiseWholeSum1 / originWholeSum, 10))
print("a-n2 key agreement rate", noiseWholeSum2, "/", originWholeSum, "=", round(noiseWholeSum2 / originWholeSum, 10))
print("a-n3 key agreement rate", noiseWholeSum3, "/", originWholeSum, "=", round(noiseWholeSum3 / originWholeSum, 10))
print("a-n4 key agreement rate", noiseWholeSum4, "/", originWholeSum, "=", round(noiseWholeSum4 / originWholeSum, 10))
print("\033[0;32;40ma-b average distance", round(ab_dist / times, 8), "\033[0m")
print("ae1 average distance", round(ae1_dist / times, 8))
print("ae2 average distance", round(ae2_dist / times, 8))
print("an1 average distance", round(an1_dist / times, 8))
print("an2 average distance", round(an2_dist / times, 8))
print("an3 average distance", round(an3_dist / times, 8))
print("an4 average distance", round(an4_dist / times, 8))
print("\033[0;32;40ma-b average correlation", round(ab_corr / times, 8), "\033[0m")
print("ae1 average correlation", round(ae1_corr / times, 8))
print("ae2 average correlation", round(ae2_corr / times, 8))
print("an1 average correlation", round(an1_corr / times, 8))
print("an2 average correlation", round(an2_corr / times, 8))
print("an3 average correlation", round(an3_corr / times, 8))
print("an4 average correlation", round(an4_corr / times, 8))
print("times", times)
