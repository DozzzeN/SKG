import csv
import math
import time
from tkinter import messagebox

import numpy as np
from dtw import dtw
from dtw import accelerated_dtw
from matplotlib import pyplot as plt
from scipy.fft import dct
from scipy.io import loadmat
from scipy.spatial import distance
from scipy.stats import pearsonr, boxcox

import newhope


def search(data, p):
    for i in range(len(data)):
        if p == data[i]:
            return i
    return len(data)


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
        # r = np.sum(data1[i] * data1[i] + data2[i] * data2[i])
        # x_list.append(np.exp(-0.5 * r))
        # x_list.append(np.exp(-0.5 * r))
        r = data2[i] / data1[i]
        x_list.append(math.atan(r) / math.pi + 0.5)
        x_list.append(math.atan(r) / math.pi + 0.5)

    return x_list


fileName = ["../../data/data_mobile_indoor_1.mat",
            "../../data/data_mobile_outdoor_1.mat",
            "../../data/data_static_outdoor_1.mat",
            "../../data/data_static_indoor_1.mat"
            ]

# 样本排序，无噪音，用value做匹配，安全性好
# mi1 		0.7434 		0.0 		 1.6 		 1.1894 	 sort the value
# mo1 		0.6979 		0.0 		 1.6 		 1.1166 	 sort the value
# so1 		0.7068 		0.0 		 1.6 		 1.1309 	 sort the value
# si1 		0.7278 		0.0 		 1.6 		 1.1645 	 sort the value

# 样本排序，无噪音，用index做匹配，安全性差（无意义，都是1到128）
# mi1 		0.9532 		0.0 		 1.6 		 1.5252		 sort the value
# mo1 		0.9939 		0.0 		 1.6 		 1.5903		 sort the value
# so1 		0.9092 		0.0 		 1.6 		 1.4547		 sort the value
# si1 		0.9091 		0.0 		 1.6 		 1.4545		 sort the value

# 样本索引，无噪音，用index做匹配，安全性好
# mi1       0.7888      0.0          1.6         1.262       index
# mo1       0.7209      0.0          1.6         1.1534      index
# so1       0.7179      0.0          1.6         1.1487      index
# si1       0.7512      0.0          1.6         1.202       index

# 采用NewHope算法
# mi1 		 1.0 		 1.0 		 2.0 		 2.0 		 NewHope
# mo1 		 1.0 		 1.0 		 2.0 		 2.0 		 NewHope
# so1 		 1.0 		 1.0 		 2.0 		 2.0 		 NewHope
# si1 		 1.0 		 1.0 		 2.0 		 2.0 		 NewHope

isShow = True
print("file", "\t", "bit", "\t", "key", "\t", "KGR", "\t", "KGR with error free", "\t", "mode")
for f in fileName:
    # print(f)
    rawData = loadmat(f)
    CSIa1Orig = rawData['A'][:, 0]
    CSIb1Orig = rawData['A'][:, 1]
    dataLen = len(CSIa1Orig)

    segLen = 4
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
    withoutSort = False
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
        CSIe2Orig = loadmat("../../skyglow/Scenario2-Office-LoS-eve_NLoS/data_eave_LOS_EVE_NLOS.mat")['A'][:, 0]

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

        # other transformations
        # tmpCSIa1 = np.array(integral_sq_derivative_increment(tmpCSIa1, tmpNoise)) * tmpCSIa1
        # tmpCSIb1 = np.array(integral_sq_derivative_increment(tmpCSIb1, tmpNoise)) * tmpCSIb1
        # tmpCSIe1 = np.array(integral_sq_derivative_increment(tmpCSIe1, tmpNoise)) * tmpCSIe1
        # tmpCSIe2 = np.array(integral_sq_derivative_increment(tmpCSIe2, tmpNoise)) * tmpCSIe2
        # print("correlation a-b", pearsonr(tmpCSIa1, tmpCSIb1)[0])
        # print("correlation a-e1", pearsonr(tmpCSIa1, tmpCSIe1)[0])
        # print("correlation a-e2", pearsonr(tmpCSIa1, tmpCSIe2)[0])
        # print("correlation a-n1", pearsonr(tmpCSIa1, tmpNoise1)[0])
        # print("correlation a-n2", pearsonr(tmpCSIa1, tmpNoise2)[0])
        # print("correlation a-n3", pearsonr(tmpCSIa1, tmpNoise3)[0])
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

        # NewHope key exchange protocol - h1 and h2 serve as the private keys
        s1 = tmpCSIa1Ind
        s1, (b, a) = newhope.keygen(s1)
        s2 = tmpCSIb1Ind
        keyb, (u, r), v_prime = newhope.sharedB(s2, (b, a))
        keya, v = newhope.sharedA((u, r), s1)

        a_list_number = np.tile(keya, int(256 / len(keyb)))
        b_list_number = np.tile(keyb, int(256 / len(keyb)))
        e1_list_number = np.random.randint(0, 256, 256)
        e2_list_number = np.random.randint(0, 256, 256)
        n1_list_number = np.random.randint(0, 256, 256)
        n2_list_number = np.random.randint(0, 256, 256)
        n3_list_number = np.random.randint(0, 256, 256)

        # print(sum(abs(np.array(v) - np.array(v_prime))))
        # print(keyb)
        # print(keya)
        # exit()

        # minEpiIndClosenessLsb = np.zeros(int(keyLen / segLen), dtype=int)
        # minEpiIndClosenessLse1 = np.zeros(int(keyLen / segLen), dtype=int)
        # minEpiIndClosenessLse2 = np.zeros(int(keyLen / segLen), dtype=int)
        # minEpiIndClosenessLsn1 = np.zeros(int(keyLen / segLen), dtype=int)
        # minEpiIndClosenessLsn2 = np.zeros(int(keyLen / segLen), dtype=int)
        # minEpiIndClosenessLsn3 = np.zeros(int(keyLen / segLen), dtype=int)
        #
        # tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)

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

        # permutation = list(range(int(keyLen / segLen)))
        # combineMetric = list(zip(tmpCSIa1IndReshape, permutation))
        # np.random.seed(staInd)
        # np.random.shuffle(combineMetric)
        # tmpCSIa1IndReshape, permutation = zip(*combineMetric)
        # tmpCSIa1Ind = np.hstack((tmpCSIa1IndReshape))
        #
        # for i in range(int(keyLen / segLen)):
        #     epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]
        #
        #     epiIndClosenessLsb = np.zeros(int(keyLen / segLen))
        #     epiIndClosenessLse1 = np.zeros(int(keyLen / segLen))
        #     epiIndClosenessLse2 = np.zeros(int(keyLen / segLen))
        #     epiIndClosenessLsn1 = np.zeros(int(keyLen / segLen))
        #     epiIndClosenessLsn2 = np.zeros(int(keyLen / segLen))
        #     epiIndClosenessLsn3 = np.zeros(int(keyLen / segLen))
        #
        #     for j in range(int(keyLen / segLen)):
        #         epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
        #         epiInde1 = tmpCSIe1Ind[j * segLen: (j + 1) * segLen]
        #         epiInde2 = tmpCSIe2Ind[j * segLen: (j + 1) * segLen]
        #         epiIndn1 = tmpCSIn1Ind[j * segLen: (j + 1) * segLen]
        #         epiIndn2 = tmpCSIn2Ind[j * segLen: (j + 1) * segLen]
        #         epiIndn3 = tmpCSIn3Ind[j * segLen: (j + 1) * segLen]
        #
        #         epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))
        #         epiIndClosenessLse1[j] = sum(abs(epiInde1 - np.array(epiInda1)))
        #         epiIndClosenessLse2[j] = sum(abs(epiInde2 - np.array(epiInda1)))
        #         epiIndClosenessLsn1[j] = sum(abs(epiIndn1 - np.array(epiInda1)))
        #         epiIndClosenessLsn2[j] = sum(abs(epiIndn2 - np.array(epiInda1)))
        #         epiIndClosenessLsn3[j] = sum(abs(epiIndn3 - np.array(epiInda1)))
        #
        #     minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)
        #     minEpiIndClosenessLse1[i] = np.argmin(epiIndClosenessLse1)
        #     minEpiIndClosenessLse2[i] = np.argmin(epiIndClosenessLse2)
        #     minEpiIndClosenessLsn1[i] = np.argmin(epiIndClosenessLsn1)
        #     minEpiIndClosenessLsn2[i] = np.argmin(epiIndClosenessLsn2)
        #     minEpiIndClosenessLsn3[i] = np.argmin(epiIndClosenessLsn3)
        #
        # # a_list_number = list(range(int(keyLen / segLen)))
        # a_list_number = list(permutation)
        # b_list_number = list(minEpiIndClosenessLsb)
        # e1_list_number = list(minEpiIndClosenessLse1)
        # e2_list_number = list(minEpiIndClosenessLse2)
        # n1_list_number = list(minEpiIndClosenessLsn1)
        # n2_list_number = list(minEpiIndClosenessLsn2)
        # n3_list_number = list(minEpiIndClosenessLsn3)

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

        # coding = ""
        # for i in range(len(a_list)):
        #     coding += a_list[i]
        # codings += coding + "\n"
        #
        # with open('./key/' + fileName + ".txt", 'a', ) as f:
        #     f.write(codings)

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
