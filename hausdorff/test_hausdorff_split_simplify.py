import math
import random
import sys
import time
from tkinter import messagebox

import numpy as np
from matplotlib import pyplot as plt
from numpy.fft import fft
from scipy import signal
from scipy.fft import dct
from scipy.io import loadmat


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


# 优化算法的计算复杂度O((n+m)log(n+m)),暂时使用穷举
# 标准单向hausdorff距离
# h(A,B) = max  min ||ai-bj||
#          ai∈A bj∈B
def standard_hd(x, y):
    h1 = 0
    for xi in x:
        shortest = sys.maxsize
        for yi in y:
            d = round(math.pow(xi[0] - yi[0], 2) + math.pow(xi[1] - yi[1], 2), 10)
            if d < shortest:
                shortest = d
        if shortest > h1:
            h1 = shortest

    h2 = 0
    for xi in y:
        shortest = sys.maxsize
        for yi in x:
            d = round(math.pow(xi[0] - yi[0], 2) + math.pow(xi[1] - yi[1], 2), 10)
            if d < shortest:
                shortest = d
        if shortest > h2:
            h2 = shortest
    return max(h1, h2)


# 平均单向hausdorff距离的效果差些
def average_hd(x, y):
    h = 0
    for xi in x:
        shortest = sys.maxsize
        for yi in y:
            d = round(math.pow(xi[0] - yi[0], 2) + math.pow(xi[1] - yi[1], 2), 10)
            if d < shortest:
                shortest = d
        h += shortest
    return h / len(x)


# 添加数组头元素以构成封闭的多边形
def makePolygon(list):
    listPolygon = []
    for i in range(len(list)):
        listTmp = []
        for j in range(len(list[i])):
            listTmp.append(list[i][j])
        listTmp.append(list[i][0])
        listPolygon.append(listTmp)
    return listPolygon


# 将三维数组转为一维数组
def toOneDim(list):
    oneDim = []
    for i in range(len(list)):
        tmp = 0
        for j in range(len(list[i])):
            tmp += (list[i][j][0] + list[i][j][1])
            # tmp += (list[i][j][0] * list[i][j][1])
        oneDim.append(round(tmp, 8))
    return oneDim


fileName = ["../data/data_mobile_indoor_1.mat",
            "../data/data_mobile_outdoor_1.mat",
            "../data/data_static_outdoor_1.mat",
            "../data/data_static_indoor_1.mat"
            ]

# file 	     bit 	         key 	     KGR 	         KGR with error free 	 mode
# mi1 		 0.9537 		 0.4 		 0.125 		     0.1192 		 no sorting
# mo1 		 0.8725 		 0.0 		 0.125 		     0.1091 		 no sorting
# so1 		 0.9548 		 0.0588 	 0.125 		     0.1193 		 no sorting
# si1 		 0.9723 		 0.2361 	 0.125 		     0.1215 		 no sorting

# mi1 		 0.9831 		 0.7 		 0.0893 		 0.0878 		 no sorting
# mo1 		 1.0 		     1.0 		 0.0893 		 0.0893 		 no sorting
# so1 		 0.9802 		 0.5 		 0.0893 		 0.0875 		 no sorting
# si1 		 0.9952 		 0.8269 	 0.0893 		 0.0889 		 no sorting

# mean consistency
# mi1 		 1.0 		     1.0 		 0.0357 		 0.0357 		 no sorting
# mo1 		 1.0 		     1.0 		 0.0357 		 0.0357 		 no sorting
# so1 		 1.0 		     1.0 		 0.0357 		 0.0357 		 no sorting
# si1 		 1.0 		     1.0 		 0.0357 		 0.0357 		 no sorting

# mi1 		 0.9825 		 0.8 		 0.0893 		 0.0877 		 index
# mo1 		 1.0 		     1.0 		 0.0893 		 0.0893 		 index
# so1 		 0.9906 		 0.6667 	 0.0893 		 0.0884 		 index
# si1 		 0.9969 		 0.8462 	 0.0893 		 0.089 		     index

# mean consistency
# mi1 		 1.0 		     1.0 		 0.0357 		 0.0357 		 index
# mo1 		 1.0 		     1.0 		 0.0357 		 0.0357 		 index
# so1 		 1.0 		     1.0 		 0.0357 		 0.0357 		 index
# si1 		 1.0 		     1.0 		 0.0357 		 0.0357 		 index

isShow = False
print("file", "\t", "bit", "\t", "key", "\t", "KGR", "\t", "KGR with error free", "\t", "mode")
for f in fileName:
    # print(f)
    rawData = loadmat(f)
    CSIa1Orig = rawData['A'][:, 0]
    CSIb1Orig = rawData['A'][:, 1]

    dataLen = len(CSIa1Orig)

    segLen = 7
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
            # print("no sorting")
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
            # tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
            # tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
            # tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
            # tmpCSIe2 = tmpCSIe2 - np.mean(tmpCSIe2)

            # mean consistency
            tmpCSIa1 = tmpCSIa1 - (np.mean(tmpCSIa1) - np.mean(tmpCSIb1))
            tmpCSIb1 = tmpCSIb1 - (np.mean(tmpCSIb1) - np.mean(tmpCSIb1))
            tmpCSIe1 = tmpCSIe1 - (np.mean(tmpCSIe1) - np.mean(tmpCSIb1))
            tmpCSIe2 = tmpCSIe2 - (np.mean(tmpCSIe2) - np.mean(tmpCSIb1))

            tmpPulse = signal.square(
                2 * np.pi * 1 / segLen * np.linspace(0, np.pi * 0.5 * int(keyLen / segLen),
                                                     keyLen))  ## Rectangular pulse

            tmpCSIa1 = tmpPulse * tmpCSIa1
            tmpCSIb1 = tmpPulse * tmpCSIb1
            tmpCSIe1 = tmpPulse * tmpCSIe1

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

        tmpCSIa1Back = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1))
        tmpCSIb1Back = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1))
        tmpCSIe1Back = (tmpCSIe1 - np.min(tmpCSIe1)) / (np.max(tmpCSIe1) - np.min(tmpCSIe1))
        tmpCSIe2Back = (tmpCSIe2 - np.min(tmpCSIe2)) / (np.max(tmpCSIe2) - np.min(tmpCSIe2))
        tmpNoise1Back = (tmpNoise1 - np.min(tmpNoise1)) / (np.max(tmpNoise1) - np.min(tmpNoise1))
        tmpNoise2Back = (tmpNoise2 - np.min(tmpNoise2)) / (np.max(tmpNoise2) - np.min(tmpNoise2))
        tmpNoise3Back = (tmpNoise3 - np.min(tmpNoise3)) / (np.max(tmpNoise3) - np.min(tmpNoise3))

        tmpCSIa1 = []
        tmpCSIb1 = []
        tmpCSIe1 = []
        tmpCSIe2 = []
        tmpCSIn1 = []
        tmpCSIn2 = []
        tmpCSIn3 = []

        for i in range(int(keyLen / segLen)):
            tmpCSIa1.append(sum(tmpCSIa1Back[i * segLen:(i + 1) * segLen]))
            tmpCSIb1.append(sum(tmpCSIb1Back[i * segLen:(i + 1) * segLen]))
            tmpCSIe1.append(sum(tmpCSIe1Back[i * segLen:(i + 1) * segLen]))
            tmpCSIe2.append(sum(tmpCSIe2Back[i * segLen:(i + 1) * segLen]))
            tmpCSIn1.append(sum(tmpNoise1Back[i * segLen:(i + 1) * segLen]))
            tmpCSIn2.append(sum(tmpNoise2Back[i * segLen:(i + 1) * segLen]))
            tmpCSIn3.append(sum(tmpNoise3Back[i * segLen:(i + 1) * segLen]))

        # 形成三维数组，其中第三维是一对坐标值
        # 数组的长度由param调节
        param = 3
        step = int(math.pow(2, param))
        tmpCSIa1Reshape = np.array(tmpCSIa1).reshape((int(len(tmpCSIa1) / step / 2), step, 2))
        tmpCSIb1Reshape = np.array(tmpCSIb1).reshape((int(len(tmpCSIb1) / step / 2), step, 2))
        tmpCSIe1Reshape = np.array(tmpCSIe1).reshape((int(len(tmpCSIe1) / step / 2), step, 2))
        tmpCSIe2Reshape = np.array(tmpCSIe2).reshape((int(len(tmpCSIe2) / step / 2), step, 2))
        tmpCSIn1Reshape = np.array(tmpCSIn1).reshape((int(len(tmpCSIn1) / step / 2), step, 2))
        tmpCSIn2Reshape = np.array(tmpCSIn2).reshape((int(len(tmpCSIn2) / step / 2), step, 2))
        tmpCSIn3Reshape = np.array(tmpCSIn3).reshape((int(len(tmpCSIn3) / step / 2), step, 2))

        # 降维以用于后续的排序
        oneDimCSIa1 = toOneDim(tmpCSIa1Reshape)
        oneDimCSIb1 = toOneDim(tmpCSIb1Reshape)
        oneDimCSIe1 = toOneDim(tmpCSIe1Reshape)
        oneDimCSIe2 = toOneDim(tmpCSIe2Reshape)
        oneDimCSIn1 = toOneDim(tmpCSIn1Reshape)
        oneDimCSIn2 = toOneDim(tmpCSIn2Reshape)
        oneDimCSIn3 = toOneDim(tmpCSIn3Reshape)

        # 计算hd距离和多边形的顺序无关，可以任意洗牌
        CSIa1Back = [[] for _ in range(len(tmpCSIa1Reshape))]
        CSIb1Back = [[] for _ in range(len(tmpCSIb1Reshape))]
        CSIe1Back = [[] for _ in range(len(tmpCSIe1Reshape))]
        CSIe2Back = [[] for _ in range(len(tmpCSIe2Reshape))]
        CSIn1Back = [[] for _ in range(len(tmpCSIn1Reshape))]
        CSIn2Back = [[] for _ in range(len(tmpCSIn2Reshape))]
        CSIn3Back = [[] for _ in range(len(tmpCSIn3Reshape))]

        rand_out_polygon = list(range(len(tmpCSIa1Reshape)))
        rand_in_polygon = list(range(step))

        random.shuffle(rand_out_polygon)
        random.shuffle(rand_in_polygon)
        for i in range(len(tmpCSIa1Reshape)):
            for j in range(step):
                CSIa1Back[i].append(tmpCSIa1Reshape[rand_out_polygon[i]][rand_in_polygon[j]])

        # 在数组a后面加上a[0]使之成为一个首尾封闭的多边形
        sortCSIa1Add = makePolygon(tmpCSIa1Reshape)
        sortCSIb1Add = makePolygon(tmpCSIb1Reshape)
        sortCSIe1Add = makePolygon(tmpCSIe1Reshape)
        sortCSIe2Add = makePolygon(tmpCSIe2Reshape)
        sortCSIn1Add = makePolygon(tmpCSIn1Reshape)
        sortCSIn2Add = makePolygon(tmpCSIn2Reshape)
        sortCSIn3Add = makePolygon(tmpCSIn3Reshape)

        # 初始化各个计算出的hd值
        ab_max = 0
        ae1_max = 0
        ae2_max = 0
        an1_max = 0
        an2_max = 0
        an3_max = 0

        # 最后各自的密钥
        a_list = []
        b_list = []
        e1_list = []
        e2_list = []
        n1_list = []
        n2_list = []
        n3_list = []

        a_list_number = []
        b_list_number = []
        e1_list_number = []
        e2_list_number = []
        n1_list_number = []
        n2_list_number = []
        n3_list_number = []

        minEpiIndClosenessLsa = np.zeros(len(CSIa1Back), dtype=int)
        minEpiIndClosenessLsb = np.zeros(len(CSIa1Back), dtype=int)
        minEpiIndClosenessLse1 = np.zeros(len(CSIa1Back), dtype=int)
        minEpiIndClosenessLse2 = np.zeros(len(CSIa1Back), dtype=int)
        minEpiIndClosenessLsn1 = np.zeros(len(CSIa1Back), dtype=int)
        minEpiIndClosenessLsn2 = np.zeros(len(CSIa1Back), dtype=int)
        minEpiIndClosenessLsn3 = np.zeros(len(CSIa1Back), dtype=int)

        for i in range(len(CSIa1Back)):
            epiIndClosenessLsa = np.zeros(len(CSIa1Back))
            epiIndClosenessLsb = np.zeros(len(CSIa1Back))
            epiIndClosenessLse1 = np.zeros(len(CSIa1Back))
            epiIndClosenessLse2 = np.zeros(len(CSIa1Back))
            epiIndClosenessLsn1 = np.zeros(len(CSIa1Back))
            epiIndClosenessLsn2 = np.zeros(len(CSIa1Back))
            epiIndClosenessLsn3 = np.zeros(len(CSIa1Back))

            for j in range(len(CSIa1Back)):
                epiIndClosenessLsa[j] = standard_hd(CSIa1Back[i], tmpCSIa1Reshape[j])
                epiIndClosenessLsb[j] = standard_hd(CSIa1Back[i], tmpCSIb1Reshape[j])
                epiIndClosenessLse1[j] = standard_hd(CSIa1Back[i], tmpCSIe1Reshape[j])
                epiIndClosenessLse2[j] = standard_hd(CSIa1Back[i], tmpCSIe2Reshape[j])
                epiIndClosenessLsn1[j] = standard_hd(CSIa1Back[i], tmpCSIn1Reshape[j])
                epiIndClosenessLsn2[j] = standard_hd(CSIa1Back[i], tmpCSIn2Reshape[j])
                epiIndClosenessLsn3[j] = standard_hd(CSIa1Back[i], tmpCSIn3Reshape[j])

            minEpiIndClosenessLsa[i] = np.argmin(epiIndClosenessLsa)
            minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)
            minEpiIndClosenessLse1[i] = np.argmin(epiIndClosenessLse1)
            minEpiIndClosenessLse2[i] = np.argmin(epiIndClosenessLse2)
            minEpiIndClosenessLsn1[i] = np.argmin(epiIndClosenessLsn1)
            minEpiIndClosenessLsn2[i] = np.argmin(epiIndClosenessLsn2)
            minEpiIndClosenessLsn3[i] = np.argmin(epiIndClosenessLsn3)

            # # 绘图
            # xa, ya = zip(*sortCSIa1Add[minEpiIndClosenessLsa[i]])
            # xb, yb = zip(*sortCSIb1Add[minEpiIndClosenessLsb[i]])
            # xe1, ye1 = zip(*sortCSIe1Add[minEpiIndClosenessLse1[i]])
            # xe2, ye2 = zip(*sortCSIe2Add[minEpiIndClosenessLse2[i]])
            # xn1, yn1 = zip(*sortCSIn1Add[minEpiIndClosenessLsn1[i]])
            # xn2, yn2 = zip(*sortCSIn2Add[minEpiIndClosenessLsn2[i]])
            # xn3, yn3 = zip(*sortCSIn3Add[minEpiIndClosenessLsn3[i]])
            # plt.figure()
            # plt.plot(xa, ya, color="red", linewidth=2.5, label="a")
            # plt.plot(xb, yb, color="blue", linewidth=1, label="b")
            # plt.plot(xe1, ye1, color="black", linewidth=1, label="e1")
            # plt.plot(xe2, ye2, color="green", linewidth=1, label="e2")
            # # plt.plot(xn, yn, color="yellow", linewidth=2.5, label="n") # 数量级差别太大，不方便显示
            # plt.legend(loc='upper left')
            # plt.show()

        a_list_number = list(minEpiIndClosenessLsa)
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

        # print("ab_max", ab_max, "ae1_max", ae1_max, "ae2_max", ae2_max, "an1_max",
        #       an1_max, "an2_max", an2_max, "an3_max", an3_max)
        # print("keys of a:", a_list)
        # print("keys of a:", a_list_number)
        # print("keys of b:", b_list)
        # print("keys of b:", b_list_number)
        # print("keys of e1:", e1_list)
        # print("keys of e2:", e2_list)
        # print("keys of n1:", n1_list)
        # print("keys of n2:", n2_list)
        # print("keys of n3:", n3_list)

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
        # print("times", times)
    spiltFileName = f.split("_")
    print("\033[0;35;40m" + spiltFileName[1][0] + spiltFileName[2][0] + spiltFileName[3][0], "\t\t",
          round(correctSum / originSum, 4), "\t\t",
          round(correctWholeSum / originWholeSum, 4), "\t\t",
          round(originSum / times / keyLen, 4), "\t\t",
          round(correctSum / times / keyLen, 4), "\t\t", operationMode + "\033[0m")
messagebox.showinfo("提示", "测试结束")
