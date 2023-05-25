import math
import os
import random
import shutil
import sys

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import loadmat
from shapely.geometry import Polygon


def is_in_poly(p, polygon):
    # https://blog.csdn.net/leviopku/article/details/111224539
    """
    :param p: [x, y]
    :param polygon: [[], [], [], [], ...]
    :return:
    """
    px, py = p
    is_in = False
    for i, corner in enumerate(polygon):
        next_i = i + 1 if i + 1 < len(polygon) else 0
        x1, y1 = corner
        x2, y2 = polygon[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in


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

    for xi in y:
        shortest = sys.maxsize
        for yi in x:
            d = round(math.pow(xi[0] - yi[0], 2) + math.pow(xi[1] - yi[1], 2), 10)
            if d < shortest:
                shortest = d
        h += shortest
    return h / len(x) / 2


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
            tmp += np.sum(list[i][j])
            # tmp += (list[i][j][0] * list[i][j][1])
        oneDim.append(round(tmp, 8))
    return oneDim


# 数组第二维的所有内容求和
def sumEachDim(list, index):
    res = 0
    for i in range(len(list[index])):
        res += (list[index][i][0] + list[index][i][1])
        # res += (list[index][i][0] * list[index][i][1])
    return round(res, 10)


def genRandomStep(len, lowBound, highBound):
    length = 0
    randomStep = []
    # 少于三则无法分，因为至少要划分出一个三角形
    while len - length >= lowBound:
        random.seed(0)
        step = random.randint(lowBound, highBound)
        randomStep.append(step)
        length += step
    return randomStep


def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


rawData = loadmat('../data/data_mobile_indoor_1.mat')

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

dataLen = len(CSIa1Orig)

CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)

CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')
CSIb1Orig = smooth(CSIb1Orig, window_len=15, window='flat')
CSIe1Orig = smooth(CSIe1Orig, window_len=15, window="flat")

CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
CSIe1OrigBack = CSIe1Orig.copy()

noise = np.random.normal(loc=-1, scale=1, size=dataLen)  ## Multiplication item normal distribution

intvl = 7
keyLen = 256
addNoise = False

originSum = 0
correctSum = 0
randomSum = 0
noiseSum = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum = 0
noiseWholeSum = 0

times = 0
for staInd in range(0, dataLen, keyLen * intvl):
    endInd = staInd + keyLen * intvl
    print("range:", staInd, endInd)

    if endInd >= len(CSIa1Orig):
        break

    times += 1

    CSIa1Orig = CSIa1OrigBack.copy()
    CSIb1Orig = CSIb1OrigBack.copy()
    CSIe1Orig = CSIe1OrigBack.copy()

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
    tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]

    tmpNoise = noise[range(staInd, endInd, 1)]

    tmpCSIa1 = tmpCSIa1 - (np.mean(tmpCSIa1) - np.mean(tmpCSIb1))  # Mean value consistency

    # linspace函数生成元素为50的等间隔数列，可以指定第三个参数为元素个数
    # signal.square返回周期性的方波波形
    tmpPulse = signal.square(
        2 * np.pi * 1 / intvl * np.linspace(0, np.pi * 0.5 * keyLen, keyLen * intvl))  ## Rectangular pulse

    if addNoise:
        tmpCSIa1 = tmpPulse * (np.float_power(np.abs(tmpCSIa1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIa1))
        tmpCSIb1 = tmpPulse * (np.float_power(np.abs(tmpCSIb1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIb1))
        tmpCSIe1 = tmpPulse * (np.float_power(np.abs(tmpCSIe1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIe1))
    else:
        tmpCSIa1 = tmpPulse * tmpCSIa1
        tmpCSIb1 = tmpPulse * tmpCSIb1
        tmpCSIe1 = tmpPulse * tmpCSIe1

    CSIa1Orig[range(staInd, endInd, 1)] = tmpCSIa1
    CSIb1Orig[range(staInd, endInd, 1)] = tmpCSIb1
    CSIe1Orig[range(staInd, endInd, 1)] = tmpCSIe1

    sortCSIa1 = []
    sortCSIb1 = []
    sortCSIe1 = []
    sortNoise = []

    for i in range(keyLen):
        sortCSIa1.append(sum(tmpCSIa1[i * intvl:(i + 1) * intvl]))
        sortCSIb1.append(sum(tmpCSIb1[i * intvl:(i + 1) * intvl]))
        sortCSIe1.append(sum(tmpCSIe1[i * intvl:(i + 1) * intvl]))
        sortNoise.append(sum(tmpNoise[i * intvl:(i + 1) * intvl]))

    np.random.seed(0)
    sortASend = np.random.normal(np.mean(sortCSIa1), np.std(sortCSIa1), len(sortCSIa1))

    # 形成三维数组，其中第三维是一对坐标值
    # 数组的长度随机生成
    sortCSIa1Copy = sortCSIa1.copy()
    sortCSIb1Copy = sortCSIb1.copy()
    sortCSIe1Copy = sortCSIe1.copy()
    sortCSIn1Copy = sortNoise.copy()
    sortASendCopy = sortASend.copy()

    sortCSIa1Copy = np.array(sortCSIa1Copy).reshape((int(len(sortCSIa1) / 2), 2))
    sortCSIb1Copy = np.array(sortCSIb1Copy).reshape((int(len(sortCSIb1) / 2), 2))
    sortCSIe1Copy = np.array(sortCSIe1Copy).reshape((int(len(sortCSIe1) / 2), 2))
    sortCSIn1Copy = np.array(sortCSIn1Copy).reshape((int(len(sortNoise) / 2), 2))
    sortASendCopy = np.array(sortASendCopy).reshape((int(len(sortASend) / 2), 2))

    sortCSIa1Split = []
    sortCSIb1Split = []
    sortCSIe1Split = []
    sortNoiseSplit = []
    sortASendSplit = []

    randomStep = genRandomStep(int(keyLen / 2), 3, 7)
    # print("randomStep", randomStep)
    startIndex = 0
    for i in range(len(randomStep)):
        # 由于随机产生step的算法不一定刚好满足step之和等于keyLen/2，故在每次复制值的时候需要判断
        if startIndex >= len(sortCSIa1Copy) or len(sortCSIa1Copy) - startIndex < 3:
            break
        sortCSIa1Split.append(sortCSIa1Copy[startIndex:startIndex + randomStep[i]])
        sortCSIb1Split.append(sortCSIb1Copy[startIndex:startIndex + randomStep[i]])
        sortCSIe1Split.append(sortCSIe1Copy[startIndex:startIndex + randomStep[i]])
        sortNoiseSplit.append(sortCSIn1Copy[startIndex:startIndex + randomStep[i]])
        sortASendSplit.append(sortASendCopy[startIndex:startIndex + randomStep[i]])
        startIndex = startIndex + randomStep[i]

    sortCSIa1 = sortCSIa1Split
    sortCSIb1 = sortCSIb1Split
    sortCSIe1 = sortCSIe1Split
    sortNoise = sortNoiseSplit
    sortASend = sortASendSplit

    CSIa1Back = sortCSIa1
    CSIb1Back = sortCSIb1
    CSIe1Back = sortCSIe1
    CSIn1Back = sortNoise
    ASendBack = sortASend

    # 凸包检查
    # for i in range(len(ASendBack)):
    #     ASendBack[i] = list(Polygon(ASendBack[i]).convex_hull.exterior.coords)[0:-1]
    #     CSIa1Back[i] = list(Polygon(CSIa1Back[i]).convex_hull.exterior.coords)[0:-1]
    #     CSIb1Back[i] = list(Polygon(CSIb1Back[i]).convex_hull.exterior.coords)[0:-1]
    #     CSIe1Back[i] = list(Polygon(CSIe1Back[i]).convex_hull.exterior.coords)[0:-1]
    #     CSIn1Back[i] = list(Polygon(CSIn1Back[i]).convex_hull.exterior.coords)[0:-1]

    # # ASend不能与CSIa1相交
    # for i in range(len(ASendBack)):
    #     is_in = Polygon(ASendBack[i]).intersects(Polygon(sortCSIa1[i]))
    #     while is_in is True:
    #         for j in range(len(ASendBack[i])):
    #             ASendBack[i][j] = [random.uniform(0, np.log10(np.abs(_max - _min))),
    #                                random.uniform(0, np.log10(np.abs(_max - _min)))]
    #         is_in = Polygon(ASendBack[i]).intersects(Polygon(sortCSIa1[i]))

    # # 凸包检查
    # for i in range(len(ASendBack)):
    #     ASendBack[i] = list(Polygon(ASendBack[i]).convex_hull.exterior.coords)[0:-1]
    #     CSIa1Back[i] = list(Polygon(CSIa1Back[i]).convex_hull.exterior.coords)[0:-1]
    #     CSIb1Back[i] = list(Polygon(CSIb1Back[i]).convex_hull.exterior.coords)[0:-1]
    #     CSIe1Back[i] = list(Polygon(CSIe1Back[i]).convex_hull.exterior.coords)[0:-1]
    #     CSIn1Back[i] = list(Polygon(CSIn1Back[i]).convex_hull.exterior.coords)[0:-1]

    # 降维以用于后续的排序
    oneDimCSIa1 = toOneDim(CSIa1Back)
    oneDimCSIb1 = toOneDim(CSIb1Back)
    oneDimCSIe1 = toOneDim(CSIe1Back)
    oneDimCSIn1 = toOneDim(CSIn1Back)
    oneDimASend = toOneDim(ASendBack)

    # 在数组a后面加上a[0]使之成为一个首尾封闭的多边形
    sortCSIa1Add = makePolygon(CSIa1Back)
    sortCSIb1Add = makePolygon(CSIb1Back)
    sortCSIe1Add = makePolygon(CSIe1Back)
    sortCSIn1Add = makePolygon(CSIn1Back)
    sortASendAdd = makePolygon(ASendBack)

    # 初始化各个计算出的hd值
    aa_max = 0
    ab_max = 0
    ae_max = 0
    an_max = 0

    # 最后各自的密钥
    a_list = []
    b_list = []
    e_list = []
    n_list = []

    a_list_number = []
    b_list_number = []
    e_list_number = []
    n_list_number = []

    minEpiIndClosenessLsa = np.zeros(len(ASendBack), dtype=int)
    minEpiIndClosenessLsb = np.zeros(len(ASendBack), dtype=int)
    minEpiIndClosenessLse = np.zeros(len(ASendBack), dtype=int)
    minEpiIndClosenessLsn = np.zeros(len(ASendBack), dtype=int)

    for i in range(len(ASendBack)):
        epiIndClosenessLsa = np.zeros(len(ASendBack))
        epiIndClosenessLsb = np.zeros(len(ASendBack))
        epiIndClosenessLse = np.zeros(len(ASendBack))
        epiIndClosenessLsn = np.zeros(len(ASendBack))

        for j in range(len(ASendBack)):
            epiIndClosenessLsa[j] = standard_hd(ASendBack[i], CSIa1Back[j])
            epiIndClosenessLsb[j] = standard_hd(ASendBack[i], CSIb1Back[j])
            epiIndClosenessLse[j] = standard_hd(ASendBack[i], CSIe1Back[j])
            epiIndClosenessLsn[j] = standard_hd(ASendBack[i], CSIn1Back[j])

        minEpiIndClosenessLsa[i] = np.argmin(epiIndClosenessLsa)
        minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)
        minEpiIndClosenessLse[i] = np.argmin(epiIndClosenessLse)
        minEpiIndClosenessLsn[i] = np.argmin(epiIndClosenessLsn)

    a_list_number = list(minEpiIndClosenessLsa)
    b_list_number = list(minEpiIndClosenessLsb)
    e_list_number = list(minEpiIndClosenessLse)
    n_list_number = list(minEpiIndClosenessLsn)

    print("keys of a:", len(a_list_number), a_list_number)
    print("keys of b:", len(b_list_number), b_list_number)
    print("keys of e:", len(e_list_number), e_list_number)
    print("keys of n:", len(n_list_number), n_list_number)

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

    originSum += sum1
    correctSum += sum2
    randomSum += sum3
    noiseSum += sum4

    originWholeSum += 1
    correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
    randomWholeSum1 = randomWholeSum + 1 if sum3 == sum1 else randomWholeSum
    noiseWholeSum = noiseWholeSum + 1 if sum4 == sum1 else noiseWholeSum

    print("a-b", sum2 / sum1)
    print("a-e", sum3 / sum1)
    print("a-n", sum4 / sum1)
    print("----------------------")

print("a-b all", correctSum, "/", originSum, "=", correctSum / originSum)
print("a-e all", randomSum, "/", originSum, "=", randomSum / originSum)
print("a-n all", noiseSum, "/", originSum, "=", noiseSum / originSum)
print("\033[0;34;40ma-b bit agreement rate", correctSum, "/", originSum, "=", round(correctSum / originSum, 10),
      "\033[0m")
print("a-e bit agreement rate", randomSum, "/", originSum, "=", round(randomSum / originSum, 10))
print("a-n bit agreement rate", noiseSum, "/", originSum, "=", round(noiseSum / originSum, 10))
print("\033[0;34;40ma-b key agreement rate", correctWholeSum, "/", originWholeSum, "=",
      round(correctWholeSum / originWholeSum, 10), "\033[0m")
print("a-e key agreement rate", randomWholeSum, "/", originWholeSum, "=",
      round(randomWholeSum / originWholeSum, 10))
print("a-n key agreement rate", noiseWholeSum, "/", originWholeSum, "=",
      round(noiseWholeSum / originWholeSum, 10))
