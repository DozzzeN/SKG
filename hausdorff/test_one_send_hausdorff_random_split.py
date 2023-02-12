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
            tmp += (list[i][j][0] + list[i][j][1])
            # tmp += (list[i][j][0] * list[i][j][1])
        oneDim.append(round(tmp, 10))
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

# rawData = loadmat('../data/data_static_indoor_1.mat')
#
# CSIa1OrigRaw = rawData['A'][:, 0]
# CSIb1OrigRaw = rawData['A'][:, 1]
#
# CSIa1Orig = []
# CSIb1Orig = []
# for i in range(2000):
#     CSIa1Orig.append(CSIa1OrigRaw[i])
#     CSIb1Orig.append(CSIb1OrigRaw[i])
# for i in range(5000):
#     CSIa1Orig.append(CSIa1OrigRaw[i + 20000])
#     CSIb1Orig.append(CSIb1OrigRaw[i + 20000])
#
# CSIa1Orig = np.array(CSIa1Orig)
# CSIb1Orig = np.array(CSIb1Orig)

# CSIe1Orig = CSIb1Orig.copy()

# a10rig_mean = np.mean(CSIa1Orig)

# for i in range(int(len(CSIb1Orig) / 2), len(CSIb1Orig)):
#     CSIa1Orig[i] = a10rig_mean
#     CSIb1Orig[i] = a10rig_mean

# for i in range(len(CSIb1Orig)):
#     CSIe1Orig[i] = a10rig_mean
#
dataLen = len(CSIa1Orig)  # 6745

CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)

CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')
CSIb1Orig = smooth(CSIb1Orig, window_len=15, window='flat')
CSIe1Orig = smooth(CSIe1Orig, window_len=15, window="flat")

CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
CSIe1OrigBack = CSIe1Orig.copy()

noise = np.random.normal(loc=-1, scale=1, size=dataLen)  ## Multiplication item normal distribution
noiseAdd = np.random.normal(loc=0, scale=10, size=dataLen)  ## Addition item normal distribution

sft = 2
intvl = 2 * sft + 1
keyLen = 256
addNoise = False

originSum = 0
correctSum = 0
randomSum = 0
noiseSum = 0

codings = ""
# for ii in range(0, 5):

del_file('./figures/')

for staInd in range(0, 10 * intvl + 1, intvl):
    endInd = staInd + keyLen * intvl
    print("range:", staInd, endInd)

    if not os.path.exists('./figures/' + str(staInd)):
        os.mkdir('./figures/' + str(staInd))

    if endInd > len(CSIa1Orig):
        break

    CSIa1Orig = CSIa1OrigBack.copy()
    CSIb1Orig = CSIb1OrigBack.copy()
    CSIe1Orig = CSIe1OrigBack.copy()

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
    tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]

    tmpNoise = noise[range(staInd, endInd, 1)]
    tmpNoiseAdd = noiseAdd[range(staInd, endInd, 1)]

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
            noiseTmp = noise[aIndVec]

            sortCSIa1[ii] = np.mean(CSIa1Tmp)  ## Metric 1: Mean
            sortCSIb1[jj - permLen] = np.mean(CSIb1Tmp)  # 只赋值一次
            sortCSIe1[jj - permLen] = np.mean(CSIe1Tmp)
            sortNoise[ii] = np.mean(noiseTmp)

    _max = max(sortCSIa1)
    _min = min(sortCSIa1)

    # sortCSIa1 = sortCSIa1 / (_max - _min) - _min / (_max - _min)
    # sortCSIb1 = sortCSIb1 / (_max - _min) - _min / (_max - _min)
    # sortCSIe1 = sortCSIe1 / (_max - _min) - _min / (_max - _min)
    # sortNoise = sortNoise / (_max - _min) - _min / (_max - _min)

    # sortCSIa1是原始算法中排序前的数据
    sortCSIa1 = np.log10(np.abs(sortCSIa1)) * 100
    sortCSIb1 = np.log10(np.abs(sortCSIb1)) * 100
    sortCSIe1 = np.log10(np.abs(sortCSIe1)) * 100
    sortNoise = np.log10(np.abs(sortNoise)) * 100

    # 准备A发送的数据
    # copyCSIa1 = sortCSIa1.copy()
    # random.shuffle(copyCSIa1)
    # sortASend = np.array(copyCSIa1)

    sortASend = np.random.normal(np.mean(sortCSIa1), np.std(sortCSIa1), len(sortCSIa1))

    # sortASend = sortCSIa1.copy()

    # sortASend = np.random.randint(0, np.log10(np.abs(_max-_min)), size=len(sortCSIa1))  # 随机分布

    # 形成三维数组，其中第三维是一对坐标值
    # 数组的长度随机生成
    sortCSIa1Copy = sortCSIa1.copy()
    sortCSIb1Copy = sortCSIb1.copy()
    sortCSIe1Copy = sortCSIe1.copy()
    sortCSIn1Copy = sortNoise.copy()
    sortASendCopy = sortASend.copy()

    sortCSIa1Copy = sortCSIa1Copy.reshape(int(len(sortCSIa1) / 2), 2)
    sortCSIb1Copy = sortCSIb1Copy.reshape(int(len(sortCSIb1) / 2), 2)
    sortCSIe1Copy = sortCSIe1Copy.reshape(int(len(sortCSIe1) / 2), 2)
    sortCSIn1Copy = sortCSIn1Copy.reshape(int(len(sortNoise) / 2), 2)
    sortASendCopy = sortASendCopy.reshape(int(len(sortASend) / 2), 2)

    sortCSIa1Split = []
    sortCSIb1Split = []
    sortCSIe1Split = []
    sortNoiseSplit = []
    sortASendSplit = []

    randomStep = genRandomStep(int(keyLen / 2), 3, 7)
    print("randomStep", randomStep)
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

    # sortCSIa1 = sortCSIa1.reshape(int(len(sortCSIa1) / step / 2), step, 2)
    # sortCSIb1 = sortCSIb1.reshape(int(len(sortCSIb1) / step / 2), step, 2)
    # sortCSIe1 = sortCSIe1.reshape(int(len(sortCSIe1) / step / 2), step, 2)
    # sortNoise = sortNoise.reshape(int(len(sortNoise) / step / 2), step, 2)
    # sortASend = sortASend.reshape(int(len(sortASend) / step / 2), step, 2)

    # # 向量旋转
    # for i in range(len(sortASend)):
    #     minx = 0
    #     miny = 0
    #     angle = random.uniform(0, math.pi / 2)
    #     for j in range(len(sortASend[i])):
    #         x = sortASend[i][j][0]
    #         y = sortASend[i][j][1]
    #         sortASend[i][j][0] = x * math.cos(angle) - y * math.sin(angle)
    #         sortASend[i][j][1] = x * math.sin(angle) + y * math.cos(angle)
    #         minx = min(minx, sortASend[i][j][0])
    #         miny = min(miny, sortASend[i][j][1])
    #     for j in range(len(sortASend[i])):
    #         sortASend[i][j][0] -= minx
    #         sortASend[i][j][1] -= miny

    # 计算hd距离和多边形的顺序无关，可以任意洗牌
    # CSIa1Back = [[] for _ in range(len(sortCSIa1))]
    # CSIb1Back = [[] for _ in range(len(sortCSIb1))]
    # CSIe1Back = [[] for _ in range(len(sortCSIe1))]
    # CSIn1Back = [[] for _ in range(len(sortNoise))]
    # ASendBack = [[] for _ in range(len(sortASend))]

    # 随机打乱
    # random.shuffle(sortCSIa1)
    # random.shuffle(sortCSIb1)
    # random.shuffle(sortCSIe1)
    # random.shuffle(sortNoise)

    # rand_out_polygon = list(range(len(sortCSIa1)))
    # rand_in_polygon = list(range(step))

    # random.shuffle(rand_out_polygon)
    # random.shuffle(rand_in_polygon)
    # for i in range(len(sortCSIa1)):
    #     for j in range(step):
    #         CSIa1Back[i].append(sortCSIa1[rand_out_polygon[i]][rand_in_polygon[j]])

    # random.shuffle(rand_out_polygon)
    # random.shuffle(rand_in_polygon)
    # for i in range(len(sortCSIb1)):
    #     for j in range(step):
    #         CSIb1Back[i].append(sortCSIb1[rand_out_polygon[i]][rand_in_polygon[j]])

    # random.shuffle(rand_out_polygon)
    # random.shuffle(rand_in_polygon)
    # for i in range(len(sortCSIe1)):
    #     for j in range(step):
    #         CSIe1Back[i].append(sortCSIe1[rand_out_polygon[i]][rand_in_polygon[j]])

    # random.shuffle(rand_out_polygon)
    # random.shuffle(rand_in_polygon)
    # for i in range(len(sortNoise)):
    #     for j in range(step):
    #         CSIn1Back[i].append(sortNoise[rand_out_polygon[i]][rand_in_polygon[j]])

    # random.shuffle(rand_out_polygon)
    # random.shuffle(rand_in_polygon)
    # for i in range(len(sortASend)):
    #     for j in range(step):
    #         ASendBack[i].append(sortASend[rand_out_polygon[i]][rand_in_polygon[j]])

    # 平移
    # for i in range(len(ASendBack)):
    #     for j in range(len(ASendBack[i])):
    #         ASendBack[i][j][0] = CSIa1Back[i][j][0] * 1.5 + 0.3
    #         ASendBack[i][j][1] = CSIa1Back[i][j][1] * 1.5 + 0.3

    CSIa1Back = sortCSIa1
    CSIb1Back = sortCSIb1
    CSIe1Back = sortCSIe1
    CSIn1Back = sortNoise
    ASendBack = sortASend

    # 凸包检查
    for i in range(len(ASendBack)):
        ASendBack[i] = list(Polygon(ASendBack[i]).convex_hull.exterior.coords)[0:-1]
        CSIa1Back[i] = list(Polygon(CSIa1Back[i]).convex_hull.exterior.coords)[0:-1]
        CSIb1Back[i] = list(Polygon(CSIb1Back[i]).convex_hull.exterior.coords)[0:-1]
        CSIe1Back[i] = list(Polygon(CSIe1Back[i]).convex_hull.exterior.coords)[0:-1]
        CSIn1Back[i] = list(Polygon(CSIn1Back[i]).convex_hull.exterior.coords)[0:-1]

    # ASend不能与CSIa1相交
    for i in range(len(ASendBack)):
        is_in = Polygon(ASendBack[i]).intersects(Polygon(sortCSIa1[i]))
        while is_in is True:
            for j in range(len(ASendBack[i])):
                ASendBack[i][j] = [random.uniform(0, np.log10(np.abs(_max - _min))),
                                   random.uniform(0, np.log10(np.abs(_max - _min)))]
            is_in = Polygon(ASendBack[i]).intersects(Polygon(sortCSIa1[i]))

    # 凸包检查
    for i in range(len(ASendBack)):
        ASendBack[i] = list(Polygon(ASendBack[i]).convex_hull.exterior.coords)[0:-1]
        CSIa1Back[i] = list(Polygon(CSIa1Back[i]).convex_hull.exterior.coords)[0:-1]
        CSIb1Back[i] = list(Polygon(CSIb1Back[i]).convex_hull.exterior.coords)[0:-1]
        CSIe1Back[i] = list(Polygon(CSIe1Back[i]).convex_hull.exterior.coords)[0:-1]
        CSIn1Back[i] = list(Polygon(CSIn1Back[i]).convex_hull.exterior.coords)[0:-1]

    # 降维以用于后续的排序
    oneDimCSIa1 = toOneDim(CSIa1Back)
    oneDimCSIb1 = toOneDim(CSIb1Back)
    oneDimCSIe1 = toOneDim(CSIe1Back)
    oneDimCSIn1 = toOneDim(CSIn1Back)
    oneDimASend = toOneDim(ASendBack)

    # # ASend不能与CSIa1相交
    # for i in range(len(sortASendAdd)):
    #     for j in range(len(sortASendAdd[i])):
    #         is_in = is_in_poly(sortASendAdd[i][j], sortCSIa1[i])
    #         while is_in is True:
    #             sortASendAdd[i][j] = [np.random.randint(0, np.mean(sortCSIa1)),
    #                                   np.random.randint(0, np.mean(sortCSIa1))]
    #             is_in = is_in_poly(sortASendAdd[i][j], sortCSIa1[i])

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

    for i in range(0, int(len(sortCSIa1Add) / 5)):
        xa1, ya1 = zip(*sortCSIa1Add[5 * i])
        xa2, ya2 = zip(*sortCSIa1Add[5 * i + 1])
        xa3, ya3 = zip(*sortCSIa1Add[5 * i + 2])
        xa4, ya4 = zip(*sortCSIa1Add[5 * i + 3])
        xa5, ya5 = zip(*sortCSIa1Add[5 * i + 4])
        plt.figure()
        plt.plot(xa1, ya1, label="a" + str(5 * i) + '-' + str(len(xa1) - 1) + 'polygon')
        plt.plot(xa2, ya2, label="a" + str(5 * i + 1) + '-' + str(len(xa2) - 1) + 'polygon')
        plt.plot(xa3, ya3, label="a" + str(5 * i + 2) + '-' + str(len(xa3) - 1) + 'polygon')
        plt.plot(xa4, ya4, label="a" + str(5 * i + 3) + '-' + str(len(xa4) - 1) + 'polygon')
        plt.plot(xa5, ya5, label="a" + str(5 * i + 4) + '-' + str(len(xa5) - 1) + 'polygon')
        plt.legend(loc='lower left')
        plt.savefig('./figures/' + str(staInd) + '/' + 'a-polygon' + str(i) + '.png')
        plt.close()

    for i in range(0, int(len(sortCSIb1Add) / 5)):
        xb1, yb1 = zip(*sortCSIb1Add[5 * i])
        xb2, yb2 = zip(*sortCSIb1Add[5 * i + 1])
        xb3, yb3 = zip(*sortCSIb1Add[5 * i + 2])
        xb4, yb4 = zip(*sortCSIb1Add[5 * i + 3])
        xb5, yb5 = zip(*sortCSIb1Add[5 * i + 4])
        plt.figure()
        plt.plot(xb1, yb1, label="b" + str(5 * i) + '-' + str(len(xb1) - 1) + 'polygon')
        plt.plot(xb2, yb2, label="b" + str(5 * i + 1) + '-' + str(len(xb2) - 1) + 'polygon')
        plt.plot(xb3, yb3, label="b" + str(5 * i + 2) + '-' + str(len(xb3) - 1) + 'polygon')
        plt.plot(xb4, yb4, label="b" + str(5 * i + 3) + '-' + str(len(xb4) - 1) + 'polygon')
        plt.plot(xb5, yb5, label="b" + str(5 * i + 4) + '-' + str(len(xb5) - 1) + 'polygon')
        plt.legend(loc='lower left')
        plt.savefig('./figures/' + str(staInd) + '/' + 'b-polygon' + str(i) + '.png')
        plt.close()

    for i in range(len(sortASendAdd)):
        xs, ys = zip(*sortASendAdd[i])
        xa, ya = zip(*sortCSIa1Add[i])
        xb, yb = zip(*sortCSIb1Add[i])
        xe, ye = zip(*sortCSIe1Add[i])
        xn, yn = zip(*sortCSIn1Add[i])
        plt.figure()
        plt.plot(xs, ys, color="green", linewidth=2.5, label="s" + str(i))
        plt.plot(xa, ya, color="red", linewidth=2.5, label="a" + str(i))
        plt.plot(xb, yb, color="blue", linewidth=2.5, label="b" + str(i))
        plt.plot(xe, ye, color="black", linewidth=2.5, label="e" + str(i))
        # plt.plot(xn, yn, color="yellow", linewidth=2.5, label="n") # 数量级差别太大，不方便显示
        plt.legend(loc='upper left')
        plt.savefig('./figures/' + str(staInd) + '/' + str(i) + '.png')
        plt.close()
        # plt.show()

    all_aa_hd = []
    for i in range(len(ASendBack)):
        aa_hd = sys.maxsize
        ab_hd = sys.maxsize
        ae_hd = sys.maxsize
        an_hd = sys.maxsize

        aa_index = 0
        ab_index = 0
        ae_index = 0
        an_index = 0
        for j in range(len(CSIa1Back)):
            # 整体计算两个集合中每个多边形的hd值，取最匹配的（hd距离最接近的两个多边形）
            # aa_d = standard_hd(ASendBack[i], CSIa1Back[j])
            # ab_d = standard_hd(ASendBack[i], CSIb1Back[j])
            # ae_d = standard_hd(ASendBack[i], CSIe1Back[j])
            # an_d = standard_hd(ASendBack[i], CSIn1Back[j])
            aa_d = Polygon(ASendBack[i]).hausdorff_distance(Polygon(CSIa1Back[j]))
            ab_d = Polygon(ASendBack[i]).hausdorff_distance(Polygon(CSIb1Back[j]))
            ae_d = Polygon(ASendBack[i]).hausdorff_distance(Polygon(CSIe1Back[j]))
            an_d = Polygon(ASendBack[i]).hausdorff_distance(Polygon(CSIn1Back[j]))
            all_aa_hd.append(aa_d)
            if aa_d < aa_hd:
                aa_hd = aa_d
                aa_index = j
            if ab_d < ab_hd:
                ab_hd = ab_d
                ab_index = j
            if ae_d < ae_hd:
                ae_hd = ae_d
                ae_index = j
            if an_d < an_hd:
                an_hd = an_d
                an_index = j
        if aa_index != ab_index:
            print("not equal aa_index", aa_index, "aa_hd", aa_hd)
            print("not equal ab_index", ab_index, "ab_hd", ab_hd)

        if aa_index == ab_index:
            print("equal aa_index", aa_index, "aa_hd", aa_hd)
            print("equal ab_index", ab_index, "ab_hd", ab_hd)

        # 反过来用ASend找A里面哪个最接近会使得a-e大幅度增加
        # for j in range(len(ASendBack)):
        #     # 整体计算两个集合中每个多边形的hd值，取最匹配的（hd距离最接近的两个多边形）
        #     # aa_d = standard_hd(ASendBack[i], CSIa1Back[j])
        #     # ab_d = standard_hd(ASendBack[i], CSIb1Back[j])
        #     # ae_d = standard_hd(ASendBack[i], CSIe1Back[j])
        #     # an_d = standard_hd(ASendBack[i], CSIn1Back[j])
        #     aa_d = Polygon(ASendBack[j]).hausdorff_distance(Polygon(CSIa1Back[i]))
        #     ab_d = Polygon(ASendBack[j]).hausdorff_distance(Polygon(CSIb1Back[i]))
        #     ae_d = Polygon(ASendBack[j]).hausdorff_distance(Polygon(CSIe1Back[i]))
        #     an_d = Polygon(ASendBack[j]).hausdorff_distance(Polygon(CSIn1Back[i]))
        #     if aa_d < aa_hd:
        #         aa_hd = aa_d
        #         aa_index = j
        #     if ab_d < ab_hd:
        #         ab_hd = ab_d
        #         ab_index = j
        #     if ae_d < ae_hd:
        #         ae_hd = ae_d
        #         ae_index = j
        #     if an_d < an_hd:
        #         an_hd = an_d
        #         an_index = j
        # if aa_index != ab_index:
        #     print("aa_index", aa_index, "aa_hd", aa_hd)
        #     print("ab_index", ab_index, "ab_hd", ab_hd)

        # 将横纵坐标之和的值作为排序标准进行排序，然后进行查找，基于原数组的位置作为密钥值
        a_list.append(np.where(np.array(oneDimCSIa1) == np.array(sumEachDim(CSIa1Back, aa_index)))[0][0])
        b_list.append(np.where(np.array(oneDimCSIb1) == np.array(sumEachDim(CSIb1Back, ab_index)))[0][0])
        e_list.append(np.where(np.array(oneDimCSIe1) == np.array(sumEachDim(CSIe1Back, ae_index)))[0][0])
        n_list.append(np.where(np.array(oneDimCSIn1) == np.array(sumEachDim(CSIn1Back, an_index)))[0][0])
        # print("\n")
        # print("\033[0;32;40mCSIa1", ASendBack[i], "\033[0m")
        # print("aa_hd", aa_hd, "\033[0;32;40mCSIa1", CSIa1Back[aa_index], "\033[0m")
        # print("ab_hd", ab_hd, "\033[0;32;40mCSIb1", CSIb1Back[ab_index], "\033[0m")
        # print("ae_hd", ae_hd, "CSIe1", CSIe1Back[ae_index])
        # print("an_hd", an_hd, "CSIn1", CSIn1Back[an_index])

        # 比较各个独立计算的hd值的差异
        aa_max = max(aa_max, aa_hd)
        ab_max = max(ab_max, ab_hd)
        ae_max = max(ae_max, ae_hd)
        an_max = max(an_max, an_hd)

        # 绘图
        xs, ys = zip(*sortASendAdd[i])
        xa, ya = zip(*sortCSIa1Add[aa_index])
        xb, yb = zip(*sortCSIb1Add[ab_index])
        xe, ye = zip(*sortCSIe1Add[ae_index])
        xn, yn = zip(*sortCSIn1Add[an_index])
        plt.figure()
        plt.plot(xs, ys, color="green", linewidth=2.5, label="s")
        plt.plot(xa, ya, color="red", linewidth=2.5, label="a")
        plt.plot(xb, yb, color="blue", linewidth=2.5, label="b")
        plt.plot(xe, ye, color="black", linewidth=2.5, label="e")
        # plt.plot(xn, yn, color="yellow", linewidth=2.5, label="n") # 数量级差别太大，不方便显示
        plt.legend(loc='upper left')
        plt.savefig('./figures/' + str(staInd) + '/sabe' + str(i) + '.png')
        plt.close()
        # plt.show()

        # plt.figure()
        # plt.plot(xs, ys, color="green", linewidth=2.5, label="s")
        # plt.plot(xb, yb, color="blue", linewidth=2.5, label="b")
        # plt.legend(loc='upper left')
        # plt.show()
        # plt.close()
        #
        # plt.figure()
        # plt.plot(xs, ys, color="green", linewidth=2.5, label="s")
        # plt.plot(xe, ye, color="black", linewidth=2.5, label="e")
        # plt.legend(loc='upper left')
        # plt.show()
        # plt.close()

    # plt.close()
    all_aa_hd.sort()
    # 差分
    increment_all_hd = np.array(all_aa_hd[1:]) - np.array(all_aa_hd[:-1])
    print("most_increment", max(increment_all_hd), min(increment_all_hd))
    print("all_aa_hd", min(all_aa_hd), max(all_aa_hd), np.std(all_aa_hd))
    print("aa_max", aa_max, "ab_max", ab_max, "ae_max", ae_max, "an_max", an_max)
    print("keys of a:", len(a_list), a_list)
    print("keys of b:", len(b_list), b_list)
    print("keys of e:", len(e_list), e_list)
    print("keys of n:", len(n_list), n_list)

    sum1 = len(a_list)
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for i in range(0, sum1):
        sum2 += (a_list[i] - b_list[i] == 0)
        sum3 += (a_list[i] - e_list[i] == 0)
        sum4 += (a_list[i] - n_list[i] == 0)

    print("a-b", sum2 / sum1)
    print("a-e", sum3 / sum1)
    print("a-n", sum4 / sum1)
    print("----------------------")
    originSum += sum1
    correctSum += sum2
    randomSum += sum3
    noiseSum += sum4

    # 编码密钥
    # char_weights = []
    # weights = Counter(a_list)  # 得到list中元素出现次数
    # for i in range(len(a_list)):
    #     char_weights.append((a_list[i], weights[a_list[i]]))
    # tree = HuffmanTree(char_weights)
    # tree.get_code()
    # HuffmanTree.codings += "\n"

    for i in range(len(a_list)):
        codings += bin(a_list[i])[2:] + "\n"

with open('../edit_distance/evaluations/key.txt', 'a', ) as f:
    f.write(codings)

print("a-b all", correctSum, "/", originSum, "=", correctSum / originSum)
print("a-e all", randomSum, "/", originSum, "=", randomSum / originSum)
print("a-n all", noiseSum, "/", originSum, "=", noiseSum / originSum)
