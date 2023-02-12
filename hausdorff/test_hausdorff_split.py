import math
import random
import sys

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
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


# 数组第二维的所有内容求和
def sumEachDim(list, index):
    res = 0
    for i in range(len(list[index])):
        res += (list[index][i][0] + list[index][i][1])
        # res += (list[index][i][0] * list[index][i][1])
    return round(res, 8)


rawData = loadmat('../data/data_mobile_indoor_1.mat')

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
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

CSIa1Orig = np.array(CSIa1Orig)
CSIb1Orig = np.array(CSIb1Orig)

plt.figure()
plt.plot(CSIa1Orig, color="red", linewidth=.05, label="a")
# plt.plot(CSIb1Orig, color="blue", linewidth=.05, label="b")
plt.legend(loc='upper left')
plt.show()

rand_CSI = list(range(len(CSIa1Orig)))
random.shuffle(rand_CSI)

CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()

for i in range(len(CSIa1Orig)):
    CSIa1Orig[i] = CSIa1OrigBack[rand_CSI[i]]
    CSIb1Orig[i] = CSIb1OrigBack[rand_CSI[i]]

plt.figure()
plt.plot(CSIa1Orig, color="red", linewidth=.05, label="a")
# plt.plot(CSIb1Orig, color="blue", linewidth=.05, label="b")
plt.legend(loc='upper left')
plt.show()

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
keyLen = 1024
addNoise = False

originSum = 0
correctSum = 0
randomSum = 0
noiseSum = 0

codings = ""
# for ii in range(0, 5):
for staInd in range(0, 10 * intvl + 1, intvl):
    endInd = staInd + keyLen * intvl
    print("range:", staInd, endInd)
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

    # 归一化
    # _max = max(max(sortCSIa1), max(sortCSIb1), max(sortCSIe1), max(sortNoise))
    # _min = min(min(sortCSIa1), min(sortCSIb1), min(sortCSIe1), min(sortNoise))

    # sortCSIa1 = sortCSIa1 / (_max - _min) - _min / (_max - _min)
    # sortCSIb1 = sortCSIb1 / (_max - _min) - _min / (_max - _min)
    # sortCSIe1 = sortCSIe1 / (_max - _min) - _min / (_max - _min)
    # sortNoise = sortNoise / (_max - _min) - _min / (_max - _min)

    # sortCSIa1是原始算法中排序前的数据
    sortCSIa1 = np.log10(np.abs(sortCSIa1))
    sortCSIb1 = np.log10(np.abs(sortCSIb1))
    sortCSIe1 = np.log10(np.abs(sortCSIe1))
    sortNoise = np.log10(np.abs(sortNoise))

    # 形成三维数组，其中第三维是一对坐标值
    # 数组的长度由param调节
    param = 2
    step = int(math.pow(2, param))
    sortCSIa1 = sortCSIa1.reshape(int(len(sortCSIa1) / step / 2), step, 2)
    sortCSIb1 = sortCSIb1.reshape(int(len(sortCSIb1) / step / 2), step, 2)
    sortCSIe1 = sortCSIe1.reshape(int(len(sortCSIe1) / step / 2), step, 2)
    sortNoise = sortNoise.reshape(int(len(sortNoise) / step / 2), step, 2)

    # 降维以用于后续的排序
    oneDimCSIa1 = toOneDim(sortCSIa1)
    oneDimCSIb1 = toOneDim(sortCSIb1)
    oneDimCSIe1 = toOneDim(sortCSIe1)
    oneDimCSIn1 = toOneDim(sortNoise)

    # 初始化一维数组，作为基准数组
    # 使用排序后的数组进行匹配效果不好
    sortCSIa1Back = np.sort(oneDimCSIa1, axis=0)
    sortCSIb1Back = np.sort(oneDimCSIb1, axis=0)
    sortCSIe1Back = np.sort(oneDimCSIe1, axis=0)
    sortCSIn1Back = np.sort(oneDimCSIn1, axis=0)

    # 计算hd距离和多边形的顺序无关，可以任意洗牌
    CSIa1Back = [[] for _ in range(len(sortCSIa1))]
    CSIb1Back = [[] for _ in range(len(sortCSIb1))]
    CSIe1Back = [[] for _ in range(len(sortCSIe1))]
    CSIn1Back = [[] for _ in range(len(sortNoise))]

    # 随机打乱
    # random.shuffle(sortCSIa1)
    # random.shuffle(sortCSIb1)
    # random.shuffle(sortCSIe1)
    # random.shuffle(sortNoise)

    rand_out_polygon = list(range(len(sortCSIa1)))
    rand_in_polygon = list(range(step))

    random.shuffle(rand_out_polygon)
    random.shuffle(rand_in_polygon)
    for i in range(len(sortCSIa1)):
        for j in range(step):
            CSIa1Back[i].append(sortCSIa1[rand_out_polygon[i]][rand_in_polygon[j]])

    random.shuffle(rand_out_polygon)
    random.shuffle(rand_in_polygon)
    for i in range(len(sortCSIb1)):
        for j in range(step):
            CSIb1Back[i].append(sortCSIb1[rand_out_polygon[i]][rand_in_polygon[j]])

    random.shuffle(rand_out_polygon)
    random.shuffle(rand_in_polygon)
    for i in range(len(sortCSIe1)):
        for j in range(step):
            CSIe1Back[i].append(sortCSIe1[rand_out_polygon[i]][rand_in_polygon[j]])

    random.shuffle(rand_out_polygon)
    random.shuffle(rand_in_polygon)
    for i in range(len(sortNoise)):
        for j in range(step):
            CSIn1Back[i].append(sortNoise[rand_out_polygon[i]][rand_in_polygon[j]])

    # 在数组a后面加上a[0]使之成为一个首尾封闭的多边形
    sortCSIa1Add = makePolygon(CSIa1Back)
    sortCSIb1Add = makePolygon(CSIb1Back)
    sortCSIe1Add = makePolygon(CSIe1Back)
    sortCSIn1Add = makePolygon(CSIn1Back)

    # 初始化各个计算出的hd值
    ab_max = 0
    ae_max = 0
    an_max = 0

    # 最后各自的密钥
    a_list = []
    b_list = []
    e_list = []
    n_list = []

    plt.close()
    for i in range(len(CSIa1Back)):
        ab_hd = sys.maxsize
        ae_hd = sys.maxsize
        an_hd = sys.maxsize

        ab_index = 0
        ae_index = 0
        an_index = 0
        for j in range(len(CSIa1Back)):
            # 整体计算两个集合中每个多边形的hd值，取最匹配的（hd距离最接近的两个多边形）
            ab_d = standard_hd(CSIa1Back[i], CSIb1Back[j])
            ae_d = standard_hd(CSIa1Back[i], CSIe1Back[j])
            an_d = standard_hd(CSIa1Back[i], CSIn1Back[j])
            if ab_d < ab_hd:
                ab_hd = ab_d
                ab_index = j
            if ae_d < ae_hd:
                ae_hd = ae_d
                ae_index = j
            if an_d < an_hd:
                an_hd = an_d
                an_index = j

        # 将横纵坐标之和的值作为排序标准进行排序，然后进行查找，基于原数组的位置作为密钥值
        # 多边形内排序后相加后的结果不准确，float精度丢失，使得无法找到匹配的
        a_list.append(np.where(oneDimCSIa1 == sumEachDim(CSIa1Back, i))[0][0])
        b_list.append(np.where(oneDimCSIb1 == sumEachDim(CSIb1Back, ab_index))[0][0])
        e_list.append(np.where(oneDimCSIe1 == sumEachDim(CSIe1Back, ae_index))[0][0])
        n_list.append(np.where(oneDimCSIn1 == sumEachDim(CSIn1Back, an_index))[0][0])
        # print("\n")
        # print("\033[0;32;40mCSIa1", CSIa1Back[i], "\033[0m")
        # print("ab_hd", ab_hd, "\033[0;32;40mCSIb1", CSIb1Back[ab_index], "\033[0m")
        # print("ae_hd", ae_hd, "CSIe1", CSIe1Back[ae_index])
        # print("an_hd", an_hd, "CSIn1", CSIn1Back[an_index])

        # 比较各个独立计算的hd值的差异
        ab_max = max(ab_max, ab_hd)
        ae_max = max(ae_max, ae_hd)
        an_max = max(an_max, an_hd)

        # 绘图
        xa, ya = zip(*sortCSIa1Add[i])
        xb, yb = zip(*sortCSIb1Add[ab_index])
        xe, ye = zip(*sortCSIe1Add[ae_index])
        xn, yn = zip(*sortCSIn1Add[an_index])
        # plt.figure()
        # plt.plot(xa, ya, color="red", linewidth=2.5, label="a")
        # plt.plot(xb, yb, color="blue", linewidth=1, label="b")
        # plt.plot(xe, ye, color="black", linewidth=1, label="e")
        # plt.plot(xn, yn, color="yellow", linewidth=2.5, label="n") # 数量级差别太大，不方便显示
        # plt.legend(loc='upper left')
        # plt.show()

    # plt.close()
    print("ab_max", ab_max, "ae_max", ae_max, "an_max", an_max)
    print("keys of a:", a_list)
    print("keys of b:", b_list)
    print("keys of e:", e_list)
    print("keys of n:", n_list)

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
        codings += bin(a_list[i]) + "\n"

with open('../edit_distance/evaluations/key.txt', 'a', ) as f:
    f.write(codings)

print("a-b all", correctSum, "/", originSum, "=", correctSum / originSum)
print("a-e all", randomSum, "/", originSum, "=", randomSum / originSum)
print("a-n all", noiseSum, "/", originSum, "=", noiseSum / originSum)
