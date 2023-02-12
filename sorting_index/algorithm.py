import math
import random
import sys
from collections import deque
from itertools import chain

import numpy as np
import scipy.stats
from dtw import dtw, accelerated_dtw
from pyentrp import entropy as ent
import scipy.signal
from scipy.signal import medfilt
from scipy.special import rel_entr
from scipy.stats import pearsonr

from select_max_diff import genDiffMatrix, genDiffPerm


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


def splitEntropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, segLen, dataLen):
    # 先整体shuffle一次
    shuffleInd = np.random.permutation(dataLen)
    CSIa1Orig = CSIa1Orig[shuffleInd]
    CSIb1Orig = CSIb1Orig[shuffleInd]
    CSIe1Orig = CSIe1Orig[shuffleInd]

    sortCSIa1Reshape = CSIa1Orig[0:segLen * int(dataLen / segLen)]
    sortCSIb1Reshape = CSIb1Orig[0:segLen * int(dataLen / segLen)]
    sortCSIe1Reshape = CSIe1Orig[0:segLen * int(dataLen / segLen)]

    sortCSIa1Reshape = sortCSIa1Reshape.reshape(int(len(sortCSIa1Reshape) / segLen), segLen)
    sortCSIb1Reshape = sortCSIb1Reshape.reshape(int(len(sortCSIb1Reshape) / segLen), segLen)
    sortCSIe1Reshape = sortCSIe1Reshape.reshape(int(len(sortCSIe1Reshape) / segLen), segLen)
    n = len(sortCSIa1Reshape)
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

    _CSIa1Orig = []
    _CSIb1Orig = []
    _CSIe1Orig = []

    for i in range(len(sortCSIa1Reshape)):
        for j in range(len(sortCSIa1Reshape[i])):
            _CSIa1Orig.append(sortCSIa1Reshape[i][j])
            _CSIb1Orig.append(sortCSIb1Reshape[i][j])
            _CSIe1Orig.append(sortCSIe1Reshape[i][j])

    return np.array(_CSIa1Orig), np.array(_CSIb1Orig), np.array(_CSIe1Orig)


def findMinDiff(list, value):
    if list is False or len(list) == 0:
        return value
    l = len(list)
    min_diff = sys.maxsize
    for i in range(l):
        diff = abs(list[i] - value)
        min_diff = min(min_diff, diff)
    return min_diff


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


def findMinX(list):
    l = len(list)
    min_X = sys.maxsize
    for i in range(l):
        min_X = min(min_X, list[i][0])
    return min_X


def findMinY(list):
    l = len(list)
    min_Y = sys.maxsize
    for i in range(l):
        min_Y = min(min_Y, list[i][1])
    return min_Y


def findMaxX(list):
    l = len(list)
    max_X = 0
    for i in range(l):
        max_X = max(max_X, list[i][0])
    return max_X


def findMaxY(list):
    l = len(list)
    max_Y = 0
    for i in range(l):
        max_Y = max(max_Y, list[i][1])
    return max_Y


def genArray(list):
    res = []
    for i in range(len(list)):
        tmp = 0
        for j in range(len(list[i])):
            tmp += list[i][j]
        res.append(tmp)
    return res


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


# 等间隔选取selected个点
def genSample(list, ratio):
    selected = int(ratio * len(list[0]))
    # print("从", len(list[0]), "中选", selected)
    res = []
    for i in range(len(list)):
        tmp = 0
        if selected >= len(list[i]) or selected == 0:
            tmp += sum(list[i])
        else:
            for j in range(selected):
                tmp += list[i][j]
        res.append(tmp)
    return res


# 等间隔选取selected个点
# 相乘或相加影响不大
# def genSample(list, ratio):
#     selected = int(ratio * len(list[0]))
#     # print("从", len(list[0]), "中选", selected)
#     res = []
#     for i in range(len(list)):
#         tmp = 1
#         if selected >= len(list[i]) or selected == 0:
#             tmp *= sum(list[i])
#         else:
#             for j in range(selected):
#                 tmp *= list[i][j]
#         res.append(tmp)
#     return res


# 随机添加噪音点
def insertNoise(data, noise, index, ratio):
    res = list(data.copy())
    cur = 0
    for i in range(int(ratio * len(data))):
        res.insert(index[i] + cur, noise[index[i]])
        cur += 1  # 防止添加的点过于集中，按照原始数组长度随机生成的索引不能保证在添加完以后，还是原来的位置
    return res


def compare(list1, list2):
    manhattan_distance = lambda x, y: np.abs(x - y)
    metric = dtw(list1, list2, dist=manhattan_distance)[0]
    if metric >= len(list1):
        return 1
    else:
        return 0
    # if np.sum(list1) >= np.sum(list2):
    #     # if np.mean(list1) >= np.mean(list2):
    #     # if np.prod(list1, axis=0) >= np.prod(list2, axis=0):
    #     # if np.var(list1) >= np.std(list2):
    #     return 1
    # else:
    #     return 0


def nextPermutation(nums):
    if len(nums) <= 1:
        return
    for i in range(len(nums) - 2, -1, -1):
        if nums[i] < nums[i + 1]:
            for k in range(len(nums) - 1, i, -1):
                if nums[k] > nums[i]:
                    nums[i], nums[k] = nums[k], nums[i]
                    nums[i + 1:] = sorted(nums[i + 1:])
                    break
            break

        else:
            if i == 0:
                nums.sort()


# 按照从小到大的顺序依次增加threshold
# 如原始数据为[2,8,6,4],threshold=3
# [2,8,6,4]->[2,4,6,8]->[2+3,4+6,6+9,8+12]->[5,10,15,20]->[5,20,15,10]
def disperse(data, threshold):
    data = np.array(data)
    sortOrder = np.argsort(data)
    sortOrderOrder = np.argsort(sortOrder)
    sortedData = data[sortOrder]
    step = threshold
    for i in range(len(sortedData)):
        sortedData[i] += step
        step += threshold
    return sortedData[sortOrderOrder]


# 按照从小到大的顺序依次增加threshold
# unit个一组求和合并再进行排序
def level_disperse(raw, threshold, unit):
    data = []
    bucket = []
    for i in range(0, len(raw), unit):
        tmp = 0
        for j in range(unit):
            if i + j >= len(raw):
                break
            tmp += raw[i + j]
        data.append(tmp)

        tmp = []
        for j in range(unit):
            if i + j >= len(raw):
                break
            tmp.append(raw[i + j])
        bucket.append(tmp)
    data = np.array(data)
    bucket = np.array(bucket)

    sortOrder = np.argsort(data)
    sortOrderOrder = np.argsort(sortOrder)
    sortedData = data[sortOrder]
    bucket = bucket[sortOrder]
    step = threshold
    for i in range(len(sortedData)):
        # sortedData[i] += step
        for j in range(unit):
            if j >= len(bucket[i]):
                break
            bucket[i][j] += (step / unit)
        step += threshold
    res = chain.from_iterable(bucket[sortOrderOrder])
    return list(res)


# print(level_disperse([9, 10, 5, 3, 1, 7], 2, 6))
# print(level_disperse([9, 8, 5, 2, 1, 7], 2, 6))
# print(disperse([5, 3, 1, 7, 9, 10], 2))
# print(disperse([5, 2, 1, 7, 9, 8], 2))


# 滑动平均将长序列变成短序列，窗口有覆盖
def rescale(short, long):
    step = len(long) - len(short) + 1
    res = []
    for i in range(len(short)):
        tmp = 0
        for j in range(step):
            if i + j < len(long):
                tmp += long[i + j]
        res.append(round(tmp / step))
    return res


# 滑动平均将长序列变成短序列，窗口大小为2，无覆盖
def adj_rescale(long):
    step = 2
    res = []
    for i in range(0, len(long), step):
        tmp = 0
        for j in range(step):
            if i + j < len(long):
                tmp += long[i + j]
        res.append(round(tmp / step))
    return res


# 插值重复的短序列，将短序列变成长序列
def expand_rescale(short):
    res = []
    for i in range(len(short)):
        res.append(short[i])
        res.append(short[i])
    return res


# 相邻取平均
def adjacentMean(data):
    res = []
    for i in range(0, len(data), 2):
        res.append((data[i] + data[i + 1]) / 2)
    return res


# print(rescale([1, 2, 3], [2, 3, 4, 5, 6]))
# print(adj_rescale([1, 2, 3, 4]))
# print(expand_rescale([1, 2, 3, 4]))


# 重用删除的blots，对数据滑动窗口连加处理
def moving_sum(data, step):
    res = []
    for i in range(0, len(data), step):
        tmp = 0
        for j in range(step):
            tmp += data[(i + j) % len(data)]
        res.append(tmp)
    return res


# 重用删除的blots，对数据滑动窗口连加处理
def successive_moving_sum(data, step):
    res = []
    for i in range(len(data)):
        tmp = 0
        for j in range(step):
            tmp += data[(i + j) % len(data)]
        res.append(tmp)
    return res


def overlap_moving_sum(data, step, window):
    res = []
    for i in range(0, len(data), window):
        tmp = 0
        for j in range(step):
            tmp += data[(i + j) % len(data)]
        res.append(tmp)
    return res


# print(list(range(10)))
# print(moving_sum(list(range(10))), 2)
# print(successive_moving_sum(list(range(10)), 3))
# print(overlap_moving_sum(list(range(10)), 3, 2))

def kl_metric(data1, data2):
    count1 = list(np.unique(data1, return_counts=True)[1])
    count2 = list(np.unique(data2, return_counts=True)[1])

    if len(count1) > len(count2):
        for i in range(len(count1) - len(count2)):
            count2.append(count1[i])
    elif len(count2) > len(count1):
        for i in range(len(count2) - len(count1)):
            count1.append(count2[i])

    return sum(rel_entr(count1, count2))


def shannonEntropy(data):
    count = np.bincount(data)
    probs = count[np.nonzero(count)] / len(data)
    return - np.sum(probs * np.log(probs)) / np.log(len(probs))


# def dtw_metric(data1, data2):
#     distance = lambda x, y: np.abs(x - y)
#     data1 = np.array(data1)
#     data2 = np.array(data2)
#     # return dtw(data1, data2, dist=distance)[0]
#     return accelerated_dtw(data1, data2, dist=distance)[0]


# def dtw_metric_slope_weighting(s1, s2, X1=2, X2=2, X3=3):
# def dtw_metric(s1, s2, X1=1, X2=1, X3=0.5):
#     dtw = {}
#     for i in range(len(s1)):
#         dtw[(i, -1)] = float('inf')
#         dtw[(i, -2)] = float('inf')
#         dtw[(i, -3)] = float('inf')
#     for i in range(len(s2)):
#         dtw[(-1, i)] = float('inf')
#         dtw[(-2, i)] = float('inf')
#         dtw[(-3, i)] = float('inf')
#     dtw[(-1, -1)] = 0
#
#     for i in range(len(s1)):
#         for j in range(len(s2)):
#             dist = abs(s1[i] - s2[j])
#             dtw[(i, j)] = dist + min(X1 * dtw[(i - 1, j)], X2 * dtw[(i, j - 1)], X3 * dtw[i - 1, j - 1])
#
#     return dtw[len(s1) - 1, len(s2) - 1]

# def dtw_metric_step_pattern(s1, s2):
# def dtw_metric(s1, s2):
#     dtw = {}
#     for i in range(len(s1)):
#         dtw[(i, -1)] = float('inf')
#         dtw[(i, -2)] = 0
#     for i in range(len(s2)):
#         dtw[(-1, i)] = float('inf')
#         dtw[(-2, i)] = 0
#     dtw[(-1, -1)] = 0
#     dtw[(-1, -2)] = 0
#     dtw[(-2, -1)] = 0
#
#     for i in range(len(s1)):
#         for j in range(len(s2)):
#             dist = abs(s1[i] - s2[j])
#             dtw[(i, j)] = dist + min(dtw[(i - 1, j - 2)], dtw[(i - 2, j - 1)], dtw[i - 1, j - 1])
#
#     return dtw[len(s1) - 1, len(s2) - 1]


def dtw_metric_restraint(s1, s2, region):
    dtw = {}
    for i in range(len(s1)):
        dtw[(-1, i)] = float('inf')
    for i in range(len(s2)):
        dtw[(i, -1)] = float('inf')
    dtw[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            if region[i][j] != 0:
                dist = abs(s1[i] - s2[j])
            else:
                dist = float('inf')
            dtw[(i, j)] = dist + min(dtw[(i - 1, j)], dtw[(i, j - 1)], dtw[i - 1, j - 1])

    i = len(s1) - 1
    j = len(s2) - 1

    path = []
    path.append([i, j])
    while i > 0 or j > 0:
        previous = min(dtw[(i - 1, j)], dtw[(i, j - 1)], dtw[(i - 1, j - 1)])
        if previous == dtw[(i - 1, j - 1)]:
            i -= 1
            j -= 1
        elif previous == dtw[(i, j - 1)]:
            j -= 1
        elif previous == dtw[(i - 1, j)]:
            i -= 1
        path.append([i, j])

    # return dtw[len(s1) - 1, len(s2) - 1], path
    return dtw[len(s1) - 1, len(s2) - 1]


def pathRegion(dimension, width=1):
    res = np.zeros(shape=(dimension, dimension), dtype=np.int8)
    for i in range(dimension):
        for j in range(dimension):
            # 路径可达位置
            if i == j:
                res[i][j] = 1
                for k in range(width + 1):
                    if j - k >= 0:
                        res[i][j - k] = 1
                    if j + k <= dimension - 1:
                        res[i][j + k] = 1
    return res


def dtw_metric(data1, data2):
    # region = pathRegion(len(data1), 1)
    region = pathRegion(len(data1), len(data1))  # 未简化搜索区域
    return dtw_metric_restraint(data1, data2, region)


def euclidean_metric(data1, data2):
    if isinstance(data1, list) is False:
        return abs(data1 - data2)
    res = 0
    for i in range(len(data1)):
        res += abs(data1[i] - data2[i])
    return res
    # for i in range(len(data1)):
    #     res += math.pow(data1[i] - data2[i], 2)
    # return math.sqrt(res)


# print(dtw_metric([1, 2, 2, 3], [3, 4, 4, 5]))
# print(dtw_metric([1, 2, 3, 2], [3, 4, 4, 5]))


def pearson_metric(data1, data2):
    p = pearsonr(data1, data2)[0]
    if np.isnan(p):
        p = 0
    return p


# 一阶差分
def difference(data):
    res = []
    res.append(0)
    for i in range(len(data) - 1):
        res.append(data[i + 1] - data[i])
        # 精度更细
        # res.append(((data[i] - data[i - 1]) + (data[i + 1] - data[i - 1]) / 2) / 2)
    return res


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


# 定义计算离散点积分的函数
def integral(x, y):
    integrals = []
    integrals.append(0)
    for i in range(len(y) - 1):  # 计算梯形的面积
        integrals.append((y[i] + y[i + 1]) * (x[i + 1] - x[i]) / 2)
    return integrals


# 定义计算离散点积分的函数
def integral_from_start(x, y):
    import scipy
    from scipy.integrate import simps  # 用于计算积分
    integrals = []
    for i in range(len(y)):  # 计算梯形的面积，由于是累加，所以是切片"i+1"
        integrals.append(scipy.integrate.trapz(y[:i + 1], x[:i + 1]))
    return integrals


# 对于平稳数据，先差分，后平方，再积分还是趋于平稳
# 比较平滑，且单调递增
def derivative_sq_integral_smth(data):
    index = list(range(len(data)))
    diff = derivative(index, data)
    square = np.power(diff, 2)
    return integral_from_start(index, square)


# 与derivative_sq_integral_smth接近
def diff_sq_integral_smth(data):
    index = list(range(len(data)))
    diff = difference(data)
    diff_diff = difference(diff)
    second = integral_from_start(index, np.array(data) * np.array(diff_diff))
    return np.array(diff) * np.array(data) - np.array(second)


# 过于平缓，且趋近于0
def diff_sq_integral_stationary(data):
    index = list(range(len(data)))
    diff = difference(data)
    square = np.power(diff, 2)
    return integral(index, square)


# 杂乱，是所有方法里面最好的
def diff_sq_integral_rough(data):
    index = list(range(len(data)))
    diff = difference(data)
    diff_diff = difference(diff)
    second = integral(index, np.array(data) * np.array(diff_diff))
    return np.array(diff) * np.array(data) - np.array(second)


def integral_sq_derivative_increment(data):
    index = list(range(len(data)))
    intgrl = integral_from_start(index, data)
    square = np.power(intgrl, 2)
    diff = derivative(index, square)
    return diff


def integral_sq_derivative(data):
    index = list(range(len(data)))
    intgrl = integral(index, data)
    square = np.power(intgrl, 2)
    diff = derivative(index, square)
    return diff


# 第一个点处恒为0
def integral_sq_diff(data):
    index = list(range(len(data)))
    intgrl = integral(index, data)
    square = np.power(intgrl, 2)
    diff = difference(square)
    return diff


# # y = [1, 4, 9, 16, 25]
# y = [21, 29, 24, 32, 20, 21, 23, 21, 32, 23, 43, 14, 31, 30, 25]
# plt.figure()
# plt.plot(y, "r")
# plt.plot(integral_sq_diff(y), "c")
# plt.plot(integral_sq_derivative(y), "y")
# plt.plot(diff_sq_integral_rough(y), "k")
# plt.show()

# print(pearson_metric(expand_rescale([1, 2, 3, 4]), [10, 15, 15, 16, 19, 20, 21, 22]))


def sortPermOfA(data, interval):
    data_seg = []
    data_back = data.copy()
    data_back.sort()
    for i in range(0, len(data_back), interval):
        tmp = []
        for j in range(interval):
            tmp.append(data_back[i + j])
        tmp.append(int(i / interval))
        data_seg.append(tmp)
    perm = np.random.permutation(list(range(len(data_seg))))
    data_seg = np.array(data_seg)[perm]
    # [原始数据 原始数据在原始序列中分段后的排序索引 置换后的索引]
    data_perm = []
    for i in range(len(data_seg)):
        for j in range(len(data_seg[i]) - 1):
            data_perm.append([data_seg[i][j], data_seg[i][interval], i])
    code = []
    for i in range(len(data)):
        for j in range(len(data_perm)):
            if math.isclose(data[i], data_perm[j][0], rel_tol=1e-5):
                code.append(data_perm[j][2])
    return data_perm, code


def sortPermOfB(data_perm, data, interval):
    simple_perm = []
    for i in range(int(len(data_perm) / interval)):
        for j in range(len(data_perm)):
            if i == data_perm[j][1]:
                simple_perm.append([i, data_perm[j][2]])
                break
    data_back = data.copy()
    data_back.sort()

    data_seg = []
    for i in range(0, len(data_back), interval):
        tmp = []
        for j in range(interval):
            tmp.append(data_back[i + j])
        data_seg.append(tmp)

    data_index = []
    for i in range(len(data_seg)):
        for j in range(interval):
            data_index.append([data_seg[i][j], i])

    code = []
    for i in range(len(data_index)):
        index = data_index[i][1]
        data_index[i].append(simple_perm[index][1])

    for i in range(len(data)):
        for j in range(len(data_index)):
            if math.isclose(data[i], data_index[j][0], rel_tol=1e-5):
                code.append(data_index[j][2])

    return code


def diffFilter(data):
    res = 0
    for i in range(len(data) - 1):
        res += abs(data[i + 1] - data[i])
    return res


def medianFilter(data):
    n = len(data)
    data_sort = data.copy()
    data_sort.sort()
    if n % 2 == 1:
        return data_sort[int(n / 2)]
    else:
        return (data_sort[int(n / 2)] + data_sort[int(n / 2) - 1]) / 2


def sampleSum(data, select):
    n = len(data)
    data_sort = data.copy()
    data_sort.sort()
    start = int(n / 2) - int(select / 2)
    res = sum(data_sort[start:start + select])
    return res


def sortSegPermOfA(data, segLen):
    data_seg = []
    data_back = data.copy()
    data_back.sort()
    # 排序后按照interval的长度分段
    for i in range(0, len(data_back), segLen):
        tmp = []
        for j in range(segLen):
            tmp.append(data_back[i + j])
        tmp.append(int(i / segLen))  # 原始的索引
        data_seg.append(tmp)
    # 找出间距最小的分段
    # data_seg_sort = []
    # for i in range(len(data_seg)):
    #     data_seg_sort.append(sum(data_seg[i][0: interval]))
    # data_seg_sort.sort()
    # min_diff = sys.maxsize
    # min_index = -1
    # for i in range(len(data_seg_sort) - 1):
    #     min_diff = min(min_diff, abs(data_seg_sort[i + 1] - data_seg_sort[i]))
    # for i in range(len(data_seg_sort) - 1):
    #     if math.isclose(min_diff, abs(data_seg_sort[i + 1] - data_seg_sort[i]), rel_tol=1e-5):
    #         min_index = i
    # for i in range(len(data_seg) - 1, - 1, -1):
    #     if data_seg[i][len(data_seg[i]) - 1] == min_index:
    #         del data_seg[i]
    # 删除
    # for i in range(len(data_seg) - 2, -1, -1):
    #     if sum(data_seg[i + 1][0: interval]) - sum(data_seg[i][0: interval]) < 0.15:
    #         del data_seg[i + 1]
    # 置换
    perm = np.random.permutation(list(range(len(data_seg))))
    data_seg = np.array(data_seg)[perm]
    data_seg = data_seg.tolist()
    publish = []
    data_seg_back = []
    for i in range(len(data_seg)):
        # data_seg_back.append(np.mean(data_seg[i][0: len(data_seg[i]) - 1]))
        data_seg_back.append(sum(data_seg[i][0: len(data_seg[i]) - 1]))
    for j in range(len(data_seg)):
        for k in range(len(data_seg[j]) - 1):
            for i in range(len(data)):
                if math.isclose(data[i], data_seg[j][k], rel_tol=1e-5):
                    publish.append(i)
                    # 不重复选
                    data[i] = sys.maxsize
                    break
    return data_seg_back, publish, list(perm)


def sortSegPermOfB(publish, data, segLen):
    data_perm = []
    for i in range(len(publish)):
        data_perm.append(data[publish[i]])
    data_seg = []
    for i in range(0, len(data_perm), segLen):
        # tmp = 0
        # sample = data_perm[i:i + segLen].copy()
        # sample.sort()
        # data_seg.append(sample[3] + sample[2] + sample[0] + sample[1])
        # 欧式距离，平方和
        # tmp = 0
        # for j in range(segLen):
        #     tmp += (data_perm[i + j] * data_perm[i + j])
        # data_seg.append(tmp)
        # 求和
        data_seg.append(sum(data_perm[i:i + segLen]))
        # 求积
        # tmp = 1
        # for j in range(segLen):
        #     tmp *= data_perm[i + j]
        # data_seg.append(tmp)
        # 采样式求平均
        # data_seg.append(sampleSum(data_perm[i:i + segLen], segLen - 2))
        # 中值滤波
        # data_seg.append(medianFilter(data_perm[i:i + segLen]))
    perm = list(np.argsort(data_seg))
    perm = list(np.argsort(perm))
    return data_seg, perm


# r1, r2, r3 = sortSegPermOfA([1, 3, 5, 2, 4, 6], 2)
# print(r1, r2, r3)
# print(sortSegPermOfB(r2, [2, 4, 6, 3, 5, 7], 2))

# def refSortSegPermOfB(publish, data, segLen):
#     data_perm = []
#     for i in range(len(publish)):
#         data_perm.append(data[publish[i]])
#     data_seg = []
#     for i in range(0, len(data_perm), segLen):
#         # 选取重叠值
#         tmp_mean = []
#         for j in range(i, i + segLen - 1):
#             if data_perm[j] <= data_perm[j + 1]:
#                 tmp_mean.append(data_perm[j])
#         data_seg.append(np.mean(tmp_mean))
#     perm = list(np.argsort(data_seg))
#     perm = list(np.argsort(perm))
#     return data_seg, perm


def refSortSegPermOfB(publish, data, segLen):
    data_seg = []
    data_back = data.copy()
    data2 = data.copy()
    data_back.sort()
    # 排序后按照interval的长度分段
    for i in range(0, len(data_back), segLen):
        tmp = []
        for j in range(segLen):
            tmp.append(data_back[i + j])
        tmp.append(int(i / segLen))  # 原始的索引
        data_seg.append(tmp)
    data_index = []
    for j in range(len(data_seg)):
        for k in range(len(data_seg[j]) - 1):
            for i in range(len(data)):
                if math.isclose(data[i], data_seg[j][k], rel_tol=1e-5):
                    data_index.append(i)
                    # 不重复选
                    data[i] = sys.maxsize
                    break
    # 为什么有这个bug
    data_index = data_index[0: int(len(data_index) / segLen) * segLen]
    publishReshape = np.array(publish).reshape(int(len(publish) / segLen), segLen)
    indexReshape = np.array(data_index).reshape(int(len(data_index) / segLen), segLen)

    possibleIndex = []
    for i in range(len(publishReshape)):
        maxIntersection = 0
        for j in range(len(indexReshape)):
            intersectionCard = len(list(set(publishReshape[i]) & set(indexReshape[j])))
            maxIntersection = max(maxIntersection, intersectionCard)
        for j in range(len(indexReshape)):
            intersection = list(set(publishReshape[i]) & set(indexReshape[j]))
            # if maxIntersection == len(intersection):
            #     possibleIndex.append(intersection)
            #     break
            jump = False
            if maxIntersection == len(intersection):
                # possibleIndex.append(intersection)
                # break
                # 必须包含两个中位数
                for k in range(len(indexReshape[j])):
                    if indexReshape[j][k] == publishReshape[i][int(segLen / 2) - 1] or \
                        indexReshape[j][k] == publishReshape[i][int(segLen / 2)]:
                        possibleIndex.append(intersection)
                        jump = True
                        break
            if jump is True:
                break

    data_perm = []
    for i in range(len(possibleIndex) - 1, -1, -1):
        # 去掉长度小于3的分段
        if len(possibleIndex[i]) <= 3:
            del possibleIndex[i]
            continue
        tmp = []
        for j in range(len(possibleIndex[i])):
            tmp.append(data2[possibleIndex[i][j]])
        data_perm.append(tmp)
    # 由于possibleIndex是倒序遍历，故反转一下
    data_perm.reverse()
    data_res = []
    for i in range(len(data_perm)):
        data_res.append(np.mean(data_perm[i]))
    perm = list(np.argsort(data_res))
    perm = list(np.argsort(perm))
    return possibleIndex, data_res, perm


def refSortSegPermOfA(publish, data):
    data_seg = []
    for i in range(len(publish)):
        tmp = []
        for j in range(len(publish[i])):
            tmp.append(data[publish[i][j]])
        data_seg.append(np.mean(tmp))
    perm = list(np.argsort(data_seg))
    perm = list(np.argsort(perm))
    return data_seg, perm


# a_metric, publish, a_list_number = sortSegPermOfA([1, 3, 5, 2, 4, 6], 3)
# print(a_metric, publish, a_list_number)
# print(refSortSegPermOfB(publish, [2, 4, 3, 6, 5, 7], 3))


def adaSortSegPermOfA(data, segLen):
    data_seg = []
    data_back = data.copy()
    data_back.sort()
    random_interval = []
    while sum(random_interval) < len(data):
        ri = np.random.randint(5, segLen + 1)
        if len(data) - ri < segLen:
            random_interval.append(len(data) - ri)
            break
        random_interval.append(ri)
    start = 0
    for i in range(len(random_interval)):
        tmp = []
        tmp += data_back[start:start + random_interval[i]]
        start += random_interval[i]
        tmp.append(i)
        data_seg.append(tmp)
    # 找出间距最小的分段
    # for de in range(5):
    #     data_seg_sort = []
    #     for i in range(len(data_seg)):
    #         data_seg_sort.append(sum(data_seg[i][0: interval]))
    #     data_seg_sort.sort()
    #     min_diff = sys.maxsize
    #     min_index = -1
    #     for i in range(len(data_seg_sort) - 1):
    #         min_diff = min(min_diff, abs(data_seg_sort[i + 1] - data_seg_sort[i]))
    #     for i in range(len(data_seg_sort) - 1):
    #         if math.isclose(min_diff, abs(data_seg_sort[i + 1] - data_seg_sort[i]), rel_tol=1e-5):
    #             min_index = i
    #     for i in range(len(data_seg) - 1, - 1, -1):
    #         if data_seg[i][len(data_seg[i]) - 1] == min_index:
    #             del data_seg[i]
    # 置换
    perm = np.random.permutation(list(range(len(data_seg))))
    data_seg = np.array(data_seg)[perm]
    publish = []
    data_seg_back = []
    for i in range(len(data_seg)):
        data_seg_back.append(sum(data_seg[i][0: len(data_seg[i]) - 1]))
    for j in range(len(data_seg)):
        tmp = []
        for k in range(len(data_seg[j]) - 1):
            for i in range(len(data)):
                if math.isclose(data[i], data_seg[j][k], rel_tol=1e-5):
                    tmp.append(i)
                    data[i] = sys.maxsize
                    break
        publish.append(tmp)

    return data_seg_back, publish, list(perm)


def adaSortSegPermOfB(publish, data):
    data_seg = []
    for i in range(len(publish)):
        tmp = 0
        for j in range(len(publish[i])):
            tmp += data[publish[i][j]]
        data_seg.append(tmp)
        # tmp = 0
        # sample = data_perm[i:i + interval].copy()
        # sample.sort()
        # data_seg.append(sample[3] + sample[2] + sample[0] + sample[1])
        # 欧式距离，平方和
        # data_seg.append(euclidean_metric(data_perm[i:i + interval], np.zeros(interval)))
        # 求积
        # tmp = 1
        # for j in range(interval):
        #     tmp *= data_perm[i + j]
        # data_seg.append(tmp)
        # 采样式求平均
        # data_seg.append(sampleSum(data_perm[i:i + interval], interval - 2))
        # 中值滤波
        # data_seg.append(medianFilter(data_perm[i:i + interval]))
    perm = list(np.argsort(data_seg))
    perm = list(np.argsort(perm))
    return data_seg, perm


# print(sortSegPermOfA([1, 3, 5, 2, 4, 6], 2))
# print(sortSegPermOfB(sortSegPermOfA([1, 3, 5, 2, 4, 6], 2)[0], [2, 4, 6, 3, 5, 7], 2))

# partner = "himself"
partner = "noise"

# linear
# perm_threshold = 0.2
# blot_threshold = 10

# square
# perm_threshold = 15
# blot_threshold = 50

# exponent
# perm_threshold = 10
# blot_threshold = 30

# base
# perm_threshold = 5
# blot_threshold = 10
perm_threshold = 8
# blot_threshold = 10
# 不同层用不同阈值来删除，最后一层为0
# blot_threshold = [10, 10, 10, 10, 10]
blot_threshold = [10, 10, 10, 5, 0]


# cosine
# perm_threshold = 0.5
# blot_threshold = 0.5

# logarithm
# perm_threshold = 0.5
# blot_threshold = 1

# box-cox
# perm_threshold = 0.05
# blot_threshold = 0.1

# reciprocal
# perm_threshold = 0.5
# blot_threshold = 1

# tangent
# perm_threshold = 1
# blot_threshold = 10

# remainder
# perm_threshold = 1
# blot_threshold = 10

# quotient
# perm_threshold = 0.5
# blot_threshold = 1.5

# difference
# perm_threshold = 0.1
# blot_threshold = 0.5

# integral_square_derivative
# perm_threshold = 0.5
# blot_threshold = 0.8

# integral_square_derivative
# dtw_slope_weighting
# perm_threshold = 1.2
# blot_threshold = 1.8

# diff
# perm_threshold = 0.5
# blot_threshold = 1.5

# diff_sq_integral_rough
# perm_threshold = 1
# blot_threshold = 1.5

# 0
# list 待排序的数组
# length 比较中所需的最小比较单元的长度
# 连续metric差值在阈值之内的值进行置换，返回置换结果
def levelMetricSortPerm(data, length, noise, metric):
    extend_list = data.copy()
    noise_list = noise.copy()
    for i in range(2 ** math.ceil(np.log2(len(data))) - len(data)):
        extend_list.append(data[i - len(data)])
    for i in range(2 ** math.ceil(np.log2(len(noise))) - len(noise)):
        noise_list.append(noise[i - len(noise)])

    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]

    perm = np.arange(0, len(extend_list))
    cnt = 0
    # while True:
    while cnt <= 50:
        cnt += 1
        min_diffs = sys.maxsize
        combine = list(zip(extend_list, perm))

        metrics = []
        cur = 1
        for i in range(1, max_level):
            tmp_metrics = []
            for j in range(cur, cur + 2 ** i, 2):
                l1l = index[j][0] - 1
                l1r = index[j][1]
                l2l = index[j + 1][0] - 1
                l2r = index[j + 1][1]
                if l1r - l1l < length:
                    break
                if partner == "himself":
                    # 自己与自己相比
                    step = int((l1r - l1l) / 2)
                    tmp_metrics.append(
                        eval(metric + '_metric(extend_list[l1l: l1l + step], extend_list[l1l + step: l1r])'))
                    tmp_metrics.append(
                        eval(metric + '_metric(extend_list[l2l: l2l + step], extend_list[l2l + step: l2r])'))
                else:
                    # 与噪音点相比
                    tmp_metrics.append(eval(metric + '_metric(extend_list[l1l: l1r], noise_list[l1l: l1r])'))
                    tmp_metrics.append(eval(metric + '_metric(extend_list[l2l: l2r], noise_list[l2l: l2r])'))
            metrics.append(tmp_metrics)
            cur = cur + 2 ** i

            min_diff = sys.maxsize

            # ------------------------------------------------------
            # 使用metric的一阶差分排序
            # tmp_metrics = difference(tmp_metrics)
            # ------------------------------------------------------

            sort_metrics = tmp_metrics.copy()
            sort_metrics.sort()
            for j in range(len(sort_metrics) - 1):
                min_diff = min(min_diff, abs(sort_metrics[j + 1] - sort_metrics[j]))
            min_diffs = min(min_diffs, min_diff)

        # if min_diffs < 0.02:  # static
        # if min_diffs < 0.2:  # static
        if min_diffs < perm_threshold:
            # print("\033[0;32;40mmin_diff", min_diffs, "\033[0m")  # green
            np.random.shuffle(combine)
            extend_list, perm = zip(*combine)
        else:
            print("\033[0;33;40mmin_diff", min_diffs, "\033[0m")  # yellow
            break
    print("\033[0;32;40mmin_diff", min_diffs, "\033[0m")  # green

    # ------------------------------------------------------
    # 使用metric的一阶差分排序
    # metrics_back = metrics.copy()
    # metrics = []
    # for i in range(len(metrics_back)):
    #     diff_metric = difference(metrics_back[i])
    #     metrics.append(diff_metric)
    # ------------------------------------------------------

    for i in range(0, len(metrics)):
        tmp = np.argsort(metrics[i])
        code.append(tmp)

    return_code = []
    for i in range(1, len(intervals) - 1):
        for j in range(len(code[i])):
            return_code.append(code[i][j])
        if intervals[i] < length:
            break

    # print("perm", perm)
    return return_code, np.array(perm)


# 1
# list 待排序的数组
# length 比较中所需的最小比较单元的长度
def levelMetricSort(data, length, noise, perm, metric):
    extend_list = data.copy()
    noise_list = noise.copy()
    for i in range(2 ** math.ceil(np.log2(len(data))) - len(data)):
        extend_list.append(data[i - len(data)])
    for i in range(2 ** math.ceil(np.log2(len(noise))) - len(noise)):
        noise_list.append(noise[i - len(noise)])

    extend_list = np.array(extend_list)
    extend_list = extend_list[perm]
    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]

    metrics = []
    cur = 1
    for i in range(1, max_level):
        tmp_metrics = []
        for j in range(cur, cur + 2 ** i, 2):
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l2l = index[j + 1][0] - 1
            l2r = index[j + 1][1]
            if l1r - l1l < length:
                break
            if partner == "himself":
                # 自己与自己相比
                step = int((l1r - l1l) / 2)
                tmp_metrics.append(eval(metric + '_metric(extend_list[l1l: l1l + step], extend_list[l1l + step: l1r])'))
                tmp_metrics.append(eval(metric + '_metric(extend_list[l2l: l2l + step], extend_list[l2l + step: l2r])'))
            else:
                # 与噪音点相比
                tmp_metrics.append(eval(metric + '_metric(extend_list[l1l: l1r], noise_list[l1l: l1r])'))
                tmp_metrics.append(eval(metric + '_metric(extend_list[l2l: l2r], noise_list[l2l: l2r])'))
        metrics.append(tmp_metrics)
        cur = cur + 2 ** i

    for i in range(0, len(metrics)):
        tmp = np.argsort(metrics[i])
        code.append(tmp)

    return_code = []
    for i in range(1, len(intervals) - 1):
        for j in range(len(code[i])):
            return_code.append(code[i][j])
        if intervals[i] < length:
            break

    return return_code


# 2
# list 待排序的数组
# length 比较中所需的最小比较单元的长度
# 根据perm先置换，然后B生成密钥，返回要删除的点
def lossyLevelMetricSortOfB(data, length, noise, perm, metric):
    extend_list = data.copy()
    noise_list = noise.copy()
    for i in range(2 ** math.ceil(np.log2(len(data))) - len(data)):
        extend_list.append(data[i - len(data)])
    for i in range(2 ** math.ceil(np.log2(len(noise))) - len(noise)):
        noise_list.append(noise[i - len(noise)])

    extend_list = np.array(extend_list)
    extend_list = extend_list[perm]
    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]

    # 丢弃容易出错的位置
    blots = []

    metrics = []
    origin_metrics = []
    cur = 1
    for i in range(1, max_level):
        tmp_metrics = []
        blot = []
        for j in range(cur, cur + 2 ** i, 2):
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l2l = index[j + 1][0] - 1
            l2r = index[j + 1][1]
            if l1r - l1l < length:
                break
            if partner == "himself":
                # 自己与自己相比
                step = int((l1r - l1l) / 2)
                tmp_metrics.append(eval(metric + '_metric(extend_list[l1l: l1l + step], extend_list[l1l + step: l1r])'))
                tmp_metrics.append(eval(metric + '_metric(extend_list[l2l: l2l + step], extend_list[l2l + step: l2r])'))
            else:
                # 与噪音点相比
                tmp_metrics.append(eval(metric + '_metric(extend_list[l1l: l1r], noise_list[l1l: l1r])'))
                tmp_metrics.append(eval(metric + '_metric(extend_list[l2l: l2r], noise_list[l2l: l2r])'))
        # 选出相邻差距最小的点，若差距过小，则挑出来进行删除

        # ------------------------------------------------------
        # 使用metric的一阶差分排序
        # tmp_metrics = difference(tmp_metrics)
        # ------------------------------------------------------

        # 选出差距接近阈值的度量值，标记出左度量值以待删除
        origin_metrics.append(tmp_metrics.copy())
        sort_metrics = tmp_metrics.copy()
        sort_metrics.sort()
        for j in range(len(sort_metrics) - 1):
            if abs(sort_metrics[j + 1] - sort_metrics[j]) < blot_threshold:
                for k in range(len(tmp_metrics)):
                    if tmp_metrics[k] == sort_metrics[j + 1]:
                        blot.append([i - 1, k])

        blots.append(blot)
        # 从后往前删除，防止待删除的索引溢出
        for j in range(len(blot) - 1, -1, -1):
            blot.sort()  # 防止索引溢出
            drop = blot[j][1]
            if -1 < drop < len(tmp_metrics):
                del tmp_metrics[drop]
        metrics.append(tmp_metrics)
        cur = cur + 2 ** i

    min_diffs = sys.maxsize
    for i in range(len(metrics)):
        sort_metrics = metrics[i].copy()
        sort_metrics.sort()
        min_diff = sys.maxsize
        for j in range(len(sort_metrics) - 1):
            min_diff = min(min_diff, abs(sort_metrics[j + 1] - sort_metrics[j]))
        min_diffs = min(min_diffs, min_diff)
    print("\033[0;36;40mmin_diff after deleting", min_diffs, "\033[0m")  # cyan

    if min_diffs < blot_threshold:
        print("error: bad blot")

    for i in range(0, len(metrics)):
        tmp = np.argsort(metrics[i])
        code.append(tmp)

    return_code = []
    for i in range(1, len(intervals) - 1):
        for j in range(len(code[i])):
            return_code.append(code[i][j])
        if intervals[i] < length:
            break

    # print("blots", blots)
    return return_code, blots


# 3
# list 待排序的数组
# length 比较中所需的最小比较单元的长度
# 根据perm先置换，再根据blots删除点，返回密钥
# window=0表示不重用blots，否则表示重用blots时使用滑动窗口的宽度
def lossyLevelMetricSortOfA(data, length, noise, perm, blots, metric, window=0):
    extend_list = data.copy()
    noise_list = noise.copy()
    for i in range(2 ** math.ceil(np.log2(len(data))) - len(data)):
        extend_list.append(data[i - len(data)])
    for i in range(2 ** math.ceil(np.log2(len(noise))) - len(noise)):
        noise_list.append(noise[i - len(noise)])

    extend_list = np.array(extend_list)
    extend_list = extend_list[perm]
    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]

    metrics = []
    recycle_bin = []
    cur = 1
    for i in range(1, max_level):
        tmp_metrics = []
        recycle = []
        for j in range(cur, cur + 2 ** i, 2):
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l2l = index[j + 1][0] - 1
            l2r = index[j + 1][1]
            if l1r - l1l < length:
                break
            if partner == "himself":
                # 自己与自己相比
                step = int((l1r - l1l) / 2)
                tmp_metrics.append(eval(metric + '_metric(extend_list[l1l: l1l + step], extend_list[l1l + step: l1r])'))
                tmp_metrics.append(eval(metric + '_metric(extend_list[l2l: l2l + step], extend_list[l2l + step: l2r])'))
            else:
                # 与噪音点相比
                tmp_metrics.append(eval(metric + '_metric(extend_list[l1l: l1r], noise_list[l1l: l1r])'))
                tmp_metrics.append(eval(metric + '_metric(extend_list[l2l: l2r], noise_list[l2l: l2r])'))

        # ------------------------------------------------------
        # 使用metric的一阶差分排序
        # tmp_metrics = difference(tmp_metrics)
        # ------------------------------------------------------

        # 根据blots删除差距接近阈值的左度量值
        if blots[i - 1] is not None and len(blots[i - 1]) != 0:
            # 从后往前删除，防止待删除的索引超出已删除的数据
            for j in range(len(blots[i - 1]) - 1, -1, -1):
                drop = blots[i - 1][j][1]
                if -1 < drop < len(tmp_metrics):
                    # 回收删除的blots
                    recycle.append(tmp_metrics[drop])
                    del tmp_metrics[drop]
        metrics.append(tmp_metrics)
        recycle_bin.append(recycle)
        cur = cur + 2 ** i

    all_diffs = []
    for i in range(len(metrics)):
        tmp = metrics[i].copy()
        tmp.sort()
        tmp_diffs = []
        for j in range(len(tmp) - 1):
            tmp_diffs.append(abs(tmp[j + 1] - tmp[j]))
        all_diffs.append(tmp_diffs)
    # print("all_diffs", all_diffs)
    # print("all_metrics", metrics)

    for i in range(len(metrics)):
        retain_tmp = np.argsort(metrics[i])
        if window > 0:
            # 回收的blots产生的code
            blots_tmp = np.argsort(successive_moving_sum(recycle_bin[i], window))
            # print("blots_tmp", blots_tmp)
            tmp = np.hstack((retain_tmp, blots_tmp))
            # print("tmp", tmp)
            code.append(tmp)
        else:
            code.append(retain_tmp)

    min_diffs = sys.maxsize
    for i in range(len(metrics)):
        sort_metrics = metrics[i].copy()
        sort_metrics.sort()
        min_diff = sys.maxsize
        for j in range(len(sort_metrics) - 1):
            min_diff = min(min_diff, abs(sort_metrics[j + 1] - sort_metrics[j]))
        min_diffs = min(min_diffs, min_diff)

    print("\033[3;35;40mmin_diff", min_diffs, "\033[0m")  # pink
    # if min_diffs < blot_threshold:
    #     print("error: bad combined blot")

    return_code = []
    for i in range(1, len(intervals) - 1):
        for j in range(len(code[i])):
            return_code.append(code[i][j])
        if intervals[i] < length:
            break

    return return_code


def cleanBlot(key, blots):
    for i in range(len(blots)):
        if len(blots[i]) != 0:
            for j in range(len(blots[i])):
                index = 2 * (2 ** blots[i][j][0] - 1) + blots[i][j][1]
                del key[index]
    return key


# 4
# list 待排序的数组
# length 比较中所需的最小比较单元的长度
def simpleLevelMetricSort(data, length, noise, metric):
    extend_list = data.copy()
    noise_list = noise.copy()
    for i in range(2 ** math.ceil(np.log2(len(data))) - len(data)):
        extend_list.append(data[i - len(data)])
    for i in range(2 ** math.ceil(np.log2(len(noise))) - len(noise)):
        noise_list.append(noise[i - len(noise)])

    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]

    min_diffs = sys.maxsize
    metrics = []
    cur = 1
    for i in range(1, max_level):
        tmp_metrics = []
        for j in range(cur, cur + 2 ** i, 2):
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l2l = index[j + 1][0] - 1
            l2r = index[j + 1][1]
            if l1r - l1l < length:
                break
            if partner == "himself":
                # 自己与自己相比
                step = int((l1r - l1l) / 2)
                tmp_metrics.append(eval(metric + '_metric(extend_list[l1l: l1l + step], extend_list[l1l + step: l1r])'))
                tmp_metrics.append(eval(metric + '_metric(extend_list[l2l: l2l + step], extend_list[l2l + step: l2r])'))
            else:
                # 与噪音点相比
                tmp_metrics.append(eval(metric + '_metric(extend_list[l1l: l1r], noise_list[l1l: l1r])'))
                tmp_metrics.append(eval(metric + '_metric(extend_list[l2l: l2r], noise_list[l2l: l2r])'))

        min_diff = sys.maxsize
        sort_metrics = tmp_metrics.copy()
        sort_metrics.sort()
        for j in range(len(sort_metrics) - 1):
            min_diff = min(min_diff, abs(sort_metrics[j + 1] - sort_metrics[j]))
        min_diffs = min(min_diffs, min_diff)

        metrics.append(tmp_metrics)
        cur = cur + 2 ** i

    print("\033[0;32;40mmin_diff", min_diffs, "\033[0m")

    for i in range(0, len(metrics)):
        tmp = np.argsort(metrics[i])
        code.append(tmp)

    return_code = []
    for i in range(1, len(intervals) - 1):
        for j in range(len(code[i])):
            return_code.append(code[i][j])
        if intervals[i] < length:
            break

    return return_code


# 5
# list 待排序的数组
# length 比较中所需的最小比较单元的长度
# 连续metric差值在阈值之内的值进行置换，返回置换结果
def levelNoiseMetricSortPerm(data, length, noise, metric):
    extend_list = data.copy()
    designed_noise = []
    noise_list = noise
    # noise_list = list(np.random.normal(loc=np.mean(noise),
    #                                    scale=np.std(noise, ddof=1),
    #                                    size=len(noise) * 100))
    for i in range(2 ** math.ceil(np.log2(len(data))) - len(data)):
        extend_list.append(data[i - len(data)])

    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]

    min_diffs = sys.maxsize

    # 生成随机噪音点，满足阈值的要求
    interval_length = 2
    lower_bound = 5.5
    # lower_bound = 10  # random way points
    # upper_bound = 5000
    upper_bound = sys.maxsize
    two_samples_metrics = []
    four_samples_metrics = []
    eight_samples_metrics = []
    sixteen_samples_metrics = []

    # 针对最底层数据生成噪音点
    for i in range(0, len(extend_list), interval_length):
        tmp_noise_list = random.sample(noise_list, interval_length)
        two_tmp_metric = eval(metric + '_metric(extend_list[i: i + interval_length], tmp_noise_list)')
        if findMinDiff(two_samples_metrics, two_tmp_metric) >= lower_bound \
                and findMinDiff(two_samples_metrics, two_tmp_metric) <= upper_bound:
            two_samples_metrics.append(two_tmp_metric)
            designed_noise += tmp_noise_list
        else:
            while True:
                tmp_noise_list = random.sample(noise_list, interval_length)
                # raw的[0:2]与insert比较
                two_tmp_metric = eval(metric + '_metric(extend_list[i: i + interval_length], tmp_noise_list)')
                if findMinDiff(two_samples_metrics, two_tmp_metric) >= lower_bound \
                        and findMinDiff(two_samples_metrics, two_tmp_metric) <= upper_bound:
                    two_samples_metrics.append(two_tmp_metric)
                    # insert插入到noise[0:2]
                    designed_noise += tmp_noise_list
                    break

    # 检查倒数第二层
    for i in range(0, len(extend_list), interval_length * 2):
        four_tmp_metric = eval(metric + '_metric(extend_list[i: i + interval_length * 2], '
                                        'designed_noise[i: i + interval_length * 2])')
        if findMinDiff(four_samples_metrics, four_tmp_metric) >= lower_bound \
                and findMinDiff(four_samples_metrics, four_tmp_metric) <= upper_bound:
            four_samples_metrics.append(four_tmp_metric)
        else:
            while True:
                tmp_noise_list = random.sample(noise_list, interval_length)
                # 只改变最右端的一个interval的噪音点
                updated_noise = designed_noise[i: i + interval_length] + tmp_noise_list
                # raw的[0:4]与noise的[0:2]||insert比较
                four_tmp_metric = eval(metric + '_metric(extend_list[i: i + interval_length * 2], updated_noise)')
                # double check可能影响的下层
                # raw的[2:4]与insert比较
                two_tmp_metric = eval(metric + '_metric(extend_list[i + interval_length: i + interval_length * 2], '
                                               'tmp_noise_list)')
                # 不用重新计算two_samples_metrics，因为只多了一个对比的点，即原始更新噪音点的metric没有更新，但是不会影响结果
                if findMinDiff(four_samples_metrics, four_tmp_metric) >= lower_bound \
                        and findMinDiff(four_samples_metrics, four_tmp_metric) <= upper_bound \
                        and findMinDiff(two_samples_metrics, two_tmp_metric) >= lower_bound \
                        and findMinDiff(two_samples_metrics, two_tmp_metric) <= upper_bound:
                    four_samples_metrics.append(four_tmp_metric)
                    two_samples_metrics.append(two_tmp_metric)
                    # insert插入到noise[2:4]
                    designed_noise[i + interval_length: i + interval_length * 2] = tmp_noise_list
                    break

    # 检查倒数第三层
    for i in range(0, len(extend_list), interval_length * 4):
        eight_tmp_metric = eval(metric + '_metric(extend_list[i: i + interval_length * 4], '
                                         'designed_noise[i: i + interval_length * 4])')
        if findMinDiff(eight_samples_metrics, eight_tmp_metric) >= lower_bound \
                and findMinDiff(eight_samples_metrics, eight_tmp_metric) <= upper_bound:
            eight_samples_metrics.append(eight_tmp_metric)
        else:
            while True:
                tmp_noise_list = random.sample(noise_list, interval_length)
                # 只改变最右端的一个interval的噪音点
                updated_noise = designed_noise[i: i + interval_length * 3] + tmp_noise_list
                # raw的[0:8]与noise的[0:6]||insert比较
                eight_tmp_metric = eval(metric + '_metric(extend_list[i: i + interval_length * 4], updated_noise)')
                # double check可能影响的下层
                # raw的[6:8]与insert比较
                two_tmp_metric = eval(metric + '_metric(extend_list[i + interval_length * 3: i + interval_length * 4], '
                                               'tmp_noise_list)')
                updated_noise = designed_noise[i + interval_length * 2: i + interval_length * 3] + tmp_noise_list
                # raw的[4:8]与noise的[4:6]||insert比较
                four_tmp_metric = eval(
                    metric + '_metric(extend_list[i + interval_length * 2: i + interval_length * 4], '
                             'updated_noise)')
                # 不用重新计算xxx_samples_metrics，因为只多了一个对比的点，即原始更新噪音点的metric没有更新，但是不会影响结果
                if findMinDiff(eight_samples_metrics, eight_tmp_metric) >= lower_bound \
                        and findMinDiff(eight_samples_metrics, eight_tmp_metric) <= upper_bound \
                        and findMinDiff(four_samples_metrics, four_tmp_metric) >= lower_bound \
                        and findMinDiff(four_samples_metrics, four_tmp_metric) <= upper_bound \
                        and findMinDiff(two_samples_metrics, two_tmp_metric) >= lower_bound \
                        and findMinDiff(two_samples_metrics, two_tmp_metric) <= upper_bound:
                    eight_samples_metrics.append(eight_tmp_metric)
                    four_samples_metrics.append(four_tmp_metric)
                    two_samples_metrics.append(two_tmp_metric)
                    # insert插入到noise[6:8]
                    designed_noise[i + interval_length * 3: i + interval_length * 4] = tmp_noise_list
                    break

    # 检查倒数第四层（第二层）
    for i in range(0, len(extend_list), interval_length * 8):
        sixteen_tmp_metric = eval(metric + '_metric(extend_list[i: i + interval_length * 8], '
                                           'designed_noise[i: i + interval_length * 8])')
        if findMinDiff(sixteen_samples_metrics, sixteen_tmp_metric) >= lower_bound \
                and findMinDiff(sixteen_samples_metrics, sixteen_tmp_metric) <= upper_bound:
            sixteen_samples_metrics.append(sixteen_tmp_metric)
        else:
            while True:
                tmp_noise_list = random.sample(noise_list, interval_length)
                # 只改变最右端的一个interval的噪音点
                updated_noise = designed_noise[i: i + interval_length * 7] + tmp_noise_list
                # raw的[0:16]与noise的[0:14]||insert比较
                sixteen_tmp_metric = eval(metric + '_metric(extend_list[i: i + interval_length * 8], updated_noise)')
                # double check可能影响的下层
                # raw的[14:16]与insert比较
                two_tmp_metric = eval(metric + '_metric(extend_list[i + interval_length * 7: i + interval_length * 8], '
                                               'tmp_noise_list)')
                updated_noise = designed_noise[i + interval_length * 6: i + interval_length * 7] + tmp_noise_list
                # raw的[12:16]与noise的[12:14]||insert比较
                four_tmp_metric = eval(
                    metric + '_metric(extend_list[i + interval_length * 6: i + interval_length * 8], '
                             'updated_noise)')
                updated_noise = designed_noise[i + interval_length * 4: i + interval_length * 7] + tmp_noise_list
                # raw的[8:16]与noise的[8:14]||insert比较
                eight_tmp_metric = eval(
                    metric + '_metric(extend_list[i + interval_length * 4: i + interval_length * 8], '
                             'updated_noise)')
                # 不用重新计算xxx_samples_metrics，因为只多了一个对比的点，即原始更新噪音点的metric没有更新，但是不会影响结果
                if findMinDiff(sixteen_samples_metrics, sixteen_tmp_metric) >= lower_bound \
                        and findMinDiff(sixteen_samples_metrics, sixteen_tmp_metric) <= upper_bound \
                        and findMinDiff(eight_samples_metrics, eight_tmp_metric) >= lower_bound \
                        and findMinDiff(eight_samples_metrics, eight_tmp_metric) <= upper_bound \
                        and findMinDiff(four_samples_metrics, four_tmp_metric) >= lower_bound \
                        and findMinDiff(four_samples_metrics, four_tmp_metric) <= upper_bound \
                        and findMinDiff(two_samples_metrics, two_tmp_metric) >= lower_bound \
                        and findMinDiff(two_samples_metrics, two_tmp_metric) <= upper_bound:
                    sixteen_samples_metrics.append(sixteen_tmp_metric)
                    eight_samples_metrics.append(eight_tmp_metric)
                    four_samples_metrics.append(four_tmp_metric)
                    two_samples_metrics.append(two_tmp_metric)
                    # insert插入到noise[14:16]
                    designed_noise[i + interval_length * 7: i + interval_length * 8] = tmp_noise_list
                    break

    # plt.figure()
    # plt.plot(designed_noise)
    # plt.show()

    metrics = []
    cur = 1
    for i in range(1, max_level):
        tmp_metrics = []
        for j in range(cur, cur + 2 ** i, 2):
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l2l = index[j + 1][0] - 1
            l2r = index[j + 1][1]
            if l1r - l1l < length:
                break
            if partner == "himself":
                # 自己与自己相比
                step = int((l1r - l1l) / 2)
                tmp_metrics.append(
                    eval(metric + '_metric(extend_list[l1l: l1l + step], extend_list[l1l + step: l1r])'))
                tmp_metrics.append(
                    eval(metric + '_metric(extend_list[l2l: l2l + step], extend_list[l2l + step: l2r])'))
            else:
                # 与噪音点相比
                tmp_metrics.append(eval(metric + '_metric(extend_list[l1l: l1r], designed_noise[l1l: l1r])'))
                tmp_metrics.append(eval(metric + '_metric(extend_list[l2l: l2r], designed_noise[l2l: l2r])'))
        metrics.append(tmp_metrics)
        cur = cur + 2 ** i

        min_diff = sys.maxsize
        sort_metrics = tmp_metrics.copy()
        sort_metrics.sort()
        for j in range(len(sort_metrics) - 1):
            min_diff = min(min_diff, abs(sort_metrics[j + 1] - sort_metrics[j]))
        min_diffs = min(min_diffs, min_diff)

    # 检验差值diff是否满足阈值
    if min_diffs <= lower_bound:
        print("\033[0;33;40mmin_diff", "poor", min_diffs, "\033[0m")
    else:
        print("\033[0;32;40mmin_diff", min_diffs, "\033[0m")

    for i in range(0, len(metrics)):
        tmp = np.argsort(metrics[i])
        code.append(tmp)

    return_code = []
    for i in range(1, len(intervals) - 1):
        for j in range(len(code[i])):
            return_code.append(code[i][j])
        if intervals[i] < length:
            break

    # adj_corr = []
    # for i in range(0, len(metrics) - 1):
    #     t1 = np.random.permutation(range(len(metrics[i])))
    #     t2 = np.random.permutation(range(len(metrics[i + 1])))
    #     adj_corr.append([dtw_metric(np.argsort(metrics[i]), np.argsort(metrics[i + 1])), dtw_metric(t1, t2)])

    # 展示所有度量结果和对应的密钥
    # y = list(chain.from_iterable(metrics))
    # x = range(len(y))
    # plt.figure()
    # plt.plot(x, y)
    #
    # bbox_color = ['black', 'blue', 'yellow', 'green']
    # # 转换为度量的顺序索引
    # code_indices = []
    # for i in range(0, len(metrics)):
    #     tmp = np.argsort(np.argsort(metrics[i]))
    #     code_indices.append(tmp)
    # code_indices = list(chain.from_iterable(code_indices))
    # for a, b in zip(x, y):
    #     if a < 2:
    #         plt.text(a, b, '%d' % code_indices[a], ha='center', va='bottom', fontsize=9,
    #                  bbox=dict(facecolor=bbox_color[0], alpha=1), color = "white")
    #     elif a < 2 + 4:
    #         plt.text(a, b, '%d' % code_indices[a], ha='center', va='bottom', fontsize=9,
    #                  bbox=dict(facecolor=bbox_color[1], alpha=1), color = "white")
    #     elif a < 2 + 4 + 8:
    #         plt.text(a, b, '%d' % code_indices[a], ha='center', va='bottom', fontsize=9,
    #                  bbox=dict(facecolor=bbox_color[2], alpha=1))
    #     elif a < 2 + 4 + 8 + 16:
    #         plt.text(a, b, '%d' % code_indices[a], ha='center', va='bottom', fontsize=9,
    #                  bbox=dict(facecolor=bbox_color[3], alpha=1), color = "white")
    # plt.show()

    return return_code, designed_noise


# 6
# list 待排序的数组
# length 比较中所需的最小比较单元的长度
# 连续metric差值在阈值之内的值进行置换，返回置换结果
def nativeLevelNoiseMetricSortPerm(data, length, noise, metric):
    extend_list = data.copy()
    designed_noise = []
    noise_list = noise
    # noise_list = list(np.random.normal(loc=np.mean(noise),
    #                                    scale=np.std(noise, ddof=1),
    #                                    size=len(noise) * 100))
    for i in range(2 ** math.ceil(np.log2(len(data))) - len(data)):
        extend_list.append(data[i - len(data)])

    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]

    min_diffs = sys.maxsize

    # 生成随机噪音点，满足阈值的要求
    interval_length = 2
    lower_bound = 5.5
    # upper_bound = 5000
    upper_bound = sys.maxsize

    # while True:
    cnt = 0
    while cnt <= 50:
        cnt += 1
        min_diffs = sys.maxsize
        tmp_noise_list = random.sample(noise_list, len(noise_list))

        metrics = []
        cur = 1
        for i in range(1, max_level):
            tmp_metrics = []
            for j in range(cur, cur + 2 ** i, 2):
                l1l = index[j][0] - 1
                l1r = index[j][1]
                l2l = index[j + 1][0] - 1
                l2r = index[j + 1][1]
                if l1r - l1l < length:
                    break
                if partner == "himself":
                    # 自己与自己相比
                    step = int((l1r - l1l) / 2)
                    tmp_metrics.append(
                        eval(metric + '_metric(extend_list[l1l: l1l + step], extend_list[l1l + step: l1r])'))
                    tmp_metrics.append(
                        eval(metric + '_metric(extend_list[l2l: l2l + step], extend_list[l2l + step: l2r])'))
                else:
                    # 与噪音点相比
                    tmp_metrics.append(eval(metric + '_metric(extend_list[l1l: l1r], tmp_noise_list[l1l: l1r])'))
                    tmp_metrics.append(eval(metric + '_metric(extend_list[l2l: l2r], tmp_noise_list[l2l: l2r])'))
            metrics.append(tmp_metrics)
            cur = cur + 2 ** i

            min_diff = sys.maxsize
            sort_metrics = tmp_metrics.copy()
            sort_metrics.sort()
            for j in range(len(sort_metrics) - 1):
                min_diff = min(min_diff, abs(sort_metrics[j + 1] - sort_metrics[j]))
            min_diffs = min(min_diffs, min_diff)

        # 检验差值diff是否满足阈值
        if min_diffs <= lower_bound:
            print("\033[0;32;40mmin_diff", min_diffs, "\033[0m")  # green
            tmp_noise_list = random.sample(noise_list, len(noise_list))
        else:
            print("\033[0;33;40mmin_diff", min_diffs, "\033[0m")  # yellow
            break
    print("\033[0;32;40mmin_diff", min_diffs, "\033[0m")  # green

    for i in range(0, len(metrics)):
        tmp = np.argsort(metrics[i])
        code.append(tmp)

    return_code = []
    for i in range(1, len(intervals) - 1):
        for j in range(len(code[i])):
            return_code.append(code[i][j])
        if intervals[i] < length:
            break

    # adj_corr = []
    # for i in range(0, len(metrics) - 1):
    #     t1 = np.random.permutation(range(len(metrics[i])))
    #     t2 = np.random.permutation(range(len(metrics[i + 1])))
    #     adj_corr.append([dtw_metric(np.argsort(metrics[i]), np.argsort(metrics[i + 1])), dtw_metric(t1, t2)])

    # 展示所有度量结果和对应的密钥
    # y = list(chain.from_iterable(metrics))
    # x = range(len(y))
    # plt.figure()
    # plt.plot(x, y)
    #
    # bbox_color = ['black', 'blue', 'yellow', 'green']
    # # 转换为度量的顺序索引
    # code_indices = []
    # for i in range(0, len(metrics)):
    #     tmp = np.argsort(np.argsort(metrics[i]))
    #     code_indices.append(tmp)
    # code_indices = list(chain.from_iterable(code_indices))
    # for a, b in zip(x, y):
    #     if a < 2:
    #         plt.text(a, b, '%d' % code_indices[a], ha='center', va='bottom', fontsize=9,
    #                  bbox=dict(facecolor=bbox_color[0], alpha=1), color = "white")
    #     elif a < 2 + 4:
    #         plt.text(a, b, '%d' % code_indices[a], ha='center', va='bottom', fontsize=9,
    #                  bbox=dict(facecolor=bbox_color[1], alpha=1), color = "white")
    #     elif a < 2 + 4 + 8:
    #         plt.text(a, b, '%d' % code_indices[a], ha='center', va='bottom', fontsize=9,
    #                  bbox=dict(facecolor=bbox_color[2], alpha=1))
    #     elif a < 2 + 4 + 8 + 16:
    #         plt.text(a, b, '%d' % code_indices[a], ha='center', va='bottom', fontsize=9,
    #                  bbox=dict(facecolor=bbox_color[3], alpha=1), color = "white")
    # plt.show()

    return return_code, tmp_noise_list


# 7
# list 待排序的数组
# length 比较中所需的最小比较单元的长度
# 连续metric差值在阈值之内的值进行置换，返回置换结果
def levelMetricSortSiftPerm(data, length, noise, metric):
    extend_list = data.copy()
    noise_list = noise.copy()
    for i in range(2 ** math.ceil(np.log2(len(data))) - len(data)):
        extend_list.append(data[i - len(data)])
    for i in range(2 ** math.ceil(np.log2(len(noise))) - len(noise)):
        noise_list.append(noise[i - len(noise)])

    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]

    min_diffs = sys.maxsize

    metrics = []
    perms = []
    cur = 1
    for i in range(1, max_level):
        data1 = []
        data2 = []
        for j in range(cur, cur + 2 ** i, 2):
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l2l = index[j + 1][0] - 1
            l2r = index[j + 1][1]
            if l1r - l1l < length:
                break
            if partner == "himself":
                # 自己与自己相比
                step = int((l1r - l1l) / 2)
            else:
                # 与噪音点相比
                data1.append(extend_list[l1l: l1r])
                data1.append(extend_list[l2l: l2r])
                data2.append(noise_list[l1l: l1r])
                data2.append(noise_list[l2l: l2r])

        res = genDiffMatrix(data1, data2, perm_threshold)
        if res is not None:
            perms.append(res[0])
            metrics.append(res[1])
        cur = cur + 2 ** i

        # min_diff = sys.maxsize
        #
        # sort_metrics = tmp_metrics.copy()
        # sort_metrics.sort()
        # for j in range(len(sort_metrics) - 1):
        #     min_diff = min(min_diff, abs(sort_metrics[j + 1] - sort_metrics[j]))
        # min_diffs = min(min_diffs, min_diff)

    # if min_diffs < perm_threshold:
    #     # print("\033[0;32;40mmin_diff", min_diffs, "\033[0m")  # green
    #     np.random.shuffle(combine)
    #     extend_list, perm = zip(*combine)
    # else:
    #     print("\033[0;33;40mmin_diff", min_diffs, "\033[0m")  # yellow

    # print("\033[0;32;40mmin_diff", min_diffs, "\033[0m")  # green

    for i in range(0, len(metrics)):
        tmp = np.argsort(metrics[i])
        code.append(tmp)

    return_code = []
    for i in range(1, len(intervals) - 1):
        for j in range(len(code[i])):
            return_code.append(code[i][j])
        if intervals[i] < length:
            break

    return return_code, perms


# 9
# list 待排序的数组
# length 比较中所需的最小比较单元的长度
# 根据perm先置换，然后B生成密钥，返回要删除的点
def levelMetricSortSiftPermOfB(data, length, noise, perm, metric):
    extend_list = data.copy()
    noise_list = noise.copy()
    for i in range(2 ** math.ceil(np.log2(len(data))) - len(data)):
        extend_list.append(data[i - len(data)])
    for i in range(2 ** math.ceil(np.log2(len(noise))) - len(noise)):
        noise_list.append(noise[i - len(noise)])

    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]

    # 丢弃容易出错的位置
    blots = []

    metrics = []
    origin_metrics = []
    cur = 1
    for i in range(1, max_level):
        tmp_metrics = []
        blot = []
        data1 = []
        data2 = []
        for j in range(cur, cur + 2 ** i, 2):
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l2l = index[j + 1][0] - 1
            l2r = index[j + 1][1]
            if l1r - l1l < length:
                break
            if partner == "himself":
                # 自己与自己相比
                step = int((l1r - l1l) / 2)
            else:
                # 与噪音点相比
                data1.append(extend_list[l1l: l1r])
                data1.append(extend_list[l2l: l2r])
                data2.append(noise_list[l1l: l1r])
                data2.append(noise_list[l2l: l2r])
        # 选出相邻差距最小的点，若差距过小，则挑出来进行删除

        # ------------------------------------------------------
        # 使用metric的一阶差分排序
        # tmp_metrics = difference(tmp_metrics)
        # ------------------------------------------------------

        data_perm = np.array(data2)[perm[i - 1]]
        for j in range(len(perm[i - 1])):
            tmp_metrics.append(euclidean_metric(data1[j], data_perm[j]))

        # 选出差距接近阈值的度量值，标记出左度量值以待删除
        origin_metrics.append(tmp_metrics.copy())
        sort_metrics = tmp_metrics.copy()
        sort_metrics.sort()
        for j in range(len(sort_metrics) - 1):
            if abs(sort_metrics[j + 1] - sort_metrics[j]) < blot_threshold:
                for k in range(len(tmp_metrics)):
                    if tmp_metrics[k] == sort_metrics[j + 1]:
                        blot.append([i - 1, k])

        blots.append(blot)
        # 从后往前删除，防止待删除的索引溢出
        for j in range(len(blot) - 1, -1, -1):
            blot.sort()  # 防止索引溢出
            drop = blot[j][1]
            if -1 < drop < len(tmp_metrics):
                del tmp_metrics[drop]
        metrics.append(tmp_metrics)
        cur = cur + 2 ** i

    min_diffs = sys.maxsize
    for i in range(len(metrics)):
        sort_metrics = metrics[i].copy()
        sort_metrics.sort()
        min_diff = sys.maxsize
        for j in range(len(sort_metrics) - 1):
            min_diff = min(min_diff, abs(sort_metrics[j + 1] - sort_metrics[j]))
        min_diffs = min(min_diffs, min_diff)
    print("\033[0;36;40mmin_diff after deleting", min_diffs, "\033[0m")  # cyan

    if min_diffs < blot_threshold:
        print("error: bad blot")

    for i in range(0, len(metrics)):
        tmp = np.argsort(metrics[i])
        code.append(tmp)

    return_code = []
    for i in range(1, len(intervals) - 1):
        for j in range(len(code[i])):
            return_code.append(code[i][j])
        if intervals[i] < length:
            break

    # print("blots", blots)
    return return_code, blots


# 10
# list 待排序的数组
# length 比较中所需的最小比较单元的长度
# 根据perm先置换，再根据blots删除点，返回密钥
# window=0表示不重用blots，否则表示重用blots时使用滑动窗口的宽度
def levelMetricSortSiftPermOfA(data, length, noise, perm, blots, metric, window=0):
    extend_list = data.copy()
    noise_list = noise.copy()
    for i in range(2 ** math.ceil(np.log2(len(data))) - len(data)):
        extend_list.append(data[i - len(data)])
    for i in range(2 ** math.ceil(np.log2(len(noise))) - len(noise)):
        noise_list.append(noise[i - len(noise)])

    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]

    metrics = []
    recycle_bin = []
    cur = 1
    for i in range(1, max_level):
        tmp_metrics = []
        recycle = []
        data1 = []
        data2 = []
        for j in range(cur, cur + 2 ** i, 2):
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l2l = index[j + 1][0] - 1
            l2r = index[j + 1][1]
            if l1r - l1l < length:
                break
            if partner == "himself":
                # 自己与自己相比
                step = int((l1r - l1l) / 2)
            else:
                # 与噪音点相比
                data1.append(extend_list[l1l: l1r])
                data1.append(extend_list[l2l: l2r])
                data2.append(noise_list[l1l: l1r])
                data2.append(noise_list[l2l: l2r])

        # ------------------------------------------------------
        # 使用metric的一阶差分排序
        # tmp_metrics = difference(tmp_metrics)
        # ------------------------------------------------------

        data_perm = np.array(data2)[perm[i - 1]]
        for j in range(len(perm[i - 1])):
            tmp_metrics.append(euclidean_metric(data1[j], data_perm[j]))

        # 根据blots删除差距接近阈值的左度量值
        if blots[i - 1] is not None and len(blots[i - 1]) != 0:
            # 从后往前删除，防止待删除的索引超出已删除的数据
            for j in range(len(blots[i - 1]) - 1, -1, -1):
                drop = blots[i - 1][j][1]
                if -1 < drop < len(tmp_metrics):
                    # 回收删除的blots
                    recycle.append(tmp_metrics[drop])
                    del tmp_metrics[drop]
        metrics.append(tmp_metrics)
        recycle_bin.append(recycle)
        cur = cur + 2 ** i

    all_diffs = []
    for i in range(len(metrics)):
        tmp = metrics[i].copy()
        tmp.sort()
        tmp_diffs = []
        for j in range(len(tmp) - 1):
            tmp_diffs.append(abs(tmp[j + 1] - tmp[j]))
        all_diffs.append(tmp_diffs)
    # print("all_diffs", all_diffs)
    # print("all_metrics", metrics)

    for i in range(len(metrics)):
        retain_tmp = np.argsort(metrics[i])
        if window > 0:
            # 回收的blots产生的code
            blots_tmp = np.argsort(successive_moving_sum(recycle_bin[i], window))
            # print("blots_tmp", blots_tmp)
            tmp = np.hstack((retain_tmp, blots_tmp))
            # print("tmp", tmp)
            code.append(tmp)
        else:
            code.append(retain_tmp)

    min_diffs = sys.maxsize
    for i in range(len(metrics)):
        sort_metrics = metrics[i].copy()
        sort_metrics.sort()
        min_diff = sys.maxsize
        for j in range(len(sort_metrics) - 1):
            min_diff = min(min_diff, abs(sort_metrics[j + 1] - sort_metrics[j]))
        min_diffs = min(min_diffs, min_diff)

    print("\033[3;35;40mmin_diff", min_diffs, "\033[0m")  # pink
    # if min_diffs < blot_threshold:
    #     print("error: bad combined blot")

    return_code = []
    for i in range(1, len(intervals) - 1):
        for j in range(len(code[i])):
            return_code.append(code[i][j])
        if intervals[i] < length:
            break

    return return_code


############################################
# 11
# list 待排序的数组
# length 比较中所需的最小比较单元的长度
# 连续metric差值在阈值之内的值进行置换，返回置换结果
def levelMetricDoubleSortSiftPerm(data, length, noise, metric):
    extend_list = data.copy()
    noise_list = noise.copy()
    for i in range(2 ** math.ceil(np.log2(len(data))) - len(data)):
        extend_list.append(data[i - len(data)])
    for i in range(2 ** math.ceil(np.log2(len(noise))) - len(noise)):
        noise_list.append(noise[i - len(noise)])

    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]

    min_diffs = sys.maxsize

    metrics = []
    perms = []
    cur = 1
    for i in range(1, max_level):
        tmp_metrics = []
        tmp_perms = []
        for j in range(cur, cur + 2 ** i, 2):
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l2l = index[j + 1][0] - 1
            l2r = index[j + 1][1]
            if l1r - l1l < length:
                break
            if partner == "himself":
                # 自己与自己相比
                step = int((l1r - l1l) / 2)
            else:
                # 与噪音点相比
                if metric == "euclidean":
                    perm_data_1 = genDiffPerm(extend_list[l1l: l1r], noise_list[l1l: l1r] + max(extend_list[l1l: l1r]))
                    tmp_perms.append(perm_data_1)
                    tmp_metrics.append(
                        euclidean_metric(extend_list[l1l: l1r], np.array(noise_list[l1l: l1r])[perm_data_1]))

                    perm_data_2 = genDiffPerm(extend_list[l2l: l2r], noise_list[l2l: l2r] + max(extend_list[l2l: l2r]))
                    tmp_perms.append(perm_data_2)
                    tmp_metrics.append(
                        euclidean_metric(extend_list[l2l: l2r], np.array(noise_list[l2l: l2r])[perm_data_2]))

        metrics.append(tmp_metrics)
        perms.append(tmp_perms)
        cur = cur + 2 ** i

        # min_diff = sys.maxsize
        #
        # sort_metrics = tmp_metrics.copy()
        # sort_metrics.sort()
        # for j in range(len(sort_metrics) - 1):
        #     min_diff = min(min_diff, abs(sort_metrics[j + 1] - sort_metrics[j]))
        # min_diffs = min(min_diffs, min_diff)

    # if min_diffs < perm_threshold:
    #     # print("\033[0;32;40mmin_diff", min_diffs, "\033[0m")  # green
    #     np.random.shuffle(combine)
    #     extend_list, perm = zip(*combine)
    # else:
    #     print("\033[0;33;40mmin_diff", min_diffs, "\033[0m")  # yellow

    # print("\033[0;32;40mmin_diff", min_diffs, "\033[0m")  # green

    for i in range(0, len(metrics)):
        tmp = np.argsort(metrics[i])
        code.append(tmp)

    return_code = []
    for i in range(1, len(intervals) - 1):
        for j in range(len(code[i])):
            return_code.append(code[i][j])
        if intervals[i] < length:
            break

    return return_code, perms


# 13
# list 待排序的数组
# length 比较中所需的最小比较单元的长度
# 根据perm先置换，然后B生成密钥，返回要删除的点
def levelMetricDoubleSortSiftPermOfB(data, length, noise, perm, metric):
    extend_list = data.copy()
    noise_list = noise.copy()
    for i in range(2 ** math.ceil(np.log2(len(data))) - len(data)):
        extend_list.append(data[i - len(data)])
    for i in range(2 ** math.ceil(np.log2(len(noise))) - len(noise)):
        noise_list.append(noise[i - len(noise)])

    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]

    # 丢弃容易出错的位置
    blots = []

    metrics = []
    origin_metrics = []
    cur = 1
    for i in range(1, max_level):
        tmp_metrics = []
        blot = []
        perm_index = 0
        for j in range(cur, cur + 2 ** i, 2):
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l2l = index[j + 1][0] - 1
            l2r = index[j + 1][1]
            if l1r - l1l < length:
                break
            if partner == "himself":
                # 自己与自己相比
                step = int((l1r - l1l) / 2)
            else:
                # 与噪音点相比
                # 根据公开的置换序列进行置换
                # noise1 = noise_list[l1l: l1r]
                # noise2 = noise_list[l2l: l2r]
                # perm1 = perm[i - 1][perm_index]
                # perm2 = perm[i - 1][perm_index + 1]
                # tmp_metrics.append(
                #     eval(metric + '_metric(extend_list[l1l: l1r], np.array(noise1)[perm1])'))
                # tmp_metrics.append(
                #     eval(metric + '_metric(extend_list[l2l: l2r], np.array(noise2)[perm2])'))
                if metric == "euclidean":
                    perm_data_1 = genDiffPerm(extend_list[l1l: l1r], noise_list[l1l: l1r] + max(extend_list[l1l: l1r]))
                    tmp_metrics.append(
                        euclidean_metric(extend_list[l1l: l1r], np.array(noise_list[l1l: l1r])[perm_data_1]))

                    perm_data_2 = genDiffPerm(extend_list[l2l: l2r], noise_list[l2l: l2r] + max(extend_list[l2l: l2r]))
                    tmp_metrics.append(
                        euclidean_metric(extend_list[l2l: l2r], np.array(noise_list[l2l: l2r])[perm_data_2]))

        perm_index += 1

        # 选出相邻差距最小的点，若差距过小，则挑出来进行删除

        # ------------------------------------------------------
        # 使用metric的一阶差分排序
        # tmp_metrics = difference(tmp_metrics)
        # ------------------------------------------------------

        # 选出差距接近阈值的度量值，标记出右度量值以待删除
        origin_metrics.append(tmp_metrics.copy())
        sort_metrics = tmp_metrics.copy()
        sort_metrics.sort()
        for j in range(len(sort_metrics) - 1):
            if abs(sort_metrics[j + 1] - sort_metrics[j]) < blot_threshold[i - 1]:
                for k in range(len(tmp_metrics)):
                    if math.isclose(tmp_metrics[k], sort_metrics[j + 1], rel_tol=1e-5):
                        blot.append([i - 1, k])

        blots.append(blot)
        metrics.append(tmp_metrics)
        cur = cur + 2 ** i

    min_diffs = []
    for i in range(len(metrics)):
        if len(metrics[i]) < 2:
            min_diffs.append(-1)
        else:
            sort_metrics = metrics[i].copy()
            sort_metrics.sort()
            min_diff = abs(sort_metrics[1] - sort_metrics[0])
            for j in range(1, len(sort_metrics) - 1):
                min_diff = min(min_diff, abs(sort_metrics[j + 1] - sort_metrics[j]))
            min_diffs.append(min_diff)
    print("\033[0;36;40mmin_diff after deleting", min_diffs, "\033[0m")  # cyan

    for i in range(0, len(metrics)):
        tmp = np.argsort(metrics[i])
        code.append(tmp)

    return_code = []
    for i in range(1, len(intervals) - 1):
        for j in range(len(code[i])):
            return_code.append(code[i][j])
        if intervals[i] < length:
            break

    # print("blots", blots)
    return return_code, blots


# 14
# list 待排序的数组
# length 比较中所需的最小比较单元的长度
# 根据perm先置换，再根据blots删除点，返回密钥
# window=0表示不重用blots，否则表示重用blots时使用滑动窗口的宽度
def levelMetricDoubleSortSiftPermOfA(caller, data, length, noise, perm, blots, metric, window=0):
    extend_list = data.copy()
    noise_list = noise.copy()
    for i in range(2 ** math.ceil(np.log2(len(data))) - len(data)):
        extend_list.append(data[i - len(data)])
    for i in range(2 ** math.ceil(np.log2(len(noise))) - len(noise)):
        noise_list.append(noise[i - len(noise)])

    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        k = q.popleft()
        index.append(k)
        if k[1] != k[0]:
            q.append([k[0], math.floor((k[0] + k[1]) / 2)])
            q.append([math.ceil((k[0] + k[1]) / 2), k[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]

    # 原始数据和容易出错的数据
    data = []
    blots_data = []
    # 二次筛选（重用blot）的删除的部分
    double_blots = []

    metrics = []
    blots_metrics = []

    recycle_bin = []
    cur = 1
    for i in range(1, max_level):
        tmp_metrics = []
        recycle = []
        tmp_data = []
        blot_data = []
        perm_index = 0
        for j in range(cur, cur + 2 ** i, 2):
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l2l = index[j + 1][0] - 1
            l2r = index[j + 1][1]
            if l1r - l1l < length:
                break
            if partner == "himself":
                # 自己与自己相比
                step = int((l1r - l1l) / 2)
            else:
                # 与噪音点相比
                # noise1 = noise_list[l1l: l1r]
                # noise2 = noise_list[l2l: l2r]
                # perm1 = perm[i - 1][perm_index]
                # perm2 = perm[i - 1][perm_index + 1]
                # tmp_metrics.append(
                #     eval(metric + '_metric(extend_list[l1l: l1r], np.array(noise1)[perm1])'))
                # tmp_metrics.append(
                #     eval(metric + '_metric(extend_list[l2l: l2r], np.array(noise2)[perm2])'))

                if metric == "euclidean":
                    perm_data_1 = genDiffPerm(extend_list[l1l: l1r], noise_list[l1l: l1r] + max(extend_list[l1l: l1r]))
                    tmp_data.append([extend_list[l1l: l1r], list(np.array(noise_list[l1l: l1r])[perm_data_1])])
                    tmp_metrics.append(
                        euclidean_metric(extend_list[l1l: l1r], np.array(noise_list[l1l: l1r])[perm_data_1]))

                    perm_data_2 = genDiffPerm(extend_list[l2l: l2r], noise_list[l2l: l2r] + max(extend_list[l2l: l2r]))
                    tmp_data.append([extend_list[l2l: l2r], list(np.array(noise_list[l2l: l2r])[perm_data_2])])
                    tmp_metrics.append(
                        euclidean_metric(extend_list[l2l: l2r], np.array(noise_list[l2l: l2r])[perm_data_2]))
        perm_index += 1

        data.append(tmp_data)

        # ------------------------------------------------------
        # 使用metric的一阶差分排序
        # tmp_metrics = difference(tmp_metrics)
        # ------------------------------------------------------

        # data_perm = np.array(data2)[perm[i - 1]]
        # for j in range(len(perm[i - 1])):
        #     tmp_metrics.append(euclidean_metric(data1[j], data_perm[j]))

        # 还原出blots对应的原始数据与其待比较的数据
        sort_metrics = tmp_metrics.copy()
        sort_metrics.sort()

        if blots[i - 1] is not None and len(blots[i - 1]) != 0:
            for j in range(len(blots[i - 1])):
                # drop为阈值接近的右度量值
                drop = blots[i - 1][j][1]
                if 0 < drop < len(sort_metrics):
                    # 只选出一个接近的值，不然A和B对于某个度量值可能会选出多个近似的值
                    for k in range(len(tmp_metrics)):
                        if math.isclose(tmp_metrics[k], sort_metrics[drop], rel_tol=1e-5):
                            blot_data.append(data[i - 1][k])
                            break
                    for k in range(len(tmp_metrics)):
                        if math.isclose(tmp_metrics[k], sort_metrics[drop - 1], rel_tol=1e-5):
                            blot_data.append(data[i - 1][k])
                            break
        blots_data.append(blot_data)

        # 根据blots删除差距接近阈值的右度量值
        if blots[i - 1] is not None and len(blots[i - 1]) != 0:
            # 从后往前删除，防止待删除的索引超出已删除的数据
            for j in range(len(blots[i - 1]) - 1, -1, -1):
                drop = blots[i - 1][j][1]
                if -1 < drop < len(tmp_metrics):
                    # 回收删除的blots
                    recycle.append(tmp_metrics[drop])
                    del tmp_metrics[drop]
        metrics.append(tmp_metrics)
        recycle_bin.append(recycle)
        cur = cur + 2 ** i

    all_diffs = []
    for i in range(len(metrics)):
        tmp = metrics[i].copy()
        tmp.sort()
        tmp_diffs = []
        for j in range(len(tmp) - 1):
            tmp_diffs.append(abs(tmp[j + 1] - tmp[j]))
        all_diffs.append(tmp_diffs)
    # print("all_diffs", all_diffs)
    # print("all_metrics", metrics)

    for i in range(len(metrics)):
        # if recycle_bin[i] is not None and len(recycle_bin[i]) != 0:
        # print("recycle_bin", recycle_bin[i])
        # print(np.argsort(recycle_bin[i]))
        retain_tmp = np.argsort(metrics[i])
        double_blot = []
        if window > 0:
            # 回收的blots产生的code
            blots_tmp = np.argsort(successive_moving_sum(recycle_bin[i], window))
            tmp = np.hstack((retain_tmp, blots_tmp))
            code.append(tmp)
        else:
            # 由欧式距离降级为普通单个点的排序
            blots_metric = []
            # 以一个segment中的第一个点为例
            # which_point = 0
            # for j in range(0, len(blots_data[i]), 2):
            #     blots_metric.append(
            #         euclidean_metric(blots_data[i][j][0][which_point], blots_data[i][j][1][which_point]))
            # 最小和最大求差
            for j in range(0, len(blots_data[i]), 2):
                blots_metric.append(euclidean_metric(max(blots_data[i][j][0]), max(blots_data[i][j][1])) +
                                    euclidean_metric(min(blots_data[i][j][0]), min(blots_data[i][j][1])))
            sort_metrics = blots_metric.copy()
            sort_metrics.sort()
            for j in range(len(sort_metrics) - 1):
                if abs(sort_metrics[j + 1] - sort_metrics[j]) < 1:
                    for k in range(len(blots_metric)):
                        if math.isclose(blots_metric[k], sort_metrics[j + 1], rel_tol=1e-5):
                            double_blot.append([i, k])
            double_blots.append(double_blot)
            code.append(retain_tmp)

    return_code = []
    for i in range(1, len(intervals) - 1):
        for j in range(len(code[i])):
            return_code.append(code[i][j])
        if intervals[i] < length:
            break

    return return_code, double_blots


# 15
# list 待排序的数组
# length 比较中所需的最小比较单元的长度
# 根据perm先置换，再根据blots删除点，返回密钥
# window=0表示不重用blots，否则表示重用blots时使用滑动窗口的宽度
def levelMetricDoubleSortSiftDoublePermOfB(caller, data, length, noise, perm, blots, double_blots, metric, window=0):
    extend_list = data.copy()
    noise_list = noise.copy()
    for i in range(2 ** math.ceil(np.log2(len(data))) - len(data)):
        extend_list.append(data[i - len(data)])
    for i in range(2 ** math.ceil(np.log2(len(noise))) - len(noise)):
        noise_list.append(noise[i - len(noise)])

    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        k = q.popleft()
        index.append(k)
        if k[1] != k[0]:
            q.append([k[0], math.floor((k[0] + k[1]) / 2)])
            q.append([math.ceil((k[0] + k[1]) / 2), k[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]

    # 原始数据和容易出错的数据
    data = []
    blots_data = []

    metrics = []
    blots_metrics = []

    recycle_bin = []
    cur = 1
    for i in range(1, max_level):
        tmp_metrics = []
        recycle = []
        tmp_data = []
        blot_data = []
        perm_index = 0
        for j in range(cur, cur + 2 ** i, 2):
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l2l = index[j + 1][0] - 1
            l2r = index[j + 1][1]
            if l1r - l1l < length:
                break
            if partner == "himself":
                # 自己与自己相比
                step = int((l1r - l1l) / 2)
            else:
                # 与噪音点相比
                # noise1 = noise_list[l1l: l1r]
                # noise2 = noise_list[l2l: l2r]
                # perm1 = perm[i - 1][perm_index]
                # perm2 = perm[i - 1][perm_index + 1]
                # tmp_metrics.append(
                #     eval(metric + '_metric(extend_list[l1l: l1r], np.array(noise1)[perm1])'))
                # tmp_metrics.append(
                #     eval(metric + '_metric(extend_list[l2l: l2r], np.array(noise2)[perm2])'))

                if metric == "euclidean":
                    perm_data_1 = genDiffPerm(extend_list[l1l: l1r], noise_list[l1l: l1r] + max(extend_list[l1l: l1r]))
                    tmp_data.append([extend_list[l1l: l1r], list(np.array(noise_list[l1l: l1r])[perm_data_1])])
                    tmp_metrics.append(
                        euclidean_metric(extend_list[l1l: l1r], np.array(noise_list[l1l: l1r])[perm_data_1]))

                    perm_data_2 = genDiffPerm(extend_list[l2l: l2r], noise_list[l2l: l2r] + max(extend_list[l2l: l2r]))
                    tmp_data.append([extend_list[l2l: l2r], list(np.array(noise_list[l2l: l2r])[perm_data_2])])
                    tmp_metrics.append(
                        euclidean_metric(extend_list[l2l: l2r], np.array(noise_list[l2l: l2r])[perm_data_2]))
        perm_index += 1

        data.append(tmp_data)

        # ------------------------------------------------------
        # 使用metric的一阶差分排序
        # tmp_metrics = difference(tmp_metrics)
        # ------------------------------------------------------

        # 还原出blots对应的原始数据与其待比较的数据
        sort_metrics = tmp_metrics.copy()
        sort_metrics.sort()

        if blots[i - 1] is not None and len(blots[i - 1]) != 0:
            for j in range(len(blots[i - 1])):
                # drop为阈值接近的右度量值
                drop = blots[i - 1][j][1]
                if 0 < drop < len(sort_metrics):
                    # 只选出一个接近的值，不然A和B对于某个度量值可能会选出多个近似的值
                    for k in range(len(tmp_metrics)):
                        if math.isclose(tmp_metrics[k], sort_metrics[drop], rel_tol=1e-5):
                            blot_data.append(data[i - 1][k])
                            break
                    for k in range(len(tmp_metrics)):
                        if math.isclose(tmp_metrics[k], sort_metrics[drop - 1], rel_tol=1e-5):
                            blot_data.append(data[i - 1][k])
                            break
        blots_data.append(blot_data)

        # 根据blots删除差距接近阈值的右度量值
        if blots[i - 1] is not None and len(blots[i - 1]) != 0:
            # 从后往前删除，防止待删除的索引超出已删除的数据
            for j in range(len(blots[i - 1]) - 1, -1, -1):
                drop = blots[i - 1][j][1]
                if -1 < drop < len(tmp_metrics):
                    # 回收删除的blots
                    recycle.append(tmp_metrics[drop])
                    del tmp_metrics[drop]
        metrics.append(tmp_metrics)
        recycle_bin.append(recycle)
        cur = cur + 2 ** i

    double_min_diffs = []
    for i in range(len(double_blots)):
        if double_blots[i] is not None and len(double_blots[i]) != 0:
            if window > 0:
                # 回收的blots产生的code
                blots_tmp = np.argsort(successive_moving_sum(recycle_bin[i], window))
                code.append(blots_tmp)
            else:
                # 由欧式距离降级为普通单个点的排序
                blots_metric = []
                # 以一个segment中的第一个点为例
                which_point = 0
                for j in range(0, len(blots_data[i]), 2):
                    blots_metric.append(
                        euclidean_metric(blots_data[i][j][0][which_point], blots_data[i][j][1][which_point]))
                # 最小和最大求差
                # for j in range(0, len(blots_data[i]), 2):
                #     blots_metric.append(euclidean_metric(max(blots_data[i][j][0]), max(blots_data[i][j][1])) -
                #                         euclidean_metric(min(blots_data[i][j][0]), min(blots_data[i][j][1])))
                for j in range(len(double_blots[i]) - 1, -1, -1):
                    drop = double_blots[i][j][1]
                    if -1 < drop < len(blots_metric):
                        del blots_metric[drop]
                sort_metrics = blots_metric.copy()
                sort_metrics.sort()
                double_min_diff = sys.maxsize
                for j in range(len(sort_metrics) - 1):
                    double_min_diff = min(double_min_diff, abs(sort_metrics[j + 1] - sort_metrics[j]))
                double_min_diffs.append(double_min_diff)
                blots_metrics.append(blots_metric)
                blots_tmp = list(np.argsort(blots_metric))
                if caller == "alice" or caller == "bob":
                    print("key", blots_metric, blots_tmp, double_min_diff)
                code.append(blots_tmp)

    # if caller != "none":
    #     print("\033[3;35;40mdouble_min_diff", min(double_min_diffs), "\033[0m")  # pink

    min_diffs = []
    for i in range(len(metrics)):
        if len(metrics[i]) < 2:
            min_diffs.append(-1)
        else:
            sort_metrics = metrics[i].copy()
            sort_metrics.sort()
            min_diff = abs(sort_metrics[1] - sort_metrics[0])
            for j in range(1, len(sort_metrics) - 1):
                min_diff = min(min_diff, abs(sort_metrics[j + 1] - sort_metrics[j]))
            if (caller == "alice" or caller == "bob") and min_diff < blot_threshold[i]:
                print("\033[0;33;40mmin_diff", "poor", min_diff, "\033[0m")
            min_diffs.append(min_diff)
    if caller != "none":
        print("\033[3;35;40mmin_diff", min_diffs, "\033[0m")  # pink
    # if min_diffs < blot_threshold:
    #     print("error: bad combined blot")

    return_code = []
    for i in range(1, len(code) - 1):
        for j in range(len(code[i])):
            return_code.append(code[i][j])
        if intervals[i] < length:
            break

    return return_code


######===================================================#######
######===================================================#######
######===================================================#######
######===================================================#######
# list 待排序的数组
# length 比较中所需的最小比较单元的长度
def singleMetricSort(data, length, noise):
    distance = lambda x, y: np.abs(x - y)
    extend_list = data.copy()
    noise_list = noise.copy()
    for i in range(2 ** math.ceil(np.log2(len(data))) - len(data)):
        extend_list.append(data[i - len(data)])
    for i in range(2 ** math.ceil(np.log2(len(noise))) - len(noise)):
        noise_list.append(noise[i - len(noise)])

    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]

    cur = 1
    for i in range(1, max_level):
        cur_code = []
        for j in range(cur, cur + 2 ** i, 2):
            tmp = linear_code[int((len(linear_code)) / 2)]
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l2l = index[j + 1][0] - 1
            l2r = index[j + 1][1]
            step = int((l1r - l1l) / 2)
            # 计算dtw时需要使用
            # if step == 0:
            #     break
            # metric1 = sum(rel_entr(extend_list[l1l: l1l + step], extend_list[l1l + step: l1r]))
            # metric2 = sum(rel_entr(extend_list[l2l: l2l + step], extend_list[l2l + step: l2r]))
            # metric1 = sum(rel_entr(extend_list[l1l: l1r], noise_list[l1l: l1r]))
            # metric2 = sum(rel_entr(extend_list[l2l: l2r], noise_list[l2l: l2r]))
            # metric1 = dtw(extend_list[l1l: l1l + step], extend_list[l1l + step: l1r], dist=distance)[0]
            # metric2 = dtw(extend_list[l2l: l2l + step], extend_list[l2l + step: l2r], dist=distance)[0]
            metric1 = dtw(extend_list[l1l: l1r], noise_list[l1l: l1r], dist=distance)[0]
            metric2 = dtw(extend_list[l2l: l2r], noise_list[l2l: l2r], dist=distance)[0]
            if metric1 >= metric2:
                tmp += "0"
            else:
                tmp += "1"
            # 记录当前code，用于子节点使用
            linear_code.append(tmp)
            # 记录当前code，用于输出
            cur_code.append(tmp)
        cur = cur + 2 ** i
        code.append(cur_code)

    return_code = []
    for i in range(1, len(intervals)):
        if intervals[i] < length:
            return_code = code[i - 1]
            break

    for i in range(len(return_code)):
        return_code[i] = return_code[i][1:]  # 去掉根节点的遍历后的密钥
        return_code[i] = return_code[i][-1]  # 只保留最后一层
    return return_code


# list 待排序的数组
# length 比较中所需的最小比较单元的长度
def binaryTreeMetricSort(data, length, noise):
    distance = lambda x, y: np.abs(x - y)
    extend_list = data.copy()
    for i in range(2 ** math.ceil(np.log2(len(data))) - len(data)):
        extend_list.append(0)

    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]

    # metrics = []
    # cur = 1
    # for i in range(1, max_level):
    #     tmpMetrics = []
    #     for j in range(cur, cur + 2 ** i, 2):
    #         l1l = index[j][0] - 1
    #         l1r = index[j][1]
    #         l2l = index[j + 1][0] - 1
    #         l2r = index[j + 1][1]
    #         if l1r - l1l < length:
    #             break
    #         tmpMetrics.append(dtw(extend_list[l1l: l1r], extend_list[l2l: l2r], dist=distance)[0])
    #     metrics.append(tmpMetrics)
    #     cur = cur + 2 ** i

    cur = 1
    for i in range(1, max_level):
        cur_code = []
        for j in range(cur, cur + 2 ** i, 2):
            tmp1 = linear_code[int(j / 2)]
            tmp2 = linear_code[int(j / 2)]
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l2l = index[j + 1][0] - 1
            l2r = index[j + 1][1]
            step = int((l1r - l1l) / 2)
            # 计算dtw时需要使用
            if step == 0:
                break
            metric1 = sum(rel_entr(extend_list[l1l: l1l + step], extend_list[l1l + step: l1r]))
            metric2 = sum(rel_entr(extend_list[l2l: l2l + step], extend_list[l2l + step: l2r]))
            # metric1 = sum(rel_entr(extend_list[l1l: l1r], noise[l1l: l1r]))
            # metric2 = sum(rel_entr(extend_list[l2l: l2r], noise[l2l: l2r]))
            # metric1 = dtw(extend_list[l1l: l1l + step], extend_list[l1l + step: l1r], dist=distance)[0]
            # metric2 = dtw(extend_list[l2l: l2l + step], extend_list[l2l + step: l2r], dist=distance)[0]
            # metric1 = dtw(extend_list[l1l: l1r], noise[l1l: l1r], dist=distance)[0]
            # metric2 = dtw(extend_list[l2l: l2r], noise[l2l: l2r], dist=distance)[0]
            if metric1 >= metric2:
                tmp1 += "0"
                tmp2 += "1"
            else:
                tmp1 += "1"
                tmp2 += "0"
            # 记录当前code，用于子节点使用
            linear_code.append(tmp1)
            linear_code.append(tmp2)
            # 记录当前code，用于输出
            cur_code.append(tmp1)
            cur_code.append(tmp2)
        cur = cur + 2 ** i
        code.append(cur_code)

    return_code = []
    for i in range(1, len(intervals)):
        if intervals[i] < length:
            return_code = code[i - 1]
            break

    for i in range(len(return_code)):
        return_code[i] = return_code[i][1:]  # 去掉根节点的遍历后的密钥
        # return_code[i] = return_code[i][-1]  # 只保留最后一层
    return return_code


# list 待排序的数组
# length 比较中所需的最小比较单元的长度
def binaryTreeSortPerm(data, length, closeness):
    extend_list = data.copy()
    for i in range(2 ** math.ceil(np.log2(len(data))) - len(data)):
        extend_list.append(0)

    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]
    cur = 1

    # 置换序列
    perm = np.arange(0, len(extend_list))
    combine = list(zip(extend_list, perm))
    # extend_list = np.array(extend_list)

    sums = []
    for i in range(1, max_level):
        for j in range(cur, cur + 2 ** i, 2):
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l2l = index[j + 1][0] - 1
            l2r = index[j + 1][1]
            if l1r - l1l < length:
                break
            sum1 = np.sum(extend_list[l1l: l1r])
            sum2 = np.sum(extend_list[l2l: l2r])
            sums.append([sum1, sum2])
        cur = cur + 2 ** i
    min_interval = abs(sums[0][0] - sums[0][1])
    for i in range(len(sums)):
        min_interval = min(min_interval, abs(sums[i][0] - sums[i][1]))

    cnt = 0  # 确保所有置换都试过一次
    # while min_interval <= closeness:
    while min_interval <= closeness and cnt < 10000:
        np.random.shuffle(combine)
        extend_list, perm = zip(*combine)
        # nextPermutation(perm)
        # extend_list = extend_list[perm]

        cur = 1
        sums = []
        for i in range(1, max_level):
            for j in range(cur, cur + 2 ** i, 2):
                l1l = index[j][0] - 1
                l1r = index[j][1]
                l2l = index[j + 1][0] - 1
                l2r = index[j + 1][1]
                if l1r - l1l < length:
                    break
                sum1 = np.sum(extend_list[l1l: l1r])
                sum2 = np.sum(extend_list[l2l: l2r])
                sums.append([sum1, sum2])
            cur = cur + 2 ** i
        min_interval = abs(sums[0][0] - sums[0][1])
        for i in range(len(sums)):
            min_interval = min(min_interval, abs(sums[i][0] - sums[i][1]))
        cnt += 1

    cur = 1
    for i in range(1, max_level):
        cur_code = []
        for j in range(cur, cur + 2 ** i, 2):
            tmp1 = linear_code[int(j / 2)]
            tmp2 = linear_code[int(j / 2)]
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l2l = index[j + 1][0] - 1
            l2r = index[j + 1][1]
            sum1 = np.sum(extend_list[l1l: l1r])
            sum2 = np.sum(extend_list[l2l: l2r])
            if abs(sum1 - sum2) <= closeness and l1r - l1l >= length:
                print("\033[0;31;40m error \033[0m")
            if sum1 >= sum2:
                tmp1 += "0"
                tmp2 += "1"
            else:
                tmp1 += "1"
                tmp2 += "0"
            # 记录当前code，用于子节点使用
            linear_code.append(tmp1)
            linear_code.append(tmp2)
            # 记录当前code，用于输出
            cur_code.append(tmp1)
            cur_code.append(tmp2)
        cur = cur + 2 ** i
        code.append(cur_code)

    return_code = []
    for i in range(1, len(intervals)):
        if intervals[i] < length:
            return_code = code[i - 1]
            break

    for i in range(len(return_code)):
        return_code[i] = return_code[i][1:]
    return return_code, np.array(perm)


# list 待排序的数组
# length 比较中所需的最小比较单元的长度
def binaryTreeSort(data, length, perm):
    extend_list = data.copy()
    for i in range(2 ** math.ceil(np.log2(len(data))) - len(data)):
        extend_list.append(0)

    extend_list = np.array(extend_list)
    extend_list = extend_list[perm]
    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])
            # 为什么不写成q.append([math.floor((l[0] + l[1]) / 2) + 1, l[1]])
            # 因为要保证生成的二叉树是完全二叉树，同时除了根节点外，中间节点的range不能为1（如下面的[7 7]）
            # 例如使用ceil时，长度为7的数组产生的二叉树为
            #                      [1 7] --从第1个到第7个
            #                   [1 4] [4 7]
            #             [1 2] [3 4] [4 5] [6 7]
            # [1 1] [2 2] [3 3] [4 4] [4 4] [5 5] [6 6] [7 7]
            # 这里最底层出现的两次[4 4]与中间节点[1 4][4 7]中重复使用的4并不会影响密钥的随机性
            # 否则使用floor+1时的二叉树为
            #                [1 7]
            #             [1 4] [5 7]
            #       [1 2] [3 4] [5 6] [7 7]
            # [1 1] [2 2] [3 3] [4 4] [5 5] [6 6]
            # 不仅[7 7]过早出现在中间节点，而且叶子节点中没有[7 7]

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]
    cur = 1
    for i in range(1, max_level):
        cur_code = []
        for j in range(cur, cur + 2 ** i, 2):
            tmp1 = linear_code[int(j / 2)]
            tmp2 = linear_code[int(j / 2)]
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l2l = index[j + 1][0] - 1
            l2r = index[j + 1][1]
            # print(cosine(extend_list[l1l: l1r], extend_list[l2l: l2r]))
            if compare(extend_list[l1l: l1r], extend_list[l2l: l2r]):
                tmp1 += "0"
                tmp2 += "1"
            else:
                tmp1 += "1"
                tmp2 += "0"
            # 记录当前code，用于子节点使用
            linear_code.append(tmp1)
            linear_code.append(tmp2)
            # 记录当前code，用于输出
            cur_code.append(tmp1)
            cur_code.append(tmp2)
        cur = cur + 2 ** i
        code.append(cur_code)

    return_code = []
    for i in range(1, len(intervals)):
        if intervals[i] < length:
            return_code = code[i - 1]
            break

    for i in range(len(return_code)):
        return_code[i] = return_code[i][1:]
    return return_code


# list 待排序的数组
# length 比较中所需的最小比较单元的长度
# pyramid shape
def quadTreeSort(list, length):
    extend_list = list.copy()
    for i in range(2 ** math.ceil(np.log2(len(list))) - len(list)):
        extend_list.append(0)

    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] - l[0] >= 3:
            q.append([l[0], l[0] + math.floor((l[1] - l[0]) / 4)])
            q.append([l[0] + math.ceil((l[1] - l[0]) / 4), l[0] + math.floor((l[1] - l[0]) / 2)])
            q.append([l[0] + math.ceil((l[1] - l[0]) / 2), l[0] + math.floor((l[1] - l[0]) * 3 / 4)])
            q.append([l[0] + math.ceil((l[1] - l[0]) * 3 / 4), l[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为4
    max_level = math.ceil(np.log10(len(index) * 3 + 1) / np.log10(4))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 4 ** i
    intervals.append(0)
    linear_code.append("1")
    code = [["1"]]
    cur = 1
    for i in range(1, max_level):
        cur_code = []
        for j in range(cur, cur + 4 ** i, 4):
            tmp1 = linear_code[int(j / 4)]
            tmp2 = linear_code[int(j / 4)]
            tmp3 = linear_code[int(j / 4)]
            tmp4 = linear_code[int(j / 4)]
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l2l = index[j + 1][0] - 1
            l2r = index[j + 1][1]
            l3l = index[j + 2][0] - 1
            l3r = index[j + 2][1]
            l4l = index[j + 3][0] - 1
            l4r = index[j + 3][1]
            sum1 = np.sum(list[l1l: l1r])
            sum2 = np.sum(list[l2l: l2r])
            sum3 = np.sum(list[l3l: l3r])
            sum4 = np.sum(list[l4l: l4r])
            # 四组求和值分别比大小，按照大小顺序分别编码为1 2 3 4
            sort_sum = [[sum1, 1], [sum2, 2], [sum3, 3], [sum4, 4]]
            sort_sum.sort(key=lambda x: (x[0]))
            for k in range(len(sort_sum)):
                if sort_sum[k][1] == 1:
                    tmp1 += str(k + 1)
                elif sort_sum[k][1] == 2:
                    tmp2 += str(k + 1)
                elif sort_sum[k][1] == 3:
                    tmp3 += str(k + 1)
                elif sort_sum[k][1] == 4:
                    tmp4 += str(k + 1)
            # 记录当前code，用于子节点使用
            linear_code.append(tmp1)
            linear_code.append(tmp2)
            linear_code.append(tmp3)
            linear_code.append(tmp4)
            # 记录当前code，用于输出
            cur_code.append(tmp1)
            cur_code.append(tmp2)
            cur_code.append(tmp3)
            cur_code.append(tmp4)
        cur = cur + 4 ** i
        code.append(cur_code)

    return_code = []
    for i in range(0, len(intervals)):
        if intervals[i] < length:
            return_code = code[i - 1]

    for i in range(len(return_code)):
        return_code[i] = return_code[i][1:]
    return return_code


def findMinSumDiff(data):
    sort_data = data.copy()
    sort_data.sort()
    minSumDiff = sys.maxsize
    for i in range(1, len(sort_data)):
        minSumDiff = min(minSumDiff, abs(sort_data[i] - sort_data[i - 1]))
    return minSumDiff


# list 待排序的数组
# length 比较中所需的最小比较单元的长度
def binaryLevelSortPerm(data, length, closeness):
    extend_list = data.copy()
    for i in range(2 ** math.ceil(np.log2(len(data))) - len(data)):
        extend_list.append(0)

    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]
    cur = 1

    # 置换序列
    perm = np.arange(0, len(extend_list))
    combine = list(zip(extend_list, perm))

    sums = []
    for i in range(1, max_level):
        cur_sum = []
        for j in range(cur, cur + 2 ** i):
            ll = index[j][0] - 1
            lr = index[j][1]
            if lr - ll < length:
                break
            cur_sum.append(sum(extend_list[ll:lr]))
        if cur_sum:
            sums.append(cur_sum)
        cur = cur + 2 ** i
    min_interval = sys.maxsize
    for i in range(len(sums)):
        min_interval = min(min_interval, findMinSumDiff(sums[i]))

    cnt = 0  # 确保所有置换都试过一次
    while min_interval <= closeness and cnt < 100000:
        # while min_interval <= closeness:
        np.random.shuffle(combine)
        extend_list, perm = zip(*combine)

        cur = 1
        sums = []
        for i in range(1, max_level):
            cur_sum = []
            for j in range(cur, cur + 2 ** i):
                ll = index[j][0] - 1
                lr = index[j][1]
                if lr - ll < length:
                    break
                cur_sum.append(sum(extend_list[ll:lr]))
            if cur_sum:
                sums.append(cur_sum)
            cur = cur + 2 ** i
        min_interval = sys.maxsize
        for i in range(len(sums)):
            min_interval = min(min_interval, findMinSumDiff(sums[i]))
        cnt += 1

    cur = 1
    for i in range(1, max_level):
        cur_code = []
        cur_sum = []
        for j in range(cur, cur + 2 ** i):
            ll = index[j][0] - 1
            lr = index[j][1]
            if lr - ll < length:
                break
            cur_sum.append(sum(extend_list[ll:lr]))
        sort_sum = np.argsort(cur_sum)
        # 记录当前code，用于子节点使用
        for j in range(len(sort_sum)):
            father = int((len(linear_code) - 1) / 2)
            # linear_code.append(linear_code[father] + str(sort_sum[j]) + ',')
            linear_code.append(linear_code[father] + str(sort_sum[j]))
            cur_code.append(linear_code[father] + str(sort_sum[j]))
        cur = cur + 2 ** i
        code.append(cur_code)

    return_code = []
    for i in range(1, len(intervals)):
        if intervals[i] < length:
            return_code = code[i - 1]
            break

    for i in range(len(return_code)):
        return_code[i] = return_code[i][1:]

    return return_code, np.array(perm)


# list 待排序的数组
# length 比较中所需的最小比较单元的长度
def binaryLevelSort(list, length, perm):
    extend_list = list.copy()
    for i in range(2 ** math.ceil(np.log2(len(list))) - len(list)):
        extend_list.append(0)

    extend_list = np.array(extend_list)
    extend_list = extend_list[perm]
    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]
    cur = 1
    for i in range(1, max_level):
        cur_code = []
        cur_sum = []
        for j in range(cur, cur + 2 ** i):
            ll = index[j][0] - 1
            lr = index[j][1]
            cur_sum.append(sum(extend_list[ll:lr]))
        sort_sum = np.argsort(cur_sum)
        # 记录当前code，用于子节点使用
        for j in range(len(sort_sum)):
            father = int((len(linear_code) - 1) / 2)
            linear_code.append(linear_code[father] + str(sort_sum[j]))
            # linear_code.append(linear_code[father] + str(sort_sum[j]) + ',')
            cur_code.append(linear_code[father] + str(sort_sum[j]))
        cur = cur + 2 ** i
        code.append(cur_code)

    return_code = []
    for i in range(1, len(intervals)):
        if intervals[i] < length:
            return_code = code[i - 1]
            break

    for i in range(len(return_code)):
        return_code[i] = return_code[i][1:]

    return return_code
