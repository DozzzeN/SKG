import math
import random
import sys
import time
from collections import deque
from itertools import chain

import numpy as np
from dtw import accelerated_dtw
from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import loadmat

from algorithm import findMinDiff, partner, smooth, genSample


def dtw_metric(data1, data2):
    distance = lambda x, y: np.abs(x - y)
    data1 = np.array(data1)
    data2 = np.array(data2)
    # return dtw(data1, data2, dist=distance)[0]
    return accelerated_dtw(data1, data2, dist=distance)[0]


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
            min_diff = min(min_diff, sort_metrics[j + 1] - sort_metrics[j])
        min_diffs = min(min_diffs, min_diff)

        metrics.append(tmp_metrics)
        cur = cur + 2 ** i

    # print("\033[0;32;40mmin_diff", min_diffs, "\033[0m")

    for i in range(0, len(metrics)):
        tmp = np.argsort(metrics[i])
        code.append(tmp)

    return_code = []
    for i in range(1, len(intervals)):
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
                two_tmp_metric = eval(metric + '_metric(extend_list[i: i + interval_length], tmp_noise_list)')
                if findMinDiff(two_samples_metrics, two_tmp_metric) >= lower_bound \
                        and findMinDiff(two_samples_metrics, two_tmp_metric) <= upper_bound:
                    two_samples_metrics.append(two_tmp_metric)
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
                four_tmp_metric = eval(metric + '_metric(extend_list[i: i + interval_length * 2], updated_noise)')
                # double check可能影响的下层
                two_tmp_metric = eval(metric + '_metric(extend_list[i + interval_length: i + interval_length * 2], '
                                               'tmp_noise_list)')
                # 不用重新计算two_samples_metrics，因为只多了一个对比的点，即原始更新噪音点的metric没有更新，但是不会影响结果
                if findMinDiff(four_samples_metrics, four_tmp_metric) >= lower_bound \
                        and findMinDiff(four_samples_metrics, four_tmp_metric) <= upper_bound \
                        and findMinDiff(two_samples_metrics, two_tmp_metric) >= lower_bound \
                        and findMinDiff(two_samples_metrics, two_tmp_metric) <= upper_bound:
                    four_samples_metrics.append(four_tmp_metric)
                    two_samples_metrics.append(two_tmp_metric)
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
                eight_tmp_metric = eval(metric + '_metric(extend_list[i: i + interval_length * 4], updated_noise)')
                # double check可能影响的下层
                two_tmp_metric = eval(metric + '_metric(extend_list[i + interval_length * 3: i + interval_length * 4], '
                                               'tmp_noise_list)')
                four_tmp_metric = eval(
                    metric + '_metric(extend_list[i + interval_length * 2: i + interval_length * 4], '
                             'tmp_noise_list)')
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
                sixteen_tmp_metric = eval(metric + '_metric(extend_list[i: i + interval_length * 8], updated_noise)')
                # double check可能影响的下层
                two_tmp_metric = eval(metric + '_metric(extend_list[i + interval_length * 7: i + interval_length * 8], '
                                               'tmp_noise_list)')
                four_tmp_metric = eval(
                    metric + '_metric(extend_list[i + interval_length * 6: i + interval_length * 8], '
                             'tmp_noise_list)')
                eight_tmp_metric = eval(
                    metric + '_metric(extend_list[i + interval_length * 4: i + interval_length * 8], '
                             'tmp_noise_list)')
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
            min_diff = min(min_diff, sort_metrics[j + 1] - sort_metrics[j])
        min_diffs = min(min_diffs, min_diff)

    # 检验差值diff是否满足阈值
    # if min_diffs <= lower_bound:
    #     print("\033[0;33;40mmin_diff", "poor", min_diffs, "\033[0m")

    for i in range(0, len(metrics)):
        tmp = np.argsort(metrics[i])
        code.append(tmp)

    return_code = []
    for i in range(1, len(intervals)):
        for j in range(len(code[i])):
            return_code.append(code[i][j])
        if intervals[i] < length:
            break

    corr = []
    random_corr = []
    for i in range(0, len(metrics) - 2):
        t1 = np.random.permutation(range(len(metrics[i])))
        t2 = np.random.permutation(range(len(metrics[i + 1])))
        corr.append(dtw_metric(np.argsort(metrics[i]), np.argsort(metrics[i + 1])))
        random_corr.append(dtw_metric(t1, t2))

    # 展示所有度量结果和对应的密钥
    y = list(chain.from_iterable(metrics))
    x = range(len(y))
    plt.figure()
    plt.plot(x, y)

    bbox_color = ['black', 'blue', 'yellow', 'green']
    # 转换为度量的顺序索引
    code_indices = []
    for i in range(0, len(metrics)):
        tmp = np.argsort(np.argsort(metrics[i]))
        code_indices.append(tmp)
    code_indices = list(chain.from_iterable(code_indices))
    for a, b in zip(x, y):
        if a < 2:
            plt.text(a, b, '%d' % code_indices[a], ha='center', va='bottom', fontsize=9,
                     bbox=dict(facecolor=bbox_color[0], alpha=1), color="white")
        elif a < 2 + 4:
            plt.text(a, b, '%d' % code_indices[a], ha='center', va='bottom', fontsize=9,
                     bbox=dict(facecolor=bbox_color[1], alpha=1), color="white")
        elif a < 2 + 4 + 8:
            plt.text(a, b, '%d' % code_indices[a], ha='center', va='bottom', fontsize=9,
                     bbox=dict(facecolor=bbox_color[2], alpha=1))
        elif a < 2 + 4 + 8 + 16:
            plt.text(a, b, '%d' % code_indices[a], ha='center', va='bottom', fontsize=9,
                     bbox=dict(facecolor=bbox_color[3], alpha=1), color="white")
    plt.show()

    return return_code, designed_noise, [corr, random_corr]


fileName = "../data/data_static_indoor_1_r_m.mat"
rawData = loadmat(fileName)
csv = open("./level_corr.csv", "a+")
csv.write("\n")

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

dataLen = len(CSIa1Orig)

CSIa1Orig = np.array(CSIa1Orig)
CSIb1Orig = np.array(CSIb1Orig)
CSIn1Orig = np.random.normal(loc=-100, scale=1000, size=dataLen)
# CSIa1OrigMean = np.mean(CSIa1Orig)
# CSIa1OrigStd = np.std(CSIa1Orig, ddof=1)
# randomLoc = np.random.randint(CSIa1OrigMean - CSIa1OrigMean / 2, - CSIa1OrigMean / 2) if CSIa1OrigMean < 0 \
#     else np.random.randint(- CSIa1OrigMean / 2, CSIa1OrigMean - CSIa1OrigMean / 2)
# randomScale = np.random.randint(CSIa1OrigStd / 2, CSIa1OrigStd + CSIa1OrigStd / 2)
# CSIn1Orig = np.random.normal(loc=randomLoc, scale=randomScale, size=dataLen)
# CSIn1Orig = np.random.normal(loc=0, scale=np.std(CSIa1Orig, ddof=1), size=dataLen)
# CSIn1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)
CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)

CSIi1Orig = loadmat('../data/data_static_indoor_1_r_m.mat')['A'][:, 0]
# CSIi1Orig = CSIi1Orig + np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)
# CSIi1Orig = CSIi1Orig + np.random.normal(loc=0, scale=np.std(CSIi1Orig, ddof=1), size=dataLen)
np.random.shuffle(CSIi1Orig)

# entropyTime = time.time()
# CSIa1Orig, CSIb1Orig, CSIe1Orig = splitEntropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, 5, dataLen)
# print("--- entropyTime %s seconds ---" % (time.time() - entropyTime))

CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')
CSIb1Orig = smooth(CSIb1Orig, window_len=15, window='flat')
CSIe1Orig = smooth(CSIe1Orig, window_len=15, window="flat")
CSIn1Orig = smooth(CSIn1Orig, window_len=15, window="flat")
CSIi1Orig = smooth(CSIi1Orig, window_len=15, window="flat")

CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
CSIe1OrigBack = CSIe1Orig.copy()
CSIn1OrigBack = CSIn1Orig.copy()
CSIi1OrigBack = CSIi1Orig.copy()

# intvl = 5
intvl = 1
keyLen = 128
interval_length = 2
addNoise = False
insertRatio = 1
ratio = 1
closeness = 5
metric = "dtw"

for segLen in range(4, 16):
    print("segLen", segLen)
    adjCorrSum = []
    randomCorrSum = []

    t1 = 0
    t2 = 0
    t3 = 0
    r1 = 0
    r2 = 0
    r3 = 0

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
        CSIi1Orig = CSIi1OrigBack.copy()

        # CSIa1Orig[range(staInd, endInd, 1)] = CSIa1Orig[range(staInd, endInd, 1)] - np.mean(
        #     CSIa1Orig[range(staInd, endInd, 1)])
        # CSIb1Orig[range(staInd, endInd, 1)] = CSIb1Orig[range(staInd, endInd, 1)] - np.mean(
        #     CSIb1Orig[range(staInd, endInd, 1)])
        # CSIe1Orig[range(staInd, endInd, 1)] = CSIe1Orig[range(staInd, endInd, 1)] - np.mean(
        #     CSIe1Orig[range(staInd, endInd, 1)])
        # CSIn1Orig[range(staInd, endInd, 1)] = CSIn1Orig[range(staInd, endInd, 1)] - np.mean(
        #     CSIn1Orig[range(staInd, endInd, 1)])
        # CSIi1Orig[range(staInd, endInd, 1)] = CSIi1Orig[range(staInd, endInd, 1)] - np.mean(
        #     CSIi1Orig[range(staInd, endInd, 1)])

        # tmpCSIa1 = np.fft.fft(CSIa1Orig[range(staInd, endInd, 1)])
        # tmpCSIb1 = np.fft.fft(CSIb1Orig[range(staInd, endInd, 1)])
        # tmpCSIe1 = np.fft.fft(CSIe1Orig[range(staInd, endInd, 1)])
        # tmpNoise = np.fft.fft(CSIn1Orig[range(staInd, endInd, 1)])
        # tmpCSIi1 = np.fft.fft(CSIi1Orig[range(staInd, endInd, 1)])

        tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
        tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
        tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
        tmpNoise = CSIn1Orig[range(staInd, endInd, 1)]
        tmpCSIi1 = CSIi1Orig[range(staInd, endInd, 1)]

        # print(np.mean(tmpCSIa1), np.std(tmpCSIa1, ddof=1))
        # print(np.mean(tmpCSIb1), np.std(tmpCSIb1, ddof=1))
        # print(np.mean(tmpCSIe1), np.std(tmpCSIe1, ddof=1))
        # print(np.mean(tmpNoise), np.std(tmpNoise, ddof=1))
        # print(np.mean(tmpCSIi1), np.std(tmpCSIi1, ddof=1))

        # linspace函数生成元素为50的等间隔数列，可以指定第三个参数为元素个数
        # signal.square返回周期性的方波波形
        tmpPulse = signal.square(
            2 * np.pi * 1 / intvl * np.linspace(0, np.pi * 0.5 * keyLen, keyLen * intvl))  ## Rectangular pulse

        # if staInd >= 40000:
        #     plt.plot(CSIa1Orig, 'r')
        #     # plt.plot(CSIa1Orig,'b')
        #     plt.show()
        #
        #     plt.plot(tmpCSIa1, 'r')
        #     # plt.plot(tmpCSIb1,'b')
        #     plt.show()
        if addNoise:
            # tmpCSIa1 = (np.float_power(np.abs(tmpCSIa1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIa1))
            # tmpCSIb1 = (np.float_power(np.abs(tmpCSIb1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIb1))
            # tmpCSIe1 = (np.float_power(np.abs(tmpCSIe1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIe1))
            # tmpCSIa1 = tmpPulse * np.float_power(np.abs(tmpCSIa1), tmpNoise)
            # tmpCSIb1 = tmpPulse * np.float_power(np.abs(tmpCSIb1), tmpNoise)
            # tmpCSIe1 = tmpPulse * np.float_power(np.abs(tmpCSIe1), tmpNoise)
            tmpCSIa1 = tmpCSIa1 * tmpNoise
            tmpCSIb1 = tmpCSIb1 * tmpNoise
            tmpCSIe1 = tmpCSIe1 * tmpNoise
            tmpNoise = tmpNoise * tmpNoise
            # tmpCSIa1 = tmpCSIa1 + tmpNoise
            # tmpCSIb1 = tmpCSIb1 + tmpNoise
            # tmpCSIe1 = tmpCSIe1 + tmpNoise
            # tmpNoise = tmpNoise + tmpNoise
        else:
            # tmpCSIa1 = tmpPulse * tmpCSIa1
            # tmpCSIb1 = tmpPulse * tmpCSIb1
            # tmpCSIe1 = tmpPulse * tmpCSIe1
            tmpCSIa1 = tmpCSIa1
            tmpCSIb1 = tmpCSIb1
            tmpCSIe1 = tmpCSIe1
            tmpNoise = tmpNoise
        # if staInd >= 40000:
        #     plt.plot(tmpCSIa1, 'g')
        #     # plt.plot(tmpCSIb1,'y')
        #     plt.show()

        # 去除直流分量
        # tmpCSIa1 = tmpCSIa1 - (np.mean(tmpCSIa1) - np.mean(tmpCSIb1))  # Mean value consistency
        tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
        tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
        tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
        tmpCSIi1 = tmpCSIi1 - np.mean(tmpCSIi1)
        # tmpNoise = tmpNoise - np.mean(tmpNoise)

        sortCSIa1 = tmpCSIa1
        sortCSIb1 = tmpCSIb1
        sortCSIe1 = tmpCSIe1
        sortNoise = tmpNoise
        sortCSIi1 = tmpCSIi1

        # CSIa1Orig[range(staInd, endInd, 1)] = tmpCSIa1
        # CSIb1Orig[range(staInd, endInd, 1)] = tmpCSIb1
        # CSIe1Orig[range(staInd, endInd, 1)] = tmpCSIe1
        # CSIn1Orig[range(staInd, endInd, 1)] = tmpNoise
        # CSIi1Orig[range(staInd, endInd, 1)] = tmpCSIi1
        #
        # permLen = len(range(staInd, endInd, intvl))
        # origInd = np.array([xx for xx in range(staInd, endInd, intvl)])
        #
        # sortCSIa1 = np.zeros(permLen)
        # sortCSIb1 = np.zeros(permLen)
        # sortCSIe1 = np.zeros(permLen)
        # sortNoise = np.zeros(permLen)
        # sortCSIi1 = np.zeros(permLen)
        #
        # for ii in range(permLen):
        #     aIndVec = np.array([aa for aa in range(origInd[ii], origInd[ii] + intvl, 1)])  ## for non-permuted CSIa1
        #
        #     for jj in range(permLen, permLen * 2):
        #         bIndVec = np.array([bb for bb in range(origInd[jj - permLen], origInd[jj - permLen] + intvl, 1)])
        #
        #         CSIa1Tmp = CSIa1Orig[aIndVec]
        #         CSIb1Tmp = CSIb1Orig[bIndVec]
        #         CSIe1Tmp = CSIe1Orig[bIndVec]
        #         CSIn1Tmp = CSIn1Orig[aIndVec]
        #         CSIi1Tmp = CSIi1Orig[aIndVec]
        #
        #         sortCSIa1[ii] = np.mean(CSIa1Tmp)  ## Metric 1: Mean
        #         sortCSIb1[jj - permLen] = np.mean(CSIb1Tmp)  # 只赋值一次
        #         sortCSIe1[jj - permLen] = np.mean(CSIe1Tmp)
        #         sortNoise[ii - permLen] = np.mean(CSIn1Tmp)
        #         sortCSIi1[ii - permLen] = np.mean(CSIi1Tmp)

        # sortCSIa1是原始算法中排序前的数据
        # sortCSIa1 = np.log10(np.abs(sortCSIa1))
        # sortCSIb1 = np.log10(np.abs(sortCSIb1))
        # sortCSIe1 = np.log10(np.abs(sortCSIe1))
        # sortNoise = np.log10(np.abs(sortNoise))

        # 取原数据的一部分来reshape
        sortCSIa1Reshape = sortCSIa1[0:segLen * int(len(sortCSIa1) / segLen)]
        sortCSIb1Reshape = sortCSIb1[0:segLen * int(len(sortCSIb1) / segLen)]
        sortCSIe1Reshape = sortCSIe1[0:segLen * int(len(sortCSIe1) / segLen)]
        sortNoiseReshape = sortNoise[0:segLen * int(len(sortNoise) / segLen)]
        sortCSIi1Reshape = sortCSIi1[0:segLen * int(len(sortCSIi1) / segLen)]

        sortCSIa1Reshape = sortCSIa1Reshape.reshape(int(len(sortCSIa1Reshape) / segLen), segLen)
        sortCSIb1Reshape = sortCSIb1Reshape.reshape(int(len(sortCSIb1Reshape) / segLen), segLen)
        sortCSIe1Reshape = sortCSIe1Reshape.reshape(int(len(sortCSIe1Reshape) / segLen), segLen)
        sortNoiseReshape = sortNoiseReshape.reshape(int(len(sortNoiseReshape) / segLen), segLen)
        sortCSIi1Reshape = sortCSIi1Reshape.reshape(int(len(sortCSIi1Reshape) / segLen), segLen)

        # 归一化
        # sortCSIa1 = []
        # sortCSIb1 = []
        # sortCSIe1 = []
        # sortNoise = []
        # sortCSIi1 = []
        #
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

        # sortCSIa1 = np.array(genArray(sortCSIa1Reshape))
        # sortCSIb1 = np.array(genArray(sortCSIb1Reshape))
        # sortCSIe1 = np.array(genArray(sortCSIe1Reshape))
        # sortNoise = np.array(genArray(sortNoiseReshape))

        sortCSIa1 = np.array(genSample(sortCSIa1Reshape, ratio))
        sortCSIb1 = np.array(genSample(sortCSIb1Reshape, ratio))
        sortCSIe1 = np.array(genSample(sortCSIe1Reshape, ratio))
        sortNoise = np.array(genSample(sortNoiseReshape, ratio))
        sortCSIi1 = np.array(genSample(sortCSIi1Reshape, ratio))

        # plt.plot(sortCSIa1, "b")
        # plt.show()
        # plt.plot(sortCSIb1, "y")
        # plt.show()
        # plt.plot(sortCSIe1, "r")
        # plt.show()

        # 这样插入噪音点和sortCSIa1数据过于接近，效果不好，不如在一个大范围内产生的噪音点好
        # sortCSIi1 = np.random.normal(loc=np.mean(sortCSIa1), scale=np.std(sortCSIa1, ddof=1), size=len(sortCSIa1))
        # np.random.shuffle(sortCSIi1)
        # insertIndex = np.random.permutation(len(sortCSIa1))
        # sortCSIa1 = insertNoise(sortCSIa1, sortCSIi1, insertIndex, insertRatio)
        # sortCSIb1 = insertNoise(sortCSIb1, sortCSIi1, insertIndex, insertRatio)
        # sortCSIe1 = insertNoise(sortCSIe1, sortCSIi1, insertIndex, insertRatio)
        # sortNoise = insertNoise(sortNoise, sortCSIi1, insertIndex, insertRatio)

        # plt.plot(sortCSIa1, "b")
        # plt.plot(sortCSIb1, "y")
        # plt.show()
        # plt.plot(sortCSIb1, "y")
        # plt.plot(sortCSIe1, "r")
        # plt.plot(sortNoise, "r")
        # plt.show()

        # 最后各自的密钥
        a_list = []
        b_list = []
        e_list = []
        n_list = []

        sortCSIa1 = np.array(sortCSIa1)
        sortCSIb1 = np.array(sortCSIb1)
        sortCSIe1 = np.array(sortCSIe1)
        sortNoise = np.array(sortNoise)

        # b找出删除的位置blots，发给a，a进行删除
        # _, perm = sortMethod[0](list(sortCSIa1), interval_length, list(sortNoise), metric)
        # b_list_number, blots = sortMethod[2](list(sortCSIb1), interval_length, list(sortNoise), perm, metric)
        # a_list_number = sortMethod[3](list(sortCSIa1), interval_length, list(sortNoise), perm, blots, metric)
        # e_list_number = sortMethod[3](list(sortCSIe1), interval_length, list(sortNoise), perm, blots, metric)
        # n_list_number = sortMethod[3](list(sortNoise), interval_length, list(sortNoise), perm, blots, metric)

        # 各自删除，不发送blots
        # _, perm = sortMethod[0](list(sortCSIa1), interval_length, list(sortNoise), metric)
        # b_list_number, _ = sortMethod[2](list(sortCSIb1), interval_length, list(sortNoise), perm, metric)
        # a_list_number, _ = sortMethod[2](list(sortCSIa1), interval_length, list(sortNoise), perm, metric)
        # e_list_number, _ = sortMethod[2](list(sortCSIe1), interval_length, list(sortNoise), perm, metric)
        # n_list_number, _ = sortMethod[2](list(sortNoise), interval_length, list(sortNoise), perm, metric)

        # 不添加噪音，置换
        # _, perm = sortMethod[0](list(sortCSIa1), interval_length, list(sortNoise), metric)
        # b_list_number = sortMethod[1](list(sortCSIb1), interval_length, list(sortNoise), perm, metric)
        # a_list_number = sortMethod[1](list(sortCSIa1), interval_length, list(sortNoise), perm, metric)
        # e_list_number = sortMethod[1](list(sortCSIe1), interval_length, list(sortNoise), perm, metric)
        # n_list_number = sortMethod[1](list(sortNoise), interval_length, list(sortNoise), perm, metric)

        # 设计对比的噪音点
        # a_list_number, designed_noise = sortMethod[5](list(sortCSIa1), interval_length, list(sortCSIi1), metric)
        # a_list_number, designed_noise = sortMethod[5](list(sortCSIa1), interval_length, list(sortNoise), metric)
        # sortInsrt = list(np.random.normal(loc=np.mean(sortCSIa1) * 2,
        #                                scale=np.std(sortCSIa1, ddof=1) * 10,
        #                                size=len(sortCSIa1) * 10))
        # sortInsrt = list(np.random.laplace(loc=np.mean(sortCSIa1),
        #                                 scale=10,
        #                                 size=len(sortCSIa1) * 10))
        scale = 2
        offset = 100
        sortInsrt = list(np.random.uniform(
            np.min(sortCSIa1) * scale - offset,
            np.max(sortCSIa1) * scale + offset,
            len(sortCSIa1) * 100))
        sortEaves = list(np.random.uniform(
            np.min(sortCSIa1) * scale - offset,
            np.max(sortCSIa1) * scale + offset,
            len(sortCSIa1)))
        # sortEaves = list(np.random.uniform(
        #     np.mean(sortCSIa1) - scale * np.std(sortCSIa1, ddof=1),
        #     np.mean(sortCSIa1) + scale * np.std(sortCSIa1, ddof=1),
        #     len(sortCSIa1)))
        # sortEaves = list(np.random.normal(np.mean(sortCSIa1), np.std(sortCSIa1, ddof=1) * 10, len(sortCSIa1)))

        # plt.figure()
        # plt.plot(sortCSIa1, 'r')
        # plt.plot(sortCSIb1, 'b')
        # plt.plot(sortCSIe1, 'k')
        # plt.plot(sortEaves, 'g')
        # plt.show()

        a_list_number, designed_noise, corrs = levelNoiseMetricSortPerm(list(sortCSIa1), interval_length,
                                                                        list(sortInsrt),
                                                                        metric)
        b_list_number = simpleLevelMetricSort(list(sortCSIb1), interval_length, list(designed_noise), metric)
        e_list_number = simpleLevelMetricSort(list(sortEaves), interval_length, list(designed_noise), metric)
        # e_list_number = sortMethod[4](list(sortCSIe1), interval_length, list(designed_noise), metric)
        n_list_number = simpleLevelMetricSort(list(sortNoise), interval_length, list(designed_noise), metric)

        # 噪音点差距不大
        # a_list_number = sortMethod[4](list(sortCSIa1), interval_length, list(sortCSIi1), metric)
        # b_list_number = sortMethod[4](list(sortCSIb1), interval_length, list(sortCSIi1), metric)
        # e_list_number = sortMethod[4](list(sortCSIe1), interval_length, list(sortCSIi1), metric)
        # n_list_number = sortMethod[4](list(sortNoise), interval_length, list(sortCSIi1), metric)

        # 只添加噪音点
        # a_list_number = sortMethod[4](list(sortCSIa1), interval_length, list(sortNoise), metric)
        # b_list_number = sortMethod[4](list(sortCSIb1), interval_length, list(sortNoise), metric)
        # e_list_number = sortMethod[4](list(sortCSIe1), interval_length, list(sortNoise), metric)
        # n_list_number = sortMethod[4](list(sortNoise), interval_length, list(sortNoise), metric)

        # simpleLevelMetricSort
        # a_list_number = sortMethod(list(sortCSIa1), interval_length)
        # b_list_number = sortMethod(list(sortCSIb1), interval_length)
        # e_list_number = sortMethod(list(sortCSIe1), interval_length)
        # n_list_number = sortMethod(list(sortNoise), interval_length)

        # a_list = a_list_number
        # b_list = b_list_number
        # e_list = e_list_number
        # n_list = n_list_number

        # scale = 1
        # for i in range(len(a_list_number)):
        #     a_list.append(int(a_list_number[i] / scale))
        # for i in range(len(b_list_number)):
        #     b_list.append(int(b_list_number[i] / scale))
        # for i in range(len(e_list_number)):
        #     e_list.append(int(e_list_number[i] / scale))
        # for i in range(len(n_list_number)):
        #     n_list.append(int(n_list_number[i] / scale))

        # 转为十进制
        # for i in range(len(a_list_number)):
        #     a_list.append(int(a_list_number[i], 2))
        # for i in range(len(b_list_number)):
        #     b_list.append(int(b_list_number[i], 2))
        # for i in range(len(e_list_number)):
        #     e_list.append(int(e_list_number[i], 2))
        # for i in range(len(n_list_number)):
        #     n_list.append(int(n_list_number[i], 2))

        # 转成层序密钥
        a_level_number = []
        b_level_number = []
        e_level_number = []
        n_level_number = []
        i = 0
        step = 1
        while i < len(a_list_number):
            a_level_number.append(list(a_list_number[i: i + 2 ** step]))
            i = i + 2 ** step
            step += 1
        i = 0
        step = 1
        while i < len(b_list_number):
            b_level_number.append(list(b_list_number[i: i + 2 ** step]))
            i = i + 2 ** step
            step += 1
        i = 0
        step = 1
        while i < len(e_list_number):
            e_level_number.append(list(e_list_number[i: i + 2 ** step]))
            i = i + 2 ** step
            step += 1
        i = 0
        step = 1
        while i < len(n_list_number):
            n_level_number.append(list(n_list_number[i: i + 2 ** step]))
            i = i + 2 ** step
            step += 1

        # 转成二进制
        for i in range(len(a_level_number)):
            for j in range(len(a_level_number[i])):
                number = bin(int(a_level_number[i][j]))[2:].zfill(i + 1)
                a_list += number
        for i in range(len(b_level_number)):
            for j in range(len(b_level_number[i])):
                number = bin(int(b_level_number[i][j]))[2:].zfill(i + 1)
                b_list += number
        for i in range(len(e_level_number)):
            for j in range(len(e_level_number[i])):
                number = bin(int(e_level_number[i][j]))[2:].zfill(i + 1)
                e_list += number
        for i in range(len(n_level_number)):
            for j in range(len(n_level_number[i])):
                number = bin(int(n_level_number[i][j]))[2:].zfill(i + 1)
                n_list += number

        # 转为二进制
        for i in range(len(a_list_number)):
            a_list += bin(int(a_list_number[i]))[2:]
        for i in range(len(b_list_number)):
            b_list += bin(int(b_list_number[i]))[2:]
        for i in range(len(e_list_number)):
            e_list += bin(int(e_list_number[i]))[2:]
        for i in range(len(n_list_number)):
            n_list += bin(int(n_list_number[i]))[2:]

        # 对齐密钥，随机补全
        for i in range(len(a_list) - len(e_list)):
            e_list += str(np.random.randint(0, 2))
        for i in range(len(a_list) - len(n_list)):
            n_list += str(np.random.randint(0, 2))

        # print("keys of a:", len(a_list), a_list)
        # print("keys of a:", len(a_list_number), a_list_number)
        # print("keys of b:", len(b_list), b_list)
        # print("keys of b:", len(b_list_number), b_list_number)
        # print("keys of e:", len(e_list), e_list)
        # print("keys of e:", len(e_list_number), e_list_number)
        # print("keys of n:", len(n_list), n_list)
        # print("keys of n:", len(n_list_number), n_list_number)

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

        # if sum2 != sum1:
        #     for i in range(len(a_list_number)):
        #         if a_list_number[i] != b_list_number[i]:
        #             print(i, a_list_number[i], b_list_number[i])

        if sum2 != sum1:
            pass
            # b_list_number, blots = sortMethod[2](list(sortCSIb1), interval_length, list(sortNoise), perm, "dtw")
            # a_list_number, blots = sortMethod[2](list(sortCSIa1), interval_length, list(sortNoise), perm, "dtw")

        # a_list_number = sortMethod(list(sortCSIa1), interval_length, list(sortNoise))
        # b_list_number = sortMethod(list(sortCSIb1), interval_length, list(sortNoise))

        # if sum2 == sum1:
        #     print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
        # else:
        #     print("\033[0;31;40ma-b", "bad", sum2, sum2 / sum1, "\033[0m")
        # print("a-e", sum3, sum3 / sum1)
        # print("a-n", sum4, sum4 / sum1)
        # print("----------------------")
        originSum += sum1
        correctSum += sum2
        randomSum += sum3
        noiseSum += sum4

        originWholeSum += 1
        correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
        randomWholeSum = randomWholeSum + 1 if sum3 == sum1 else randomWholeSum
        noiseWholeSum = noiseWholeSum + 1 if sum4 == sum1 else noiseWholeSum

        adjCorrSum.append(corrs[0])
        randomCorrSum.append(corrs[1])
        print("adj_corr", corrs[0])
        print("random_corr", corrs[1])

        if len(corrs[0]) == 3:
            t1 += corrs[0][0]
            t2 += corrs[0][1]
            t3 += corrs[0][2]
            r1 += corrs[1][0]
            r2 += corrs[1][1]
            r3 += corrs[1][2]
        elif len(corrs[0]) == 2:
            t1 += corrs[0][0]
            t2 += corrs[0][1]
            t3 += 0
            r1 += corrs[1][0]
            r2 += corrs[1][1]
            r3 += 0
        elif len(corrs[0]) == 1:
            t1 += corrs[0][0]
            t2 += 0
            t3 += 0
            r1 += corrs[1][0]
            r2 += 0
            r3 += 0

        coding = ""
        for i in range(len(a_list)):
            coding += a_list[i]
        codings += coding + "\n"

    # with open('./key.txt', 'a', ) as f:
    #     f.write(codings)

    # print("a-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10))
    # print("a-e all", randomSum, "/", originSum, "=", round(randomSum / originSum, 10))
    # print("a-n all", noiseSum, "/", originSum, "=", round(noiseSum / originSum, 10))
    # print("a-b whole match", correctWholeSum, "/", originWholeSum, "=", round(correctWholeSum / originWholeSum, 10))
    # print("a-e whole match", randomWholeSum, "/", originWholeSum, "=", round(randomWholeSum / originWholeSum, 10))
    # print("a-n whole match", noiseWholeSum, "/", originWholeSum, "=", round(noiseWholeSum / originWholeSum, 10))
    # print("times", times)

    print("adjCorrMean", [round(t1 / times, 5), round(t2 / times, 5), round(t3 / times, 5)])
    print("randomCorrMean", [round(r1 / times, 5), round(r2 / times, 5), round(r3 / times, 5)])
    csv.write(str(round(t1 / times, 5)) + ' ' + str(round(t2 / times, 5)) + ' ' + str(round(t3 / times, 5)) + ',' +
              str(round(r1 / times, 5)) + ' ' + str(round(r2 / times, 5)) + ' ' + str(round(r3 / times, 5)) + '\n')
