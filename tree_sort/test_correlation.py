import math
import sys
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pywt
from dtw import accelerated_dtw
from scipy.io import loadmat
from scipy.stats import pearsonr, spearmanr

from algorithm import smooth, genSample, insertNoise


def dtw_metric(data1, data2):
    distance = lambda x, y: np.abs(x - y)
    data1 = np.array(data1)
    data2 = np.array(data2)
    # return dtw(data1, data2, dist=distance)[0]
    return accelerated_dtw(data1, data2, dist=distance)[0]


# list 待排序的数组
# length 比较中所需的最小比较单元的长度
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
    cnts = 0
    while True:
        cnts += 1

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
                # tmp_metrics.append(sum(rel_entr(extend_list[l1l: l1r], noise_list[l1l: l1r])))
                # tmp_metrics.append(sum(rel_entr(extend_list[l2l: l2r], noise_list[l2l: l2r])))
                tmp_metrics.append(eval(metric + '_metric(extend_list[l1l: l1r], noise_list[l1l: l1r])'))
                tmp_metrics.append(eval(metric + '_metric(extend_list[l2l: l2r], noise_list[l2l: l2r])'))
            metrics.append(tmp_metrics)
            cur = cur + 2 ** i

            min_diff = sys.maxsize
            sort_metrics = tmp_metrics.copy()
            sort_metrics.sort()
            for j in range(len(sort_metrics) - 1):
                min_diff = min(min_diff, sort_metrics[j + 1] - sort_metrics[j])
            min_diffs = min(min_diffs, min_diff)

        if min_diffs < 1:
            # if min_diffs < 0.002:  # pearson
            np.random.shuffle(combine)
            extend_list, perm = zip(*combine)
        else:
            # print("\033[0;32;40mmin_diff", min_diffs, "\033[0m")
            break

        if cnts > 10:
            break

    for i in range(0, len(metrics)):
        tmp = np.argsort(metrics[i])
        code.append(tmp)

    return_code = []
    for i in range(1, len(intervals)):
        for j in range(len(code[i])):
            return_code.append(code[i][j])
        if intervals[i] < length:
            break

    return return_code, np.array(perm)


# list 待排序的数组
# length 比较中所需的最小比较单元的长度
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

    # 计算相关性
    corr = 0
    times = 0

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
            tmp_metrics.append(eval(metric + '_metric(extend_list[l1l: l1r], noise_list[l1l: l1r])'))
            tmp_metrics.append(eval(metric + '_metric(extend_list[l2l: l2r], noise_list[l2l: l2r])'))

            # 测试与noise的相关性
            p1 = pearsonr(extend_list[l1l: l1r], noise_list[l1l: l1r])[0]
            p2 = pearsonr(extend_list[l2l: l2r], noise_list[l2l: l2r])[0]
            p1 = 0 if math.isnan(p1) else p1
            p2 = 0 if math.isnan(p2) else p2
            corr = corr + p1 + p2
            times += 2

        # 选出相邻差距最小的点，若差距过小，则挑出来进行删除
        origin_metrics.append(tmp_metrics.copy())
        sort_metrics = tmp_metrics.copy()
        sort_metrics.sort()
        for j in range(len(sort_metrics) - 1):
            if sort_metrics[j + 1] - sort_metrics[j] < 1:
                for k in range(len(tmp_metrics)):
                    if tmp_metrics[k] == sort_metrics[j]:
                        blot.append([i - 1, k])

        blots.append(blot)
        # 从后往前删除，防止待删除的索引溢出
        for j in range(len(blot) - 1, -1, -1):
            blot.sort()  # 防止索引溢出
            drop = blot[j][1]
            del tmp_metrics[drop]
        metrics.append(tmp_metrics)
        cur = cur + 2 ** i

    for i in range(0, len(metrics)):
        tmp = np.argsort(metrics[i])
        code.append(tmp)

    return_code = []
    for i in range(1, len(intervals)):
        for j in range(len(code[i])):
            return_code.append(code[i][j])
        if intervals[i] < length:
            break

    return return_code, blots, corr / times


# list 待排序的数组
# length 比较中所需的最小比较单元的长度
def lossyLevelMetricSortOfA(data, length, noise, perm, blots, metric):
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

    # 计算相关性
    corr = 0
    times = 0

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
            tmp_metrics.append(eval(metric + '_metric(extend_list[l1l: l1r], noise_list[l1l: l1r])'))
            tmp_metrics.append(eval(metric + '_metric(extend_list[l2l: l2r], noise_list[l2l: l2r])'))

            # 测试与noise的相关性
            p1 = pearsonr(extend_list[l1l: l1r], noise_list[l1l: l1r])[0]
            p2 = pearsonr(extend_list[l2l: l2r], noise_list[l2l: l2r])[0]
            p1 = 0 if math.isnan(p1) else p1
            p2 = 0 if math.isnan(p2) else p2
            corr = corr + p1 + p2
            times += 2

        if blots[i - 1] != None and len(blots[i - 1]) != 0:
            # 从后往前删除，防止待删除的索引超出已删除的数据
            for j in range(len(blots[i - 1]) - 1, -1, -1):
                drop = blots[i - 1][j][1]
                del tmp_metrics[drop]
        metrics.append(tmp_metrics)
        cur = cur + 2 ** i

    for i in range(0, len(metrics)):
        tmp = np.argsort(metrics[i])
        code.append(tmp)

    return_code = []
    for i in range(1, len(intervals)):
        for j in range(len(code[i])):
            return_code.append(code[i][j])
        if intervals[i] < length:
            break

        return return_code, corr / times


csv = open("./corr.csv", "a+")

fileName = "../data/data_mobile_indoor_1_r.mat"
rawData = loadmat(fileName)

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

dataLen = len(CSIa1Orig)

print(pearsonr(CSIa1Orig, CSIb1Orig)[0])
print(spearmanr(CSIa1Orig, CSIb1Orig)[0])

# plt.figure()
# plt.subplot(2, 1, 1)
# plt.plot(CSIa1Orig[:200])

# wavelet = pywt.Wavelet('sym4')
# maxLevel = pywt.dwt_max_level(len(CSIa1Orig), wavelet.dec_len)
# threshold = 0.04
# # 小波分解
# coefficient_a = pywt.wavedec(CSIa1Orig, 'sym4', level=maxLevel)
# coefficient_b = pywt.wavedec(CSIb1Orig, 'sym4', level=maxLevel)
# # 噪声滤波
# for i in range(1, len(coefficient_a)):
#     coefficient_a[i] = pywt.threshold(coefficient_a[i], threshold * max(coefficient_a[i]))
#     coefficient_b[i] = pywt.threshold(coefficient_b[i], threshold * max(coefficient_b[i]))
# rec_a = pywt.waverec(coefficient_a, 'sym4')
# rec_b = pywt.waverec(coefficient_b, 'sym4')

# plt.subplot(2, 1, 2)
# plt.plot(reca[:200])
# plt.show()

# print(pearsonr(rec_a, rec_b)[0])
# print(spearmanr(rec_a, rec_b)[0])

CSIa1Orig = np.array(CSIa1Orig)
CSIb1Orig = np.array(CSIb1Orig)
CSIn1Orig = np.random.normal(loc=-100, scale=1000, size=dataLen)
CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)

CSIi1Orig = loadmat('../data/data_mobile_indoor_1_r.mat')['A'][:, 0]
np.random.shuffle(CSIi1Orig)

# entropyTime = time.time()
# CSIa1Orig, CSIb1Orig, CSIe1Orig = splitEntropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, 5, dataLen)
# print("--- entropyTime %s seconds ---" % (time.time() - entropyTime))

CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')
CSIb1Orig = smooth(CSIb1Orig, window_len=15, window='flat')
CSIe1Orig = smooth(CSIe1Orig, window_len=15, window="flat")
CSIn1Orig = smooth(CSIn1Orig, window_len=15, window="flat")
CSIi1Orig = smooth(CSIi1Orig, window_len=15, window="flat")

print(pearsonr(CSIa1Orig, CSIb1Orig)[0])
print(spearmanr(CSIa1Orig, CSIb1Orig)[0])

# rec_a = smooth(rec_a, window_len=15, window='flat')
# rec_b = smooth(rec_b, window_len=15, window='flat')

# print(pearsonr(rec_a, rec_b)[0])
# print(spearmanr(rec_a, rec_b)[0])

CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
CSIe1OrigBack = CSIe1Orig.copy()
CSIn1OrigBack = CSIn1Orig.copy()
CSIi1OrigBack = CSIi1Orig.copy()

intvl = 5
keyLen = 128
interval_length = 4
addNoise = False
insertRatio = 1
ratio = 1
metric = "dtw"

codings = ""
# for segLen in range(15, 1, -1):
for segLen in range(2, 11):
    print("segLen", segLen)
    corr_sum_ab = 0
    corr_sum_ae = 0
    corr_sum_an = 0

    inserted_corr_sum_ab = 0
    inserted_corr_sum_ae = 0
    inserted_corr_sum_an = 0

    segment_corr_sum_a_noise = 0
    segment_corr_sum_b_noise = 0
    segment_corr_sum_e_noise = 0
    segment_corr_sum_n_noise = 0

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

        # 去除直流分量
        # tmpCSIa1 = tmpCSIa1 - (np.mean(tmpCSIa1) - np.mean(tmpCSIb1))  # Mean value consistency
        tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
        tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
        # tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
        # tmpCSIi1 = tmpCSIi1 - np.mean(tmpCSIi1)
        # tmpNoise = tmpNoise - np.mean(tmpNoise)

        if addNoise:
            # tmpCSIa1 = tmpPulse * (np.float_power(np.abs(tmpCSIa1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIa1))
            # tmpCSIb1 = tmpPulse * (np.float_power(np.abs(tmpCSIb1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIb1))
            # tmpCSIe1 = tmpPulse * (np.float_power(np.abs(tmpCSIe1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIe1))
            # tmpCSIa1 = np.float_power(np.abs(tmpCSIa1), tmpNoise)
            # tmpCSIb1 = np.float_power(np.abs(tmpCSIb1), tmpNoise)
            # tmpCSIe1 = np.float_power(np.abs(tmpCSIe1), tmpNoise)
            tmpCSIa1 = tmpCSIa1 * tmpNoise
            tmpCSIb1 = tmpCSIb1 * tmpNoise
            tmpCSIe1 = tmpCSIe1 * tmpNoise
            # tmpCSIa1 = tmpCSIa1 + tmpNoise
            # tmpCSIb1 = tmpCSIb1 + tmpNoise
            # tmpCSIe1 = tmpCSIe1 + tmpNoise
            # tmpCSI = list(zip(tmpCSIa1, tmpCSIb1, tmpCSIe1))
            # np.random.shuffle(tmpCSI)
            # tmpCSIa1N, tmpCSIb1N, tmpCSIe1N = zip(*tmpCSI)
            # tmpCSIa1N, tmpCSIb1N, tmpCSIe1N = splitEntropyPerm(tmpCSIa1, tmpCSIb1, tmpCSIe1, 5, len(tmpCSIa1))
            # tmpCSIa1 = tmpCSIa1 + tmpCSIa1N
            # tmpCSIb1 = tmpCSIb1 + tmpCSIb1N
            # tmpCSIe1 = tmpCSIe1 + tmpCSIe1N
            # tmpCSIa1 = tmpCSIa1 * np.fft.fft(tmpCSIa1N)
            # tmpCSIb1 = tmpCSIb1 * np.fft.fft(tmpCSIb1N)
            # tmpCSIe1 = tmpCSIe1 * np.fft.fft(tmpCSIe1N)
        else:
            # tmpCSIa1 = tmpPulse * tmpCSIa1
            # tmpCSIb1 = tmpPulse * tmpCSIb1
            # tmpCSIe1 = tmpPulse * tmpCSIe1
            tmpCSIa1 = tmpCSIa1
            tmpCSIb1 = tmpCSIb1
            tmpCSIe1 = tmpCSIe1
            tmpNoise = tmpNoise

        CSIa1Orig[range(staInd, endInd, 1)] = tmpCSIa1
        CSIb1Orig[range(staInd, endInd, 1)] = tmpCSIb1
        CSIe1Orig[range(staInd, endInd, 1)] = tmpCSIe1
        CSIn1Orig[range(staInd, endInd, 1)] = tmpNoise
        CSIi1Orig[range(staInd, endInd, 1)] = tmpCSIi1

        permLen = len(range(staInd, endInd, intvl))
        origInd = np.array([xx for xx in range(staInd, endInd, intvl)])

        sortCSIa1 = np.zeros(permLen)
        sortCSIb1 = np.zeros(permLen)
        sortCSIe1 = np.zeros(permLen)
        sortNoise = np.zeros(permLen)
        sortCSIi1 = np.zeros(permLen)

        for ii in range(permLen):
            aIndVec = np.array([aa for aa in range(origInd[ii], origInd[ii] + intvl, 1)])  ## for non-permuted CSIa1

            for jj in range(permLen, permLen * 2):
                bIndVec = np.array([bb for bb in range(origInd[jj - permLen], origInd[jj - permLen] + intvl, 1)])

                CSIa1Tmp = CSIa1Orig[aIndVec]
                CSIb1Tmp = CSIb1Orig[bIndVec]
                CSIe1Tmp = CSIe1Orig[bIndVec]
                CSIn1Tmp = CSIn1Orig[aIndVec]
                CSIi1Tmp = CSIi1Orig[aIndVec]

                sortCSIa1[ii] = np.mean(CSIa1Tmp)
                sortCSIb1[jj - permLen] = np.mean(CSIb1Tmp)
                sortCSIe1[jj - permLen] = np.mean(CSIe1Tmp)
                sortNoise[ii - permLen] = np.mean(CSIn1Tmp)
                sortCSIi1[ii - permLen] = np.mean(CSIi1Tmp)

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

        sortCSIa1 = np.array(genSample(sortCSIa1Reshape, ratio))
        sortCSIb1 = np.array(genSample(sortCSIb1Reshape, ratio))
        sortCSIe1 = np.array(genSample(sortCSIe1Reshape, ratio))
        sortNoise = np.array(genSample(sortNoiseReshape, ratio))
        sortCSIi1 = np.array(genSample(sortCSIi1Reshape, ratio))

        corr_ab = pearsonr(sortCSIa1, sortCSIb1)[0]
        corr_ae = pearsonr(sortCSIa1, sortCSIe1)[0]
        corr_an = pearsonr(sortCSIa1, sortNoise)[0]
        # corr_ab = spearmanr(sortCSIa1, sortCSIb1)[0]
        # corr_ae = spearmanr(sortCSIa1, sortCSIe1)[0]
        # corr_an = spearmanr(sortCSIa1, sortNoise)[0]
        print("a-b", corr_ab)
        print("a-e", corr_ae)
        print("a-n", corr_an)
        corr_sum_ab += corr_ab
        corr_sum_ae += corr_ae
        corr_sum_an += corr_an

        # 这样插入噪音点和sortCSIa1数据过于接近，效果不好，不如在一个大范围内产生的噪音点好
        # sortCSIi1 = np.random.normal(loc=np.mean(sortCSIa1), scale=np.std(sortCSIa1, ddof=1), size=len(sortCSIa1))
        np.random.shuffle(sortCSIi1)
        insertIndex = np.random.permutation(len(sortCSIa1))
        sortCSIa1 = insertNoise(sortCSIa1, sortCSIi1, insertIndex, insertRatio)
        sortCSIb1 = insertNoise(sortCSIb1, sortCSIi1, insertIndex, insertRatio)
        sortCSIe1 = insertNoise(sortCSIe1, sortCSIi1, insertIndex, insertRatio)
        sortNoise = insertNoise(sortNoise, sortCSIi1, insertIndex, insertRatio)

        inserted_corr_ab = pearsonr(sortCSIa1, sortCSIb1)[0]
        inserted_corr_ae = pearsonr(sortCSIa1, sortCSIe1)[0]
        inserted_corr_an = pearsonr(sortCSIa1, sortNoise)[0]
        # inserted_corr_ab = spearmanr(sortCSIa1, sortCSIb1)[0]
        # inserted_corr_ae = spearmanr(sortCSIa1, sortCSIe1)[0]
        # inserted_corr_an = spearmanr(sortCSIa1, sortNoise)[0]
        print("after inserting")
        print("a-b", inserted_corr_ab)
        print("a-e", inserted_corr_ae)
        print("a-n", inserted_corr_an)
        inserted_corr_sum_ab += inserted_corr_ab
        inserted_corr_sum_ae += inserted_corr_ae
        inserted_corr_sum_an += inserted_corr_an

        sortCSIa1 = np.array(sortCSIa1)
        sortCSIb1 = np.array(sortCSIb1)
        sortCSIe1 = np.array(sortCSIe1)
        sortNoise = np.array(sortNoise)

        # _, perm = levelMetricSortPerm(list(sortCSIa1), interval_length, list(sortNoise), metric)
        # _, blots, segment_corr_b_noise = lossyLevelMetricSortOfB(list(sortCSIb1), interval_length, list(sortNoise),
        #                                                          perm, metric)
        # _, segment_corr_a_noise = lossyLevelMetricSortOfA(list(sortCSIa1), interval_length, list(sortNoise), perm,
        #                                                   blots, metric)
        # _, segment_corr_e_noise = lossyLevelMetricSortOfA(list(sortCSIe1), interval_length, list(sortNoise), perm,
        #                                                   blots, metric)
        # _, segment_corr_n_noise = lossyLevelMetricSortOfA(list(sortNoise), interval_length, list(sortNoise), perm,
        #                                                   blots, metric)
        #
        # segment_corr_sum_a_noise += segment_corr_a_noise
        # segment_corr_sum_b_noise += segment_corr_b_noise
        # segment_corr_sum_e_noise += segment_corr_e_noise
        # segment_corr_sum_n_noise += segment_corr_n_noise

    print("\033[0;31;40ma-b", corr_sum_ab / times, "\033[0m")
    print("\033[0;31;40ma-e", corr_sum_ae / times, "\033[0m")
    print("\033[0;31;40ma-n", corr_sum_an / times, "\033[0m")
    print("after inserting")
    print("\033[0;31;40ma-b", inserted_corr_sum_ab / times, "\033[0m")
    print("\033[0;31;40ma-e", inserted_corr_sum_ae / times, "\033[0m")
    print("\033[0;31;40ma-n", inserted_corr_sum_an / times, "\033[0m")
    # print("segment")
    # print("\033[0;31;40ma", segment_corr_sum_a_noise / times, "\033[0m")
    # print("\033[0;31;40mb", segment_corr_sum_b_noise / times, "\033[0m")
    # print("\033[0;31;40me", segment_corr_sum_e_noise / times, "\033[0m")
    # print("\033[0;31;40mn", segment_corr_sum_n_noise / times, "\033[0m")

    csv.write(str(round(corr_sum_ab / times, 10)) + ',' + str(round(corr_sum_ae / times, 10)) + ',' +
              str(round(corr_sum_an / times, 10)) + ',' + str(round(inserted_corr_sum_ab / times, 10)) + ',' +
              str(round(inserted_corr_sum_ae / times, 10)) + ',' +
              str(round(inserted_corr_sum_an / times, 10)) + ',' + '\n')

    # csv.write(str(round(segment_corr_sum_a_noise / times, 10)) + ',' +
    #           str(round(segment_corr_sum_b_noise / times, 10)) + ',' +
    #           str(round(segment_corr_sum_e_noise / times, 10)) + ',' +
    #           str(round(segment_corr_sum_n_noise / times, 10)) + ',' + '\n')
