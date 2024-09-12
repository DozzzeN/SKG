import hashlib
import itertools
import math
import random
import sys
import time
from collections import Counter
from datetime import datetime

import graycode
import scipy.stats
from dtaidistance import dtw_ndim

from segmentation.methods import find_opt_segment_method, segment_sequence, compute_min_dtw, \
    find_sub_opt_segment_method_down, find_sub_opt_segment_method_up, find_opt_segment_method_from_candidate, \
    search_segment_method_with_matching, find_segments, find_all_cover_intervals_iter_up, get_segment_lengths, \
    find_all_cover_intervals_iter, search_segment_method_with_min_match_dtw, find_special_array, step_discrete_corr, \
    find_peaks_in_segments, search_segment_method, search_segment_method_with_offset, find_min_threshold, find_offset, \
    dtw_metric, find_plateau_length, generate_segment_lengths, normal2uniform, smooth, compute_threshold, \
    diff_sq_integral_rough, find_opt_segment_method_cond, search_segment_method_with_min_cross_dtw, \
    find_special_array_min_mean_var, search_segment_method_with_min_dtw, find_special_array_min_min_mean_var, \
    search_index_with_segment, find_special_array_mean_var, find_sub_opt_segment_method_down_cond, \
    find_special_array_min_mean, find_sub_opt_segment_method_merge, find_sub_opt_segment_method_merge_heap_fast, \
    find_gaps, merge_gaps, find_covering_intervals, find_sub_opt_segment_method_sliding_threshold, \
    find_sub_opt_segment_method_sliding, find_out_of_range_intervals, find_before_intervals, compute_all_dtw, \
    common_pca, find_sub_opt_shuffle_method_sliding_threshold, find_best_matching_pair, \
    find_best_matching_pair_filter, find_best_matching_pair_shuffled, find_sub_opt_shuffle_method_sliding, \
    find_best_matching_pair_threshold, compute_min_euclidean, find_best_matching_pair_threshold_filter

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr, boxcox, spearmanr, kurtosis, skew

from segmentation.test_partition import partition


# 将当前代码和结果写入文件
# writeToFile()

# fileName = ["../data/data_mobile_indoor_1.mat",
#             "../data/data_mobile_outdoor_1.mat",
#             "../data/data_static_outdoor_1.mat",
#             "../data/data_static_indoor_1.mat"
#             ]

# 基于自适应分段索引匹配的密钥生成
fileName = ["../data/data_mobile_indoor_1.mat"]

# 统计时间

isPrint = False
isUp = False
isCoxBox = False
isCorrect = True
isDegrade = False
withIndexValue = False
withProb = False
withNewMatrix = False

isShuffle = False
isSegSort = False
isValueShuffle = False

# 是否纠错
rec = False

segLen = 4
segNum = 4
keyLen = segNum * segLen
tell = True

# 是否排序
# withoutSorts = [True, False]
withoutSorts = [False]
# 是否添加噪声
# addNoises = ["pca", "mul", "add", ""]
addNoises = ["mul"]
segment_option = ""
print(segment_option)

for f in fileName:
    for addNoise in addNoises:
        for withoutSort in withoutSorts:
            print(f)
            rawData = loadmat(f)

            if f.find("data_alignment") != -1:
                CSIa1Orig = rawData['csi'][:, 0]
                CSIb1Orig = rawData['csi'][:, 1]
            elif f.find("csi") != -1:
                CSIa1Orig = rawData['testdata'][:, 0]
                CSIb1Orig = rawData['testdata'][:, 1]
            else:
                CSIa1Orig = rawData['A'][:, 0]
                CSIb1Orig = rawData['A'][:, 1]

            if f == "../data/data_NLOS.mat":
                # 先整体shuffle一次
                seed = np.random.randint(100000)
                np.random.seed(seed)
                shuffleInd = np.random.permutation(len(CSIa1Orig))
                CSIa1Orig = CSIa1Orig[shuffleInd]
                CSIb1Orig = CSIb1Orig[shuffleInd]

            # 为了使用cox-box处理，将滤波放在总循环前面，结果与old版本的有略微的不一致
            CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
            CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')

            if isCoxBox:
                CSIa1Orig = (CSIa1Orig - np.min(CSIa1Orig)) / (np.max(CSIa1Orig) - np.min(CSIa1Orig))
                CSIb1Orig = (CSIb1Orig - np.min(CSIb1Orig)) / (np.max(CSIb1Orig) - np.min(CSIb1Orig))
                CSIa1Orig = scipy.stats.boxcox(np.abs(CSIa1Orig) + 1e-4)[0]
                CSIb1Orig = scipy.stats.boxcox(np.abs(CSIb1Orig) + 1e-4)[0]
                CSIa1Orig = normal2uniform(CSIa1Orig) * 2
                CSIb1Orig = normal2uniform(CSIb1Orig) * 2
                CSIa1Orig = np.array(CSIa1Orig)
                CSIb1Orig = np.array(CSIb1Orig)

            dataLen = len(CSIa1Orig)
            print("dataLen", dataLen)

            print("segLen", segLen)
            print("keyLen", keyLen / segLen)

            indices = list(itertools.permutations(range(4)))

            originSum = 0
            correctSum = 0

            originDecSum = 0
            correctDecSum = 0

            originWholeSum = 0
            correctWholeSum = 0

            originSegSum = 0
            correctSegSum = 0

            segmentMaxDist = []
            evenMaxDist = []
            badSegments = 0

            times = 0
            overhead = 0

            shuffling_time = []
            search_time = []

            if withoutSort:
                if addNoise != "":
                    print("no sorting")
                elif addNoise == "":
                    print("no sorting and no perturbation")
            else:
                if addNoise == "":
                    print("no perturbation")
                elif addNoise == "mul":
                    print("normal mul")
                elif addNoise == "add":
                    print("normal add")
            if isShuffle:
                print("with shuffle")
            if isSegSort:
                print("with segSort")

            dataLenLoop = dataLen
            keyLenLoop = keyLen

            isRunBack = False
            runBackCounts = 0

            shuffle_patterns = []
            if f == "../data/data_static_indoor_1.mat":
                dataLenLoop = int(dataLen / 5.5)
                keyLenLoop = int(keyLen / 5)
            staInd = -keyLenLoop
            while staInd < dataLenLoop:
                staInd += keyLenLoop
                start = time.time_ns()
                keyLen = segNum * segLen
                endInd = staInd + keyLen
                if endInd >= len(CSIa1Orig):
                    break

                # 如果有错误，则退化成原始分段
                findError = False

                tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
                tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]

                if isRunBack:
                    # 乘随机矩阵
                    np.random.seed(staInd + runBackCounts)
                    randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
                    tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)

                    runBackCounts += 1
                else:
                    runBackCounts = 0

                # 目的是把加噪音+无排序的结果降下来
                if addNoise == "mul" and isRunBack is False:
                    np.random.seed(10000)
                    randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
                    # randomMatrix = np.random.uniform(0, 1, size=(keyLen, keyLen))
                    # randomMatrix = np.random.normal(0, 1, size=(keyLen, keyLen))
                    # 均值化
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
                    tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)
                else:
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)


                # 寻找最优噪声矩阵
                if withNewMatrix:
                    resCSIa1 = [0] * int(keyLen / 2) + [1] * int(keyLen / 2)
                    np.random.seed(staInd)
                    resCSIa1 = np.random.permutation(resCSIa1)

                    # 无影响
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)

                    randomMatrix = np.linalg.lstsq(np.array(tmpCSIa1).reshape(1, -1), np.array(resCSIa1).reshape(1, -1), rcond=None)[0]
                    randomMatrix = randomMatrix - np.mean(randomMatrix)
                    randomMatrix = (randomMatrix - np.min(randomMatrix)) / (np.max(randomMatrix) - np.min(randomMatrix))
                    np.random.seed(staInd)
                    randomMatrix += np.random.uniform(0, 1, (keyLen, keyLen))
                    # randomMatrix += np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
                    tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
                    tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)

                    # 无影响
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)

                # 最后各自的密钥
                a_list = []
                b_list = []

                # without sorting
                if withIndexValue:
                    # 将测量值和其索引结合成二维数组
                    tmpCSIa1Index = np.array(tmpCSIa1).argsort().argsort()
                    tmpCSIb1Index = np.array(tmpCSIb1).argsort().argsort()

                    # 将index和value放缩到同样的区间内
                    tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1))
                    tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1))
                    tmpCSIa1Index = (tmpCSIa1Index - np.min(tmpCSIa1Index)) / (np.max(tmpCSIa1Index) - np.min(tmpCSIa1Index))
                    tmpCSIb1Index = (tmpCSIb1Index - np.min(tmpCSIb1Index)) / (np.max(tmpCSIb1Index) - np.min(tmpCSIb1Index))

                    tmpCSIa1Ind = np.array(list(zip(tmpCSIa1, tmpCSIa1Index)))
                    tmpCSIb1Ind = np.array(list(zip(tmpCSIb1, tmpCSIb1Index)))
                    tmpCSIa1Prod = tmpCSIa1 * tmpCSIa1Index
                    tmpCSIb1Prod = tmpCSIb1 * tmpCSIb1Index
                else:
                    if withoutSort:
                        tmpCSIa1Ind = np.array(tmpCSIa1)
                        tmpCSIb1Ind = np.array(tmpCSIb1)
                    else:
                        tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
                        tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()

                minEpiIndClosenessLsb = np.zeros(int(keyLen / segLen), dtype=int)
                # with segSort
                if isSegSort:
                    if withoutSort is False:
                        for i in range(int(keyLen / segLen)):
                            epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]
                            epiIndb1 = tmpCSIb1Ind[i * segLen:(i + 1) * segLen]

                            np.random.seed(i)
                            combineEpiIndx1 = list(zip(epiInda1, epiIndb1))
                            np.random.shuffle(combineEpiIndx1)
                            epiInda1, epiIndb1 = zip(*combineEpiIndx1)

                            tmpCSIa1Ind[i * segLen:(i + 1) * segLen] = epiInda1
                            tmpCSIb1Ind[i * segLen:(i + 1) * segLen] = epiIndb1
                        # print(pearsonr(tmpCSIa1Ind, tmpCSIb1Ind)[0])

                if segment_option == "find_search":
                    start_time = time.time_ns()
                    shuffling_start = time.time_ns()

                    # 滑动窗口分段
                    if withIndexValue:
                        threshold = 1
                    else:
                        if withoutSort:
                            threshold = 0.2
                        else:
                            threshold = 15
                    # shuffle_method_ori = find_sub_opt_shuffle_method_sliding_threshold(
                    #     tmpCSIa1Ind, tmpCSIa1Ind, 4, indices, threshold)
                    shuffle_method_ori = find_sub_opt_shuffle_method_sliding(tmpCSIa1Ind, tmpCSIa1Ind, 4, indices)
                    # shuffle_method_ori = find_all_sub_opt_shuffle_method_sliding(tmpCSIa1Ind, tmpCSIa1Ind, 4, indices)
                    shuffling_end = time.time_ns()
                    shuffling_time.append((shuffling_end - shuffling_start) / 1e9)
                    for index in shuffle_method_ori:
                        shuffle_patterns.append("".join(np.array(index).astype(str)))

                    if isRunBack is False:
                        print("staInd", staInd, "shuffle_method", shuffle_method_ori)

                    if withIndexValue:
                        tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIa1Ind))
                        tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIb1Ind))
                    else:
                        tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)
                        tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(int(keyLen / segLen), segLen)
                    tmpCSIa1IndReshape = [list(tmpCSIa1IndReshape[i][list(shuffle_method_ori[i])]) for i in
                                          range(len(tmpCSIa1IndReshape))]

                    # tmpCSIa1EvenSegment = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIa1Ind))
                    # dist_opt_seg = compute_min_dtw(tmpCSIa1IndReshape, tmpCSIa1IndReshape)
                    # dist_even_seg = compute_min_dtw(tmpCSIa1EvenSegment, tmpCSIa1EvenSegment)
                    # if (dist_opt_seg < dist_even_seg):
                    #     badSegments += 1
                    #     tmpCSIa1IndReshape = tmpCSIa1EvenSegment
                    #     segment_method_ori = [segLen] * int(keyLen / segLen)
                    #     min_length = min(segment_method_ori)
                    #     max_length = max(segment_method_ori)
                    #     num_segments = len(segment_method_ori)
                    #     measurements_len = sum(segment_method_ori)
                    # segmentMaxDist.append(dist_opt_seg)
                    # evenMaxDist.append(dist_even_seg)

                elif segment_option == "sub_window":
                    start_time = time.time_ns()
                    if withIndexValue:
                        threshold = 2
                    else:
                        if withoutSort:
                            threshold = 0.2
                        else:
                            threshold = 15
                    shuffle_method = find_sub_opt_shuffle_method_sliding_threshold(
                        tmpCSIa1Ind, tmpCSIa1Ind, 4, indices, threshold)
                    print("staInd", staInd, "shuffle_method", shuffle_method)

                    if withIndexValue:
                        tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIa1Ind))
                        tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIb1Ind))
                    else:
                        tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)
                        tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(int(keyLen / segLen), segLen)

                    tmpCSIa1IndReshape = [list(tmpCSIa1IndReshape[i][list(shuffle_method[i])]) for i in
                                          range(len(tmpCSIa1IndReshape))]
                    tmpCSIb1IndReshape = [list(tmpCSIb1IndReshape[i][list(shuffle_method[i])]) for i in
                                          range(len(tmpCSIb1IndReshape))]
                elif segment_option == "sub_window_ind":
                    start_time = time.time_ns()
                    if withIndexValue:
                        threshold = 2
                    else:
                        if withoutSort:
                            threshold = 0.2
                        else:
                            threshold = 15
                    shuffle_method_A = find_sub_opt_shuffle_method_sliding_threshold(
                        tmpCSIa1Ind, tmpCSIa1Ind, 4, indices, threshold)
                    shuffle_method_B = find_sub_opt_shuffle_method_sliding_threshold(
                        tmpCSIb1Ind, tmpCSIb1Ind, 4, indices, threshold)
                    print("staInd", staInd, "shuffle_method", shuffle_method_A, shuffle_method_B)

                    tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)
                    tmpCSIa1IndReshape = [list(tmpCSIa1IndReshape[i][list(shuffle_method_A[i])]) for i in
                                          range(len(tmpCSIa1IndReshape))]
                    tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(int(keyLen / segLen), segLen)
                    tmpCSIb1IndReshape = [list(tmpCSIb1IndReshape[i][list(shuffle_method_B[i])]) for i in
                                          range(len(tmpCSIb1IndReshape))]
                else:
                    start_time = time.time_ns()
                    # 原方法：固定分段
                    tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)
                    tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(int(keyLen / segLen), segLen)

                tmpCSIa1Bck = tmpCSIa1Ind.copy()
                permutation = list(range(int(keyLen / segLen)))
                combineMetric = list(zip(tmpCSIa1IndReshape, permutation))
                np.random.seed(staInd)
                np.random.shuffle(combineMetric)
                tmpCSIa1IndReshape, permutation = zip(*combineMetric)

                if segment_option == "find_search":
                    search_start = time.time_ns()
                    tmpCSIa1BckReshape = np.array(tmpCSIa1Bck).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIa1Ind))
                    threshold = compute_min_euclidean(tmpCSIa1BckReshape, tmpCSIb1IndReshape)

                    # indices_a, shuffle_methods_a = find_best_matching_pair(tmpCSIa1IndReshape, tmpCSIa1BckReshape)
                    # indices_b, shuffle_methods_b = find_best_matching_pair(tmpCSIa1IndReshape, tmpCSIb1IndReshape)
                    # indices_a, shuffle_methods_a = find_best_matching_pair_filter(tmpCSIa1IndReshape, tmpCSIa1BckReshape)
                    # indices_b, shuffle_methods_b = find_best_matching_pair_filter(tmpCSIa1IndReshape, tmpCSIb1IndReshape)
                    # indices_a, shuffle_methods_a = find_best_matching_pair_threshold(tmpCSIa1IndReshape, tmpCSIa1BckReshape, threshold)
                    # indices_b, shuffle_methods_b = find_best_matching_pair_threshold(tmpCSIa1IndReshape, tmpCSIb1IndReshape, threshold)
                    indices_a, shuffle_methods_a = find_best_matching_pair_threshold_filter(tmpCSIa1IndReshape, tmpCSIa1BckReshape, threshold)
                    indices_b, shuffle_methods_b = find_best_matching_pair_threshold_filter(tmpCSIa1IndReshape, tmpCSIb1IndReshape, threshold)

                    print("shuffle_methods_a", shuffle_methods_a)
                    print("shuffle_methods_b", shuffle_methods_b)

                    search_end = time.time_ns()
                    search_time.append((search_end - search_start) / 1e9)

                    indices_a = [index[1] for index in indices_a]
                    indices_b = [index[1] for index in indices_b]

                    a_list_number = list(indices_a)
                    b_list_number = list(indices_b)

                    if isPrint:
                        if a_list_number != b_list_number:
                            print("staInd", staInd, "error")
                        else:
                            print("staInd", staInd)
                        print("a", a_list_number, "segment_method", shuffle_method_ori)
                        print("b", b_list_number, "segment_method", shuffle_methods_b)
                        print()

                    # correction
                    key_hash = hash(tuple(a_list_number))
                    if key_hash != hash(tuple(b_list_number)):
                        indices_b_filter, shuffle_methods_b = find_best_matching_pair_filter(tmpCSIa1IndReshape, tmpCSIb1IndReshape)
                        indices_b_filter = [index[1] for index in indices_b_filter]
                        b_list_number = list(indices_b_filter)
                        # ind
                        if key_hash != hash(tuple(b_list_number)):
                            if withIndexValue:
                                threshold = 1
                            else:
                                if withoutSort:
                                    threshold = 0.2
                                else:
                                    threshold = 15
                            shuffle_methods_b = find_sub_opt_shuffle_method_sliding_threshold(
                                tmpCSIb1Ind, tmpCSIb1Ind, 4, indices, threshold)
                            tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIb1Ind))
                            tmpCSIb1IndReshape = [list(tmpCSIb1IndReshape[i][list(shuffle_methods_b[i])]) for i in
                                                  range(len(tmpCSIb1IndReshape))]
                            indices_b_ind = find_best_matching_pair_shuffled(tmpCSIa1IndReshape, tmpCSIb1IndReshape)
                            b_list_number = list(indices_b_ind)
                        if isPrint:
                            print("correction")
                            if a_list_number != b_list_number:
                                print("staInd", staInd, "error again")
                            else:
                                print("staInd", staInd)
                            print("a", a_list_number, "segment_method", shuffle_method_ori)
                            print("b", b_list_number, "segment_method", shuffle_methods_b)
                            print()

                    # 回退多次
                    if a_list_number != b_list_number and runBackCounts < 5:
                        if isPrint:
                            print("回退" + str(runBackCounts) + "次")
                        isRunBack = True
                        staInd -= keyLenLoop
                        continue
                    elif a_list_number != b_list_number and isRunBack is True:
                        runBackCounts = 0
                        isRunBack = False
                    else:
                        runBackCounts = 0
                        isRunBack = False

                    if a_list_number == b_list_number:
                        runBackCounts = 0

                else:
                    for i in range(int(keyLen / segLen)):
                        epiInda1 = tmpCSIa1IndReshape[i]

                        epiIndClosenessLsb = np.zeros(int(keyLen / segLen))

                        for j in range(int(keyLen / segLen)):
                            epiIndb1 = tmpCSIb1IndReshape[j]

                            # 欧式距离度量更好
                            epiIndClosenessLsb[j] = sum(np.square(epiIndb1 - np.array(epiInda1)))
                            # epiIndClosenessLsb[j] = dtw_metric(epiIndb1, np.array(epiInda1))
                            # epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))
                            # epiIndClosenessLsb[j] = distance.cosine(epiIndb1, np.array(epiInda1))
                            # epiIndClosenessLsb[j] = abs(sum(epiIndb1) - sum(np.array(epiInda1)))

                        minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)

                    a_list_number = list(permutation)
                    b_list_number = list(minEpiIndClosenessLsb)

                # 转成二进制
                max_digit = str(math.ceil(np.log2(len(a_list_number))))
                for i in range(len(a_list_number)):
                    a_list += str('{:0' + max_digit + 'b}').format(graycode.tc_to_gray_code(a_list_number[i]))
                for i in range(len(b_list_number)):
                    b_list += str('{:0' + max_digit + 'b}').format(graycode.tc_to_gray_code(b_list_number[i]))

                # for i in range(len(a_list_number)):
                #     number = bin(a_list_number[i])[2:].zfill(int(np.log2(len(a_list_number))))
                #     a_list += number
                # for i in range(len(b_list_number)):
                #     number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
                #     b_list += number

                sum1 = min(len(a_list), len(b_list))
                sum2 = 0
                sum3 = 0

                for i in range(0, sum1):
                    sum2 += (a_list[i] == b_list[i])

                end = time.time_ns()
                overhead += end - start
                # 从分段开始记时

                # 自适应纠错
                if sum1 != sum2 and rec:
                    if tell:
                        # print("correction")
                        # a告诉b哪些位置出错，b对其纠错
                        for i in range(len(a_list_number)):
                            if a_list_number[i] != b_list_number[i]:
                                epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]

                                epiIndClosenessLsb = np.zeros(int(keyLen / segLen))

                                for j in range(int(keyLen / segLen)):
                                    epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
                                    epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))

                                # 第一个找到的错误的，将其距离置为最大，下次找到的就是第二个，作为正确结果
                                min_b = np.argmin(epiIndClosenessLsb)
                                epiIndClosenessLsb[min_b] = keyLen * segLen
                                b_list_number[i] = np.argmin(epiIndClosenessLsb)

                                b_list = []

                                max_digit = str(math.ceil(np.log2(len(b_list_number))))
                                for i in range(len(b_list_number)):
                                    b_list += str('{:0' + max_digit + 'b}').format(
                                        graycode.tc_to_gray_code(b_list_number[i]))

                                # for i in range(len(b_list_number)):
                                #     number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
                                #     b_list += number

                                sum2 = 0
                                for i in range(0, min(len(a_list), len(b_list))):
                                    sum2 += (a_list[i] == b_list[i])

                # print("key", len(a_list), a_list)

                # print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
                originSum += sum1
                correctSum += sum2

                originWholeSum += 1
                correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum

                times += 1

            print("\033[0;34;40ma-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 9), "\033[0m")
            print("\033[0;34;40ma-b whole match", correctWholeSum, "/", originWholeSum, "=",
                  round(correctWholeSum / originWholeSum, 9), "\033[0m")

            print(round(correctSum / originSum, 9), round(correctWholeSum / originWholeSum, 9),
                  round(originSum / times / keyLen, 9),
                  round(correctSum / times / keyLen, 9))
            # 分段匹配的情况
            if originDecSum != 0:
                print("\033[0;32;40ma-b seg", correctSegSum, "/", originSegSum, "=",
                      round(correctSegSum / originSegSum, 9), "\033[0m")
            if segmentMaxDist != []:
                print("segmented max distance", np.mean(segmentMaxDist), np.std(segmentMaxDist))
                print("even max distance", np.mean(evenMaxDist), np.std(evenMaxDist))
                print("badSegments", badSegments / times)
            if withoutSort:
                print("withoutSort")
            else:
                print("withSort")
            if withIndexValue:
                print("withIndexValue")
            else:
                print("withoutIndexValue")
            if len(shuffling_time) != 0:
                print("segment time", "total search time")
                # 求AB进行搜索的平均时间
                print(np.round(np.mean(shuffling_time), 9),
                      np.round(np.mean(search_time) / 2, 9))

                print("(" + str(correctSum), "/", originSum, "=", str(round(correctSum / originSum, 9)) + ")",
                      round(correctSum / originSum, 9), round(correctWholeSum / originWholeSum, 9),
                      round(originSum / times / keyLen, 9),
                      round(correctSum / times / keyLen, 9), " / ",
                      np.round(np.mean(shuffling_time), 9),
                      np.round(np.mean(search_time) / 2, 9))
            print("shuffle_patterns", Counter(shuffle_patterns))
            print("\n")
