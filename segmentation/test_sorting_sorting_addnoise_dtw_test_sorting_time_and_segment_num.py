import hashlib
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
    common_pca

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr, boxcox, spearmanr, kurtosis, skew

from segmentation.test_partition import partition, partition_with_occupy, partition_with_occupy_dp_lru

# 将当前代码和结果写入文件
# writeToFile()

# fileName = ["../data/data_mobile_indoor_1.mat",
#             "../data/data_mobile_outdoor_1.mat",
#             "../data/data_static_outdoor_1.mat",
#             "../data/data_static_indoor_1.mat"
#             ]

# 基于自适应分段索引匹配的密钥生成
fileName = ["../data/data_mobile_indoor_1.mat"]

# 统计运行时间，和分段类型

isPrint = False
isUp = False
isCoxBox = False
isCorrect = True
isDegrade = False
withIndexValue = False
withProb = False

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

            segment_time = []
            search_time = []
            final_search_time = []

            if withoutSort:
                if addNoise != "":
                    print("no sorting")
            if withoutSort:
                if addNoise == "":
                    print("no sorting and no perturbation")
            if withoutSort is False:
                if addNoise == "":
                    print("no perturbation")
                if addNoise == "mul":
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

            segment_lengths = []
            real_segment_lengths = []
            segment_with_occupy_lengths = []

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
                e_list = []

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
                elif withProb:
                    tmpCSIa1Index = np.array(tmpCSIa1).argsort().argsort()
                    tmpCSIb1Index = np.array(tmpCSIb1).argsort().argsort()

                    tmpCSIa1Ind = tmpCSIa1 * tmpCSIa1Index
                    tmpCSIb1Ind = tmpCSIb1 * tmpCSIb1Index
                else:
                    if withoutSort:
                        tmpCSIa1Ind = np.array(tmpCSIa1)
                        tmpCSIb1Ind = np.array(tmpCSIb1)
                    else:
                        tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
                        tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()

                minEpiIndClosenessLsb = np.zeros(int(keyLen / segLen), dtype=int)

                if segment_option == "find_search":
                    start_time = time.time_ns()
                    segment_start = time.time_ns()

                    # 滑动窗口分段
                    if withIndexValue:
                        # segment_method_ori = find_sub_opt_segment_method_sliding(
                        #     tmpCSIa1Ind, tmpCSIa1Ind, 3, 5)
                        segment_method_ori = find_sub_opt_segment_method_sliding_threshold(
                           tmpCSIa1Ind, tmpCSIa1Ind, 3, 5, 0.005)
                    else:
                        if withoutSort is True:
                            segment_method_ori = find_sub_opt_segment_method_sliding(
                                tmpCSIa1Ind, tmpCSIa1Ind, 3, 5)
                            # segment_method_ori = find_sub_opt_segment_method_sliding_threshold(
                            #     tmpCSIa1Ind, tmpCSIa1Ind, 3, 5, 0.005)
                        else:
                            segment_method_ori = find_sub_opt_segment_method_sliding_threshold(
                                tmpCSIa1Ind, tmpCSIa1Ind, 3, 5, 60)
                    segment_end = time.time_ns()

                    if min(segment_method_ori) < 3:
                        continue

                    min_length = min(segment_method_ori)
                    max_length = max(segment_method_ori)
                    num_segments = len(segment_method_ori)
                    measurements_len = sum(segment_method_ori)

                    segment_lengths.append(num_segments)

                    tmpCSIa1IndReshape = segment_sequence(tmpCSIa1Ind, segment_method_ori)

                    tmpCSIa1EvenSegment = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIa1Ind))
                    dist_opt_seg = compute_min_dtw(tmpCSIa1IndReshape, tmpCSIa1IndReshape)
                    dist_even_seg = compute_min_dtw(tmpCSIa1EvenSegment, tmpCSIa1EvenSegment)
                    if (dist_opt_seg < dist_even_seg):
                        badSegments += 1
                        tmpCSIa1IndReshape = tmpCSIa1EvenSegment
                        segment_method_ori = [segLen] * int(keyLen / segLen)
                        min_length = min(segment_method_ori)
                        max_length = max(segment_method_ori)
                        num_segments = len(segment_method_ori)
                        measurements_len = sum(segment_method_ori)
                    segmentMaxDist.append(dist_opt_seg)
                    evenMaxDist.append(dist_even_seg)
                else:
                    start_time = time.time_ns()
                    # 原方法：固定分段
                    tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)
                    tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(int(keyLen / segLen), segLen)

                tmpCSIa1Bck = tmpCSIa1Ind.copy()
                if segment_option != "find_search":
                    # 如果还以此分段进行划分,则超过此分段的索引会被截掉
                    permutation = list(range(int(keyLen / segLen)))
                else:
                    permutation = list(range(num_segments))
                combineMetric = list(zip(tmpCSIa1IndReshape, permutation))
                np.random.seed(staInd)
                np.random.shuffle(combineMetric)
                tmpCSIa1IndReshape, permutation = zip(*combineMetric)
                if withIndexValue:
                    tmpCSIa1Ind = np.vstack(tmpCSIa1IndReshape)
                else:
                    tmpCSIa1Ind = np.hstack((tmpCSIa1IndReshape))

                if segment_option == "find_search":
                    if findError is False:
                        search_start = time.time_ns()
                        # 计算对应位置最大差距作为阈值
                        threshold = compute_threshold(tmpCSIa1Bck, tmpCSIb1Ind)
                        if withIndexValue:
                            base_threshold = threshold
                            threshold /= 4
                        else:
                            if withoutSort is True:
                                base_threshold = threshold
                                threshold /= 4
                            else:
                                threshold = 3
                        # 在阈值内匹配相近的索引
                        if withIndexValue:
                            all_segments_A = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIa1Bck, threshold)
                            all_segments_B = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIb1Ind, threshold)
                        else:
                            if withoutSort is True:
                                all_segments_A = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIa1Bck, threshold)
                                all_segments_B = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIb1Ind, threshold)
                            else:
                                all_segments_A = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIa1Bck, threshold + 1)
                                all_segments_B = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIb1Ind, threshold + 1)
                        # 根据相近的索引组合成若干个子分段
                        if isUp:
                            segments_A = find_segments(all_segments_A, 2, keyLen)
                            segments_B = find_segments(all_segments_B, 2, keyLen)
                        else:
                            if withIndexValue:
                                # 有时候找到的分段总个数不等于所使用的测量值个数，故需要重新规定总长度进行搜索
                                segments_A = find_segments(all_segments_A, 3, measurements_len)
                                segments_B = find_segments(all_segments_B, 3, measurements_len)

                                while len(segments_B) < int(len(segments_A) / 2) and threshold < base_threshold:
                                    threshold += base_threshold / 8
                                    all_segments_B = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIb1Ind,
                                                                                         threshold)
                                    segments_B = find_segments(all_segments_B, 3, measurements_len)
                            else:
                                if withoutSort is True:
                                    # 有时候找到的分段总个数不等于所使用的测量值个数，故需要重新规定总长度进行搜索
                                    segments_A = find_segments(all_segments_A, 3, measurements_len)
                                    segments_B = find_segments(all_segments_B, 3, measurements_len)

                                    while len(segments_B) < int(len(segments_A) / 2) and threshold < base_threshold:
                                        threshold += base_threshold / 8
                                        all_segments_B = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIb1Ind,
                                                                                             threshold)
                                        segments_B = find_segments(all_segments_B, 3, measurements_len)
                                else:
                                    segments_A = find_segments(all_segments_A, 3, keyLen)
                                    segments_B = find_segments(all_segments_B, 3, keyLen)

                        if len(segments_A) == 0:
                            if isDegrade:
                                findError = True
                            else:
                                continue

                        if len(segments_B) == 0:
                            segments_B = segments_A

                        if withIndexValue:
                            gaps = find_gaps(segments_B, [0, measurements_len])
                        else:
                            if withoutSort is True:
                                gaps = find_gaps(segments_B, [0, measurements_len])
                            else:
                                gaps = find_gaps(segments_B, [0, keyLen])
                        if len(gaps) != 0:
                            segments_B_bck = segments_B.copy()

                            add_intervals = find_covering_intervals(segments_A, gaps)
                            if len(add_intervals) == 0:
                                if isDegrade:
                                    findError = True
                                else:
                                    continue
                            segments_B.extend(add_intervals)
                            segments_B.sort(key=lambda x: x[0])

                            # 考虑间隙的具体位置，再计算可能的分段数，速度过慢
                            # segments_B_with_occupy = partition_with_occupy_dp_lru(keyLen, 3, 5, add_intervals)
                            # 统一将间隙放入最前面，后面的统一进行分段
                            remaining_length = keyLen - sum([end - start for start, end in add_intervals])
                            segments_B_with_occupy = partition(remaining_length, 3, 5)
                            if len(segments_B_with_occupy) != 0:
                                segment_with_occupy_lengths.extend([len(seg) for seg in segments_B_with_occupy])

                        # 根据子分段构成一个覆盖总分段长度的组合
                        if isUp:
                            segment_method_A = find_all_cover_intervals_iter_up(segments_A, (0, keyLen), 2)
                            segment_method_B = find_all_cover_intervals_iter_up(segments_B, (0, keyLen), 2)
                        else:
                            if withIndexValue:
                                segment_method_A = find_all_cover_intervals_iter(
                                    segments_A, (0, measurements_len), min_length, max_length)[0]
                                segment_method_B, all_segment_method_B = find_all_cover_intervals_iter(
                                    segments_B, (0, measurements_len), min_length, max_length)
                            else:
                                if withoutSort is True:
                                    segment_method_A = find_all_cover_intervals_iter(
                                        segments_A, (0, measurements_len), min_length, max_length)[0]
                                    segment_method_B, all_segment_method_B = find_all_cover_intervals_iter(
                                        segments_B, (0, measurements_len), min_length, max_length)
                                else:
                                    segment_method_A = find_all_cover_intervals_iter(
                                        segments_A, (0, keyLen), min_length, max_length)[0]
                                    segment_method_B, all_segment_method_B = find_all_cover_intervals_iter(
                                        segments_B, (0, keyLen), min_length, max_length)

                        if (withProb or keyLen >= 4 * 16) and (segment_method_A == -1 or segment_method_B == -1):
                            if isDegrade:
                                findError = True
                            else:
                                continue

                        if findError is False:
                            if len(segment_method_B) == 0:
                                # 找出最长的分段方式
                                segment_with_max_length = []
                                max_segment_length = 0
                                for k in all_segment_method_B.keys():
                                    for solution in all_segment_method_B[k]:
                                        if len(solution) > max_segment_length:
                                            segment_with_max_length = solution
                                            max_segment_length = len(solution)
                                if len(segment_with_max_length) != 0:
                                    short_segment = find_gaps(segment_with_max_length, [0, measurements_len])
                                    double_add_intervals = find_covering_intervals(segments_A, short_segment)
                                    if len(double_add_intervals) != 0:
                                        segments_B.extend(double_add_intervals)
                                        # 找出间隙之后的分段，有可能就是该分段不一致导致的无法推出连续分段
                                        double_add_before_intervals = find_before_intervals(segments_A, double_add_intervals)
                                        segments_B.extend(double_add_before_intervals)
                                        segments_B.sort(key=lambda x: x[0])
                                    else:
                                        segments_B = segments_A
                                    # 找出不在范围内的分段
                                    out_of_segment = find_out_of_range_intervals(segments_B, min_length, max_length)
                                    triple_add_intervals = find_covering_intervals(segments_A, out_of_segment)
                                    if len(triple_add_intervals) != 0:
                                        segments_B.extend(triple_add_intervals)
                                        segments_B.sort(key=lambda x: x[0])
                                    segment_method_B = find_all_cover_intervals_iter(
                                        segments_B, (0, measurements_len), min_length, max_length)[0]
                                else:
                                    segments_B = segments_A
                            if (withProb or keyLen >= 4 * 16) and segment_method_B == -1:
                                if isDegrade:
                                    findError = True
                                else:
                                    continue
                            if len(segment_method_B) == 0:
                                segments_B = segments_A
                            segment_method_B = find_all_cover_intervals_iter(
                                segments_B, (0, measurements_len), min_length, max_length)[0]
                            if (withProb or keyLen >= 4 * 16) and segment_method_B == -1:
                                if isDegrade:
                                    findError = True
                                else:
                                    continue

                    if findError is False:
                        # 根据子分段索引得到子分段长度
                        segment_length_A = get_segment_lengths(segment_method_A)
                        segment_length_B = get_segment_lengths(segment_method_B)
                        search_end = time.time_ns()

                        if len(segment_length_A) > 2000 or len(segment_length_B) > 2000:
                            if isDegrade:
                                findError = True
                            else:
                                continue

                        a_list_numbers = []
                        a_segments = []
                        a_dists = []
                        b_list_numbers = []
                        b_segments = []
                        b_dists = []
                        e_list_numbers = []
                        e_segments = []
                        e_dists = []
                        final_search_start = time.time_ns()

                        for i in range(len(segment_length_A)):
                            a_list_number_tmp, dists = search_index_with_segment(
                                tmpCSIa1Bck, tmpCSIa1Ind, np.array(segment_length_A[i]).astype(int))
                            if np.array_equal(np.sort(a_list_number_tmp),
                                              list(range(0, len(a_list_number_tmp)))) is False:
                                continue
                            a_segments.append(segment_length_A[i])
                            a_list_numbers.append(a_list_number_tmp)
                            a_dists.append(dists)
                        for i in range(len(segment_length_B)):
                            b_list_number_tmp, dists = search_index_with_segment(
                                tmpCSIb1Ind, tmpCSIa1Ind, np.array(segment_length_B[i]).astype(int))
                            if np.array_equal(np.sort(b_list_number_tmp),
                                              list(range(0, len(b_list_number_tmp)))) is False:
                                continue
                            b_segments.append(segment_length_B[i])
                            b_list_numbers.append(b_list_number_tmp)
                            b_dists.append(dists)

                        if len(a_segments) == 0 or len(b_segments) == 0:
                            if isDegrade:
                                findError = True
                            else:
                                continue

                        if findError is False:
                            max_dist_A = np.inf
                            max_dist_B = np.inf
                            mean_dist_A = np.inf
                            mean_dist_B = np.inf
                            var_dist_A = np.inf
                            var_dist_B = np.inf

                            # 按照min最小->mean最小的次序排序
                            a_min_index = find_special_array_min_mean(a_dists)[0][0]
                            a_list_number = a_list_numbers[a_min_index]
                            a_segment = a_segments[a_min_index]
                            b_min_index = find_special_array_min_mean(b_dists)[0][0]
                            b_list_number = b_list_numbers[b_min_index]
                            b_segment = b_segments[b_min_index]

                            final_search_end = time.time_ns()

                            ######################################### correction part #########################################

                            if isCorrect and a_list_number != b_list_number:
                                key_hash = hash(tuple(a_list_number))

                                for i in range(len(segment_length_B)):
                                    b_list_number_tmp, dists = search_index_with_segment(
                                        tmpCSIb1Ind, tmpCSIa1Ind, np.array(segment_length_B[i]).astype(int))
                                    if hash(tuple(b_list_number_tmp)) == key_hash:
                                        b_list_number = b_list_number_tmp
                                        b_segment = segment_length_B[i]
                                        break

                                # 仍然不正确，如果Alice的推测分段不是原始分段的话，使用原始分段
                                if a_list_number != b_list_number and segment_method_ori != a_segment:
                                    a_list_number, dists = search_index_with_segment(
                                        tmpCSIa1Bck, tmpCSIa1Ind, np.array(segment_method_ori).astype(int))
                                    key_hash = hash(tuple(a_list_number))

                                    for i in range(len(b_list_numbers)):
                                        b_list_number_tmp = b_list_numbers[i]
                                        if hash(tuple(b_list_number_tmp)) == key_hash:
                                            b_list_number = b_list_number_tmp
                                            b_segment = segment_length_B[i]
                                            break
                                # 回退多次
                                if a_list_number != b_list_number and runBackCounts < 5:
                                    isRunBack = True
                                    staInd -= keyLenLoop
                                    continue
                                    # findError = True
                                elif a_list_number != b_list_number and isRunBack is True:
                                    runBackCounts = 0
                                    isRunBack = False
                                else:
                                    runBackCounts = 0
                                    isRunBack = False

                            if a_list_number == b_list_number:
                                runBackCounts = 0
                                isRunBack = False

                            if findError is False:
                                segment_time.append((segment_end - segment_start) / 1e9)
                                search_time.append((search_end - search_start) / 1e9)
                                final_search_time.append((final_search_end - final_search_start) / 1e9)

                                sum1 = min(len(a_segment), len(b_segment))
                                sum2 = 0

                                for i in range(0, sum1):
                                    sum2 += (a_segment[i] == b_segment[i])

                                originSegSum += sum1
                                correctSegSum += sum2
                else:
                    for i in range(int(keyLen / segLen)):
                        epiInda1 = tmpCSIa1IndReshape[i]

                        epiIndClosenessLsb = np.zeros(int(keyLen / segLen))

                        for j in range(int(keyLen / segLen)):
                            epiIndb1 = tmpCSIb1IndReshape[j]

                            # 欧式距离度量更好
                            # epiIndClosenessLsb[j] = sum(np.square(epiIndb1 - np.array(epiInda1)))
                            epiIndClosenessLsb[j] = dtw_metric(epiIndb1, np.array(epiInda1))
                            # epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))
                            # epiIndClosenessLsb[j] = distance.cosine(epiIndb1, np.array(epiInda1))
                            # epiIndClosenessLsb[j] = abs(sum(epiIndb1) - sum(np.array(epiInda1)))


                        minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)

                    a_list_number = list(permutation)
                    b_list_number = list(minEpiIndClosenessLsb)

                if findError:
                    # 原方法：等长分段
                    tmpCSIa1IndReshape = np.array(tmpCSIa1Bck).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIa1Bck))
                    tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIb1Ind))

                    permutation = list(range(int(keyLen / segLen)))
                    combineMetric = list(zip(tmpCSIa1IndReshape, permutation))
                    np.random.seed(staInd)
                    np.random.shuffle(combineMetric)
                    tmpCSIa1IndReshape, permutation = zip(*combineMetric)
                    tmpCSIa1Ind = np.hstack((tmpCSIa1IndReshape))

                    for i in range(int(keyLen / segLen)):
                        epiInda1 = tmpCSIa1IndReshape[i]

                        epiIndClosenessLsb = np.zeros(int(keyLen / segLen))

                        for j in range(int(keyLen / segLen)):
                            epiIndb1 = tmpCSIb1IndReshape[j]

                            if withIndexValue:
                                # epiIndClosenessLsb[j] = dtw_ndim.distance(epiIndb1, np.array(epiInda1))
                                epiIndClosenessLsb[j] = np.sum(np.abs(epiIndb1 - np.array(epiInda1)))
                            else:
                                # 欧式距离度量更好
                                # epiIndClosenessLsb[j] = sum(np.square(epiIndb1 - np.array(epiInda1)))
                                # epiIndClosenessLsb[j] = dtw_metric(epiIndb1, np.array(epiInda1))
                                epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))
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

                real_segment_lengths.append(len(a_list_number))
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
            if len(segment_time) != 0:
                print("segment time", "total search time")
                # 求AB进行搜索的平均时间
                print(np.round(np.mean(segment_time), 9),
                      np.round((np.mean(search_time) + np.mean(final_search_time)) / 2, 9))

                print("(" + str(correctSum), "/", originSum, "=", str(round(correctSum / originSum, 9)) + ")",
                      round(correctSum / originSum, 9), round(correctWholeSum / originWholeSum, 9),
                      round(originSum / times / keyLen, 9),
                      round(correctSum / times / keyLen, 9), " / ",
                      np.round(np.mean(segment_time), 9),
                      np.round((np.mean(search_time) + np.mean(final_search_time)) / 2, 9))
            print("segment lengths", Counter(segment_lengths))
            print("real segment lengths", Counter(real_segment_lengths))
            print("segment with occupy lengths", Counter(segment_with_occupy_lengths))
            print("\n")
