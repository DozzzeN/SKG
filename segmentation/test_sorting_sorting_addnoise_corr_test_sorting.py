import csv
import hashlib
import math
import random
import time
from collections import defaultdict
from itertools import permutations
from tkinter import messagebox
from segmentation.test_permutations import unique_permutations

import numpy as np
from dtw import accelerated_dtw
from matplotlib import pyplot as plt
from scipy.fft import dct
from scipy.io import loadmat, savemat
from scipy.signal import find_peaks
from scipy.stats import pearsonr, boxcox
from scipy.spatial import distance

from segmentation.test_partition import partition


def generate_segment_lengths(segments, total_length, seed):
    segment_lengths = []
    remaining_length = total_length
    for _ in range(segments - 1):
        if remaining_length <= 1:
            break
        # 生成一个随机分段长度，范围在1到剩余长度之间
        random.seed(seed)
        segment_length = random.randint(1, remaining_length - segments + len(segment_lengths))
        segment_lengths.append(segment_length)
        remaining_length -= segment_length
    # 最后一个分段的长度为剩余长度
    last_segment_length = max(1, remaining_length)
    segment_lengths.append(last_segment_length)
    return segment_lengths


def segment_sequence(data, segment_lengths):
    segments = []
    start_index = 0
    for length in segment_lengths:
        end_index = start_index + length
        segments.append(data[start_index:end_index])
        start_index = end_index
    return segments


def dtw_metric(data1, data2):
    distance = lambda x, y: np.abs(x - y)
    data1 = np.array(data1)
    data2 = np.array(data2)
    # return dtw(data1, data2, dist=distance)[0]
    return accelerated_dtw(data1, data2, dist=distance)


def compute_min_dtw(data):
    min_dtw = np.inf
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            dtw = dtw_metric(data[i], data[j])[0]
            if dtw < min_dtw:
                min_dtw = dtw
    return min_dtw


def compute_min_corr(data, step, threshold):
    min_corr = np.inf
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            min_corr = min(min_corr, step_corr_metric(data[i], data[j], step, threshold))
    return min_corr


def compute_threshold(data1, data2):
    # data1_sort = np.sort(data1)
    # data2_sort = np.sort(data2)
    data1_sort = data1
    data2_sort = data2
    threshold = 0
    for i in range(len(data1)):
        temp = abs(data1_sort[i] - data2_sort[i])
        if temp > threshold:
            threshold = temp
    return threshold


def find_sub_opt_segment_method_down(data, min_length, num_segments, step, threshold):
    each_layer_candidate = [[[len(data)]]]
    each_layer_optimal = [[[len(data)]]]
    max_distance = [[-np.inf] for _ in range(len(data) - num_segments + 1)]
    # 遍历每一层，即每个分段长度
    for layer in range(len(data) - num_segments + 1):
        # 每层的最大距离
        segment_methods = each_layer_optimal[layer]
        # 未找到最优解或者找到最优解
        if len(segment_methods) == 0 or len(segment_methods[0]) == num_segments:
            break
        opt_segment_method = []
        each_layer_candidate.append([])
        each_layer_optimal.append([])
        for segment_method in segment_methods:
            for i in range(len(segment_method)):
                # 至少可以分割成两段
                if segment_method[i] <= 2 * min_length - 1:
                    continue
                # 分割分段
                for j in range(min_length, segment_method[i] - min_length + 1):
                    segment_method_temp = segment_method[:i] + [j, segment_method[i] - j] + segment_method[i + 1:]
                    # 已经计算过距离了，直接跳过
                    if segment_method_temp in each_layer_candidate[layer + 1]:
                        continue
                    else:
                        each_layer_candidate[layer + 1].append(segment_method_temp)
                    segment = segment_sequence(data, segment_method_temp)
                    distance_temp = compute_min_corr(segment, step, threshold)
                    # 找到了更优的分段方法，将其加入到新的分段方法中
                    if distance_temp > max_distance[layer]:
                        max_distance[layer] = distance_temp
                        opt_segment_method = [segment_method_temp]
                    # 找到了与最优分段方法相同的分段方法，将其加入到新的分段方法中
                    elif np.isclose(distance_temp, max_distance[layer], atol=1e-3):
                        if (segment_method_temp not in each_layer_candidate[layer]
                                and segment_method_temp not in opt_segment_method):
                            opt_segment_method.append(segment_method_temp)
        # 按照不同的分段个数放入此时最优的分段方法
        each_layer_optimal[layer + 1].extend(opt_segment_method)
    # print(max_distance[num_segments - 2])
    if each_layer_optimal[-1] != []:
        return each_layer_optimal[-1][0]
    else:
        return each_layer_optimal[-2][0]


def new_segment_sequence(data, segment_lengths):
    segments = []
    for i in range(1, len(segment_lengths)):
        segments.append(data[segment_lengths[i - 1]:segment_lengths[i]])
    return segments


def find_sub_opt_segment_method_up(data, base_length, num_segments, step, threshold):
    each_layer_candidate = [[[base_length for _ in range(int(len(data) / base_length))]]]
    each_layer_optimal = [[[base_length for _ in range(int(len(data) / base_length))]]]
    # 遍历每一层，即每个分段长度
    for layer in range(int(len(data) / base_length) - num_segments):
        # 每层的最大距离
        max_distance = -np.inf
        segment_methods = each_layer_optimal[layer]
        opt_segment_method = []
        each_layer_candidate.append([])
        each_layer_optimal.append([])
        for segment_method in segment_methods:
            for i in range(len(segment_method) - 1):
                segment_method_temp = (segment_method[:i] +
                                       [segment_method[i] + segment_method[i + 1]] + segment_method[i + 2:])
                # 已经计算过距离了，直接跳过
                if segment_method_temp in each_layer_candidate[layer + 1]:
                    continue
                else:
                    each_layer_candidate[layer + 1].append(segment_method_temp)
                segment = segment_sequence(data, segment_method_temp)
                distance_temp = compute_min_corr(segment, step, threshold)
                # 找到了更优的分段方法，将其加入到新的分段方法中
                if distance_temp > max_distance:
                    max_distance = distance_temp
                    opt_segment_method = [segment_method_temp]
                # 找到了与最优分段方法相同的分段方法，将其加入到新的分段方法中
                # unique只选择一个最优分段
                elif np.isclose(distance_temp, max_distance, atol=1e-3):
                    if (segment_method_temp not in each_layer_candidate[layer]
                            and segment_method_temp not in opt_segment_method):
                        opt_segment_method.append(segment_method_temp)
        # 按照不同的分段个数放入此时最优的分段方法
        each_layer_optimal[layer + 1].extend(opt_segment_method)
    return each_layer_optimal[-1][0]


# tmpCSIa1Ind = np.random.permutation(32)
# start_time = time.time_ns()
# segment_method = find_sub_opt_segment_method_down(tmpCSIa1Ind, 2, 4)
# print("time (ms):", (time.time_ns() - start_time) / 1e6)
# print(segment_method)
# print(compute_min_dtw(segment_sequence(tmpCSIa1Ind, segment_method)))
# start_time = time.time_ns()
# segment_method = find_sub_opt_segment_method_up(tmpCSIa1Ind, 2, 1)
# print("time (ms):", (time.time_ns() - start_time) / 1e6)
# print(segment_method)
# print(compute_min_dtw(segment_sequence(tmpCSIa1Ind, segment_method[0])))

def find_opt_segment_method(data, min_length, num_segments):
    # 生成所有分段方法，计算分段间的最小DTW距离
    all_segments = partition(len(data), min_length=min_length, num_partitions=num_segments)
    segment_methods = []
    # for segment in all_segments:
    #     all_permutations = list(set(permutations(segment)))
    #     for permutation in all_permutations:
    #         if permutation not in segment_methods:
    #             segment_methods.append(permutation)
    for segment in all_segments:
        segment_methods += unique_permutations(segment)

    max_distance = -np.inf
    opt_data_seg = []
    for segment_method in segment_methods:
        data_seg = segment_sequence(data, segment_method)
        dist = compute_min_dtw(data_seg)
        if dist > max_distance:
            max_distance = dist
            opt_data_seg = [segment_method]
        elif dist == max_distance:
            if segment_method not in opt_data_seg:
                opt_data_seg.append(segment_method)
    return opt_data_seg[0]


def find_opt_segment_method_from_candidate(data, min_length, max_length, num_segments):
    # 生成所有分段方法，计算分段间的最小DTW距离
    all_segments = partition(len(data), min_length=min_length, max_length=max_length, num_partitions=num_segments)
    segment_methods = []
    for segment in all_segments:
        segment_methods += unique_permutations(segment)

    max_distance = -np.inf
    opt_data_seg = []
    for segment_method in segment_methods:
        data_seg = segment_sequence(data, segment_method)
        dist = compute_min_dtw(data_seg)
        if dist > max_distance:
            max_distance = dist
            opt_data_seg = [segment_method]
        elif dist == max_distance:
            if segment_method not in opt_data_seg:
                opt_data_seg.append(segment_method)
    return opt_data_seg[0]


def segment_data(data, segments, segment_lengths):
    segmented_data = []
    start_index = 0
    for length in segment_lengths:
        end_index = start_index + length
        if end_index > len(data):
            end_index = len(data)
        segmented_data.append(data[start_index:end_index])
        start_index = end_index
    # 如果分段数超过指定数量，将剩余数据作为最后一个分段
    if len(segmented_data) < segments:
        segmented_data.append(data[start_index:])
    return segmented_data


def step_corr(data1, data2, step, threshold):
    # 根据step和阈值计算两个序列的相似性
    n, m = len(data1), len(data2)

    corr = np.zeros(n)
    for i in range(n):
        distance = np.inf
        for j in range(max(0, i - step), min(m, i + step + 1)):
            distance = min(distance, np.abs(data1[i] - data2[j]))
        if distance > threshold:
            corr[i] = 0
        # 如果一个窗内有匹配到的数据，则corr置为1
        else:
            corr[i] = 1
    return np.sum(corr)


def step_corr_metric(data1, data2, step, threshold):
    # 由于不是距离度量，故人为引入对称性保证度量的对称性
    return min(step_corr(data1, data2, step, threshold), step_corr(data2, data1, step, threshold))


def search_segment_method(dataA, dataB, segments, step, threshold):
    # 给定分段长度，通过搜索找到分段位置
    segmentA = segment_data(dataA, len(segments), segments)
    segmentB = segment_data(dataB, len(segments), segments)
    dists = []
    est_index = []
    for i in range(len(segmentA)):
        dists.append([])
        for j in range(len(segmentB)):
            # dists[-1].append(step_corr_metric(segmentA[i], segmentB[j], step, threshold))
            # dists[-1].append(dtw_metric(segmentA[i], segmentB[j])[0])
            if len(segmentA[i]) == len(segmentB[j]):
                dists[-1].append(step_corr_metric(segmentA[i], segmentB[j], step, threshold))
            elif len(segmentA[i]) > len(segmentB[j]):
                dists_tmp = []
                for k in range(len(segmentA[i]) - len(segmentB[j]) + 1):
                    dists_tmp.append(
                        step_corr_metric(segmentA[i][k:k + len(segmentB[j])], segmentB[j], step, threshold))
                dists[-1].append(max(dists_tmp))
            else:
                dists_tmp = []
                for k in range(len(segmentB[j]) - len(segmentA[i]) + 1):
                    dists_tmp.append(
                        step_corr_metric(segmentA[i], segmentB[j][k:k + len(segmentA[i])], step, threshold))
                dists[-1].append(max(dists_tmp))
        # est_index.append(np.argmin(dists[-1]))
        est_index.append(np.argmax(dists[-1]))
    return est_index


def search_segment_method_with_metric(dataA, dataB, segments, step, threshold):
    # 给定分段长度，通过搜索找到分段位置
    segmentA = segment_data(dataA, len(segments), segments)
    segmentB = segment_data(dataB, len(segments), segments)
    dists = []
    est_index = []
    min_dists = []
    for i in range(len(segmentA)):
        dists.append([])
        for j in range(len(segmentB)):
            dists[-1].append(step_corr_metric(segmentA[i], segmentB[j], step, threshold))
        est_index.append(np.argmax(dists[-1]))
        min_dists.append(max(dists[-1]))
    return est_index, min_dists


def step_discrete_corr(data1, data2, step, threshold):
    # 按照一个固定窗口计算互相关
    length = len(data1) + len(data2) - 1
    corr = np.zeros(length)
    data2 = np.flip(data2)
    # 卷积
    for i in range(length):
        tmp1 = []
        tmp2 = []
        for j in range(len(data1)):
            # 根据step选定窗口大小来进行互相关的计算
            if i - j >= 0 and i - j < len(data2):
                tmp1.append(data1[j])
                tmp2.append(data2[i - j])
        corr[i] = step_corr_metric(tmp1, tmp2, step, threshold)
    return corr


def my_find_peaks(data, height=None, distance=None):
    peaks = []
    peak_height = []
    last_peak_index = 0

    for i in range(1, len(data) - 1):
        if data[i] >= data[i - 1] and data[i] >= data[i + 1]:
            if (height is None or data[i] > height) and (distance is None or i - last_peak_index >= distance):
                peaks.append(i)
                peak_height.append(data[i])
                last_peak_index = i

    return peaks


def find_peaks_in_overlapping_segments(data, segment_length):
    # 拷贝一个数组用于删除重叠部分
    data_copy = data.copy()
    peaks = []

    last_peak_index = 0
    i = 0
    while i < len(data):
        segment = data_copy[i:i + segment_length]
        max_value = max(segment)
        # 如果最大值等于最小值，直接找到了峰值
        if max_value == min(segment):
            peaks.append(len(segment) - 1 + i)
            i = len(segment) + i
            continue
        # 如果最大值在最后一个元素，说明发现了重叠，否则找到了峰值
        if segment[-1] != max_value:
            peaks.append(np.argmax(segment) + i)
            i = len(segment) + i
            continue
        # 找出第二大的值，忽略重复的最大值
        second_max_value = -np.inf
        peak_pos = 0
        for j in range(len(segment)):
            if segment[j] < max_value and segment[j] >= second_max_value:
                peak_pos = j
                second_max_value = segment[j]
        peaks.append(peak_pos + i)
        for j in range(len(segment) - 1, -1, -1):
            if segment[j] == max_value:
                data_copy[i + j] -= second_max_value
                last_peak_index = j + i
        i = last_peak_index

    return peaks, np.array(data_copy)[peaks]


def find_peaks_in_segments(data, height, segment_length):
    peaks = []
    peak_values = []

    i = 0
    while i < len(data):
        if data[i] <= height:
            i += 1
            continue
        segment = data[i:i + segment_length]
        segment_peaks = my_find_peaks(segment, height, segment_length // 2)
        if segment_peaks:
            segment_peaks = [peak + i for peak in segment_peaks]
            # 如果分段最后一个元素大于峰值，即找到了重叠分段
            last_value = segment[-1]
            # 有时候峰值就是最大值，但还是重复了，就寻找分段后一个值
            if last_value > data[segment_peaks[-1]] or (
                    last_value == data[segment_peaks[-1]] and i + segment_length < len(data)
                    and data[i + segment_length] > height):
                j = i + segment_length
                while j < len(data):
                    if data[j] <= height:
                        break
                    j += 1
                segment = data[i:j]
                peaks_overlapping = find_peaks_in_overlapping_segments(segment, segment_length)
                peaks.extend(peaks_overlapping[0])
                peak_values.extend([int(peak_value) for peak_value in peaks_overlapping[1]])
                i = j
            else:
                peaks.extend(segment_peaks)
                peak_values.extend([int(data[peak]) for peak in segment_peaks])
                i += segment_length
        else:
            i += 1

    return peaks, peak_values


def find_plateau_length(data, height):
    # 找出最大的平台长度，即连续相等的元素的长度，同时这些元素必须大于height，防止找到0值
    segment_length = 1
    max_segment_length = 0
    for i in range(1, len(data)):
        if data[i - 1] > height:
            if data[i] == data[i - 1]:
                segment_length += 1
            else:
                max_segment_length = max(segment_length, max_segment_length)
                segment_length = 1
    return max_segment_length


def find_offset(dataA, dataB):
    # 根据元素在dataA中的位置找出dataB中元素的对应偏移
    ori_loc = {}
    for i in range(len(dataA)):
        ori_loc[dataA[i]] = i
    offsets = []
    for i in range(len(dataB)):
        offsets.append(ori_loc[dataB[i]])
    return offsets
    # 存在非整数的情况
    # threshold = []
    # for i in range(len(dataA)):
    #     threshold.append(abs(dataA[i] - dataB[i]))
    # threshold = np.median(threshold)
    # offsets = []
    # for i in range(len(dataB)):
    #     for j in range(len(dataA)):
    #         if abs(dataB[i] - dataA[j]) < threshold:
    #             offsets.append(j)
    #             break
    # return offsets


def search_segment_method_with_offset(dataA, dataB, step, threshold):
    offsets = find_offset(dataA, dataB)
    # print("offsets", offsets)
    segment_method = []
    segment_len = 0
    # 按照窗长与阈值来找分段类型
    for i in range(len(offsets)):
        # 连续的递增序列看作一个分段
        if i == 0 or abs(offsets[i] - offsets[i - 1]) < threshold:
            # 在阈值内，分段长度递增
            segment_len += 1
        else:
            within_threshold_in_step = False
            for j in range(step):
                if i - j - 1 >= 0 and abs(offsets[i] - offsets[i - j - 1]) < threshold and j + 1 <= segment_len + 1:
                    # 不在阈值内，且step内有数据在阈值内，且在阈值内的该元素与匹配元素的位置偏差小于此时分段长度+1，分段长度递增
                    segment_len += 1
                    within_threshold_in_step = True
                    break
            if not within_threshold_in_step:
                # 不在阈值内，且step内的数据也没有一个在阈值内，分段长度置为1，重新分段
                segment_method.append(segment_len)
                segment_len = 1
        if i == len(offsets) - 1:
            # 最后一个元素，直接加入分段
            segment_method.append(segment_len)
    return segment_method


def search_segment_method_with_matching(dataA, dataB, threshold):
    # 按照窗长与阈值来找分段类型
    start_index = []
    for i in range(len(dataB)):
        # 选择与当前接近的位置作为寻找起点
        current_start_index = []
        for j in range(len(dataA)):
            if abs(dataB[i] - dataA[j]) < threshold:
                current_start_index.append(j)
        start_index.append(current_start_index)
    all_segments = []
    # 将start_index的第一列放入all_segments的第一列，键为索引值，值为第一个分段起点以及其索引值
    current_segments = {}
    for i in range(len(start_index[0])):
        current_segments[start_index[0][i]] = str(0) + '-' + str(start_index[0][i])
    all_segments.append(current_segments)
    for i in range(1, len(dataB)):
        current_segments = {}
        for j in range(len(start_index[i])):
            for k in range(len(start_index[i - 1])):
                # 如果某个分段包含某个子分段，该子分段不计入
                if start_index[i][j] == start_index[i - 1][k] + 1:
                    if start_index[i - 1][k] not in all_segments[i - 1].keys():
                        # 找到某个分段的起点，将其位置作为值插入
                        all_segments[i - 1][start_index[i - 1][k]] = str(i - 1) + '-' + str(start_index[i - 1][k])
                        current_segments[start_index[i][j]] = str(i - 1) + '-' + str(start_index[i - 1][k])
                    else:
                        # 已经是某个分段的一部分，将分段起点作为值插入
                        segment_start_point = all_segments[i - 1].get(start_index[i - 1][k])
                        current_segments[start_index[i][j]] = segment_start_point
        all_segments.append(current_segments)
    return all_segments


def find_segments(arr, min_length, max_length):
    from collections import defaultdict

    # 将每个键值对转换为边界
    segments = defaultdict(set)
    for d in arr:
        for key, value in d.items():
            segments[value].add(key)

    # 合并边界形成连续的段，找出长度在[min_length, max_length]之间的段
    all_segments = []
    for key, value in segments.items():
        # 找到起始位置
        start = int(key[:str(key).find('-')])
        if len(value) >= min_length and len(value) <= max_length:
            # 将该段的起止索引加入到结果中
            if [start, start + len(value)] not in all_segments:
                all_segments.append([start, start + len(value)])

    return all_segments


def generate_subintervals(interval, min_length, max_length):
    # 给定单个区间，生成所有可能的子区间，长度在[min_length, max_length]之间
    start, end = interval
    subintervals = []
    # 生成所有子区间，确保它们的长度在min_length和max_length之间
    for sub_start in range(start, end):
        for sub_end in range(sub_start + min_length, min(end, sub_start + max_length) + 1):
            subintervals.append([sub_start, sub_end])
    return subintervals


def generate_subintervals_up(interval, base_length):
    # 给定单个区间，生成所有可能的子区间，长度在是base_length的整数倍
    start, end = interval
    subintervals = []
    # 生成所有子区间，确保它们的长度是base_length的整数倍
    for sub_start in range(start, end):
        # for sub_end in range(sub_start + min_length, min(end, sub_start + max_length) + 1):
        #     subintervals.append([sub_start, sub_end])
        for sub_end in range(sub_start + base_length, end + 1):
            if (sub_end - sub_start) % base_length == 0:
                subintervals.append([sub_start, sub_end])
    return subintervals


def generate_all_subintervals(intervals, min_length, max_length):
    # 给定一些区间，生成所有可能的子区间，长度在[min_length, max_length]之间
    all_subintervals = []
    for interval in intervals:
        subintervals = generate_subintervals(interval, min_length, max_length)
        for subinterval in subintervals:
            if subinterval not in all_subintervals:
                all_subintervals.append(subinterval)
    return all_subintervals


def generate_all_subintervals_up(intervals, base_length):
    # 给定一些区间，生成所有可能的子区间，长度在[min_length, max_length]之间
    all_subintervals = []
    for interval in intervals:
        subintervals = generate_subintervals_up(interval, base_length)
        for subinterval in subintervals:
            if subinterval not in all_subintervals:
                all_subintervals.append(subinterval)
    return all_subintervals


def find_all_cover_intervals(intervals, target_range, min_length, max_length):
    start_time = time.time_ns()

    # 给定一些区间，找到所有可能的区间分段方法，使得每个分段的长度在[min_length, max_length]之间
    def backtrack(start, current_cover, calls):
        # 如果已经覆盖到或超过目标区间的结束点
        if start >= target_range[1] and current_cover[:] not in result:
            result.append(current_cover[:])
            return

        for interval in all_subintervals:
            calls[0] += 1
            # 下一个区间的起始时间必须在当前区间的结束时间之后，且区间长度在[min_length, max_length]之间
            if interval[0] <= start < interval[1] and min_length <= interval[1] - start <= max_length:
                # 如果当前区间能覆盖start，则选择这个区间并继续寻找
                # 跳过之前分段已经使用的部分
                current_cover.append([start, interval[1]])
                backtrack(interval[1], current_cover, calls)
                current_cover.pop()

    # 生成所有可能的子任务
    all_subintervals = generate_all_subintervals(intervals, min_length, max_length)
    all_subintervals.sort(key=lambda x: (x[0], x[1]))
    result = []
    calls = [0]
    backtrack(target_range[0], [], calls)
    print(calls)
    print("time (ms):", (time.time_ns() - start_time) / 1e6)

    return result


def remove_contained_intervals(intervals):
    # 删除被包含的子任务
    intervals.sort(key=lambda x: (x[0], x[1]))
    filtered_intervals = []
    for interval in intervals:
        if not any(i[0] <= interval[0] and i[1] >= interval[1] for i in filtered_intervals):
            filtered_intervals.append(interval)
    # 再次过滤其内部，防止后插入的区间包含之前的区间
    pop_index = []
    for i in range(len(filtered_intervals) - 1):
        if (filtered_intervals[i][0] >= filtered_intervals[i + 1][0]
                and filtered_intervals[i][1] <= filtered_intervals[i + 1][1]):
            pop_index.append(i)
    # 倒序删除
    for i in range(len(pop_index) - 1, -1, -1):
        filtered_intervals.pop(pop_index[i])
    return filtered_intervals


def find_all_cover_intervals_iter(intervals, target_range, min_length, max_length):
    start_time = time.time_ns()

    def build_tree(intervals):
        tree = defaultdict(list)
        for start, end in intervals:
            tree[start].append(end)
        return tree

    all_intervals = generate_all_subintervals(intervals, min_length, max_length)
    tree = build_tree(all_intervals)

    dp = defaultdict(list)
    dp[target_range[1]].append([])

    # 调用次数
    calls = [0, 0]
    for start in range(target_range[1] - 1, target_range[0] - 1, -1):
        for end in tree[start]:
            if min_length <= end - start <= max_length:
                for subseq in dp[end]:
                    calls[0] += 1
                    dp[start].append([[start, end]] + subseq)

    # 筛选出符合条件的方案
    valid_results = []
    for solution in dp[target_range[0]]:
        calls[1] += 1
        if all(min_length <= (interval[1] - interval[0]) <= max_length for interval in solution):
            valid_results.append(solution)

    # print("time (ms):", (time.time_ns() - start_time) / 1e6)
    # print(calls)
    return valid_results


def find_all_cover_intervals_iter_up(intervals, target_range, base_length):
    start_time = time.time_ns()

    def build_tree(intervals):
        tree = defaultdict(list)
        for start, end in intervals:
            tree[start].append(end)
        return tree

    all_intervals = generate_all_subintervals_up(intervals, base_length)
    tree = build_tree(all_intervals)

    dp = defaultdict(list)
    dp[target_range[1]].append([])

    # 调用次数
    calls = [0, 0]
    for start in range(target_range[1] - 1, target_range[0] - 1, -1):
        for end in tree[start]:
            if (end - start) % base_length == 0:
                for subseq in dp[end]:
                    calls[0] += 1
                    dp[start].append([[start, end]] + subseq)

    # 筛选出符合条件的方案
    valid_results = []
    for solution in dp[target_range[0]]:
        calls[1] += 1
        if all((interval[1] - interval[0]) % base_length == 0 for interval in solution):
            valid_results.append(solution)

    # print("time (ms):", (time.time_ns() - start_time) / 1e6)
    # print(calls)
    return valid_results


def get_segment_lengths(segment_method):
    segment_lengths = []
    for segment in segment_method:
        current_segment_lengths = []
        for seg in segment:
            current_segment_lengths.append(seg[1] - seg[0])
        segment_lengths.append(current_segment_lengths)
    return segment_lengths


# [4, 8, 2, 2]
# intervals = [[0, 4], [1, 3], [3, 12], [3, 5], [5, 7], [6, 9], [7, 9], [9, 12], [10, 12], [11, 13], [12, 16], [12, 14], [13, 16]]
#
# segments = find_all_cover_intervals(intervals, (0, np.max(intervals)), 2, 10)
# segments_len1 = get_segment_lengths(segments)
# print(len(segments_len1), np.array(segments_len1, dtype=object))
# print()
#
# segments = find_all_cover_intervals(remove_contained_intervals(intervals),
#                                     (0, np.max(intervals)), 2, 10)
# segments_len2 = get_segment_lengths(segments)
# print(len(segments_len2),np.array(segments_len2, dtype=object))
# print( np.array_equal(segments_len1, segments_len2))
#
# segments = find_all_cover_intervals_iter(intervals, (0, np.max(intervals)), 2, 10)
# segments_len3 = get_segment_lengths(segments)
# print(len(segments_len3), np.array(segments_len3, dtype=object))
# print(np.array_equal(np.sort(segments_len1), np.sort(segments_len3)))
#
# intervals = remove_contained_intervals(intervals)
# segments = find_all_cover_intervals_iter_up(intervals, (0, np.max(intervals)), 2)
# segments_len4 = get_segment_lengths(segments)
# print(len(segments_len4), np.array(segments_len4, dtype=object))
# exit()

def find_special_array(arrays, epsilon=1e-5):
    # 按照最大值，均值，方差的顺序排序，同时考虑浮点数判等的精度问题
    # List to store (index, max_value, mean_value, variance, array)
    array_info = []

    for idx, arr in enumerate(arrays):
        max_value = max(arr)
        mean_value = np.mean(arr)
        var_value = np.var(arr)
        array_info.append((idx, max_value, mean_value, var_value, arr))

    # Function to compare two arrays based on the criteria
    def compare_arrays(a, b):
        idx_a, max_a, mean_a, var_a, arr_a = a
        idx_b, max_b, mean_b, var_b, arr_b = b

        if abs(max_a - max_b) > epsilon:
            return -1 if max_a < max_b else 1
        elif abs(mean_a - mean_b) > epsilon:
            return -1 if mean_a < mean_b else 1
        else:
            return -1 if var_a < var_b else 1

    # Sort the arrays using the custom compare function
    from functools import cmp_to_key
    sorted_arrays = sorted(array_info, key=cmp_to_key(compare_arrays))

    # Return the index of the array with the smallest max_value, mean_value, and variance
    return sorted_arrays


# print(find_special_array([[2, -1, 1], [1, 2, 3], [0, 0, 2]]))

# all_segments = search_segment_method_with_matching([12, 3, 9, 11, 2, 6, 10, 5, 1, 13, 7, 8, 14, 4, 0, 15],
#                                     [13, 3, 8, 10, 4, 0, 15, 1, 12, 7, 9, 14, 2, 6, 11, 5], 3)
# print(all_segments)
# segments = find_segments(all_segments, 3, 8)
# print(len(segments), segments)
# segment_method = find_all_cover_intervals(segments, (0, 16), 3, 5)
# print(len(segment_method), segment_method)
# segment_length = get_segment_lengths(segment_method)
# print(len(segment_length), segment_length)

def find_min_threshold(dataA, dataB, segment_method):
    # 根据分段方法找出分段之间最小的阈值
    offsets = find_offset(dataA, dataB)
    min_threshold = np.inf
    next_segment_pos = 0
    # 最后一个分段位置不计算
    for i in range(len(segment_method) - 1):
        next_segment_pos += segment_method[i]
        min_threshold = min(min_threshold, abs(offsets[next_segment_pos] -
                                               offsets[next_segment_pos - 1]))
    return min_threshold


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


# fileName = ["../data/data_mobile_indoor_1.mat",
#             "../data/data_mobile_outdoor_1.mat",
#             "../data/data_static_outdoor_1.mat",
#             "../data/data_static_indoor_1.mat"
#             ]

# 基于自适应分段索引匹配的密钥生成
fileName = ["../data/data_mobile_indoor_1.mat"]

isPrint = False

isShuffle = False
isSegSort = False
isValueShuffle = False

# 是否纠错
rec = False

# 是否排序
# withoutSorts = [True, False]
withoutSorts = [False]
# 是否添加噪声
# addNoises = ["mul", ""]
addNoises = ["mul"]
# 最优分段opt，次优分段sub，各自分段ind，随机分段rdm，随机分段且限制最小分段数rdm_cond，发送打乱后的数据然后推测分段
# 通过自相关确定分段类型，再根据公布的数据进行分段匹配snd
# 不根据相关系数只根据数据来进行搜索find
# 固定分段类型fix，双方从固定分段类型里面挑选
# 根据公布的索引搜索分段find_search
segment_option = "find_search"
print(segment_option)

for f in fileName:
    for addNoise in addNoises:
        for withoutSort in withoutSorts:
            print(f)
            rawData = loadmat(f)

            if f.find("data_alignment") != -1:
                CSIa1Orig = rawData['csi'][:, 0]
                CSIb1Orig = rawData['csi'][:, 1]
                CSIe1Orig = rawData['csi'][:, 2]
            elif f.find("csi") != -1:
                CSIa1Orig = rawData['testdata'][:, 0]
                CSIb1Orig = rawData['testdata'][:, 1]
            else:
                CSIa1Orig = rawData['A'][:, 0]
                CSIb1Orig = rawData['A'][:, 1]

            dataLen = len(CSIa1Orig)
            print("dataLen", dataLen)

            segLen = 4
            keyLen = 4 * segLen
            tell = True

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
                if addNoise == "mul":
                    print("no sorting")
            if withoutSort:
                if addNoise == "":
                    print("no sorting and no perturbation")
            if withoutSort is False:
                if addNoise == "":
                    print("no perturbation")
                if addNoise == "mul":
                    print("normal")
            if isShuffle:
                print("with shuffle")
            if isSegSort:
                print("with segSort")

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

                if f.find("data_alignment") != -1:
                    CSIa1Orig = rawData['csi'][:, 0]
                    CSIb1Orig = rawData['csi'][:, 1]
                    CSIe1Orig = rawData['csi'][:, 2]
                elif f.find("csi") != -1:
                    CSIa1Orig = rawData['testdata'][:, 0]
                    CSIb1Orig = rawData['testdata'][:, 1]
                else:
                    CSIa1Orig = rawData['A'][:, 0]
                    CSIb1Orig = rawData['A'][:, 1]

                if f == "../data/data_NLOS.mat":
                    # 先整体shuffle一次
                    shuffleInd = np.random.permutation(dataLen)
                    CSIa1Orig = CSIa1Orig[shuffleInd]
                    CSIb1Orig = CSIb1Orig[shuffleInd]
                seed = np.random.randint(100000)
                np.random.seed(seed)

                # 固定随机置换的种子
                # np.random.seed(0)
                # combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig))
                # np.random.shuffle(combineCSIx1Orig)
                # CSIa1Orig, CSIb1Orig = zip(*combineCSIx1Orig)
                # CSIa1Orig = np.array(CSIa1Orig)
                # CSIb1Orig = np.array(CSIb1Orig)

                CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
                CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')

                tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
                tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]

                # 目的是把加噪音+无排序的结果降下来
                if addNoise == "mul":
                    # randomMatrix = np.random.randint(0, 2, size=(keyLen, keyLen))
                    # randomMatrix = np.random.uniform(0, 1, size=(keyLen, keyLen))
                    np.random.seed(10000)
                    randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
                    # 均值化
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1))
                    tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1))
                    tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
                    tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)

                    # 相当于乘了一个置换矩阵 permutation matrix
                    # np.random.seed(0)
                    # combineCSIx1Orig = list(zip(tmpCSIa1, tmpCSIb1))
                    # np.random.shuffle(combineCSIx1Orig)
                    # tmpCSIa1, tmpCSIb1 = zip(*combineCSIx1Orig)
                    # tmpCSIa1 = np.array(tmpCSIa1)
                    # tmpCSIb1 = np.array(tmpCSIb1)
                else:
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    # tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1))
                    # tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1))

                # 最后各自的密钥
                a_list = []
                b_list = []

                # with value shuffling
                if isValueShuffle:
                    np.random.seed(10000)
                    combineCSIx1Orig = list(zip(tmpCSIa1, tmpCSIb1))
                    np.random.shuffle(combineCSIx1Orig)
                    tmpCSIa1, tmpCSIb1 = zip(*combineCSIx1Orig)
                    tmpCSIa1 = list(tmpCSIa1)
                    tmpCSIb1 = list(tmpCSIb1)

                # without sorting
                # print(pearsonr(tmpCSIa1, tmpCSIb1)[0])
                if withoutSort:
                    tmpCSIa1Ind = np.array(tmpCSIa1)
                    tmpCSIb1Ind = np.array(tmpCSIb1)
                else:
                    tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
                    tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()
                    # print(pearsonr(tmpCSIa1Ind, tmpCSIb1Ind)[0])

                    # with shuffling
                    if isShuffle:
                        np.random.seed(0)
                        combineCSIx1Orig = list(zip(tmpCSIa1Ind, tmpCSIb1Ind))
                        np.random.shuffle(combineCSIx1Orig)
                        tmpCSIa1Ind, tmpCSIb1Ind = zip(*combineCSIx1Orig)
                        tmpCSIa1Ind = list(tmpCSIa1Ind)
                        tmpCSIb1Ind = list(tmpCSIb1Ind)
                        # print(pearsonr(tmpCSIa1Ind, tmpCSIb1Ind)[0])

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

                if segment_option == "opt":
                    segment_method = find_opt_segment_method(tmpCSIa1Ind, 3, int(keyLen / segLen))
                    if len(segment_method) != int(keyLen / segLen):
                        print("未找到最优解")
                        continue
                    tmpCSIa1IndReshape = segment_sequence(tmpCSIa1Ind, segment_method)
                    tmpCSIb1IndReshape = segment_sequence(tmpCSIb1Ind, segment_method)
                    tmpCSIa1EvenSegment = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)
                    if compute_min_dtw(tmpCSIa1IndReshape) < compute_min_dtw(tmpCSIa1EvenSegment):
                        print("staInd", staInd, "最优分段最大差距", compute_min_dtw(tmpCSIa1IndReshape),
                              "平均分段最大差距", compute_min_dtw(tmpCSIa1EvenSegment))
                        badSegments += 1
                    segmentMaxDist.append(compute_min_dtw(tmpCSIa1IndReshape))
                    evenMaxDist.append(compute_min_dtw(tmpCSIa1EvenSegment))
                elif segment_option == "sub_down":
                    step = 3
                    threshold = 1
                    segment_method = find_sub_opt_segment_method_down(tmpCSIa1Ind, 3,
                                                                      int(keyLen / segLen), step, threshold)
                    if len(segment_method) != int(keyLen / segLen):
                        print("未找到次优解")
                        continue
                    print("staInd", staInd, "segment_method", segment_method)
                    tmpCSIa1IndReshape = segment_sequence(tmpCSIa1Ind, segment_method)
                    tmpCSIb1IndReshape = segment_sequence(tmpCSIb1Ind, segment_method)
                    tmpCSIa1EvenSegment = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)
                    if compute_min_corr(tmpCSIa1IndReshape, step, threshold) < compute_min_corr(tmpCSIa1EvenSegment, step, threshold):
                        print("staInd", staInd, "次优分段最大差距", compute_min_corr(tmpCSIa1IndReshape, step, threshold),
                              "平均分段最大差距", compute_min_corr(tmpCSIa1EvenSegment, step, threshold))
                        badSegments += 1
                    segmentMaxDist.append(compute_min_corr(tmpCSIa1IndReshape, step, threshold))
                    evenMaxDist.append(compute_min_corr(tmpCSIa1EvenSegment, step, threshold))
                elif segment_option == "sub_up":
                    step = 3
                    threshold = 1
                    segment_method = find_sub_opt_segment_method_up(tmpCSIa1Ind, 2,
                                                                    int(keyLen / segLen), step, 1)
                    if len(segment_method) != int(keyLen / segLen):
                        print("未找到次优解")
                        continue
                    print("staInd", staInd, "segment_method", segment_method)
                    tmpCSIa1IndReshape = segment_sequence(tmpCSIa1Ind, segment_method)
                    tmpCSIb1IndReshape = segment_sequence(tmpCSIb1Ind, segment_method)
                    tmpCSIa1EvenSegment = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)
                    if compute_min_corr(tmpCSIa1IndReshape, step, threshold) < compute_min_corr(tmpCSIa1EvenSegment, step, threshold):
                        print("staInd", staInd, "次优分段最大差距", compute_min_corr(tmpCSIa1IndReshape, step, threshold),
                              "平均分段最大差距", compute_min_dtw(tmpCSIa1EvenSegment))
                        badSegments += 1
                    segmentMaxDist.append(compute_min_corr(tmpCSIa1IndReshape, step, threshold))
                    evenMaxDist.append(compute_min_corr(tmpCSIa1EvenSegment, step, threshold))
                elif segment_option == "ind_down":
                    segment_method_A = find_sub_opt_segment_method_down(tmpCSIa1Ind, 3, int(keyLen / segLen))
                    segment_method_B = find_sub_opt_segment_method_down(tmpCSIb1Ind, 3, int(keyLen / segLen))
                    if len(segment_method_A) != int(keyLen / segLen) or len(segment_method_B) != int(keyLen / segLen):
                        print("未找到各自的次优解")
                        continue
                    print("staInd", staInd, "segment_method", segment_method_A, segment_method_B)
                    tmpCSIa1IndReshape = segment_sequence(tmpCSIa1Ind, segment_method_A)
                    tmpCSIb1IndReshape = segment_sequence(tmpCSIb1Ind, segment_method_B)
                elif segment_option == "ind_up":
                    segment_method_A = find_sub_opt_segment_method_up(tmpCSIa1Ind, 2, int(keyLen / segLen))
                    segment_method_B = find_sub_opt_segment_method_up(tmpCSIb1Ind, 2, int(keyLen / segLen))
                    if len(segment_method_A) != int(keyLen / segLen) or len(segment_method_B) != int(keyLen / segLen):
                        print("未找到各自的次优解")
                        continue
                    print("staInd", staInd, "segment_method", segment_method_A, segment_method_B)
                    tmpCSIa1IndReshape = segment_sequence(tmpCSIa1Ind, segment_method_A)
                    tmpCSIb1IndReshape = segment_sequence(tmpCSIb1Ind, segment_method_B)
                elif segment_option == "rdm":
                    segment_method = generate_segment_lengths(int(keyLen / segLen), keyLen, staInd)
                    tmpCSIa1IndReshape = segment_sequence(tmpCSIa1Ind, segment_method)
                    tmpCSIb1IndReshape = segment_sequence(tmpCSIb1Ind, segment_method)
                elif segment_option == "rdm_cond":
                    segment_methods = partition(keyLen, min_length=3, max_length=5, num_partitions=int(keyLen / segLen))
                    # for 5*64
                    # segment_methods = partition(keyLen, min_length=4, max_length=7, num_partitions=int(keyLen / segLen))
                    np.random.seed(staInd)
                    random_int = np.random.randint(0, len(segment_methods))
                    segment_method = segment_methods[random_int]
                    print("staInd", staInd, "segment_method", segment_method)
                    tmpCSIa1IndReshape = segment_sequence(tmpCSIa1Ind, segment_method)
                    tmpCSIb1IndReshape = segment_sequence(tmpCSIb1Ind, segment_method)
                elif segment_option == "snd" or segment_option == "find":
                    start_time = time.time()
                    # segment_method = find_sub_opt_segment_method(tmpCSIa1Ind, 3, int(keyLen / segLen))
                    segment_method_A = find_opt_segment_method_from_candidate(
                        tmpCSIa1Ind, segLen - 1, segLen + 1, int(keyLen / segLen))
                    segment_method_B = find_opt_segment_method_from_candidate(
                        tmpCSIb1Ind, segLen - 1, segLen + 1, int(keyLen / segLen))
                    if len(segment_method_A) != int(keyLen / segLen):
                        print("未找到次优解")
                        continue
                    print("staInd", staInd, "segment_method", segment_method_A, segment_method_B)
                    tmpCSIa1IndReshape = segment_sequence(tmpCSIa1Ind, segment_method_A)
                elif segment_option == "find_search":
                    start_time = time.time()
                    # 随机分段
                    # segment_methods = partition(keyLen, min_length=3, max_length=5, num_partitions=int(keyLen / segLen))
                    # np.random.seed(staInd)
                    # random_int = np.random.randint(0, len(segment_methods))
                    # segment_method_ori = segment_methods[random_int]

                    # 最优分段（是否固定分段个数）
                    # segment_method_ori = find_opt_segment_method(tmpCSIa1Ind, 3, int(keyLen / segLen))

                    # 次优分段（是否固定分段个数）
                    # segment_method_ori = find_sub_opt_segment_method(tmpCSIa1Ind, 3, int(keyLen / segLen))
                    # segment_method_ori1 = find_sub_opt_segment_method(tmpCSIa1Ind, 3, int(keyLen / segLen))
                    # segment_method_ori2 = find_sub_opt_segment_method(tmpCSIa1Ind, 3, int(keyLen / segLen) + 1)

                    # 固定分段（是否固定分段个数）
                    # segment_method_ori = find_opt_segment_method_from_candidate(
                    #         tmpCSIa1Ind, segLen - 1, segLen + 1, int(keyLen / segLen))
                    # segment_method_ori1 = find_opt_segment_method_from_candidate(
                    #     tmpCSIa1Ind, segLen - 1, segLen + 1, int(keyLen / segLen))
                    # segment_method_ori2 = find_opt_segment_method_from_candidate(
                    #     tmpCSIa1Ind, segLen - 1, segLen + 1, int(keyLen / segLen) + 1)
                    #
                    # _, dist1 = search_segment_method_with_metric(
                    #                     tmpCSIb1Ind, tmpCSIa1Ind, np.array(segment_method_ori1).astype(int))
                    # _, dist2 = search_segment_method_with_metric(
                    #                     tmpCSIb1Ind, tmpCSIa1Ind, np.array(segment_method_ori2).astype(int))
                    # a_dists = []
                    # a_dists.append(dist1)
                    # a_dists.append(dist2)
                    #
                    # a_min_index = find_special_array(a_dists)[0][0]
                    # if a_min_index == 0:
                    #     segment_method_ori = segment_method_ori1
                    # else:
                    #     segment_method_ori = segment_method_ori2

                    # 合并分段
                    step = 3
                    threshold = 1
                    segment_start = time.time_ns()
                    segment_method_ori = find_sub_opt_segment_method_up(tmpCSIa1Ind, 2,
                                                                        int(keyLen / segLen), step, threshold)
                    segment_end = time.time_ns()

                    print("staInd", staInd, "segment_method", segment_method_ori)
                    tmpCSIa1IndReshape = segment_sequence(tmpCSIa1Ind, segment_method_ori)

                    # segment_method_ori = [4, 4, 4, 4]
                    # tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)
                elif segment_option == "fix":
                    # segment_method = find_opt_segment_method_from_candidate(
                    #     tmpCSIa1Ind, segLen - 2, segLen + 2,  int(keyLen / segLen))
                    segment_method = find_opt_segment_method_from_candidate(
                        tmpCSIa1Ind, segLen - 1, segLen + 1, int(keyLen / segLen))
                    print("staInd", staInd, "segment_method", segment_method)
                    tmpCSIa1IndReshape = segment_sequence(tmpCSIa1Ind, segment_method)
                    tmpCSIb1IndReshape = segment_sequence(tmpCSIb1Ind, segment_method)
                elif segment_option == "fix_ind":
                    segment_method_A = find_opt_segment_method_from_candidate(
                        tmpCSIa1Ind, segLen - 1, segLen + 1, int(keyLen / segLen))
                    segment_method_B = find_opt_segment_method_from_candidate(
                        tmpCSIb1Ind, segLen - 1, segLen + 1, int(keyLen / segLen))
                    print("staInd", staInd, "segment_method_A", segment_method_A, "segment_method_B", segment_method_B)
                    tmpCSIa1IndReshape = segment_sequence(tmpCSIa1Ind, segment_method_A)
                    tmpCSIb1IndReshape = segment_sequence(tmpCSIb1Ind, segment_method_B)
                else:
                    # 原方法：固定分段
                    tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)
                    tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(int(keyLen / segLen), segLen)

                tmpCSIa1Bck = tmpCSIa1Ind.copy()
                # if a_min_index == 0:
                #     permutation = list(range(int(keyLen / segLen)))
                # else:
                #     permutation = list(range(int(keyLen / segLen) + 1))
                permutation = list(range(int(keyLen / segLen)))
                combineMetric = list(zip(tmpCSIa1IndReshape, permutation))
                np.random.seed(staInd)
                np.random.shuffle(combineMetric)
                tmpCSIa1IndReshape, permutation = zip(*combineMetric)
                tmpCSIa1Ind = np.hstack((tmpCSIa1IndReshape))

                if segment_option == "snd":
                    # 调库find_peaks找峰值
                    # step = int(keyLen / segLen / 2) - 1
                    # cross_correlation_of_A = step_discrete_corr(tmpCSIa1Bck, tmpCSIa1Ind, step, 0)
                    # cross_correlation_of_B = step_discrete_corr(tmpCSIb1Ind, tmpCSIa1Ind, step, 0)
                    # segmentsA = cross_correlation_of_A[find_peaks(cross_correlation_of_A, height=0, distance=2 * step + 1)[0]]
                    # segmentsB = cross_correlation_of_B[find_peaks(cross_correlation_of_B, height=0, distance=2 * step + 1)[0]]
                    # segment_peaks_A = np.array(segmentsA).astype(int)
                    # segment_peaks_B = np.array(segmentsB).astype(int)

                    # 给定连续峰值最大长度
                    # step = find_plateau_length(cross_correlation_of_A, 2)
                    # segmentsA = cross_correlation_of_A[find_peaks(cross_correlation_of_A, height=0, distance=step)[0]]
                    # segmentsB = cross_correlation_of_B[find_peaks(cross_correlation_of_B, height=0, distance=step)[0]]
                    # segment_peaks_A = np.array(segmentsA).astype(int)
                    # segment_peaks_B = np.array(segmentsB).astype(int)

                    # AB用固定长度找峰值
                    # step = int(keyLen / segLen / 2) - 1
                    # cross_correlation_of_A = step_discrete_corr(tmpCSIa1Bck, tmpCSIa1Ind, step, 0)
                    # cross_correlation_of_B = step_discrete_corr(tmpCSIb1Ind, tmpCSIa1Ind, step, 0)
                    # segment_peaks_A = find_peaks_in_segments(cross_correlation_of_A, 2, 2 * step + 1)[1]
                    # segment_peaks_B = find_peaks_in_segments(cross_correlation_of_B, 2, 2 * step + 1)[1]

                    # A找出峰值位置给B，B根据A的峰值位置找出自己的峰值位置
                    # step = int(keyLen / segLen / 2) - 1
                    # cross_correlation_of_A = step_discrete_corr(tmpCSIa1Bck, tmpCSIa1Ind, step, 0)
                    # cross_correlation_of_B = step_discrete_corr(tmpCSIb1Ind, tmpCSIa1Ind, step, 0)
                    # segment_peaks = find_peaks_in_segments(cross_correlation_of_A, 2, 2 * step + 1)
                    # segment_peaks_A = segment_peaks[1]
                    # segment_peaks_B = np.array(cross_correlation_of_B)[segment_peaks[0]].astype(int)

                    # A找出连续峰值最大长度给B，B以此找出自己的峰值
                    step = int(keyLen / segLen / 2) - 1
                    cross_correlation_of_A = step_discrete_corr(tmpCSIa1Bck, tmpCSIa1Ind, step, 0)
                    cross_correlation_of_B = step_discrete_corr(tmpCSIb1Ind, tmpCSIa1Ind, step, 0)
                    step = find_plateau_length(cross_correlation_of_A, 2)
                    segment_peaks_A = find_peaks_in_segments(cross_correlation_of_A, 2, step)[1]
                    segment_peaks_B = find_peaks_in_segments(cross_correlation_of_B, 2, step)[1]

                    a_list_number = search_segment_method(tmpCSIa1Bck, tmpCSIa1Ind, segment_peaks_A, step, 0)
                    b_list_number = search_segment_method(tmpCSIb1Ind, tmpCSIa1Ind, segment_peaks_B, step, 0)

                    print("segment_peaks_A", segment_peaks_A)
                    print("segment_peaks_B", segment_peaks_B)
                    print("a_list_number", a_list_number)
                    print("b_list_number", b_list_number)
                    print("time", time.time() - start_time)
                    print()

                    # plt.figure()
                    # plt.plot(cross_correlation_of_A)
                    # plt.tight_layout()
                    # plt.show()
                    #
                    # plt.figure()
                    # plt.plot(cross_correlation_of_B)
                    # plt.tight_layout()
                    # plt.show()
                elif segment_option == "find":
                    # print("original data", tmpCSIa1, tmpCSIb1)
                    print("sorted index", tmpCSIa1Bck, tmpCSIb1Ind)
                    print("published index", tmpCSIa1Ind)
                    segment_method_A = search_segment_method_with_offset(tmpCSIa1Bck, tmpCSIa1Ind, 2, 2)
                    min_threshold = find_min_threshold(tmpCSIa1Bck, tmpCSIa1Ind, segment_method_A)
                    # segment_method_B = search_segment_method_with_offset(data1_flat, data2_flat, 2, 2)
                    segment_method_B = search_segment_method_with_offset(tmpCSIb1Ind, tmpCSIa1Ind, 2, min_threshold)
                    print("search:", segment_method_A, segment_method_B)
                    print("min_threshold: ", min_threshold)
                    while len(segment_method_A) > len(segment_method_B):
                        print("length adjustment: shorten", segment_method_A, segment_method_B)
                        min_threshold -= 1
                        segment_method_B = search_segment_method_with_offset(tmpCSIb1Ind, tmpCSIa1Ind, 2, min_threshold)
                    while len(segment_method_A) < len(segment_method_B):
                        print("length adjustment: lengthen", segment_method_A, segment_method_B)
                        min_threshold += 1
                        segment_method_B = search_segment_method_with_offset(tmpCSIb1Ind, tmpCSIa1Ind, 2, min_threshold)
                    print("min_threshold after adjustment: ", min_threshold)
                    print("dataA", tmpCSIa1Bck)
                    print("dataB", tmpCSIb1Ind)
                    print("offset", find_offset(tmpCSIa1Bck, tmpCSIa1Ind), "segment", segment_method_A)
                    print("offset", find_offset(tmpCSIb1Ind, tmpCSIa1Ind), "segment", segment_method_B)
                    step = int(keyLen / segLen / 2) - 1
                    a_list_number = search_segment_method(tmpCSIa1Bck, tmpCSIa1Ind,
                                                          np.array(segment_method_A).astype(int), step, 0)
                    b_list_number = search_segment_method(tmpCSIb1Ind, tmpCSIa1Ind,
                                                          np.array(segment_method_B).astype(int), step, 0)
                    # a_list_number = search_segment_method_with_metric(tmpCSIa1Bck, tmpCSIa1Ind,
                    #                                                   np.array(segment_method_A).astype(int))
                    # b_list_number = search_segment_method_with_metric(tmpCSIb1Ind, tmpCSIa1Ind,
                    #                                                   np.array(segment_method_B).astype(int))
                    print("correct:", segment_method_A, segment_method_B)
                    print("a_list_number", a_list_number)
                    print("b_list_number", b_list_number)
                    print()
                elif segment_option == "find_search":
                    search_start = time.time_ns()
                    # 计算对应位置最大差距作为阈值
                    threshold = compute_threshold(tmpCSIa1Bck, tmpCSIb1Ind)
                    # threshold = 4
                    # 在阈值内匹配相近的索引
                    all_segments_A = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIa1Bck, threshold + 1)
                    all_segments_B = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIb1Ind, threshold + 1)
                    # 根据相近的索引组合成若干个子分段
                    # segments_A = find_segments(all_segments_A, 3, keyLen)
                    # segments_B = find_segments(all_segments_B, 3, keyLen)
                    segments_A = find_segments(all_segments_A, 2, keyLen)
                    segments_B = find_segments(all_segments_B, 2, keyLen)

                    # 根据子分段构成一个覆盖总分段长度的组合
                    # segment_method_A = find_all_cover_intervals_iter(
                    #     segments_A, (0, keyLen), 3, 5)
                    # segment_method_B = find_all_cover_intervals_iter(
                    #     segments_B, (0, keyLen), 3, 5)
                    # segment_method_A = find_all_cover_intervals_iter(segments_A, (0, keyLen), 2, 10)
                    # segment_method_B = find_all_cover_intervals_iter(segments_B, (0, keyLen), 2, 10)
                    segment_method_A = find_all_cover_intervals_iter_up(segments_A, (0, keyLen), 2)
                    segment_method_B = find_all_cover_intervals_iter_up(segments_B, (0, keyLen), 2)
                    if len(segment_method_B) == 0:
                        print("未找到合适分段1")
                        continue
                    # 根据子分段索引得到子分段长度
                    segment_length_A = get_segment_lengths(segment_method_A)
                    segment_length_B = get_segment_lengths(segment_method_B)
                    search_end = time.time_ns()

                    if isPrint:
                        print("sorted index", tmpCSIa1Bck, tmpCSIb1Ind)
                        print("published index", tmpCSIa1Ind)
                        print("threshold", threshold)
                        print("all_segments_A", all_segments_A)
                        print("all_segments_B", all_segments_B)
                        print("segments_A", segments_A)
                        print("segments_B", segments_B)
                        print("segment_length_A", segment_length_A)
                        print("segment_length_B", segment_length_B)

                    max_dist_A = np.inf
                    max_dist_B = np.inf
                    mean_dist_A = np.inf
                    mean_dist_B = np.inf
                    var_dist_A = np.inf
                    var_dist_B = np.inf
                    a_list_number = []
                    b_list_number = []
                    a_segment = []
                    b_segment = []

                    a_list_numbers = []
                    a_dists = []
                    a_segments = []
                    b_list_numbers = []
                    b_dists = []
                    b_segments = []

                    step = 3
                    threshold = 1
                    final_search_start = time.time_ns()
                    for i in range(len(segment_length_A)):
                        if len(segment_length_A[i]) != int(keyLen / segLen):
                            continue
                        # if len(segment_length_A[i]) < int(keyLen / segLen) - 1 or \
                        #         len(segment_length_A[i]) > int(keyLen / segLen) + 1:
                        #     continue
                        # if np.array_equal(np.sort(a_list_number), list(range(0, len(a_list_number)))) is False:
                        #     continue
                        a_list_number_tmp, dists = search_segment_method_with_metric(
                            tmpCSIa1Bck, tmpCSIa1Ind, np.array(segment_length_A[i]).astype(int), step, threshold)
                        a_list_numbers.append(a_list_number_tmp)
                        a_dists.append(dists)
                        a_segments.append(segment_length_A[i])
                        if isPrint:
                            print("a_list_number", a_list_number_tmp, "segment", segment_length_A[i],
                                  "max", np.max(dists), "mean", np.mean(dists), "var", np.var(dists), "dists", dists)
                    for i in range(len(segment_length_B)):
                        if len(segment_length_B[i]) != int(keyLen / segLen):
                            continue
                        # if len(segment_length_B[i]) < int(keyLen / segLen) - 1 or \
                        #         len(segment_length_B[i]) > int(keyLen / segLen) + 1:
                        #     continue
                        # if np.array_equal(np.sort(b_list_number), list(range(0, len(b_list_number)))) is False:
                        #     continue
                        b_list_number_tmp, dists = search_segment_method_with_metric(
                            tmpCSIb1Ind, tmpCSIa1Ind, np.array(segment_length_B[i]).astype(int), step, threshold)
                        b_list_numbers.append(b_list_number_tmp)
                        b_dists.append(dists)
                        b_segments.append(segment_length_B[i])
                        if isPrint:
                            print("b_list_number", b_list_number_tmp, "segment", segment_length_B[i],
                                  "max", np.max(dists), "mean", np.mean(dists), "var", np.var(dists), "dists", dists)

                    if len(b_segments) == 0:
                        print("未找到合适分段2")
                        continue
                    print("search space", len(a_segments), len(b_segments))
                    a_min_index = find_special_array(a_dists)[0][0]
                    a_list_number = a_list_numbers[a_min_index]
                    a_segment = a_segments[a_min_index]
                    b_min_index = find_special_array(b_dists)[0][0]
                    b_list_number = b_list_numbers[b_min_index]
                    b_segment = b_segments[b_min_index]
                    final_search_end = time.time_ns()

                    # if a_list_number != b_list_number:
                    #     key_hash = hash(tuple(a_list_number))
                    #
                    #     for i in range(len(segment_length_B)):
                    #         b_list_number_tmp, dists = search_segment_method_with_metric(
                    #             tmpCSIb1Ind, tmpCSIa1Ind, np.array(segment_length_B[i]).astype(int))
                    #         if hash(tuple(b_list_number_tmp)) == key_hash:
                    #             b_list_number = b_list_number_tmp
                    #             b_segment = segment_length_B[i]
                    #             break

                    # if a_list_number != b_list_number:
                    #     print("mismatch")
                    #     print("a_list_number", a_list_number, "a_segment", a_segment,
                    #           "b_list_number", b_list_number, "b_segment", b_segment)
                    #     # 找出第二小的分段
                    #     if len(segment_length_A) >= 2:
                    #         a_min_index = find_special_array(a_dists)[1][0]
                    #         a_list_number = a_list_numbers[a_min_index]
                    #         a_segment = segment_length_A[a_min_index]
                    #     if len(segment_length_B) >= 2:
                    #         b_min_index = find_special_array(b_dists)[1][0]
                    #         b_list_number = b_list_numbers[b_min_index]
                    #         b_segment = segment_length_B[b_min_index]
                    # if a_list_number != b_list_number:
                    #     # 密钥有重复的元素
                    #     if np.array_equal(np.sort(b_list_number), list(range(0, len(b_list_number)))) is False:
                    #         print("b_repeat")
                    #         max_dist_B = np.inf
                    #         mean_dist_B = np.inf
                    #         var_dist_B = np.inf
                    #
                    #         b_list_numbers = []
                    #         b_dists = []
                    #         b_segments = []
                    #         for i in range(len(segment_length_B)):
                    #             b_list_number_tmp, dists = search_segment_method_with_metric(
                    #                 tmpCSIb1Ind, tmpCSIa1Ind, np.array(segment_length_B[i]).astype(int))
                    #             if np.array_equal(np.sort(b_list_number_tmp),
                    #                               list(range(0, len(b_list_number_tmp)))) is False:
                    #                 continue
                    #             b_list_numbers.append(b_list_number_tmp)
                    #             b_dists.append(dists)
                    #             b_segments.append(segment_length_B[i])
                    #             print("b_list_number", b_list_number_tmp, "segment", segment_length_B[i],
                    #                   "max", np.max(dists), "mean", np.mean(dists), "dists", dists)
                    #             b_min_index = find_special_array(b_dists)[0][0]
                    #             b_list_number = b_list_numbers[b_min_index]
                    #             b_segment = b_segments[b_min_index]
                    #     if np.array_equal(np.sort(a_list_number), list(range(0, len(a_list_number)))) is False:
                    #         print("a_repeat")
                    #         max_dist_A = np.inf
                    #         mean_dist_A = np.inf
                    #         var_dist_A = np.inf
                    #
                    #         a_list_numbers = []
                    #         a_dists = []
                    #         a_segments = []
                    #         for i in range(len(segment_length_A)):
                    #             a_list_number_tmp, dists = search_segment_method_with_metric(
                    #                 tmpCSIa1Bck, tmpCSIa1Ind, np.array(segment_length_A[i]).astype(int))
                    #             if np.array_equal(np.sort(a_list_number_tmp),
                    #                               list(range(0, len(a_list_number_tmp)))) is False:
                    #                 continue
                    #
                    #             a_list_numbers.append(a_list_number_tmp)
                    #             a_dists.append(dists)
                    #             a_segments.append(segment_length_A[i])
                    #             print("a_list_number", a_list_number_tmp, "segment", segment_length_A[i],
                    #                   "max", np.max(dists), "mean", np.mean(dists), "dists", dists)
                    #             a_min_index = find_special_array(a_dists)[0][0]
                    #             a_list_number = a_list_numbers[a_min_index]
                    #             a_segment = a_segments[a_min_index]
                    #     # 仍然有重复的
                    #     print("a_list_number", a_list_number, "a_segment", a_segment,
                    #           "b_list_number", b_list_number, "b_segment", b_segment)
                    # if a_list_number != b_list_number:
                    #     if np.array_equal(np.sort(a_list_number), list(range(0, len(a_list_number)))) is False:
                    #         print("a also repeat")
                    #         a_segment = find_opt_segment_method_from_candidate(
                    #             tmpCSIa1Bck, segLen - 1, segLen + 1, int(keyLen / segLen))
                    #         a_list_number = search_segment_method(
                    #             tmpCSIa1Bck, tmpCSIa1Ind, np.array(a_segment).astype(int),
                    #             int(keyLen / segLen / 2) - 1, 0)
                    #     if np.array_equal(np.sort(b_list_number), list(range(0, len(b_list_number)))) is False:
                    #         print("b also repeat")
                    #         b_segment = find_opt_segment_method_from_candidate(
                    #             tmpCSIb1Ind, segLen - 1, segLen + 1, int(keyLen / segLen))
                    #         b_list_number = search_segment_method(
                    #             tmpCSIb1Ind, tmpCSIa1Ind, np.array(b_segment).astype(int),
                    #             int(keyLen / segLen / 2) - 1, 0)
                    # if a_list_number != b_list_number:
                    #     # A推导出来的和自己的次优分段个数不一致时
                    #     if len(segment_method_ori) != len(a_segment):
                    #         print("unequal length")
                    #         a_segment = segment_method_ori
                    #         a_list_number = search_segment_method(
                    #             tmpCSIa1Bck, tmpCSIa1Ind, np.array(a_segment).astype(int),
                    #             int(keyLen / segLen / 2) - 1, 0)
                    #
                    # if a_list_number != b_list_number:
                    #     key_hash = hash(tuple(a_list_number))
                    #
                    #     for i in range(len(segment_length_B)):
                    #         b_list_number_tmp, dists = search_segment_method_with_metric(
                    #             tmpCSIb1Ind, tmpCSIa1Ind, np.array(segment_length_B[i]).astype(int))
                    #         if hash(tuple(b_list_number_tmp)) == key_hash:
                    #             b_list_number = b_list_number_tmp
                    #             b_segment = segment_length_B[i]
                    #             break

                    print("a_list_number", a_list_number, "a_segment", a_segment,
                          "b_list_number", b_list_number, "b_segment", b_segment)
                    if np.array_equal(a_list_number, b_list_number) is False:
                        print("error")
                    print("time (s)", time.time() - start_time)
                    print("segment time (s)", (segment_end - segment_start) / 1e9,
                          "search time (s)", (search_end - search_start) / 1e9,
                          "final search time (s)", (final_search_end - final_search_start) / 1e9)
                    print()

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
                            epiIndClosenessLsb[j] = dtw_metric(epiIndb1, np.array(epiInda1))[0]
                            # epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))
                            # epiIndClosenessLsb[j] = distance.cosine(epiIndb1, np.array(epiInda1))
                            # epiIndClosenessLsb[j] = abs(sum(epiIndb1) - sum(np.array(epiInda1)))

                        minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)

                    a_list_number = list(permutation)
                    b_list_number = list(minEpiIndClosenessLsb)

                # 转成二进制，0填充成0000
                for i in range(len(a_list_number)):
                    number = bin(a_list_number[i])[2:].zfill(int(np.log2(len(a_list_number))))
                    a_list += number
                for i in range(len(b_list_number)):
                    number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
                    b_list += number

                sum1 = min(len(a_list), len(b_list))
                sum2 = 0

                for i in range(0, sum1):
                    sum2 += (a_list[i] == b_list[i])

                end = time.time()
                overhead += end - start
                # print("time:", end - start)

                # if sum1 != sum2:
                #     print()

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

                                for i in range(len(b_list_number)):
                                    number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
                                    b_list += number

                                sum2 = 0
                                for i in range(0, min(len(a_list), len(b_list))):
                                    sum2 += (a_list[i] == b_list[i])

                # print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
                originSum += sum1
                correctSum += sum2

                originWholeSum += 1
                correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum

            print("\033[0;34;40ma-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 9), "\033[0m")
            print("\033[0;34;40ma-b whole match", correctWholeSum, "/", originWholeSum, "=",
                  round(correctWholeSum / originWholeSum, 9), "\033[0m")

            print(round(correctSum / originSum, 9), round(correctWholeSum / originWholeSum, 9),
                  round(originSum / times / keyLen, 9),
                  round(correctSum / times / keyLen, 9))
            # 分段匹配的情况
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
            print("segment time", np.mean(segment_time))
            print("search time", np.mean(search_time))
            print("final search time", np.mean(final_search_time))
            print("\n")


