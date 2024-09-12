import copy
import heapq
import itertools
import math
import operator
import random
import sys
import time
from collections import defaultdict, OrderedDict, deque, Counter
from datetime import datetime

import numpy as np
from dtaidistance import dtw_ndim
from dtw import accelerated_dtw
from scipy.optimize import linear_sum_assignment

from segmentation.test_partition import partition
from segmentation.test_permutations import unique_permutations
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import minimize

from pyentrp import entropy as ent


# 一阶差分
def difference(data):
    res = []
    if np.ndim(data) == 1:
        res.append(0)
    for i in range(len(data) - 1):
        if np.ndim(data) == 1:
            res.append(data[i + 1] - data[i])
            # 精度更细
            # res.append(((data[i] - data[i - 1]) + (data[i + 1] - data[i - 1]) / 2) / 2)
        else:
            diff = []
            for j in range(len(data[i])):
                diff.append(data[i + 1][j] - data[i][j])
            res.append(diff)
    return res


# 定义计算离散点积分的函数
def integral(x, y):
    integrals = []
    integrals.append(0)
    for i in range(len(y) - 1):  # 计算梯形的面积
        integrals.append((y[i] + y[i + 1]) * (x[i + 1] - x[i]) / 2)
    return integrals


def diff_sq_integral_rough(data):
    index = list(range(len(data)))
    diff = difference(data)
    diff_diff = difference(diff)
    second = integral(index, np.array(data) * np.array(diff_diff))
    return np.array(diff) * np.array(data) - np.array(second)


def normal2uniform(data):
    data_reshape = np.array(data[0: 2 * int(len(data) / 2)])
    data_reshape = data_reshape.reshape(int(len(data_reshape) / 2), 2)
    x_list = []
    for i in range(len(data_reshape)):
        r = np.sum(np.square(data_reshape[i]))
        x_list.append(np.exp(-0.5 * r))

    return x_list


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
    if np.ndim(data1) == 1:
        distance = lambda x, y: np.abs(x - y)
        # ddtw
        # data1 = np.array(difference(data1))
        # data2 = np.array(difference(data2))
        # dtw
        data1 = np.array(data1)
        data2 = np.array(data2)
        # return dtw(data1, data2, dist=distance)[0]
        return accelerated_dtw(data1, data2, dist=distance)[0]
    else:
        # 二维DTW距离
        data1 = np.array(difference(data1))
        data2 = np.array(difference(data2))
        # data1 = np.array(data1)
        # data2 = np.array(data2)
        return dtw_ndim.distance(data1, data2)


# print(dtw_metric([[1, 1], [2, 2], [3, 3]], [[2, 2], [2, 4]]))


# def dtw_metric(data1, data2):
#     if np.ndim(data1) == 1:
#         distance = lambda x, y: np.abs(x - y)
#         data1 = np.array(data1)
#         data2 = np.array(data2)
#         # return dtw(data1, data2, dist=distance)[0]
#         return accelerated_dtw(data1, data2, dist=distance)[0]
#     else:
#         # 二维DTW距离
#         return dtw_ndim.distance(data1, data2)


def compute_max_index_dtw(data1, data2):
    # 相同索引下的最大差距
    max_dtw = -np.inf
    for i in range(len(data1)):
        max_dtw = max(max_dtw, dtw_metric(data1[i], data2[i]))
    return max_dtw


def compute_min_dtw(data1, data2):
    min_dtw = np.inf
    for i in range(len(data1)):
        for j in range(len(data2)):
            if i == j:
                continue
            min_dtw = min(min_dtw, dtw_metric(data1[i], data2[j]))
    return min_dtw


def compute_all_dtw(data1, data2):
    all_dtw = []
    for i in range(len(data1)):
        dtws = []
        for j in range(len(data2)):
            dtws.append(dtw_metric(data1[i], data2[j]))
        all_dtw.append(dtws)
    return all_dtw


def compute_max_min_euclidean(data1, data2):
    min_dist = - np.inf
    for i in range(len(data1)):
        current_min_dist = np.inf
        for j in range(len(data2)):
            if i == j:
                continue
            current_min_dist = min(current_min_dist, np.linalg.norm(np.array(data1[i]) - np.array(data2[j])))
        min_dist = max(min_dist, current_min_dist)
    return min_dist


def compute_min_euclidean(data1, data2):
    min_dist = np.inf
    for i in range(len(data1)):
        for j in range(len(data2)):
            if i == j:
                continue
            min_dist = min(min_dist, np.linalg.norm(np.array(data1[i]) - np.array(data2[j])))
    return min_dist


# def compute_min_dtw(data):
#     min_dtws = []
#     for i in range(len(data)):
#         min_dtw = np.inf
#         for j in range(i + 1, len(data)):
#             min_dtw = min(min_dtw, dtw_metric(data[i], data[j])])
#         min_dtws.append(min_dtw)
#     return np.mean(min_dtws)


def compute_threshold(data1, data2):
    # data1_sort = np.sort(data1)
    # data2_sort = np.sort(data2)
    data1_sort = data1
    data2_sort = data2
    threshold = 0
    for i in range(len(data1)):
        if np.ndim(data1) == 1:
            temp = abs(data1_sort[i] - data2_sort[i])
        else:
            # temp = dtw_ndim.distance(data1_sort[i], data2_sort[i])
            temp = np.sum(np.abs(data1_sort[i] - data2_sort[i]))
        if temp > threshold:
            threshold = temp
    return threshold


def find_opt_segment_method(data1, data2, min_length, num_segments):
    # 生成所有分段方法，计算分段间的最小DTW距离
    all_segments = partition(len(data1), min_length=min_length, num_partitions=num_segments)
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
        data_seg1 = segment_sequence(data1, segment_method)
        data_seg2 = segment_sequence(data2, segment_method)
        dist = compute_min_dtw(data_seg1, data_seg2)
        if dist > max_distance:
            max_distance = dist
            opt_data_seg = [segment_method]
        elif dist == max_distance:
            if segment_method not in opt_data_seg:
                opt_data_seg.append(segment_method)
    return opt_data_seg[0]


def find_opt_segment_method_cond(data1, data2, min_length, max_length, num_segments):
    # 生成所有分段方法，计算分段间的最小DTW距离
    all_segments = partition(len(data1), min_length=min_length, max_length=max_length, num_partitions=num_segments)
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
        data_seg1 = segment_sequence(data1, segment_method)
        data_seg2 = segment_sequence(data2, segment_method)
        dist = compute_min_dtw(data_seg1, data_seg2)
        if dist > max_distance:
            max_distance = dist
            opt_data_seg = [segment_method]
        elif dist == max_distance:
            if segment_method not in opt_data_seg:
                opt_data_seg.append(segment_method)
    return opt_data_seg[0]


# data1 = [11, 5, 2, 7, 10, 15, 0, 14, 13, 1, 4, 12, 3, 9, 8, 6]
# data2 = [11, 5, 2, 7, 10, 15, 0, 14, 13, 1, 4, 12, 3, 9, 8, 6]
# segment_method = [3, 5, 4, 4]
# data_seg1 = segment_sequence(data1, segment_method)
# data_seg2 = segment_sequence(data2, segment_method)
# print(compute_all_dtw(data_seg1, data_seg2))
#
# data1 = [11, 5, 2,
#          7, 10, 15, 0, 14,
#          13, 1, 4, 12,
#          3, 9, 8, 6]
# data2 = [13, 1, 4, 12,
#          3, 9, 8, 6,
#          11, 5, 2,
#          7, 10, 15, 0, 14]
# segment_method1 = [3, 5, 4, 4]
# segment_method2 = [4, 4, 3, 5]
# data_seg1 = segment_sequence(data1, segment_method1)
# data_seg2 = segment_sequence(data2, segment_method2)
# print(data_seg1)
# print(data_seg2)
# print(compute_all_dtw(data_seg1, data_seg2))
#
# data_seg2 = segment_sequence(data2, segment_method1)
# print(data_seg2)
# print(compute_all_dtw(data_seg1, data_seg2))

def find_opt_segment_method_from_candidate(data1, data2, min_length, max_length, num_segments):
    # 生成所有分段方法，计算分段间的最小DTW距离
    all_segments = partition(len(data1), min_length=min_length, max_length=max_length, num_partitions=num_segments)
    segment_methods = []
    for segment in all_segments:
        segment_methods += unique_permutations(segment)
    # 无置换
    # segment_methods = all_segments

    max_distance = -np.inf
    opt_data_seg = []
    for segment_method in segment_methods:
        data_seg1 = segment_sequence(data1, segment_method)
        data_seg2 = segment_sequence(data2, segment_method)
        dist = compute_min_dtw(data_seg1, data_seg2)
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


def find_sub_opt_segment_method_down(data1, data2, min_length, num_segments):
    cal_dtw_times = 0

    each_layer_candidate = [[[len(data1)]]]
    each_layer_optimal = [[[len(data1)]]]
    # 以下代码可能是只针对index或者value的
    # max_distance = [[-np.inf] for _ in range(len(data1) - num_segments + 1)]
    # 同时针对index和value的使用如下代码
    max_distance = [-np.inf for _ in range(len(data1) - num_segments + 1)]
    # 遍历每一层，即每个分段长度
    for layer in range(len(data1) - num_segments + 1):
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
                    segment1 = segment_sequence(data1, segment_method_temp)
                    segment2 = segment_sequence(data2, segment_method_temp)
                    cal_dtw_times += (len(segment1) * len(segment2))
                    distance_temp = compute_min_dtw(segment1, segment2)
                    # 找到了更优的分段方法，将其加入到新的分段方法中
                    if distance_temp > max_distance[layer]:
                        max_distance[layer] = distance_temp
                        opt_segment_method = [segment_method_temp]
                    # unique
                    # 找到了与最优分段方法相同的分段方法，将其加入到新的分段方法中
                    elif np.isclose(distance_temp, max_distance[layer], atol=1e-3):
                        if (segment_method_temp not in each_layer_candidate[layer]
                                and segment_method_temp not in opt_segment_method):
                            opt_segment_method.append(segment_method_temp)
        # 按照不同的分段个数放入此时最优的分段方法
        each_layer_optimal[layer + 1].extend(opt_segment_method)
    # print(max_distance[num_segments - 2])

    # print("dtw times:", cal_dtw_times)
    if each_layer_optimal[-1] != []:
        return each_layer_optimal[-1][0]
    else:
        return each_layer_optimal[-2][0]


def find_sub_opt_segment_method_down_cond(data1, data2, min_length, max_length, num_segments):
    # 估计的最大分段个数
    # num_segments = int(len(data1) / min_length)
    each_layer_candidate = [[[len(data1)]]]
    each_layer_optimal = [[[len(data1)]]]
    # 以下代码可能是只针对index或者value的
    # max_distance = [[-np.inf] for _ in range(len(data1) - num_segments + 1)]
    # 同时针对index和value的使用如下代码
    max_distance = [-np.inf for _ in range(len(data1) - num_segments + 1)]
    # 遍历每一层，即每个分段长度
    for layer in range(len(data1) - num_segments + 1):
        # 每层的最大距离
        segment_methods = each_layer_optimal[layer]
        # 未找到最优解或者找到最优解
        if len(segment_methods) == 0:
            break
        # 检查分段长度是否在范围内
        inRange = True
        for segment in segment_methods[0]:
            if segment > max_length or segment < min_length:
                inRange = False
                break
        if inRange:
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
                    segment1 = segment_sequence(data1, segment_method_temp)
                    segment2 = segment_sequence(data2, segment_method_temp)
                    distance_temp = compute_min_dtw(segment1, segment2)
                    # 找到了更优的分段方法，将其加入到新的分段方法中
                    if distance_temp > max_distance[layer]:
                        max_distance[layer] = distance_temp
                        opt_segment_method = [segment_method_temp]
                    # 注释以下内容则为找出unique的分段类型
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


# data1 = [11, 4, 12, 17, 23, 22, 27, 3, 7, 21, 24, 20, 30, 28, 19, 29, 9, 14, 8, 15, 0, 16, 10, 25, 26, 31, 2, 13, 5, 6,
#          18, 1]
# data1 = np.random.permutation(data1)
# segment = find_sub_opt_segment_method_down_cond(data1, data1, 3, 5, 8)
# print(len(segment), segment)


def new_segment_sequence(data, segment_lengths):
    segments = []
    for i in range(1, len(segment_lengths)):
        segments.append(data[segment_lengths[i - 1]:segment_lengths[i]])
    return segments


def find_sub_opt_segment_method_up(data1, data2, base_length, num_segments):
    cal_dtw_times = 0

    each_layer_candidate = [[[base_length for _ in range(int(len(data1) / base_length))]]]
    each_layer_optimal = [[[base_length for _ in range(int(len(data1) / base_length))]]]
    # 遍历每一层，即每个分段长度
    for layer in range(int(len(data1) / base_length) - num_segments):
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
                segment1 = segment_sequence(data1, segment_method_temp)
                segment2 = segment_sequence(data2, segment_method_temp)
                cal_dtw_times += (len(segment1) * len(segment2))
                distance_temp = compute_min_dtw(segment1, segment2)
                # 找到了更优的分段方法，将其加入到新的分段方法中
                if distance_temp > max_distance:
                    max_distance = distance_temp
                    opt_segment_method = [segment_method_temp]
                # 找到了与最优分段方法相同的分段方法，将其加入到新的分段方法中
                # 注释以下内容则为找出unique的分段类型
                elif np.isclose(distance_temp, max_distance, atol=1e-3):
                    if (segment_method_temp not in each_layer_candidate[layer]
                            and segment_method_temp not in opt_segment_method):
                        opt_segment_method.append(segment_method_temp)
        # 按照不同的分段个数放入此时最优的分段方法
        each_layer_optimal[layer + 1].extend(opt_segment_method)

    # print("dtw times:", cal_dtw_times)
    return each_layer_optimal[-1][0]


def compute_min_dtw_with_candidate(segments, i, j):
    # 当找到第i和第j分段之间dtw距离接近需要合并时，找出最小dtw最大的合并方式进行合并
    candidates = []
    if i + 1 < len(segments):
        new_segment = segments.copy()
        new_segment[i] = new_segment[i] + new_segment[i + 1]
        del new_segment[i + 1]
        candidates.append((compute_min_dtw(new_segment, new_segment), i, i + 1))
    if i > 0:
        new_segment = segments.copy()
        new_segment[i - 1] = new_segment[i - 1] + new_segment[i]
        del new_segment[i]
        candidates.append((compute_min_dtw(new_segment, new_segment), i - 1, i))
    if j + 1 < len(segments):
        new_segment = segments.copy()
        new_segment[j] = new_segment[j] + new_segment[j + 1]
        del new_segment[j + 1]
        candidates.append((compute_min_dtw(new_segment, new_segment), j, j + 1))
    if j > 0:
        new_segment = segments.copy()
        new_segment[j - 1] = new_segment[j - 1] + new_segment[j]
        del new_segment[j]
        candidates.append((compute_min_dtw(new_segment, new_segment), j - 1, j))
    # 合并完之后最小dtw距离最大的作为候选
    min_candidate = max(candidates, key=lambda x: x[0])
    return min_candidate


def compute_min_dtw_with_adjacent_candidate(segments1, segments2, i, j, max_length):
    # 当找到第i和第j分段之间dtw距离接近需要合并时，找出最小dtw的合并方式进行合并
    candidates = []
    if i + 1 < len(segments1) and len(segments1[i]) + len(segments1[i + 1]) <= max_length:
        candidates.append((dtw_metric(segments1[i], segments2[i + 1]), i, i + 1))
    if i > 0 and len(segments1[i - 1]) + len(segments1[i]) <= max_length:
        candidates.append((dtw_metric(segments1[i - 1], segments2[i]), i - 1, i))
    if j + 1 < len(segments1) and len(segments1[j]) + len(segments1[j + 1]) <= max_length:
        candidates.append((dtw_metric(segments1[j], segments2[j + 1]), j, j + 1))
    if j > 0 and len(segments1[j - 1]) + len(segments1[j]) <= max_length:
        candidates.append((dtw_metric(segments1[j - 1], segments2[j]), j - 1, j))
    # 合并完之后与之前接近的分段最小dtw距离的作为候选
    if len(candidates) == 0:
        return -1, -1, -1
    return min(candidates, key=lambda x: x[0])


def compute_min_dtw_with_single_candidate(segments1, segments2, i):
    # 当找到第i段长度过短时，与其左右分段具有最小dtw距离的合并
    candidates = []
    if i + 1 < len(segments1):
        candidates.append((dtw_metric(segments1[i], segments2[i + 1]), i, i + 1))
    if i > 0:
        candidates.append((dtw_metric(segments1[i - 1], segments2[i]), i - 1, i))
    return min(candidates, key=lambda x: x[0])


def find_sub_opt_segment_method_merge(data1, data2, max_length, num_segments, threshold=np.inf):
    # 初始每个元素为一个分段
    segments1 = [[elem] for elem in data1]
    segments2 = [[elem] for elem in data2]

    limited_segments = []
    while len(segments1) > num_segments:
        min_dist = float('inf')
        merge_idx = (-1, -1)

        # 找出距离小于阈值的最小距离及其对应的索引
        for i in range(len(segments1)):
            for j in range(i + 1, len(segments2)):
                if (i, j) not in limited_segments:
                    dist = dtw_metric(segments1[i], segments2[j])
                    if dist < min_dist:
                        min_dist = dist
                        merge_idx = (i, j)

        # 如果找到的最小距离大于阈值，则停止合并
        if min_dist >= threshold:
            break

        # 合并相邻的两个分段
        i, j = merge_idx
        dist, merge_i, merge_j = compute_min_dtw_with_adjacent_candidate(segments1, segments2, i, j, max_length)
        # 如果合并以后的分段长度大于max_length，则放入受限分段中不进行合并
        if dist == -1:
            limited_segments.append((i, j))
        else:
            segments1[merge_i] = segments1[merge_i] + segments1[merge_j]
            segments2[merge_i] = segments2[merge_i] + segments2[merge_j]
            del segments1[merge_j]
            del segments2[merge_j]

    segment_method = []
    for segment in segments1:
        segment_method.append(len(segment))
    return segment_method


def find_sub_opt_segment_method_merge_heap(data1, data2, max_length, num_segments, threshold=np.inf):
    cal_dtw_times = 0

    # 初始每个元素为一个分段
    segments1 = [[elem] for elem in data1]
    segments2 = [[elem] for elem in data2]

    # 创建初始的所有相邻分段对的距离堆
    heap = []
    for i in range(len(segments1)):
        for j in range(i + 1, len(segments2)):
            cal_dtw_times += 1
            dist = dtw_metric(segments1[i], segments2[j])
            heapq.heappush(heap, (dist, i, j))

    limited_segments = []
    while len(segments1) > num_segments:
        # 从堆中取出最小距离
        if len(heap) == 0:
            break

        min_dist, i, j = heapq.heappop(heap)
        if (i, j) in limited_segments:
            continue

        # 如果最小距离大于等于阈值，则停止合并
        if min_dist >= threshold:
            break

        # 合并相邻的两个分段
        cal_dtw_times += 4
        dist, merge_i, merge_j = compute_min_dtw_with_adjacent_candidate(segments1, segments2, i, j, max_length)
        if dist == -1:
            limited_segments.append((i, j))
        else:
            segments1[merge_i] = segments1[merge_i] + segments1[merge_j]
            segments2[merge_i] = segments2[merge_i] + segments2[merge_j]
            del segments1[merge_j]
            del segments2[merge_j]

        # 重新计算合并后的分段与所有其他分段的距离
        new_heap = []
        for k in range(len(segments1)):
            for l in range(k + 1, len(segments2)):
                cal_dtw_times += 1
                dist = dtw_metric(segments1[k], segments2[l])
                new_heap.append((dist, k, l))

        heapq.heapify(new_heap)
        heap = new_heap

    segment_method = [len(segment) for segment in segments1]

    # print("dtw times:", cal_dtw_times)
    return segment_method


def find_sub_opt_segment_method_merge_heap_fast(data1, data2, max_length, num_segments, threshold=np.inf, min_length=3):
    cal_dtw_times = 0

    # 初始每个元素为一个分段
    segments1 = [[elem] for elem in data1]
    segments2 = [[elem] for elem in data2]

    # 创建初始的所有分段对的距离堆
    distances = {}
    for i in range(len(segments1)):
        for j in range(i + 1, len(segments2)):
            cal_dtw_times += 1
            dist = dtw_metric(segments1[i], segments2[j])
            distances[(i, j)] = dist

    limited_segments = []
    while len(segments1) > num_segments:
        if len(distances) == 0:
            break

        # 按照dist排序，排序后python会将dict转为list
        distances = sorted(dict(distances).items(), key=operator.itemgetter(1))

        (i, j), min_dist = distances.pop(0)
        if (i, j) in limited_segments:
            continue

        # 如果最小距离大于等于阈值，则停止合并
        if min_dist >= threshold:
            break

        # 合并相邻的两个分段
        cal_dtw_times += 4
        dist, merge_s, merge_e = compute_min_dtw_with_adjacent_candidate(segments1, segments2, i, j, max_length)
        if dist == -1:
            limited_segments.append((i, j))
            continue
        segments1[merge_s] = segments1[merge_s] + segments1[merge_e]
        segments2[merge_s] = segments2[merge_s] + segments2[merge_e]
        del segments1[merge_e]
        del segments2[merge_e]

        # 删除旧的距离信息
        for i in range(len(distances) - 1, -1, -1):
            # 起始分段的首
            index1 = distances[i][0][0]
            # 结束分段的首
            index2 = distances[i][0][1]
            if merge_s == index1 or merge_s == index2 or merge_e == index1 or merge_e == index2:
                del distances[i]
            else:
                dist = distances[i][1]
                if index1 > merge_s:
                    index1 -= 1
                if index2 > merge_s:
                    index2 -= 1
                distances[i] = ((index1, index2), dist)

        # 更新堆中的距离，重新计算受影响的分段
        for i in range(len(segments1)):
            if i == merge_s:
                continue
            cal_dtw_times += 1
            dist = dtw_metric(segments1[merge_s], segments2[i])
            distances.append(((merge_s, i), dist))

    # 将短于最小长度限制的分段于其前/后分段合并
    for i in range(len(segments1) - 1, -1, -1):
        if len(segments1[i]) < min_length:
            cal_dtw_times += 2
            dist, merge_s, merge_e = compute_min_dtw_with_single_candidate(segments1, segments2, i)
            segments1[merge_s] = segments1[merge_s] + segments1[merge_e]
            segments2[merge_s] = segments2[merge_s] + segments2[merge_e]
            del segments1[merge_e]
            del segments2[merge_e]

    segment_method = [len(segment1) for segment1 in segments1]

    # print("dtw times:", cal_dtw_times)
    return segment_method


def find_sub_opt_segment_method_sliding(data1, data2, min_len, max_len, threshold=np.inf):
    cal_dtw_times = 0

    segments = []
    n = len(data1)
    i = 0

    # 随机选择第一个分段的长度
    random.seed(100000)
    first_segment_len = random.randint(min_len, max_len)
    segments.append(data1[:first_segment_len])
    i += first_segment_len

    while i < n:
        segment_found = False
        max_min_distance = 0
        best_segment = None

        # 尝试在规定的区间内找到一个合适的分段长度
        for length in range(min_len, max_len + 1):
            if i + length > n:
                break  # 超出数组范围

            new_segment = data2[i:i + length]
            min_distance = float('inf')

            # 计算新分段与之前所有分段的最小DTW距离
            for segment in segments:
                cal_dtw_times += 1
                distance = dtw_metric(segment, new_segment)
                if distance < threshold:
                    min_distance = distance

            # 选择最小DTW距离最大的分段
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_segment = new_segment

        if best_segment is not None:
            segments.append(best_segment)
            i += len(best_segment)
            segment_found = True

        # 如果在规定区间内没有找到合适的分段长度，则将剩余部分作为最后一个分段
        if not segment_found:
            segments.append(data1[i:])
            break

    # 处理最后一个分段的长度
    if len(segments[-1]) < min_len:
        segments.pop()

    segment_method = []
    for segment in segments:
        segment_method.append(len(segment))

    # print("dtw times:", cal_dtw_times)
    return segment_method


def find_sub_opt_segment_method_sliding_threshold(data1, data2, min_len, max_len, threshold):
    cal_dtw_times = 0

    segments = []
    n = len(data1)
    i = 0

    # 随机选择第一个分段的长度
    random.seed(100000)
    first_segment_len = random.randint(min_len, max_len)
    segments.append(data1[:first_segment_len])
    i += first_segment_len

    while i < n:
        # 尝试在规定的区间内找到一个合适的分段长度
        segment_found = False
        for length in range(min_len, max_len + 1):
            if i + length > n:
                break  # 超出数组范围

            new_segment = data2[i:i + length]
            valid = True

            # 检查新分段与之前所有分段的DTW距离
            for segment in segments:
                cal_dtw_times += 1
                distance = dtw_metric(segment, new_segment)
                if distance <= threshold:
                    valid = False
                    break

            if valid:
                segments.append(new_segment)
                i += length
                segment_found = True
                break

        # 如果在规定区间内没有找到合适的分段长度，则将剩余部分作为最后一个分段
        if not segment_found:
            # segments.append(data[i:])
            while i < n:
                remaining_length = n - i
                if remaining_length <= max_len:
                    if remaining_length >= min_len:
                        segments.append(data1[i:])
                    else:
                        if segments:
                            previous_segment = segments.pop()
                            combined_segment = list(previous_segment)
                            combined_segment.extend(data1[i:])
                            if len(combined_segment) <= max_len:
                                segments.append(combined_segment)
                            else:
                                segments.append(previous_segment)
                        else:
                            segments.append(data1[i:])
                    break
                else:
                    if np.ndim(data1) == 1:
                        np.random.seed(int(data1[i]))
                    else:
                        np.random.seed(int(data1[i][0]))
                    segment_length = np.random.randint(min_len, max_len + 1)
                    segments.append(data1[i:i + segment_length])
                    i += segment_length
            break

    # 处理最后一个分段的长度
    if len(segments[-1]) < min_len:
        segments.pop()

    segment_method = []
    for segment in segments:
        segment_method.append(len(segment))

    # print("dtw times:", cal_dtw_times)
    return segment_method


# np.random.seed(8)
# tmpCSIa1Ind = np.random.permutation(4 * 8)
# num_segments = 8
# max_length = 5
# min_length = 3
#
# start_time = time.time_ns()
# segment_method = find_opt_segment_method(tmpCSIa1Ind, tmpCSIa1Ind, min_length   , num_segments)
# print("time (ms):", (time.time_ns() - start_time) / 1e6)
# print(len(segment_method), segment_method)
# print(compute_min_dtw(segment_sequence(tmpCSIa1Ind, segment_method), segment_sequence(tmpCSIa1Ind, segment_method)))
# print()
#
# start_time = time.time_ns()
# segment_method = find_opt_segment_method_cond(tmpCSIa1Ind, tmpCSIa1Ind, min_length, max_length, num_segments)
# print("time (ms):", (time.time_ns() - start_time) / 1e6)
# print(len(segment_method), segment_method)
# print(compute_min_dtw(segment_sequence(tmpCSIa1Ind, segment_method), segment_sequence(tmpCSIa1Ind, segment_method)))
# print()
#
# start_time = time.time_ns()
# segment_method = find_sub_opt_segment_method_down(tmpCSIa1Ind, tmpCSIa1Ind, min_length, num_segments)
# print("time (ms):", (time.time_ns() - start_time) / 1e6)
# print(len(segment_method), segment_method)
# print(compute_min_dtw(segment_sequence(tmpCSIa1Ind, segment_method), segment_sequence(tmpCSIa1Ind, segment_method)))
# print()
#
# start_time = time.time_ns()
# segment_method = find_sub_opt_segment_method_up(tmpCSIa1Ind, tmpCSIa1Ind, 2, num_segments)
# print("time (ms):", (time.time_ns() - start_time) / 1e6)
# print(len(segment_method), segment_method)
# print(compute_min_dtw(segment_sequence(tmpCSIa1Ind, segment_method), segment_sequence(tmpCSIa1Ind, segment_method)))
# print()
#
# start_time = time.time_ns()
# segment_method = find_sub_opt_segment_method_merge(tmpCSIa1Ind, tmpCSIa1Ind, max_length, num_segments)
# print("time (ms):", (time.time_ns() - start_time) / 1e6)
# print(len(segment_method), segment_method)
# print(compute_min_dtw(segment_sequence(tmpCSIa1Ind, segment_method), segment_sequence(tmpCSIa1Ind, segment_method)))
# print()
#
# start_time = time.time_ns()
# segment_method = find_sub_opt_segment_method_merge_heap(tmpCSIa1Ind, tmpCSIa1Ind, max_length, num_segments)
# print("time (ms):", (time.time_ns() - start_time) / 1e6)
# print(len(segment_method), segment_method)
# print(compute_min_dtw(segment_sequence(tmpCSIa1Ind, segment_method), segment_sequence(tmpCSIa1Ind, segment_method)))
# print()
#
# start_time = time.time_ns()
# segment_method = find_sub_opt_segment_method_merge_heap_fast(tmpCSIa1Ind, tmpCSIa1Ind, max_length, num_segments, min_length)
# print("time (ms):", (time.time_ns() - start_time) / 1e6)
# print(len(segment_method), segment_method)
# print(compute_min_dtw(segment_sequence(tmpCSIa1Ind, segment_method), segment_sequence(tmpCSIa1Ind, segment_method)))
# print()
#
# start_time = time.time_ns()
# segment_method = find_sub_opt_segment_method_sliding(tmpCSIa1Ind, tmpCSIa1Ind, min_length, max_length, num_segments)
# print("time (ms):", (time.time_ns() - start_time) / 1e6)
# print(len(segment_method), segment_method)
# print(compute_min_dtw(segment_sequence(tmpCSIa1Ind, segment_method), segment_sequence(tmpCSIa1Ind, segment_method)))
# print()
#
# start_time = time.time_ns()
# segment_method = find_sub_opt_segment_method_sliding_threshold(tmpCSIa1Ind, tmpCSIa1Ind, min_length, max_length, 10)
# print("time (ms):", (time.time_ns() - start_time) / 1e6)
# print(len(segment_method), segment_method)
# print(compute_min_dtw(segment_sequence(tmpCSIa1Ind, segment_method), segment_sequence(tmpCSIa1Ind, segment_method)))
# print()


def find_sub_opt_shuffle_method_sliding_threshold(data1, data2, segment_length, indices, threshold=np.inf):
    cal_times = 0

    segments = []
    shuffle_methods = []
    n = len(data1)
    i = 0

    # 固定第一个分段
    # segments.append(data1[:segment_length])
    # i += segment_length
    # shuffle_methods.append(list(range(segment_length)))

    # 第一个分段随机选择
    np.random.seed(int(np.sum(data1) + np.sum(data2)))
    index_of_1 = np.random.permutation(list(range(segment_length)))
    segments.append(list(data1[:segment_length][list(index_of_1)]))
    i += segment_length
    shuffle_methods.append(list(index_of_1))

    while i < n:
        # 尝试找到一个合适的置换
        segment2 = data2[i:i + segment_length]
        best_shuffle = None

        # 检查新分段与之前所有分段的欧式距离，搜索所有的可能的置换结果，从中选一个距离小于阈值的
        for j in range(len(indices)):
            valid = True
            new_segment = list(segment2[list(indices[j])])
            for segment in segments:
                cal_times += 1
                distance = np.linalg.norm(np.array(segment) - np.array(new_segment))
                if distance < threshold:
                    valid = False
                    break

            if valid:
                best_shuffle = list(indices[j])
                segments.append(list(data1[i:i + segment_length][best_shuffle]))
                shuffle_methods.append(best_shuffle)
                i += segment_length
                break

        if best_shuffle is None:
            segments.append(list(data1[i:i + segment_length]))
            shuffle_methods.append(list(range(segment_length)))
            i += segment_length

    # print("cal times:", cal_times)
    return shuffle_methods


def find_sub_opt_shuffle_method_sliding_threshold_random(data1, data2, segment_length, indices, threshold=np.inf):
    # 置换顺序更加随机
    cal_times = 0

    segments = []
    shuffle_methods = []
    n = len(data1)
    i = 0

    # 第一个分段随机选择
    np.random.seed(int(np.sum(data1) + np.sum(data2)))
    index_of_1 = np.random.permutation(list(range(segment_length)))
    segments.append(list(data1[:segment_length][list(index_of_1)]))
    i += segment_length
    shuffle_methods.append(list(index_of_1))

    while i < n:
        # 尝试找到一个合适的置换
        segment2 = data2[i:i + segment_length]
        best_shuffle = None

        # 检查新分段与之前所有分段的欧式距离，搜索所有的可能的置换结果，从中选一个距离小于阈值的
        # 打乱总置换候选
        np.random.seed(int(np.sum(segment2) * 100))
        indices = np.random.permutation(indices)
        for j in range(len(indices)):
            valid = True
            new_segment = list(segment2[list(indices[j])])
            for segment in segments:
                cal_times += 1
                distance = np.linalg.norm(np.array(segment) - np.array(new_segment))
                if distance < threshold:
                    valid = False
                    break

            if valid:
                best_shuffle = list(indices[j])
                segments.append(list(data1[i:i + segment_length][best_shuffle]))
                shuffle_methods.append(best_shuffle)
                i += segment_length
                break

        if best_shuffle is None:
            # 随机置换
            np.random.seed(int(np.sum(segment2)) + 1)
            index_of_1 = np.random.permutation(list(range(segment_length)))
            segments.append(list(data1[i:i + segment_length][list(index_of_1)]))
            shuffle_methods.append(list(index_of_1))
            i += segment_length

    # print("cal times:", cal_times)
    return shuffle_methods


def find_sub_opt_shuffle_method_sliding_threshold_even(data1, data2, segment_length, indices, indices_counters,
                                                       threshold=np.inf):
    # 置换顺序更加随机
    cal_times = 0

    segments = []
    shuffle_methods = []
    n = len(data1)
    i = 0

    # 第一个分段随机选择
    most_common_indices = []
    most_common_indices_counter = 0
    for key, value in indices_counters.items():
        most_common_indices_counter = max(value, most_common_indices_counter)
    for key, value in indices_counters.items():
        if value == most_common_indices_counter:
            most_common_indices.append(key)

    trial = 0
    while True:
        inCommon = True
        # np.random.seed(trial + int(np.sum(data1[:segment_length]) * 100))
        np.random.seed(trial)
        trial += 1
        # index_of_1必须不在常用分段中
        index_of_1 = np.random.permutation(list(range(segment_length)))
        # 如果所有的置换类型次数都一样则说明是首次运行，跳过
        if len(most_common_indices) == math.factorial(segment_length):
            break
        for common_index in most_common_indices:
            if np.array_equal(index_of_1, common_index):
                inCommon = False
        if inCommon:
            break
    segments.append(list(data1[:segment_length][list(index_of_1)]))
    i += segment_length
    shuffle_methods.append(list(index_of_1))

    if indices_counters.get(tuple(index_of_1)) is None:
        indices_counters[tuple(index_of_1)] = 1
    else:
        indices_counters[tuple(index_of_1)] += 1

    while i < n:
        # 尝试找到一个合适的置换
        segment2 = data2[i:i + segment_length]
        best_shuffle = None

        # 检查新分段与之前所有分段的欧式距离，搜索所有的可能的置换结果，从中选一个距离小于阈值的
        # 打乱总置换候选
        np.random.seed(int(np.sum(segment2) * 100) % (2 ** 32 - 1))
        # np.random.seed(int(np.sum(segment2) * 100))
        indices = np.random.permutation(indices)
        # 再次寻找最多的置换类型，以免第一个分段选择的置换类型成为了最多的置换类型
        most_common_indices = []
        most_common_indices_counter = 0
        for key, value in indices_counters.items():
            most_common_indices_counter = max(value, most_common_indices_counter)
        for key, value in indices_counters.items():
            if value == most_common_indices_counter:
                most_common_indices.append(key)
        # 从所有可选的置换类型indices中删除最常用的置换类型，然后从剩下的置换类型中选择可用的置换类型
        new_indices = copy.deepcopy(indices)
        for common_index in most_common_indices:
            for ni in range(len(new_indices) - 1, -1, -1):
                if np.array_equal(new_indices[ni], common_index):
                    new_indices = np.delete(new_indices, ni, 0)
        for j in range(len(new_indices)):
            valid = True
            new_segment = list(segment2[list(new_indices[j])])
            for segment in segments:
                cal_times += 1
                distance = np.linalg.norm(np.array(segment) - np.array(new_segment))
                if distance < threshold:
                    valid = False
                    break

            if valid:
                best_shuffle = list(new_indices[j])
                segments.append(list(data1[i:i + segment_length][best_shuffle]))
                shuffle_methods.append(best_shuffle)
                i += segment_length
                if indices_counters.get(tuple(best_shuffle)) is None:
                    indices_counters[tuple(best_shuffle)] = 1
                else:
                    indices_counters[tuple(best_shuffle)] += 1
                break

        if best_shuffle is None:
            most_uncommon_indices = []
            most_uncommon_indices_counter = np.inf
            for key, value in indices_counters.items():
                most_uncommon_indices_counter = min(value, most_uncommon_indices_counter)
            for key, value in indices_counters.items():
                if value == most_uncommon_indices_counter:
                    most_uncommon_indices.append(key)
            # 从最不常用的置换类型中随机选一个
            np.random.seed(int(np.sum(segment2) * 200) % (2 ** 32 - 1))
            # np.random.seed(int(np.sum(segment2) * 200))
            most_uncommon_index = list(most_uncommon_indices[np.random.randint(0, len(most_uncommon_indices))])
            segments.append(list(data1[i:i + segment_length][list(most_uncommon_index)]))
            shuffle_methods.append(list(most_uncommon_index))
            if indices_counters.get(tuple(most_uncommon_index)) is None:
                indices_counters[tuple(most_uncommon_index)] = 1
            else:
                indices_counters[tuple(most_uncommon_index)] += 1
            i += segment_length

    # print("cal times:", cal_times)
    return shuffle_methods, indices_counters


def find_sub_opt_shuffle_method_sliding(data1, data2, segment_length, indices):
    cal_times = 0

    segments = []
    shuffle_methods = []
    n = len(data1)
    i = 0

    # 第一个分段随机选择
    np.random.seed(int(np.sum(data1) + np.sum(data2)))
    index_of_1 = np.random.permutation(list(range(segment_length)))
    segments.append(list(data1[:segment_length][list(index_of_1)]))
    i += segment_length
    shuffle_methods.append(list(index_of_1))

    while i < n:
        # 尝试找到一个合适的置换
        segment2 = data2[i:i + segment_length]
        max_min_distance = 0
        best_shuffle = None

        # 检查新分段与之前所有分段的欧式距离，搜索所有的可能的置换结果，从中选一个最小距离最大的
        # indices进行随机置换不影响结果，因为所有的indices都要搜索一遍
        # np.random.seed(int(np.sum(segment2)))
        # indices = np.random.permutation(indices)
        for j in range(len(indices)):
            new_segment = list(segment2[list(indices[j])])
            min_distance = float('inf')
            for segment in segments:
                cal_times += 1
                distance = np.linalg.norm(np.array(segment) - np.array(new_segment))
                if distance < min_distance:
                    min_distance = distance

            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_shuffle = list(indices[j])

        segments.append(list(data1[i:i + segment_length][best_shuffle]))
        shuffle_methods.append(best_shuffle)
        i += segment_length

    # print("cal times:", cal_times)
    return shuffle_methods


def find_all_sub_opt_shuffle_method_sliding(data1, data2, segment_length, indices):
    # 第一个分段进行穷举搜索
    all_min_distance = np.inf
    final_shuffle_methods = []
    for index in indices:
        cal_times = 0

        segments = []
        shuffle_methods = []
        n = len(data1)
        i = 0

        # 第一个分段
        segments.append(list(data1[:segment_length][list(index)]))
        i += segment_length
        shuffle_methods.append(list(index))

        while i < n:
            # 尝试找到一个合适的置换
            segment2 = data2[i:i + segment_length]
            max_min_distance = 0
            best_shuffle = None

            # 检查新分段与之前所有分段的欧式距离，搜索所有的可能的置换结果，从中选一个最小距离最大的
            for j in range(len(indices)):
                new_segment = list(segment2[list(indices[j])])
                min_distance = float('inf')
                for segment in segments:
                    cal_times += 1
                    distance = np.linalg.norm(np.array(segment) - np.array(new_segment))
                    if distance < min_distance:
                        min_distance = distance

                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_shuffle = list(indices[j])

            segments.append(list(data1[i:i + segment_length][best_shuffle]))
            shuffle_methods.append(best_shuffle)
            i += segment_length

        # print("cal times:", cal_times)
        data1_reshape = np.array(data1).reshape(int(len(data1) / segment_length), segment_length, np.ndim(data1))
        shuffled_data1 = [list(data1_reshape[i][list(shuffle_methods[i])]) for i in range(len(data1_reshape))]
        current_min_distance = compute_min_euclidean(shuffled_data1, shuffled_data1)
        if current_min_distance < all_min_distance:
            all_min_distance = current_min_distance
            final_shuffle_methods = shuffle_methods
    return final_shuffle_methods


# num_segments = 4
# np.random.seed(10)
# tmpCSIa1Bck = np.random.permutation(4 * num_segments)
#
# # 计算所有置换也很耗时
# indices = list(itertools.permutations(range(4)))
# start_time = time.time_ns()
# shuffle_method = find_sub_opt_shuffle_method_sliding_threshold(tmpCSIa1Bck, tmpCSIa1Bck, 4, indices, 30)
# print("time (ms):", (time.time_ns() - start_time) / 1e6)
# tmpCSIa1Ind = np.array(tmpCSIa1Bck).reshape(num_segments, 4)
# shuffled_data = [list(tmpCSIa1Ind[i][list(shuffle_method[i])]) for i in range(len(tmpCSIa1Ind))]
# print("shuffle_method", shuffle_method)
# print(compute_min_dtw(shuffled_data, shuffled_data))
# print()
#
# np.random.seed(10)
# start_time = time.time_ns()
# shuffle_method = find_all_sub_opt_shuffle_method_sliding(tmpCSIa1Bck, tmpCSIa1Bck, 4, indices)
# print("time (ms):", (time.time_ns() - start_time) / 1e6)
# tmpCSIa1Ind = np.array(tmpCSIa1Bck).reshape(num_segments, 4)
# shuffled_data = [list(tmpCSIa1Ind[i][list(shuffle_method[i])]) for i in range(len(tmpCSIa1Ind))]
# print("shuffle_method", shuffle_method)
# print(compute_min_dtw(shuffled_data, shuffled_data))
# print()

#
# tmpCSIa1Ind = [list(np.random.permutation(segment)) for segment in tmpCSIa1Ind]
# print(tmpCSIa1Ind)
# print(compute_min_dtw(tmpCSIa1Ind, tmpCSIa1Ind))
# print()


def min_total_euclidean_distance(A, B):
    # 找到欧式距离最小的对应数组(匈牙利算法)
    n = len(A)
    cost_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            cost_matrix[i][j] = np.linalg.norm(np.array(A[i]) - np.array(B[j]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    min_distance = cost_matrix[row_ind, col_ind].sum()

    return row_ind, col_ind, min_distance


def find_best_matching_pair(arrays1, arrays2):
    best_indices = []
    shuffle_methods = []

    for i, A in enumerate(arrays1):
        best_distance = float('inf')
        current_best = []
        shuffle_method = []
        for j, B in enumerate(arrays2):
            row_ind, col_ind, distance = min_total_euclidean_distance(A, B)
            if distance < best_distance:
                best_distance = distance
                current_best = [i, j]
                shuffle_method = list(col_ind)
        best_indices.append(current_best)
        shuffle_methods.append(shuffle_method)

    return best_indices, shuffle_methods


def find_best_matching_pair_filter(arrays1, arrays2):
    best_indices = []
    shuffle_methods = []
    used_indices = []

    for i, A in enumerate(arrays1):
        best_distance = float('inf')
        current_best = []
        shuffle_method = []
        for j, B in enumerate(arrays2):
            # 滤重
            if j in used_indices:
                continue
            row_ind, col_ind, distance = min_total_euclidean_distance(A, B)
            if distance < best_distance:
                best_distance = distance
                current_best = [i, j]
                shuffle_method = list(row_ind)
        best_indices.append(current_best)
        used_indices.append(current_best[1])
        shuffle_methods.append(shuffle_method)

    return best_indices, shuffle_methods


def find_best_matching_pair_shuffled(arrays1, arrays2):
    # 子分段已经置换，进行匹配
    best_indices = []

    for i, A in enumerate(arrays1):
        best_distance = float('inf')
        current_best = []
        for j, B in enumerate(arrays2):
            distance = np.linalg.norm(np.array(A) - np.array(B))
            if distance < best_distance:
                best_distance = distance
                current_best = [i, j]
        best_indices.append(current_best[1])

    return best_indices


# def find_best_matching_pair_shuffled(arrays1, arrays2):
#     row_ind, col_ind, distance = min_total_euclidean_distance(arrays1, arrays2)
#     return col_ind


def find_all_matching_pairs_threshold(arrays1, arrays2, threshold):
    best_indices = []
    shuffle_methods = []

    for i, A in enumerate(arrays1):
        current_best = []
        shuffle_method = []
        for j, B in enumerate(arrays2):
            row_ind, col_ind, distance = min_total_euclidean_distance(A, B)
            if distance < threshold:
                current_best.append([i, j])
                shuffle_method.append(list(col_ind))

        if len(current_best) == 0:
            row_ind, col_ind, distance = min_total_euclidean_distance(A, arrays2[0])
            current_best.append([i, 0])
            shuffle_method.append(list(col_ind))
        best_indices.append(current_best)
        shuffle_methods.append(shuffle_method)

    return best_indices, shuffle_methods


def calculate_distances(A, B, pairs):
    distances = []
    for i, j in pairs:
        distance = np.linalg.norm(np.array(A[i]) - np.array(B[j]))
        distances.append(distance)
    return sum(distances)


def combine_indices_iteratively(best_indices):
    # 从所有置换顺序中组合成所有的置换顺序，例如[[[1,2],[3,4]], [[0,1]]] -> [[[1,2],[0,1]], [[3,4],[0,1]]]
    queue = deque([[]])
    all_combinations = []

    for pairs in best_indices:
        for _ in range(len(queue)):
            current_combination = queue.popleft()
            for pair in pairs:
                new_combination = current_combination + [pair]
                queue.append(new_combination)

    while queue:
        all_combinations.append(queue.popleft())

    return all_combinations


def inverse_permutation(permutation):
    n = len(permutation)
    inverse = [0] * n

    for i, p in enumerate(permutation):
        inverse[p] = i

    return inverse


# # 示例使用
# best_indices = [
#     [[0, 2]],
#     [[1, 0], [1, 1], [1, 3]],
#     [[2, 0], [2, 1], [2, 3]],
#     [[3, 0], [3, 1], [3, 3]]
# ]
#
# result = combine_indices_iteratively(best_indices)
# for idx, combination in enumerate(result):
#     print(f"Combination {idx + 1}: {combination}")

def find_best_matching_pair_threshold(arrays1, arrays2, threshold):
    best_indices = []
    shuffle_methods = []

    for i, A in enumerate(arrays1):
        current_best = []
        shuffle_method = []
        find_best = False
        for j, B in enumerate(arrays2):
            row_ind, col_ind, distance = min_total_euclidean_distance(A, B)
            if distance < threshold:
                current_best = [i, j]
                shuffle_method = list(row_ind)
                find_best = True
                break
        if find_best:
            best_indices.append(current_best)
            shuffle_methods.append(shuffle_method)
        else:
            row_ind, col_ind, distance = min_total_euclidean_distance(A, arrays2[0])
            best_indices.append([i, 0])
            shuffle_methods.append(row_ind)

    return best_indices, shuffle_methods


def find_best_matching_pair_threshold_filter(arrays1, arrays2, threshold):
    best_indices = []
    shuffle_methods = []
    used_indices = []

    for i, A in enumerate(arrays1):
        current_best = []
        shuffle_method = []
        find_best = False
        for j, B in enumerate(arrays2):
            # 滤重
            if j in used_indices:
                continue
            row_ind, col_ind, distance = min_total_euclidean_distance(A, B)
            if distance < threshold:
                current_best = [i, j]
                shuffle_method = list(row_ind)
                find_best = True
                break
        if find_best:
            best_indices.append(current_best)
            shuffle_methods.append(shuffle_method)
            used_indices.append(current_best[1])
        else:
            for j, B in enumerate(arrays2):
                # 滤重
                if j in used_indices:
                    continue
                row_ind, col_ind, distance = min_total_euclidean_distance(A, B)
                current_best = [i, j]
                shuffle_method = list(row_ind)

            best_indices.append(current_best)
            shuffle_methods.append(shuffle_method)
            used_indices.append(current_best[1])

    return best_indices, shuffle_methods


# # 生成两组 4x4 的数组
# arrays1 = np.random.randint(0, 5, (4, 4))
# arrays2 = arrays1 + np.random.normal(0, 0.5, (4, 4))
# arrays2 = np.random.permutation(arrays2)
#
# # 找出欧式距离最小的对应数组
# best_indices, shuffle_methods = find_best_matching_pair(arrays1, arrays2)
#
# print(best_indices, shuffle_methods)

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
            # dists[-1].append(dtw_metric(segmentA[i], segmentB[j]))
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


def search_segment_method_with_min_match_dtw(dataA, dataB, segments):
    # 给定分段长度，通过搜索找到分段位置，并返回该分段位置下的距离（最小距离）
    segmentA = segment_data(dataA, len(segments), segments)
    segmentB = segment_data(dataB, len(segments), segments)
    dists = []
    est_index = []
    min_dists = []
    for i in range(len(segmentA)):
        dists.append([])
        for j in range(len(segmentB)):
            # dists[-1].append(step_corr_metric(segmentA[i], segmentB[j], step, threshold))
            dists[-1].append(dtw_metric(segmentA[i], segmentB[j]))
        est_index.append(np.argmin(dists[-1]))
        min_dists.append(min(dists[-1]))
    return est_index, min_dists


def search_segment_method_with_min_cross_dtw(dataA, dataB, segments):
    # 给定分段长度，通过搜索找到分段位置，并返回该分段位置下的距离（排除最小距离的最小距离，即第二小距离）
    segmentA = segment_data(dataA, len(segments), segments)
    segmentB = segment_data(dataB, len(segments), segments)
    dists = []
    est_index = []
    min_dists = []
    for i in range(len(segmentA)):
        dists.append([])
        for j in range(len(segmentB)):
            dists[-1].append(dtw_metric(segmentA[i], segmentB[j]))
        est_index.append(np.argmin(dists[-1]))
        # 排除最小的距离
        min_dists.append(min(np.delete(dists[-1], est_index[-1])))
    return est_index, min_dists


def search_segment_method_with_min_dtw(dataA, dataB, segments):
    # 给定分段长度，通过搜索找到分段位置，并返回该分段位置下的距离（最小距离）
    segmentA = segment_data(dataA, len(segments), segments)
    segmentB = segment_data(dataB, len(segments), segments)
    dists = []
    est_index = []
    match_min_dists = []
    cross_min_dists = []
    for i in range(len(segmentA)):
        dists.append([])
        for j in range(len(segmentB)):
            # dists[-1].append(step_corr_metric(segmentA[i], segmentB[j], step, threshold))
            dists[-1].append(dtw_metric(segmentA[i], segmentB[j]))
        est_index.append(np.argmin(dists[-1]))
        match_min_dists.append(min(dists[-1]))
        cross_min_dists.append(min(np.delete(dists[-1], est_index[-1])))
    return est_index, match_min_dists, cross_min_dists


def search_index_with_segment(data, published, segment_method):
    # 根据已推断出来的分段类型和数据，以及公布的已置换分段，猜测密钥（置换分段的索引）
    indices = []
    min_dists = []
    data_segs = segment_data(data, len(segment_method), segment_method)
    i = 0
    used_segment_indices = []
    while i < len(published):
        min_dist = np.inf
        min_dist_index = -1
        for j in range(len(segment_method)):
            # 滤重
            if j in used_segment_indices:
                continue
            published_seg = published[i:i + segment_method[j]]
            # 是否需要相同长度
            if len(published_seg) != len(data_segs[j]):
                continue
            # dist = dtw_metric(data_segs[j], published_seg)
            # dist = np.sum(np.abs(np.array(data_segs[j]) - np.array(published_seg)))
            dist = np.linalg.norm(np.array(data_segs[j]) - np.array(published_seg))
            if dist < min_dist:
                min_dist = dist
                min_dist_index = j
        indices.append(min_dist_index)
        min_dists.append(min_dist)
        used_segment_indices.append(min_dist_index)
        i += segment_method[min_dist_index]
    return indices, min_dists


# data = [11, 5, 2,
#         7, 10, 15, 0, 14,
#         13, 1, 4, 12,
#         3, 9, 8, 6]
# published = [13, 1, 4, 12,
#              3, 9, 8, 6,
#              11, 5, 2,
#              7, 10, 15, 0, 14]
# segment_method1 = [3, 5, 4, 4]
# print(search_index_with_segment(data, published, segment_method1))
# indices = search_index_with_segment(data, published, segment_method1)[0]
# data_seg = segment_sequence(data, segment_method1)
# published_segment_method1 = np.array(segment_method1)[indices]
# print(published_segment_method1)
# published_seg = segment_sequence(published, published_segment_method1)
# print(compute_min_dtw(data_seg, published_seg))


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
    # 按照阈值来找分段类型
    start_index = []
    for i in range(len(dataB)):
        # 选择与当前接近的位置作为寻找起点
        current_start_index = []
        for j in range(len(dataA)):
            if np.ndim(dataA) == 1:
                if abs(dataB[i] - dataA[j]) < threshold:
                    current_start_index.append(j)
            else:
                # if dtw_ndim.distance(dataB[i], dataA[j]) < threshold:
                if np.sum(np.abs(dataB[i] - dataA[j])) < threshold:
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
    isBreak = False
    for start in range(target_range[1] - 1, target_range[0] - 1, -1):
        if isBreak:
            break
        for end in tree[start]:
            if isBreak:
                break
            if min_length <= end - start <= max_length:
                for subseq in dp[end]:
                    calls[0] += 1
                    dp[start].append([[start, end]] + subseq)
                    # 针对于withProb和长数据(> 4*16)，否则MemoryError
                    if len(dp[start]) > 10000 and target_range[1] >= 4 * 16:
                        isBreak = True
                        break

    if isBreak:
        return -1, -1

    # 筛选出符合条件的方案
    valid_results = []
    for solution in dp[target_range[0]]:
        calls[1] += 1
        if all(min_length <= (interval[1] - interval[0]) <= max_length for interval in solution):
            valid_results.append(solution)

    # print("time (ms):", (time.time_ns() - start_time) / 1e6)
    # print(calls)
    return valid_results, dp


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


def find_gaps(intervals, target_interval):
    # 检查所有区间是否覆盖了目标区间，如果没有返回间隙组成的区间gaps
    # 按照区间的起始位置排序
    intervals.sort(key=lambda x: x[0])

    # 初始化当前覆盖的范围
    current_start, current_end = intervals[0]

    # 检查初始间隙
    gaps = []
    if current_start > target_interval[0]:
        gaps.append((target_interval[0], current_start))

    # 遍历所有区间
    for start, end in intervals:
        if start > current_end:
            # 如果区间之间有间隙，记录间隙
            gaps.append((current_end, start))
            current_end = end  # 更新当前覆盖的结束位置为当前区间的结束位置
        else:
            # 扩展当前覆盖的范围
            current_end = max(current_end, end)

        # 如果当前覆盖的范围已经包含目标区间的结束位置，可以提前退出
        if current_end >= target_interval[1]:
            break

    # 检查最后的间隙
    if current_end < target_interval[1]:
        gaps.append((current_end, target_interval[1]))

    return gaps


# 测试用例
# intervals1 = [[0, 12], [9, 19], [17, 20], [22, 24], [23, 28], [28, 32]]
# intervals2 = [[0, 12], [9, 19], [18, 23], [23, 28], [28, 32]]
# target_interval = [0, 32]
# print(find_gaps(intervals1, target_interval))
# print(find_gaps(intervals2, target_interval))


def merge_gaps(intervals, gaps):
    # 将gaps合并至子区间内，对每个gap都与其前一个区间合并
    # 按起始位置排序
    intervals.sort(key=lambda x: x[0])
    gaps.sort(key=lambda x: x[0])

    merged_intervals = []
    n = len(intervals)
    m = len(gaps)

    # 如果第一个分段前有间隙，则加入此间隙
    if intervals[0][0] > 0:
        merged_intervals.append([0, intervals[0][0]])

    i = 0
    while i < n:
        start, end = intervals[i]

        # 尝试合并间隙
        gap_index = 0
        while gap_index < m:
            gap_start, gap_end = gaps[gap_index]
            if end == gap_start:
                # 合并当前区间与间隙
                end = gap_end
                break
            gap_index += 1

        merged_intervals.append([start, end])
        i += 1

    return merged_intervals


# intervals = [[18, 21], [25, 28], [27, 31], [31, 36], [43, 46], [47, 51], [55, 60], [60, 63]]
# gaps = [(0, 18), (21, 25), (36, 43), (46, 47), (51, 55), (63, 64)]
# print(merge_gaps(intervals, gaps))


def find_covering_intervals(intervals, gaps):
    # 给定一些子区间和gaps，要求返回子区间中包含每个gaps的区间，如果单个子区间无法包含所有gap，则需要返回所有合并后能覆盖gap的子区间
    # 按起始位置排序
    intervals.sort(key=lambda x: x[0])
    gaps.sort(key=lambda x: x[0])

    covering_intervals = []

    for gap in gaps:
        gap_start, gap_end = gap
        current_cover = []
        current_end = gap_start

        for interval in intervals:
            start, end = interval

            # 检查区间是否与间隙重叠并有助于覆盖间隙
            if start <= current_end and end > current_end:
                current_cover.append(interval)
                current_end = end

                # 如果已经覆盖整个间隙，退出循环
                if current_end >= gap_end:
                    break

        # 检查是否覆盖了整个间隙
        if current_end >= gap_end:
            covering_intervals.extend(current_cover)
        else:
            return []

    return covering_intervals


def find_before_intervals(intervals, gaps):
    before_intervals = []
    for i in range(len(intervals) - 1):
        for gap in gaps:
            if gap == intervals[i]:
                before_intervals.append(intervals[i + 1])
                break

    return before_intervals


# 测试用例
# intervals1 = [[0, 3], [3, 6], [4, 7], [6, 9], [8, 13], [10, 13], [13, 18], [16, 19], [18, 22], [22, 25], [25, 28],
#     [28, 31], [31, 36], [36, 41], [41, 44], [44, 49], [49, 55], [55, 60], [60, 64]]
# gaps = [(0, 18), (21, 25), (36, 43), (46, 47), (51, 55), (63, 64)]
#
# result = find_covering_intervals(intervals1, gaps)
# print(result)
# intervals2 = [[18, 21], [25, 28], [27, 31], [31, 36], [43, 46], [47, 51], [55, 60], [60, 63]]
# intervals2.extend(result)
# print(find_gaps(intervals2, (0, 64)))

# segments_A = [[0, 5], [5, 11], [9, 14], [11, 15], [15, 19], [19, 24], [24, 30], [29, 32], [29, 33], [33, 37], [37, 41],
#               [41, 46], [46, 50], [50, 55], [55, 61], [61, 64]]
# segments_B = [[0, 3], [15, 18], [20, 24], [23, 26], [42, 46], [51, 55]]
# gaps = [(3, 15), (18, 20), (26, 42), (46, 51), (55, 64)]
# result = find_covering_intervals(segments_A, gaps)
# print(result)
# segments_B.extend(result)
# print(find_gaps(segments_B, (0, 64)))

# segments_A = [[0, 3], [3, 6], [6, 10], [10, 13], [10, 14], [13, 16], [16, 20], [20, 24], [24, 29], [29, 34], [34, 37],
#               [37, 41], [41, 44], [44, 48], [48, 52], [52, 55], [55, 58], [60, 63], [63, 64]]
# gaps = [(23, 24), (37, 41), (55, 64)]
# print(find_covering_intervals(segments_A, gaps))


def find_out_of_range_intervals(intervals, min_length, max_length):
    # 检查所有区间是否小于目标区间，如果没有返回间隙组成的区间gaps
    # 按照区间的起始位置排序
    intervals.sort(key=lambda x: x[0])

    # 初始化结果列表
    out_of_range_intervals = []

    # 遍历所有区间
    for start, end in intervals:
        length = end - start
        # 检查区间长度是否在给定范围内
        if length < min_length or length > max_length:
            out_of_range_intervals.append((start, end))

    return out_of_range_intervals


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
    # 按照最大值增序，均值增序，方差降序的顺序排序，同时考虑浮点数判等的精度问题
    array_info = []

    for idx, arr in enumerate(arrays):
        max_value = max(arr)
        mean_value = np.mean(arr)
        var_value = np.var(arr)
        array_info.append((idx, max_value, mean_value, var_value, arr))

    def compare_arrays(a, b):
        idx_a, max_a, mean_a, var_a, arr_a = a
        idx_b, max_b, mean_b, var_b, arr_b = b

        if abs(max_a - max_b) > epsilon:
            return -1 if max_a < max_b else 1
        elif abs(mean_a - mean_b) > epsilon:
            return -1 if mean_a < mean_b else 1
        else:
            return -1 if var_a < var_b else 1

    from functools import cmp_to_key
    sorted_arrays = sorted(array_info, key=cmp_to_key(compare_arrays))

    return sorted_arrays


def find_special_array_min_mean_var(arrays, epsilon=1e-5):
    # 按照最小值，均值，方差的顺序排序，同时考虑浮点数判等的精度问题
    # 先找出min最大的，如果min近似相等则找出mean最大的，如果mean近似相等则找出var最小的
    array_info = []

    for idx, arr in enumerate(arrays):
        min_value = min(arr)
        mean_value = np.mean(arr)
        var_value = np.var(arr)
        array_info.append((idx, min_value, mean_value, var_value, arr))

    def compare_arrays(a, b):
        idx_a, min_a, mean_a, var_a, arr_a = a
        idx_b, min_b, mean_b, var_b, arr_b = b

        # 比较最小值
        if abs(min_a - min_b) >= epsilon:
            return -1 if min_a > min_b else 1
        # 最小值近似相等时，比较均值
        elif abs(mean_a - mean_b) >= epsilon:
            return -1 if mean_a > mean_b else 1
        # 均值近似相等时，比较方差
        else:
            return -1 if var_a < var_b else 1

    from functools import cmp_to_key
    sorted_arrays = sorted(array_info, key=cmp_to_key(compare_arrays))

    return sorted_arrays


def find_special_array_min_min_mean_var(match_arrays, cross_arrays, epsilon=1e-5):
    # 按照最小值，均值，方差的顺序排序，同时考虑浮点数判等的精度问题
    # 先找出min最大的，如果min近似相等则找出mean最大的，如果mean近似相等则找出var最小的
    array_info = []

    for idx, match_arr in enumerate(match_arrays):
        match_min_value = min(match_arr)
        cross_min_value = min(cross_arrays[idx])
        mean_value = np.mean(cross_arrays[idx])
        array_info.append((idx, match_min_value, cross_min_value, mean_value, match_arr, cross_arrays[idx]))

    def compare_arrays(a, b):
        idx_a, match_min_a, cross_min_a, mean_a, match_arr_a, cross_arr_a = a
        idx_b, match_min_b, cross_min_b, mean_b, match_arr_b, cross_arr_b = b

        # 比较最小值
        if abs(match_min_a - match_min_b) > epsilon:
            return -1 if match_min_a < match_min_a else 1
        # 最小值近似相等时，比较最小值
        elif abs(cross_min_a - cross_min_b) >= epsilon:
            return -1 if cross_min_a > cross_min_b else 1
        # 最小值近似相等时，比较均值
        else:
            return -1 if mean_a > mean_b else 1

    from functools import cmp_to_key
    sorted_arrays = sorted(array_info, key=cmp_to_key(compare_arrays))

    return sorted_arrays


def find_special_array_mean_var(arrays, epsilon=1e-5):
    # 按照均值增序，方差降序的顺序排序，同时考虑浮点数判等的精度问题
    array_info = []

    for idx, arr in enumerate(arrays):
        mean_value = np.mean(arr)
        var_value = np.var(arr)
        array_info.append((idx, mean_value, var_value, arr))

    def compare_arrays(a, b):
        idx_a, mean_a, var_a, arr_a = a
        idx_b, mean_b, var_b, arr_b = b

        if abs(mean_a - mean_b) > epsilon:
            return -1 if mean_a < mean_b else 1
        else:
            return -1 if var_a < var_b else 1

    from functools import cmp_to_key
    sorted_arrays = sorted(array_info, key=cmp_to_key(compare_arrays))

    return sorted_arrays


def find_special_array_min_mean(arrays, epsilon=1e-5):
    # 按照最小值，均值，方差的顺序排序，同时考虑浮点数判等的精度问题
    # 先找出min最小的，如果min近似相等则找出mean最小的
    array_info = []

    for idx, arr in enumerate(arrays):
        min_value = min(arr)
        mean_value = np.mean(arr)
        array_info.append((idx, min_value, mean_value, arr))

    def compare_arrays(a, b):
        idx_a, min_a, mean_a, arr_a = a
        idx_b, min_b, mean_b, arr_b = b

        # 比较最小值
        if abs(min_a - min_b) > epsilon:
            return -1 if min_a < min_b else 1
        # 最小值近似相等时，比较均值
        else:
            return -1 if mean_a < mean_b else 1

    from functools import cmp_to_key
    sorted_arrays = sorted(array_info, key=cmp_to_key(compare_arrays))

    return sorted_arrays


# print(find_special_array_min_mean([[2, -1, 1], [1, 2, 3], [0, 0, 2], [0, 1, 1]]))


# print(find_special_array_min_mean_var([[2, -1, 1], [1, 2, 3], [0, 0, 2], [0, 1, 1]]))

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


def common_pca(ha, hb, he, k):
    ha = (ha - np.min(ha)) / (np.max(ha) - np.min(ha))
    hb = (hb - np.min(hb)) / (np.max(hb) - np.min(hb))
    he = (he - np.min(he)) / (np.max(he) - np.min(he))
    rha = np.dot(ha, ha.T) / len(ha)
    ua, sa, vha = np.linalg.svd(rha)
    # print("p", np.sum(sa) / np.sum(sa[:k]))
    vha = vha[:k, :]
    ya = vha @ ha
    yb = vha @ hb
    ye = vha @ he
    return np.array(ya), np.array(yb), np.array(ye)


def optimize_random_matrix_max(A, tau):
    # 效果差，84%
    # 数据预处理：均值化和归一化
    A_mean = np.mean(A)
    A_normalized = (A - A_mean) / np.std(A)

    n = len(A)

    # 创建模型
    model = gp.Model()

    # 隐藏求解器的输出
    model.Params.OutputFlag = 0

    # 设置求解参数以提升速度
    model.Params.Threads = 4  # 使用多线程
    model.Params.Presolve = 2  # 开启高级预处理
    model.Params.MIPFocus = 1  # 聚焦于寻找可行解
    model.Params.Cuts = 2  # 增强切割平面

    # 定义变量 K
    K = model.addVars(n, n, lb=-GRB.INFINITY, name="K")

    # 计算 AK
    AK = [[gp.LinExpr(A_normalized[i] * K[i, j]) for j in range(n)] for i in range(n)]

    # 定义目标函数：最小化 AK - A_normalized 的范数平方和
    objective = gp.quicksum((gp.quicksum(AK[i][j] for j in range(n)) - A_normalized[i]) ** 2 for i in range(n))
    model.setObjective(objective, GRB.MINIMIZE)

    # 定义约束条件：确保 AK 中每个元素之间的欧式距离之和大于 tau
    dist_sum = gp.quicksum(gp.quicksum(
        (gp.quicksum(AK[i][k] for k in range(n)) - gp.quicksum(AK[j][k] for k in range(n))) ** 2 for j in
        range(i + 1, n)) for i in range(n))
    model.addConstr(dist_sum >= tau, name="distance_constraint")

    # 设置NonConvex参数以允许非凸问题
    model.Params.NonConvex = 2

    # 求解优化问题
    model.optimize()

    return np.array([[K[i, j].X for j in range(n)] for i in range(n)])


def optimize_random_matrix_max_min(A, tau):
    # 效果差，94%
    # 数据预处理：均值化和归一化
    A_mean = np.mean(A)
    A_normalized = (A - A_mean) / np.std(A)

    n = len(A)

    # 创建模型
    model = gp.Model()

    # 隐藏求解器的输出
    model.Params.OutputFlag = 0

    # 设置求解参数以提升速度
    model.Params.Threads = 4  # 使用多线程
    model.Params.TimeLimit = 3  # 设置时间限制（秒）
    model.Params.Presolve = 2  # 开启高级预处理
    model.Params.MIPFocus = 1  # 聚焦于寻找可行解
    model.Params.Cuts = 2  # 增强切割平面

    # 定义变量 K
    K = model.addVars(n, n, lb=-1, ub=1, name="K")

    # 计算 AK
    AK = [[gp.LinExpr(A_normalized[i] * K[i, j]) for j in range(n)] for i in range(n)]

    # 定义辅助变量和约束来表示 AK 中每对元素之间的欧式距离
    distances = model.addVars(n, n, lb=0, name="distances")

    for i in range(n):
        for j in range(n):
            if i != j:
                model.addConstr(distances[i, j] == gp.quicksum((AK[i][k] - AK[j][k]) ** 2 for k in range(n)),
                                name=f"dist_{i}_{j}")

    # 定义辅助变量来表示最小距离
    min_distances = model.addVars(n, lb=0, name="min_distances")

    for i in range(n):
        model.addConstr(min_distances[i] == gp.min_([distances[i, j] for j in range(n) if i != j]),
                        name=f"min_distance_{i}")

    # 设置目标函数：最大化最小距离之和
    model.setObjective(gp.quicksum(min_distances[i] for i in range(n)), GRB.MAXIMIZE)

    # 定义目标函数：最小化 AK - A_normalized 的范数平方和
    objective = gp.quicksum((gp.quicksum(AK[i][j] for j in range(n)) - A_normalized[i]) ** 2 for i in range(n))
    model.setObjective(objective, GRB.MINIMIZE)
    # model.addConstr(objective <= tau, name="objective")

    # 设置NonConvex参数以允许非凸问题
    model.Params.NonConvex = 2

    # 求解优化问题
    model.optimize()

    return np.array([[K[i, j].X for j in range(n)] for i in range(n)])


def search_random_matrix_max_min(A, alpha):
    def objective(K_flat, A, alpha):
        # 将展平的 K 恢复为 N x N 矩阵
        N = len(A)
        K = K_flat.reshape(N, N)

        # 计算 B = A @ K
        B = np.dot(A, K)

        # 对 B 进行均值化和归一化
        B = B - np.mean(B)
        B = (B - np.min(B)) / (np.max(B) - np.min(B))

        # 计算最小欧式距离之和
        min_distance_sum = 0
        for i in range(len(B)):
            distances = np.linalg.norm(B[i] - np.delete(B, i))
            min_distance_sum += np.min(distances)
        # 计算 A-B 范数
        norm_A_B = np.linalg.norm(A - B)

        # 我们需要最大化目标函数，因此返回其负值
        return -min_distance_sum + alpha * norm_A_B

    def constraint(K_flat, A):
        # 将展平的 K 恢复为 N x N 矩阵
        N = len(A)
        K = K_flat.reshape(N, N)

        # 计算 B = A @ K
        B = np.dot(A, K)

        # 约束：B 的所有元素应在 [0, 1] 范围内
        return np.concatenate((B - 0, 1 - B))

    def optimize_matrix(A, method, alpha):
        N = len(A)
        # 初始猜测的 K
        initial_K = np.random.rand(N, N)
        initial_K_flat = initial_K.flatten()

        # 定义约束条件
        cons = ({'type': 'ineq', 'fun': constraint, 'args': (A,)})

        # 优化目标函数，包含边界和约束条件
        if method == 'slsqp' or method == 'trust-constr':
            result = minimize(objective, initial_K_flat, args=(A, alpha), method=method, constraints=cons)
        else:
            result = minimize(objective, initial_K_flat, args=(A, alpha), method=method)

        # 将优化后的 K 恢复为 N x N 矩阵
        K_optimal = result.x.reshape(N, N)

        return K_optimal

    return optimize_matrix(A, "slsqp", alpha)


def search_random_matrix_uniform(A, method, alpha):
    # 产生均匀分布的序列
    def objective(K_flat, A, alpha):
        # 将展平的 K 恢复为 N x N 矩阵
        N = len(A)
        K = K_flat.reshape(N, N)

        # 计算 B = A @ K
        B = np.dot(A, K)

        # 对 B 进行均值化和归一化
        B = B - np.mean(B)
        B = (B - np.min(B)) / (np.max(B) - np.min(B))

        # 计算最小欧式距离之和
        min_distance_sum = 0
        for i in range(len(B)):
            distances = []
            for j in range(len(B)):
                if i == j:
                    continue
                distances.append(np.square(B[i] - B[j]))
            min_distance_sum += 1 / (np.min(distances) + 1e-6)
        # 计算 A-B 欧式距离
        dist_A_B = 0
        for i in range(len(A)):
            dist_A_B += np.square(A[i] - B[i])

        # 我们需要最大化目标函数，因此返回其负值
        return min_distance_sum * alpha * dist_A_B

    def constraint(K_flat, A):
        # 将展平的 K 恢复为 N x N 矩阵
        N = len(A)
        K = K_flat.reshape(N, N)

        # 计算 B = A @ K
        B = np.dot(A, K)

        # 约束：B 的所有元素应在 [0, 1] 范围内
        return np.concatenate((B - 0, 1 - B))

    def optimize_matrix(A, method, alpha):
        N = len(A)
        # 初始猜测的 K
        initial_K = np.random.rand(N, N)
        initial_K_flat = initial_K.flatten()

        # 定义约束条件
        cons = ({'type': 'ineq', 'fun': constraint, 'args': (A,)})

        # 优化目标函数，包含边界和约束条件
        if method == 'slsqp' or method == 'trust-constr':
            result = minimize(objective, initial_K_flat, args=(A, alpha), method=method, constraints=cons)
        else:
            result = minimize(objective, initial_K_flat, args=(A, alpha), method=method)

        # 将优化后的 K 恢复为 N x N 矩阵
        K_optimal = result.x.reshape(N, N)

        return K_optimal

    return optimize_matrix(A, method, alpha)


def insert_random_numbers(arr, insert_number, seed):
    # 向数组arr中随机添加insert_number个随机数，seed是与随机数和其位置相关的随机种子
    arr = list(arr)
    for _ in range(insert_number):
        np.random.seed(seed)
        random_number = np.random.uniform(np.min(arr), np.max(arr))

        np.random.seed(seed)
        position = np.random.randint(0, len(arr))

        arr.insert(position, random_number)

    return arr


# array = np.array([1, 2, 3, 4, 5])
# N = 3  # Number of random numbers to insert
# modified_array = insert_random_numbers(array, N, 1)
# print("Modified Array:", modified_array)


def modify_segments_with_random_position(segments, insert_number, seed):
    # 插入的位置也是随机的
    # min_segments = 0
    # max_segments = 1
    min_segments = -1
    max_segments = 2

    # min_segments = np.inf
    # max_segments = -np.inf
    # for i in range(len(segments)):
    #     min_segments = min(min_segments, np.min(segments[i]))
    #     max_segments = max(max_segments, np.max(segments[i]))

    # 检查各个分段直接的DTW距离，找出最小的DTW距离的两个分段，然后向其中分段中某个位置添加一个随机数
    # 确保添加之后的DTW距离比原本的大，然后一直添加完insert_number个随机数
    def find_min_dtw_segments(segments):
        min_distance = float('inf')
        min_pair = (0, 0)

        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                distance = dtw_metric(segments[i], segments[j])
                if distance < min_distance:
                    min_distance = distance
                    min_pair = (i, j)

        return min_pair, min_distance

    def insert_random_number(segment, segments, seed):
        # 随机数的种子很多时候设置的一样，还可以优化
        np.random.seed(seed)
        if np.ndim(segments[0]) == 1:
            random_number = np.random.uniform(min_segments, max_segments)
        else:
            random_number = np.random.uniform(min_segments, max_segments, size=(np.ndim(segments[0])))

        np.random.seed(seed)
        position = random.randint(0, len(segments[i]))

        new_segment = segment[:position] + [random_number] + segment[position:]

        return new_segment, position, random_number

    def increase_dtw_distance(segments, i, j, insert_number, seed):
        modifications = []

        for _ in range(insert_number):
            current_distance = dtw_metric(segments[i], segments[j])

            count = 0
            while True:
                new_segment, position, random_number = insert_random_number(segments[i], segments, seed + count)
                new_distance = dtw_metric(new_segment, segments[j])
                count += 1

                if new_distance > current_distance:
                    segments[i] = new_segment
                    modifications.append((i, position, random_number))
                    break

        return segments, modifications

    segments = [list(segment) for segment in segments]
    modifications = []

    for _ in range(insert_number):
        min_pair, min_dist, = find_min_dtw_segments(segments)
        i, j = min_pair
        segments, new_modifications = increase_dtw_distance(segments, i, j, 1, seed)
        modifications.extend(new_modifications)

    return segments, modifications


def modify_segments_position_compared_with_one_segments(segments, insert_number, seed):
    # min_segments = 0
    # max_segments = 1
    # min_segments = -1
    # max_segments = 2
    min_segments = -1
    max_segments = 2

    # min_segments = np.inf
    # max_segments = -np.inf
    # for i in range(len(segments)):
    #     min_segments = min(min_segments, np.min(segments[i]))
    #     max_segments = max(max_segments, np.max(segments[i]))

    # 检查各个分段直接的DTW距离，找出最小的DTW距离的两个分段，然后向其中分段中某个位置添加一个随机数
    # 确保添加之后的DTW距离比原本的大，然后一直添加完insert_number个随机数
    def find_min_dtw_segments(segments):
        min_distance = float('inf')
        min_pair = (0, 0)

        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                distance = dtw_metric(segments[i], segments[j])
                if distance < min_distance:
                    min_distance = distance
                    min_pair = (i, j)

        return min_pair, min_distance

    def increase_dtw_distance(segments, i, j, insert_number, seed):
        modifications = []

        for _ in range(insert_number):
            current_distance = dtw_metric(segments[i], segments[j])

            count = 0
            while True:
                # 随机数的种子很多时候设置的一样，还可以优化
                np.random.seed(seed + count)
                count += 1

                if np.ndim(segments[0]) == 1:
                    random_number = np.random.uniform(min_segments, max_segments)
                    # random_number = np.array(2)
                else:
                    random_number = np.random.uniform(min_segments, max_segments, size=(np.ndim(segments[0])))
                    # random_number = np.array([2, -1])

                max_distance = -np.inf
                max_distance_segment = None
                max_distance_modifications = None
                for position in range(len(segments[i])):
                    new_segment = segments[i][:position] + [random_number] + segments[i][position:]
                    new_distance = dtw_metric(new_segment, segments[j])

                    # 找出最大距离和对应分段位置
                    if new_distance > max_distance:
                        max_distance = new_distance
                        max_distance_segment = new_segment
                        max_distance_modifications = (i, position, random_number)

                if max_distance > current_distance or count > 3:
                    segments[i] = max_distance_segment
                    modifications.append(max_distance_modifications)
                    break

        return segments, modifications

    segments = [list(segment) for segment in segments]
    modifications = []

    for _ in range(insert_number):
        min_pair, min_dist, = find_min_dtw_segments(segments)
        i, j = min_pair
        segments, new_modifications = increase_dtw_distance(segments, i, j, 1, seed)
        modifications.extend(new_modifications)

    return segments, modifications


def modify_segments_position_compared_with_all_segments(segments, insert_number, seed):
    # min_segments = 0
    # max_segments = 1
    # min_segments = -1
    # max_segments = 2
    min_segments = -1
    max_segments = 2

    # min_segments = np.inf
    # max_segments = -np.inf
    # for i in range(len(segments)):
    #     min_segments = min(min_segments, np.min(segments[i]))
    #     max_segments = max(max_segments, np.max(segments[i]))

    # 检查各个分段直接的DTW距离，找出最小的DTW距离的两个分段，然后向其中分段中某个位置添加一个随机数
    # 确保添加之后的DTW距离比原本的大，然后一直添加完insert_number个随机数
    def find_min_dtw_segments(segments):
        min_distance = float('inf')
        min_pair = (0, 0)

        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                distance = dtw_metric(segments[i], segments[j])
                if distance < min_distance:
                    min_distance = distance
                    min_pair = (i, j)

        return min_pair, min_distance

    def increase_dtw_distance(segments, i, j, insert_number, seed):
        modifications = []

        for _ in range(insert_number):
            current_distance = dtw_metric(segments[i], segments[j])

            count = 0
            while True:
                # 随机数的种子很多时候设置的一样，还可以优化
                np.random.seed(seed + count)
                count += 1

                if np.ndim(segments[0]) == 1:
                    random_number = np.random.uniform(min_segments, max_segments)
                    # random_number = np.array(2)
                else:
                    random_number = np.random.uniform(min_segments, max_segments, size=(np.ndim(segments[0])))
                    # random_number = np.array([2, -1])

                max_distance = -np.inf
                max_distance_segment = None
                max_distance_modifications = None
                for position in range(len(segments[i])):
                    new_segment = segments[i][:position] + [random_number] + segments[i][position:]

                    # 找出新分段与原始所有其他分段的最小距离
                    min_distance = np.inf
                    for k in range(len(segments)):
                        if i == k:
                            continue
                        new_distance = dtw_metric(new_segment, segments[k])
                        if new_distance < min_distance:
                            min_distance = new_distance
                    # 再从其中找出最大距离和对应分段位置
                    if min_distance > max_distance:
                        max_distance = min_distance
                        max_distance_segment = new_segment
                        max_distance_modifications = (i, position, random_number)

                if max_distance > current_distance or count > 3:
                    segments[i] = max_distance_segment
                    modifications.append(max_distance_modifications)
                    break

        return segments, modifications

    segments = [list(segment) for segment in segments]
    modifications = []

    for _ in range(insert_number):
        min_pair, min_dist, = find_min_dtw_segments(segments)
        i, j = min_pair
        segments, new_modifications = increase_dtw_distance(segments, i, j, 1, seed)
        modifications.extend(new_modifications)

    return segments, modifications


def modify_segments_with_even_addition(segments, insert_number, keyLen, seed):
    # min_segments = 0
    # max_segments = 1
    # min_segments = -1
    # max_segments = 2
    min_segments = -1
    max_segments = 2

    # min_segments = np.inf
    # max_segments = -np.inf
    # for i in range(len(segments)):
    #     min_segments = min(min_segments, np.min(segments[i]))
    #     max_segments = max(max_segments, np.max(segments[i]))

    # 检查各个分段直接的DTW距离，找出最小的DTW距离的两个分段，然后向其中分段中某个位置添加一个随机数
    # 确保添加之后的DTW距离比原本的大，然后一直添加完insert_number个随机数
    def find_all_min_dtw_segments(segments):
        min_pair_with_dist = []

        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                min_pair_with_dist.append((i, j, dtw_metric(segments[i], segments[j])))

        return min_pair_with_dist

    def increase_dtw_distance(segments, index, min_pair_with_dists, seed):
        modifications = []
        current_distance = None

        # 找出与当前index有关联的分段中的最小距离
        for i in range(len(min_pair_with_dists)):
            if min_pair_with_dists[i][0] == index or min_pair_with_dists[i][1] == index:
                current_distance = min_pair_with_dists[i][2]
                break

        count = 0
        while True:
            # 随机数的种子很多时候设置的一样，还可以优化
            np.random.seed(seed + count)
            count += 1

            if np.ndim(segments[0]) == 1:
                random_number = np.random.uniform(min_segments, max_segments)
                # random_number = np.array(2)
            else:
                random_number = np.random.uniform(min_segments, max_segments, size=(np.ndim(segments[0])))
                # random_number = np.array([2, -1])

            max_distance = -np.inf
            max_distance_segment = None
            max_distance_modifications = None
            for position in range(len(segments[index])):
                new_segment = segments[index][:position] + [random_number] + segments[index][position:]
                # 新的分段和原始数据中其他分段的所有距离都比原始分段对的距离小
                isSuccess = True
                for j in range(len(segments)):
                    if index == j:
                        continue
                    # 还可以对比所有分段以进行更好的距离选择
                    new_distance = dtw_metric(new_segment, segments[j])
                    if new_distance < max_distance:
                        isSuccess = False
                        break
                if isSuccess:
                    max_distance_segment = new_segment
                    max_distance_modifications = (index, position, random_number)

            if max_distance > current_distance or count > 3:
                segments[index] = max_distance_segment
                modifications.append(max_distance_modifications)
                break

        return segments, modifications

    segments = [list(segment) for segment in segments]
    modifications = []
    used_indices = []

    for i in range(insert_number):
        # 统计所有dtw距离，找出最小的dtw距离对应的insert_number个分段对
        min_pair_with_dists = find_all_min_dtw_segments(segments)
        min_pair_with_dists = sorted(min_pair_with_dists, key=(lambda x: x[2]))
        most_shortest_pair_with_dists = min_pair_with_dists[0:insert_number]
        visited_indices = []
        # 收集所有最小距离对应的分段对的索引
        for index1, index2, dist in most_shortest_pair_with_dists:
            visited_indices.append(index1)
            visited_indices.append(index2)
        # 找出出现次数最多的索引，在该分段中添加随机数据
        most_indices = sorted(dict(Counter(visited_indices)).items(), key=lambda x: x[1], reverse=True)
        remained_pair_with_dists = min_pair_with_dists[insert_number:]
        unvisited_indices = []
        for index1, index2, dist in remained_pair_with_dists:
            if index1 not in visited_indices and index2 not in visited_indices:
                unvisited_indices.append(index1)
                unvisited_indices.append(index2)
        find_index = False
        for j in range(len(most_indices)):
            min_pair_index = most_indices[j][0]
            if min_pair_index not in used_indices and len(segments[min_pair_index]) < 6:
                used_indices.append(min_pair_index)
                segments, new_modifications = increase_dtw_distance(segments, min_pair_index, min_pair_with_dists, seed)
                modifications.extend(new_modifications)
                find_index = True
                break
        if find_index is False:
            for j in range(len(unvisited_indices)):
                min_pair_index = unvisited_indices[j]
                if j not in used_indices and len(segments[min_pair_index]) < 6:
                    used_indices.append(min_pair_index)
                    segments, new_modifications = increase_dtw_distance(segments, min_pair_index, min_pair_with_dists,
                                                                        seed)
                    modifications.extend(new_modifications)
                    find_index = True
                    break
        if find_index is False:
            # 原始分段总和不足，需要额外添加数据
            curr_keyLen = sum([np.shape(segment)[0] for segment in segments])
            if keyLen > curr_keyLen:
                curr_keyLen += 1
                min_segment_length = np.inf
                for j in range(len(segments)):
                    min_segment_length = min(min_segment_length, np.shape(segments[j])[0])
                for j in range(len(segments)):
                    if np.shape(segments[j])[0] == min_segment_length:
                        segments, new_modifications = increase_dtw_distance(segments, j, min_pair_with_dists, seed)
                        modifications.extend(new_modifications)
                        break

    return segments, modifications


# Example usage
# N = 16  # Number of random numbers to insert
# np.random.seed(10000)
# segments = np.random.normal(0, 1, (N, 4))
# start_time = time.time()
# modified_segments, modifications = modify_segments_with_even_addition(segments, N, N * 4, 1)
# print("time", time.time() - start_time)
#
# print("Modified Segments:", modified_segments)
# segment_length = []
# for segment in modified_segments:
#     segment_length.append(len(segment))
# print("Segment Length:", segment_length)

# new_segments = segments.copy()
# for i, position, random_number in modifications:
#     new_segments[i].insert(position, random_number)
# print("New Segments:", new_segments)


def replace_segments_position_compared_with_all_segments_with_threshold(segments, ori_dist_threshold,
                                                                        max_same_position_dist, replace_number, seed):
    # min_segments = 0
    # max_segments = 1
    # min_segments = -1
    # max_segments = 2
    min_segments = -1
    max_segments = 2

    def find_min_dtw_segments(segments):
        min_distance = float('inf')
        min_pair = (0, 0)

        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                distance = dtw_metric(segments[i], segments[j])
                if distance < min_distance:
                    min_distance = distance
                    min_pair = (i, j)

        return min_pair, min_distance

    def increase_dtw_distance(segments, i, j, seed):
        modifications = []

        original_segment = segments[i]
        current_distance = dtw_metric(segments[i], segments[j])

        count = 0
        while True:
            np.random.seed(seed + count + int(np.sum(segments[i])))
            count += 1

            if np.ndim(segments[0]) == 1:
                random_number = np.random.uniform(min_segments, max_segments)
            else:
                random_number = np.random.uniform(min_segments, max_segments, size=(np.ndim(segments[0])))

            # 找出与其他分段最小距离中的最大者
            max_other_distance = -np.inf
            # 找出与原始分段的最小距离
            min_new_dtw_distance = np.inf
            best_new_segment = None
            best_modifications = None
            for position in range(len(segments[i])):
                new_segment = segments[i][:position] + [random_number] + segments[i][position + 1:]
                new_distance = dtw_metric(new_segment, original_segment)

                if new_distance < min_new_dtw_distance:
                    min_new_dtw_distance = new_distance
                    # 找出与其他分段的最小距离
                    min_other_distance = np.inf

                    for k in range(len(segments)):
                        if k == i:
                            continue
                        other_distance = dtw_metric(new_segment, segments[k])
                        if other_distance < min_other_distance:
                            min_other_distance = other_distance

                    if min_other_distance > max_other_distance:
                        max_other_distance = min_other_distance
                        best_new_segment = new_segment
                        best_modifications = (i, position, random_number)

            if min_new_dtw_distance < ori_dist_threshold and max_other_distance > max_same_position_dist \
                    and (max_other_distance > current_distance or count > 3):
                segments[i] = best_new_segment
                modifications.append(best_modifications)
                break

        # print("max_other_distance", max_other_distance, "min_new_dtw_distance", min_new_dtw_distance, "count", count)
        return segments, modifications

    segments = [list(segment) for segment in segments]
    modifications = []

    for number in range(replace_number):
        min_pair, min_dist = find_min_dtw_segments(segments)
        i, j = min_pair
        segments, new_modifications = increase_dtw_distance(segments, i, j, seed * number)
        modifications.extend(new_modifications)

    return segments, modifications


def replace_segments_position_compared_with_all_segments_limited_factor(segments, replace_number, seed):
    # 缩放因子有限
    # min_segments = 0
    # max_segments = 1
    min_segments = -1
    max_segments = 2

    def find_min_dtw_segments(segments):
        min_distance = float('inf')
        min_pair = (0, 0)

        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                distance = dtw_metric(segments[i], segments[j])
                if distance < min_distance:
                    min_distance = distance
                    min_pair = (i, j)

        return min_pair, min_distance

    def increase_dtw_distance(segments, i, j, seed):
        modifications = []
        if np.ndim(segments[0]) == 1:
            min_segments0 = min(np.array(segments[i]))
            max_segments0 = max(np.array(segments[i]))
        else:
            # value部分的范围
            min_segments0 = min(np.array(segments[i]).T[0])
            max_segments0 = max(np.array(segments[i]).T[0])
            # index部分的范围
            min_segments1 = min(np.array(segments[i]).T[1])
            max_segments1 = max(np.array(segments[i]).T[1])

        current_distance = dtw_metric(segments[i], segments[j])

        count = 0
        while True:
            np.random.seed((seed + count + int(np.sum(segments[i]))) % (2 ** 32 - 1))
            count += 1

            if np.ndim(segments[0]) == 1:
                # 公布的矩阵中两维数据保持一致（攻击者猜测成功率过高）
                # random_number = np.random.uniform(min_segments, max_segments)

                # 当前分段两个维度范围
                random_number = np.random.uniform(min_segments0, max_segments0)
            else:
                # 原始方法
                # random_number = np.random.uniform(min_segments, max_segments, size=(np.ndim(segments[0])))

                # 公布的矩阵中两维数据保持一致（攻击者猜测成功率过高）
                # random_number = np.random.uniform(min_segments, max_segments)
                # random_number = np.array([random_number, random_number])

                # 当前分段两个维度范围，两组数据不同
                # random_number = np.random.uniform(min(min_segments0, min_segments1), max(max_segments0, max_segments1),
                #                                   size=(np.ndim(segments[0])))
                # 当前分段当前维度范围，两组数据不同
                random_number0 = np.random.uniform(min_segments0, max_segments0)
                random_number1 = np.random.uniform(min_segments1, max_segments1)
                random_number = np.array([random_number0, random_number1])

                excess_count = 1
                while True:
                    excessive_factor = False
                    for position in range(len(segments[i])):
                        factor0 = abs(random_number[0] / segments[i][position][0])
                        factor1 = abs(random_number[1] / segments[i][position][1])
                        if factor0 > 10 or factor1 > 10:
                            excessive_factor = True
                            break
                    if excessive_factor:
                        np.random.seed((seed + count + excess_count) % (2 ** 32 - 1))
                        excess_count += 1

                        # random_number = np.random.uniform(min_segments, max_segments)
                        # random_number = np.array([random_number, random_number])

                        # random_number = np.random.uniform(min(min_segments0, min_segments1),
                        #                                   max(max_segments0, max_segments1), size=(np.ndim(segments[0])))

                        random_number0 = np.random.uniform(min_segments0, max_segments0)
                        random_number1 = np.random.uniform(min_segments1, max_segments1)
                        random_number = np.array([random_number0, random_number1])
                    else:
                        break

            # 找出与其他分段最小距离中的最大者
            max_other_distance = -np.inf
            best_new_segment = None
            best_modifications = None
            for position in range(len(segments[i])):
                new_segment = segments[i][:position] + [random_number] + segments[i][position + 1:]

                # 找出与其他分段的最小距离
                min_other_distance = np.inf
                for k in range(len(segments)):
                    if k == i:
                        continue
                    other_distance = dtw_metric(new_segment, segments[k])
                    if other_distance < min_other_distance:
                        min_other_distance = other_distance

                if min_other_distance > max_other_distance:
                    max_other_distance = min_other_distance
                    best_new_segment = new_segment
                    best_modifications = (i, position, random_number)

            if max_other_distance > current_distance or count > 10:
                segments[i] = best_new_segment
                modifications.append(best_modifications)
                break

        # print("max_other_distance", max_other_distance, "min_new_dtw_distance", min_new_dtw_distance, "count", count)
        return segments, modifications

    segments = [list(segment) for segment in segments]
    modifications = []

    for number in range(replace_number):
        min_pair, min_dist = find_min_dtw_segments(segments)
        i, j = min_pair
        segments, new_modifications = increase_dtw_distance(segments, i, j, seed * number)
        modifications.extend(new_modifications)

    return segments, modifications


def replace_segments_position_compared_with_all_segments_no_limit(segments, replace_number, seed):
    # min_segments = 0
    # max_segments = 1
    min_segments = -1
    max_segments = 2

    def find_min_dtw_segments(segments):
        min_distance = float('inf')
        min_pair = (0, 0)

        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                distance = dtw_metric(segments[i], segments[j])
                if distance < min_distance:
                    min_distance = distance
                    min_pair = (i, j)

        return min_pair, min_distance

    def increase_dtw_distance(segments, i, j, seed):
        modifications = []
        if np.ndim(segments[0]) == 1:
            min_segments0 = min(np.array(segments[i]))
            max_segments0 = max(np.array(segments[i]))
        else:
            # value部分的范围
            min_segments0 = min(np.array(segments[i]).T[0])
            max_segments0 = max(np.array(segments[i]).T[0])
            # index部分的范围
            min_segments1 = min(np.array(segments[i]).T[1])
            max_segments1 = max(np.array(segments[i]).T[1])

        current_distance = dtw_metric(segments[i], segments[j])

        count = 0
        while True:
            np.random.seed((seed + count + int(np.sum(segments[i]))) % (2 ** 32 - 1))
            count += 1

            if np.ndim(segments[0]) == 1:
                # 公布的矩阵中两维数据保持一致（攻击者猜测成功率过高）
                # random_number = np.random.uniform(min_segments, max_segments)

                # 当前分段两个维度范围
                random_number = np.random.uniform(min_segments0, max_segments0)
            else:
                # 原始方法
                # random_number = np.random.uniform(min_segments, max_segments, size=(np.ndim(segments[0])))

                # 公布的矩阵中两维数据保持一致（攻击者猜测成功率过高）
                random_number = np.random.uniform(min_segments, max_segments)
                random_number = np.array([random_number, random_number])

                # 当前分段两个维度范围，两组数据不同
                # random_number = np.random.uniform(min(min_segments0, min_segments1), max(max_segments0, max_segments1),
                #                                   size=(np.ndim(segments[0])))

                # 当前分段当前维度范围，两组数据不同
                # random_number0 = np.random.uniform(min_segments0, max_segments0)
                # random_number1 = np.random.uniform(min_segments1, max_segments1)
                # random_number = np.array([random_number0, random_number1])

            # 找出与其他分段最小距离中的最大者
            max_other_distance = -np.inf
            best_new_segment = None
            best_modifications = None
            for position in range(len(segments[i])):
                new_segment = segments[i][:position] + [random_number] + segments[i][position + 1:]

                # 找出与其他分段的最小距离
                min_other_distance = np.inf
                for k in range(len(segments)):
                    if k == i:
                        continue
                    other_distance = dtw_metric(new_segment, segments[k])
                    if other_distance < min_other_distance:
                        min_other_distance = other_distance

                if min_other_distance > max_other_distance:
                    max_other_distance = min_other_distance
                    best_new_segment = new_segment
                    best_modifications = (i, position, random_number)

            if max_other_distance > current_distance or count > 10:
                segments[i] = best_new_segment
                modifications.append(best_modifications)
                break

        # print("max_other_distance", max_other_distance, "min_new_dtw_distance", min_new_dtw_distance, "count", count)
        return segments, modifications

    segments = [list(segment) for segment in segments]
    modifications = []

    for number in range(replace_number):
        min_pair, min_dist = find_min_dtw_segments(segments)
        i, j = min_pair
        segments, new_modifications = increase_dtw_distance(segments, i, j, seed * number)
        modifications.extend(new_modifications)

    return segments, modifications


def replace_segments_position_compared_with_all_segments_no_limit_plus(segments, replace_number):
    def find_min_dtw_segments(segments):
        min_distance = float('inf')
        min_pair = (0, 0)

        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                distance = dtw_metric(segments[i], segments[j])
                if distance < min_distance:
                    min_distance = distance
                    min_pair = (i, j)

        return min_pair, min_distance

    def increase_dtw_distance(segments, i, j):
        modifications = []

        current_distance = dtw_metric(segments[i], segments[j])

        changes = np.arange(-1, 1, 0.01)
        for change in changes:
            if np.ndim(segments[0]) == 1:
                random_number = change
            else:
                random_number = np.array([change, change])

            # 找出与其他分段最小距离中的最大者
            max_other_distance = -np.inf
            best_new_segment = None
            best_modifications = None
            for position in range(len(segments[i])):
                new_segment = segments[i][:position] + [random_number] + segments[i][position + 1:]

                # 找出与其他分段的最小距离
                min_other_distance = np.inf
                for k in range(len(segments)):
                    if k == i:
                        continue
                    other_distance = dtw_metric(new_segment, segments[k])
                    if other_distance < min_other_distance:
                        min_other_distance = other_distance

                if min_other_distance > max_other_distance:
                    max_other_distance = min_other_distance
                    best_new_segment = new_segment
                    best_modifications = (i, position, random_number)

            if max_other_distance > current_distance:
                segments[i] = best_new_segment
                modifications.append(best_modifications)
                break

        # print("max_other_distance", max_other_distance, "min_new_dtw_distance", min_new_dtw_distance, "count", count)
        return segments, modifications

    segments = [list(segment) for segment in segments]
    modifications = []

    for number in range(replace_number):
        min_pair, min_dist = find_min_dtw_segments(segments)
        i, j = min_pair
        segments, new_modifications = increase_dtw_distance(segments, i, j)
        modifications.extend(new_modifications)

    return segments, modifications


def replace_segments_position_compared_with_all_segments_no_limit_pair(segments1, segments2, replace_number, seed):
    # min_segments = 0
    # max_segments = 1
    min_segments = -1
    max_segments = 2

    def find_min_dtw_segments(segments1, segments2):
        min_distance = float('inf')
        min_pair = (0, 0)

        for i in range(len(segments1)):
            for j in range(i + 1, len(segments2)):
                distance = dtw_metric(segments1[i], segments2[j])
                if distance < min_distance:
                    min_distance = distance
                    min_pair = (i, j)

        return min_pair, min_distance

    def increase_dtw_distance(segments1, segments2, i, j, seed):
        modifications = []

        current_distance = dtw_metric(segments1[i], segments2[j])

        count = 0
        while True:
            np.random.seed((seed + count + int(np.sum(segments1[i]))) % (2 ** 32 - 1))
            count += 1

            if np.ndim(segments1[0]) == 1:
                random_number = np.random.uniform(min_segments, max_segments)
            else:
                random_number = np.random.uniform(min_segments, max_segments, size=(np.ndim(segments1[0])))

            # 找出与其他分段最小距离中的最大者
            max_other_distance = -np.inf
            best_new_segment = None
            best_modifications = None
            for position in range(len(segments1[i])):
                new_segment = segments1[i][:position] + [random_number] + segments1[i][position + 1:]

                # 找出与其他分段的最小距离
                min_other_distance = np.inf
                for k in range(len(segments2)):
                    if k == i:
                        continue
                    other_distance = dtw_metric(new_segment, segments2[k])
                    if other_distance < min_other_distance:
                        min_other_distance = other_distance

                if min_other_distance > max_other_distance:
                    max_other_distance = min_other_distance
                    best_new_segment = new_segment
                    best_modifications = (i, position, random_number)

            if max_other_distance > current_distance or count > 10:
                segments1[i] = best_new_segment
                modifications.append(best_modifications)
                break

        # print("max_other_distance", max_other_distance, "min_new_dtw_distance", min_new_dtw_distance, "count", count)
        return segments1, modifications

    segments1 = [list(segment) for segment in segments1]
    segments2 = [list(segment) for segment in segments2]
    modifications = []

    for number in range(replace_number):
        min_pair, min_dist = find_min_dtw_segments(segments1, segments2)
        i, j = min_pair
        segments1, new_modifications = increase_dtw_distance(segments1, segments2, i, j, seed * number)
        modifications.extend(new_modifications)

    return segments1, modifications


def replace_segments_position_compared_with_all_segments(segments, ori_dist_threshold, replace_number, seed):
    # min_segments = 0
    # max_segments = 1
    min_segments = -1
    max_segments = 2

    def find_min_dtw_segments(segments):
        min_distance = float('inf')
        min_pair = (0, 0)

        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                distance = dtw_metric(segments[i], segments[j])
                if distance < min_distance:
                    min_distance = distance
                    min_pair = (i, j)

        return min_pair, min_distance

    def increase_dtw_distance(segments, i, j, seed):
        modifications = []

        original_segment = segments[i]
        current_distance = dtw_metric(segments[i], segments[j])

        count = 0
        while True:
            np.random.seed((seed + count + int(np.sum(segments[i]))) % (2 ** 32 - 1))
            count += 1

            if np.ndim(segments[0]) == 1:
                random_number = np.random.uniform(min_segments, max_segments)
            else:
                random_number = np.random.uniform(min_segments, max_segments, size=(np.ndim(segments[0])))

            # 找出与其他分段最小距离中的最大者
            max_other_distance = -np.inf
            # 找出与原始分段的最小距离
            min_new_dtw_distance = np.inf
            best_new_segment = None
            best_modifications = None
            for position in range(len(segments[i])):
                new_segment = segments[i][:position] + [random_number] + segments[i][position + 1:]
                new_distance = dtw_metric(new_segment, original_segment)

                if new_distance < min_new_dtw_distance:
                    min_new_dtw_distance = new_distance
                    # 找出与其他分段的最小距离
                    min_other_distance = np.inf

                    for k in range(len(segments)):
                        if k == i:
                            continue
                        other_distance = dtw_metric(new_segment, segments[k])
                        if other_distance < min_other_distance:
                            min_other_distance = other_distance

                    if min_other_distance > max_other_distance:
                        max_other_distance = min_other_distance
                        best_new_segment = new_segment
                        best_modifications = (i, position, random_number)

            if (min_new_dtw_distance < ori_dist_threshold and max_other_distance > current_distance) or count > 10:
                segments[i] = best_new_segment
                modifications.append(best_modifications)
                break

        # print("max_other_distance", max_other_distance, "min_new_dtw_distance", min_new_dtw_distance, "count", count)
        return segments, modifications

    segments = [list(segment) for segment in segments]
    modifications = []

    for number in range(replace_number):
        min_pair, min_dist = find_min_dtw_segments(segments)
        i, j = min_pair
        segments, new_modifications = increase_dtw_distance(segments, i, j, seed * number)
        modifications.extend(new_modifications)

    return segments, modifications


# segments = [
#     [1, 2, 3, 4, 5],
#     [2, 3, 4, 5, 6],
#     [3, 4, 5, 6, 7]
# ]
#
# replace_number = 2
# seed = 42
#
# modified_segments, modifications = replace_segments_position_compared_with_all_segments(segments, 1, replace_number, seed)
#
# print("Modified Segments:")
# for segment in modified_segments:
#     print(segment)
#
# print("\nModifications:")
# for mod in modifications:
#     print(mod)


def entropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, dataLen, seed):
    ts = CSIa1Orig / np.max(CSIa1Orig)
    # shanon_entropy = ent.shannon_entropy(ts)
    # perm_entropy = ent.permutation_entropy(ts, order=3, delay=1, normalize=True)
    # mulperm_entropy = ent.multiscale_permutation_entropy(ts, 3, 1, 1)
    mul_entropy = ent.multiscale_entropy(ts, 3, maxscale=1)

    cnts = 0
    entropyThres = 3
    while mul_entropy < entropyThres and cnts < 10:
        np.random.seed(seed + cnts)
        shuffleInd = np.random.permutation(dataLen)
        CSIa1Orig = CSIa1Orig[shuffleInd]
        CSIb1Orig = CSIb1Orig[shuffleInd]
        CSIe1Orig = CSIe1Orig[shuffleInd]

        ts = CSIa1Orig / np.max(CSIa1Orig)
        mul_entropy = ent.multiscale_entropy(ts, 4, maxscale=1)
        cnts += 1

    return CSIa1Orig, CSIb1Orig, CSIe1Orig


def splitEntropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, segLen, dataLen, seed):
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

        cnts = 0
        entropyThres = 2
        while a_mul_entropy < entropyThres and cnts < 10:
            np.random.seed(seed + cnts)
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


def optimize_random_block_vector_max_min(A, segment_method):
    def objective(K, A, segment_method):
        K = np.diag(K)  # 将 K 重塑为 n x n 矩阵
        B = K @ A  # B = K * A, 这里 B 是一个 1 x n 的向量

        B = B - np.mean(B)
        B = (B - np.min(B)) / (np.max(B) - np.min(B))
        B = segment_sequence(B, segment_method)

        return -compute_min_dtw(B, B)  # 因为我们使用最小化，所以取负数

    initial_K = np.random.normal(0, 1, len(A))

    optimizers = ['l-bfgs-b', 'Powell']

    optimizer = optimizers[0]
    result = minimize(objective, initial_K, args=(A, segment_method), method=optimizer)

    # 将最优解重塑为矩阵
    optimal_K = np.diag(result.x)
    return optimal_K
