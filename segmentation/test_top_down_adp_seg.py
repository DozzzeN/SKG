import itertools
import pickle
import time
from itertools import permutations
from test_permutations import unique_permutations
import numpy as np
from dtw import accelerated_dtw

from segmentation.test_partition import partition


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
    return accelerated_dtw(data1, data2, dist=distance)


def new_segment_sequence(data, segment_lengths):
    segments = []
    for i in range(1, len(segment_lengths)):
        segments.append(data[segment_lengths[i - 1]:segment_lengths[i]])
    return segments


def compute_min_dtw(data):
    min_dtw = np.inf
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            dtw = dtw_metric(data[i], data[j])[0]
            if dtw < min_dtw:
                min_dtw = dtw
    return min_dtw


# 三种方法自上而下分段，找出分段方法中的最优解：分段之间最小的DTW距离最大
# data = [3, 1, 2, 5, 4, 7, 6, 8]
# np.random.seed(0)
data = np.random.permutation(4 * 16)
min_length = 3
max_length = 5
num_segments = 16
########################################################################################
print("方法一，贪心法")
algorithm_start = time.time()
cal_dtw_times = 0
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
                cal_dtw_times += 1
                distance_temp = compute_min_dtw(segment)
                # 找到了更优的分段方法，将其加入到新的分段方法中
                if distance_temp > max_distance[layer]:
                    max_distance[layer] = distance_temp
                    opt_segment_method = [segment_method_temp]
                # 找到了与最优分段方法相同的分段方法，将其加入到新的分段方法中
                # elif np.isclose(distance_temp, max_distance[layer], atol=1e-3):
                #     if (segment_method_temp not in each_layer_candidate[layer]
                #             and segment_method_temp not in opt_segment_method):
                #         opt_segment_method.append(segment_method_temp)
    # 放入所有分段情况下的第一个最优分段方法，而不是所有的分段方法
    if len(opt_segment_method) > 0:
        each_layer_optimal[layer + 1].append(opt_segment_method[0])
    print(each_layer_optimal[layer + 1])
algorithm_end = time.time()
# print(max_distance[num_segments - 2])

each_layer_optimal.reverse()

for i in range(len(each_layer_optimal)):
    if each_layer_optimal[i] == []:
        continue
    print("分段数:", len(each_layer_optimal[i][0]))
    for segment_method in each_layer_optimal[i]:
        segment = segment_sequence(data, segment_method)
        print(segment_method, compute_min_dtw(segment))
print("贪心法一耗时:", algorithm_end - algorithm_start)
print("计算DTW次数:", cal_dtw_times)
print()
########################################################################################
print("方法二，动态规划")
algorithm_start = time.time()
cal_dtw_times = 0
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
                cal_dtw_times += 1
                distance_temp = compute_min_dtw(segment)
                # 找到了更优的分段方法，将其加入到新的分段方法中
                if distance_temp > max_distance[layer]:
                    max_distance[layer] = distance_temp
                    opt_segment_method = [segment_method_temp]
                # 找到了与最优分段方法相同的分段方法，将其加入到新的分段方法中
                elif np.isclose(distance_temp, max_distance[layer], atol=1e-3):
                    if segment_method_temp not in each_layer_candidate[
                        layer] and segment_method_temp not in opt_segment_method:
                        opt_segment_method.append(segment_method_temp)
    # 按照不同的分段个数放入此时所有的最优的分段方法
    each_layer_optimal[layer + 1].extend(opt_segment_method)
algorithm_end = time.time()

each_layer_optimal.reverse()

for i in range(len(each_layer_optimal)):
    if each_layer_optimal[i] == []:
        continue
    print("分段数:", len(each_layer_optimal[i][0]))
    for segment_method in each_layer_optimal[i]:
        segment = segment_sequence(data, segment_method)
        print(segment_method, compute_min_dtw(segment))
print("动态规划耗时:", algorithm_end - algorithm_start)
print("计算DTW次数:", cal_dtw_times)
print()
########################################################################################
print("方法三，穷举")
search_start = time.time()
cal_dtw_times = 0
# 生成所有分段方法，计算分段间的最小DTW距离
all_segments = partition(len(data), min_length=min_length, max_length=max_length, num_partitions=num_segments)
segment_methods = []
# for segment in all_segments:
#     all_permutations = list(set(permutations(segment)))
#     for permutation in all_permutations:
#         if permutation not in segment_methods:
#             segment_methods.append(permutation)
for segment in all_segments:
    segment_methods += unique_permutations(segment)
print("分段方法数:", len(segment_methods))
max_distance = -np.inf
opt_data_seg = []
for segment_method in segment_methods:
    data_seg = segment_sequence(data, segment_method)
    cal_dtw_times += 1
    dist = compute_min_dtw(data_seg)
    # print(segment_sequence(data, segment_method), dist)
    if dist > max_distance:
        max_distance = dist
        opt_data_seg = [segment_method]
    elif dist == max_distance:
        if segment_method not in opt_data_seg:
            opt_data_seg.append(segment_method)
print("最优分段方法:")
for data_seg in opt_data_seg:
    # print(segment_sequence(data, data_seg), compute_min_dtw(segment_sequence(data, data_seg)))
    print(data_seg, compute_min_dtw(segment_sequence(data, data_seg)))
search_end = time.time()
print("穷举搜索耗时:", search_end - search_start)
print("计算DTW次数:", cal_dtw_times)

# search_start = time.time()
# cal_dtw_times = 0
# for num_segments in range(1, len(data) + 1):
#     print("分段数:", num_segments)
#     # 生成所有分段方法，计算分段间的最小DTW距离
#     all_segments = partition(len(data), min_length=min_length, max_length=max_length, num_partitions=num_segments)
#     segment_methods = []
#     for segment in all_segments:
#         all_permutations = list(set(permutations(segment)))
#         for permutation in all_permutations:
#             if permutation not in segment_methods:
#                 segment_methods.append(permutation)
#
#     max_distance = -np.inf
#     opt_data_seg = []
#     for segment_method in segment_methods:
#         data_seg = segment_sequence(data, segment_method)
#         cal_dtw_times += 1
#         dist = compute_min_dtw(data_seg)
#         # print(segment_sequence(data, segment_method), dist)
#         if dist > max_distance:
#             max_distance = dist
#             opt_data_seg = [segment_method]
#         elif dist == max_distance:
#             if segment_method not in opt_data_seg:
#                 opt_data_seg.append(segment_method)
#     print("最优分段方法:")
#     for data_seg in opt_data_seg:
#         # print(segment_sequence(data, data_seg), compute_min_dtw(segment_sequence(data, data_seg)))
#         print(data_seg, compute_min_dtw(segment_sequence(data, data_seg)))
# search_end = time.time()
# print("穷举搜索耗时:", search_end - search_start)
# print("计算DTW次数:", cal_dtw_times)
