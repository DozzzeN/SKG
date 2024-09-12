import time
from itertools import permutations

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
            dtw = dtw_metric(data[i], data[j])[0] / max(len(data[i]), len(data[j]))
            # dtw = dtw_metric(data[i], data[j])[0]
            if dtw < min_dtw:
                min_dtw = dtw
    return min_dtw

# 三种方法自下而上合并，找出分段方法中的最优解：分段之间最小的DTW距离最大
# data = [3, 1, 2, 5, 4, 7, 6, 8]
# data = [7, 2, 8, 3, 4, 1, 6]
np.random.seed(0)
data = np.random.permutation(9)

########################################################################################
print("方法一，贪心法")
num_segments = 3
algorithm_start = time.time()
cal_dtw_times = 0
each_layer_candidate = [[[1 for _ in range(len(data))]]]
each_layer_optimal = [[[1 for _ in range(len(data))]]]
# 遍历每一层，即每个分段长度
for layer in range(len(data) - num_segments):
    # 每层的最大距离
    max_distance = -np.inf
    segment_methods = each_layer_optimal[layer]
    opt_segment_method = []
    each_layer_candidate.append([])
    each_layer_optimal.append([])
    for segment_method in segment_methods:
        for i in range(len(segment_method) - 1):
            segment_method_temp = segment_method[:i] + [segment_method[i] + segment_method[i + 1]] + segment_method[i + 2:]
            # 已经计算过距离了，直接跳过
            if segment_method_temp in each_layer_candidate[layer + 1]:
                continue
            else:
                each_layer_candidate[layer + 1].append(segment_method_temp)
            segment = segment_sequence(data, segment_method_temp)
            cal_dtw_times += 1
            distance_temp = compute_min_dtw(segment)
            # 找到了更优的分段方法，将其加入到新的分段方法中
            if distance_temp > max_distance:
                max_distance = distance_temp
                opt_segment_method = [segment_method_temp]
            # 找到了与最优分段方法相同的分段方法，将其加入到新的分段方法中
            elif np.isclose(distance_temp, max_distance, atol=1e-3):
                if segment_method_temp not in each_layer_candidate[layer] and segment_method_temp not in opt_segment_method:
                    opt_segment_method.append(segment_method_temp)
    # 按照不同的分段个数放入此时最优的分段方法
    each_layer_optimal[layer + 1].extend(opt_segment_method)
    # 以下方法同方法二
    # if len(opt_segment_method) > 0:
    #     each_layer_optimal[layer + 1].append(opt_segment_method[0])
algorithm_end = time.time()

if num_segments == 1:
    each_layer_optimal.remove([])
each_layer_optimal.reverse()

for i in range(len(each_layer_optimal)):
    if each_layer_optimal[i] == []:
        continue
    print("分段数:", len(each_layer_optimal[i][0]))
    for segment_method in each_layer_optimal[i]:
        segment = segment_sequence(data, segment_method)
        print(segment, compute_min_dtw(segment))
print("贪心法一耗时:", algorithm_end - algorithm_start)
print("计算DTW次数:", cal_dtw_times)

########################################################################################
print("方法二，动态规划")
algorithm_start = time.time()
cal_dtw_times = 0
for num_segments in range(1, len(data) + 1):
    segment_start = 1
    num_segment_start = len(data) // segment_start
    indices = np.zeros(len(data) + 1, dtype=int)
    for i in range(1, num_segment_start):
        indices[i] = indices[i - 1] + segment_start
    indices[num_segment_start] = len(data)
    k = num_segment_start

    # 每次选最小的DTW距离是次优的
    while k != num_segments:
        index_to_merge_position = 0
        # 最小距离最大
        distance = -np.inf
        for i in range(1, k):
            indices_temp = np.delete(indices, i)
            segment = new_segment_sequence(data, indices_temp)
            cal_dtw_times += 1
            distance_temp = compute_min_dtw(segment)
            if distance_temp > distance:
                distance = distance_temp
                index_to_merge_position = i
        indices = np.delete(indices, index_to_merge_position)
        k -= 1
    print("分段数:", num_segments)
    print(new_segment_sequence(data, indices), compute_min_dtw(new_segment_sequence(data, indices)))
algorithm_end = time.time()
print("动态规划耗时:", algorithm_end - algorithm_start)
print("计算DTW次数:", cal_dtw_times)

########################################################################################
print("方法三，穷举")
search_start = time.time()
cal_dtw_times = 0
for num_segments in range(1, len(data) + 1):
    # 生成所有分段方法，计算分段间的最小DTW距离
    all_segments = partition(len(data), 1, num_partitions=num_segments)
    segment_methods = []
    for segment in all_segments:
        all_permutations = list(set(permutations(segment)))
        for permutation in all_permutations:
            if permutation not in segment_methods:
                segment_methods.append(permutation)

    max_distance = -np.inf
    opt_data_seg = []
    for segment_method in segment_methods:
        data_seg = segment_sequence(data, segment_method)
        cal_dtw_times += 1
        dist = compute_min_dtw(data_seg)
        if dist > max_distance:
            max_distance = dist
            opt_data_seg = [segment_method]
        elif dist == max_distance:
            if segment_method not in opt_data_seg:
                opt_data_seg.append(segment_method)
    print("分段数:", num_segments)
    for data_seg in opt_data_seg:
        print(segment_sequence(data, data_seg), compute_min_dtw(segment_sequence(data, data_seg)))
search_end = time.time()
print("穷举搜索耗时:", search_end - search_start)
print("计算DTW次数:", cal_dtw_times)

########################################################################################
print("方法四，合并")
search_start = time.time()
cal_dtw_times = 0

base_length = 1
each_layer_candidate = [[[base_length for _ in range(int(len(data) / base_length))]]]
each_layer_optimal = [[[base_length for _ in range(int(len(data) / base_length))]]]
# 遍历每一层，即每个分段长度
for layer in range(int(len(data) / base_length) - 1):
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
            cal_dtw_times += 1
            distance_temp = compute_min_dtw(segment)
            # 找到了更优的分段方法，将其加入到新的分段方法中
            if distance_temp > max_distance:
                max_distance = distance_temp
                opt_segment_method = [segment_method_temp]
            # 找到了与最优分段方法相同的分段方法，将其加入到新的分段方法中
            elif np.isclose(distance_temp, max_distance, atol=1e-3):
                if (segment_method_temp not in each_layer_candidate[layer]
                        and segment_method_temp not in opt_segment_method):
                    opt_segment_method.append(segment_method_temp)
    # 按照不同的分段个数放入此时最优的分段方法
    each_layer_optimal[layer + 1].extend(opt_segment_method)

each_layer_optimal.reverse()
for segments in each_layer_optimal:
    print("分段数:", len(segments[0]))
    for segment_method in segments:
        segment = segment_sequence(data, segment_method)
        print(segment, compute_min_dtw(segment))
search_end = time.time()
print("合并搜索耗时:", search_end - search_start)
print("计算DTW次数:", cal_dtw_times)
########################################################################################
print("方法四，合并取唯一")
search_start = time.time()
cal_dtw_times = 0

base_length = 1
each_layer_candidate = [[[base_length for _ in range(int(len(data) / base_length))]]]
each_layer_optimal = [[[base_length for _ in range(int(len(data) / base_length))]]]
# 遍历每一层，即每个分段长度
for layer in range(int(len(data) / base_length) - 1):
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
            cal_dtw_times += 1
            distance_temp = compute_min_dtw(segment)
            # 找到了更优的分段方法，将其加入到新的分段方法中
            if distance_temp > max_distance:
                max_distance = distance_temp
                opt_segment_method = [segment_method_temp]
            # 找到了与最优分段方法相同的分段方法，将其加入到新的分段方法中
            # elif np.isclose(distance_temp, max_distance, atol=1e-3):
            #     if (segment_method_temp not in each_layer_candidate[layer]
            #             and segment_method_temp not in opt_segment_method):
            #         opt_segment_method.append(segment_method_temp)
    # 按照不同的分段个数放入此时最优的分段方法
    each_layer_optimal[layer + 1].extend(opt_segment_method)

each_layer_optimal.reverse()
for segment in each_layer_optimal:
    print("分段数:", len(segment[0]))
    print(segment[0], compute_min_dtw(segment_sequence(data, segment[0])))
search_end = time.time()
print("合并取唯一搜索耗时:", search_end - search_start)
print("计算DTW次数:", cal_dtw_times)
