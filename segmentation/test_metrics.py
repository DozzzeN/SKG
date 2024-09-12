import itertools

import matplotlib.pyplot as plt
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


def calculate_distance(data1, data2):
    # return (x1 - x2) ** 2 + (y1 - y2) ** 2
    return abs(data1[0] - data2[0]) + abs(data1[1] - data2[1])


def dtw_metric(data1, data2):
    distance = lambda x, y: np.abs(x - y)
    data1 = np.array(data1)
    data2 = np.array(data2)
    # return dtw(data1, data2, dist=distance)[0]
    return accelerated_dtw(data1, data2, dist=distance)


def shortest_distance(solution):
    min_distance = float('inf')  # 初始化最小距离为正无穷
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            distance = dtw_metric(solution[i], solution[j])[0]
            min_distance = min(min_distance, distance)  # 更新最小距离
    return min_distance


def total_distance(solution):
    total = 0
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            distance = dtw_metric(solution[i], solution[j])[0]
            total += distance
    return total


def all_distance(solution):
    total = []
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            distance = dtw_metric(solution[i], solution[j])[0]
            total.append(distance)
    return total


# 遍历所有分段情况，计算不同的度量指标
M = 8
segment_methods = partition(M)
max_min_distances = []
mean_sum_distances = []
mean_in_all_distances = []
var_sum_distances = []
var_in_all_distances = []

for segment_method in segment_methods:
    if len(segment_method) == 1:
        continue
    max_min_distance = float('-inf')  # 初始化最大最小距离为负无穷
    max_min_distance_solutions = []
    sum_distances = []
    in_all_distances = []
    for combination in itertools.permutations(range(M)):
        new_shape = segment_sequence(combination, segment_method)
        min_distance = shortest_distance(new_shape)
        if min_distance > max_min_distance:
            max_min_distance = min_distance
            max_min_distance_solutions = [new_shape]
        elif min_distance == max_min_distance:
            max_min_distance_solutions.append(new_shape)
        sum_distances.append(total_distance(new_shape))
        in_all_distances.append(all_distance(new_shape))

    print("当前分割方法为:", segment_method)
    print("所有结果中最大的最小距离为:", max_min_distance)
    print("对应的所有解为:")
    # for idx, solution, in enumerate(max_min_distance_solutions, start=1):
    #     print("Solution", idx, np.sort(solution).tolist(), shortest_distance(solution), total_distance(solution))
    print("共有", len(max_min_distance_solutions), "个解")
    print("所有解的平均总距离为:", np.mean(sum_distances))
    print("所有解的总距离方差为", np.var(sum_distances))
    mean_mean = np.mean([np.mean(i) for i in in_all_distances])
    print("所有解的平均平均距离为:", mean_mean)
    mean_var = np.mean([np.var(i) for i in in_all_distances])
    print("所有解的平均平均距离方差为:", mean_var)

    max_min_distances.append(max_min_distance)
    mean_sum_distances.append(np.mean(sum_distances))
    mean_in_all_distances.append(mean_mean)
    var_sum_distances.append(np.var(sum_distances))
    var_in_all_distances.append(mean_var)
    print()

# 绘图
plt.figure()
plt.plot(range(len(max_min_distances)), max_min_distances, label="Max Min Distance")
plt.plot(range(len(mean_sum_distances)), mean_sum_distances, label="Mean Sum Distance")
plt.plot(range(len(mean_in_all_distances)), mean_in_all_distances, label="Mean In All Distance")
plt.plot(range(len(var_sum_distances)), var_sum_distances, label="Var Sum Distance")
plt.plot(range(len(var_in_all_distances)), var_in_all_distances, label="Var In All Distance")
plt.legend()
# 横坐标是所有的分割方法
plt.xticks(range(len(segment_methods)), segment_methods)
# 横坐标太长，旋转45度
plt.xticks(rotation=45)
# 横坐标超出范围，自动调整
plt.tight_layout()
plt.xlabel("Segment Method")
plt.ylabel("Distance")
# 保存
# plt.savefig('distance.png')
plt.show()
