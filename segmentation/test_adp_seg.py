from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
from dtw import accelerated_dtw
from scipy.stats import pearsonr

from test_partition import partition


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


def shortest_distance(solution1, solution2):
    min_distance = float('inf')
    for i in range(len(solution1)):
        for j in range(len(solution2)):
            if i == j:
                continue
            distance = dtw_metric(solution1[i], solution2[j])[0]
            min_distance = min(min_distance, distance)  # 更新最小距离
    return min_distance


def mean_shortest_distance(solution1, solution2):
    min_distances = []
    for i in range(len(solution1)):
        min_distance = float('inf')
        for j in range(len(solution2)):
            if i == j:
                continue
            distance = dtw_metric(solution1[i], solution2[j])[0]
            min_distance = min(min_distance, distance)
        min_distances.append(min_distance)
    return np.mean(min_distances)


def total_distance(solution1, solution2):
    total = 0
    for i in range(len(solution1)):
        for j in range(len(solution2)):
            if i == j:
                continue
            distance = dtw_metric(solution1[i], solution2[j])[0]
            total += distance
    return total


def all_distance(solution1, solution2):
    total = []
    for i in range(len(solution1)):
        for j in range(len(solution2)):
            if i == j:
                continue
            distance = dtw_metric(solution1[i], solution2[j])[0]
            total.append(distance)
    return total


# 仿真分析：测试不同分段方法的密钥生成准确率，和分段之间的最小DTW距离
# 与test_metric一起分析准确率和度量指标的对应关系
M = 8
result = partition(M)
segment_methods = []
for segment in result:
    all_permutations = list(permutations(segment))
    for permutation in all_permutations:
        if permutation not in segment_methods and len(permutation) > 1:
            segment_methods.append(permutation)

match_rates = []
# 只考虑A的数据特性
min_min_distances = []
mean_min_distances = []
mean_in_all_distances = []
mean_sum_distances = []
var_in_all_distances = []
var_sum_distances = []

# 综合考虑双方数据特性
pair_min_min_distances = []
pair_mean_min_distances = []
pair_mean_in_all_distances = []
pair_mean_sum_distances = []
pair_var_in_all_distances = []
pair_var_sum_distances = []

for i in range(100):
    # 对所有方法保持数据一致
    data1 = np.random.normal(0, 1, M)
    SNR_dB = 0  # 期望的信噪比（dB）
    signal_energy = np.var(data1)
    noise_energy = signal_energy / (10 ** (SNR_dB / 10))  # 根据信噪比计算噪声能量
    noise_data = np.random.normal(0, np.sqrt(noise_energy), M)
    data2 = data1 + noise_data

    # print(pearsonr(data1, data2)[0])
    # data1 = np.argsort(data1)
    # data2 = np.argsort(data2)
    # print(pearsonr(data1, data2)[0])

    match_rate = []
    min_min_distance = []
    mean_min_distance =[]
    mean_in_all_distance = []
    mean_sum_distance = []
    var_in_all_distance = []
    var_sum_distance = []

    pair_min_min_distance = []
    pair_mean_min_distance =[]
    pair_mean_in_all_distance = []
    pair_mean_sum_distance = []
    pair_var_in_all_distance = []
    pair_var_sum_distance = []

    for segment_method in segment_methods:
        data1_seg = segment_sequence(data1, segment_method)
        data2_seg = segment_sequence(data2, segment_method)

        # np.random.seed(0)
        permutation = list(range(len(segment_method)))
        combineMetric = list(zip(data2_seg, permutation))
        np.random.shuffle(combineMetric)
        data2_seg, permutation = zip(*combineMetric)

        # 用DTW度量
        dtw = []
        est_index_dtw = np.array([])
        for i in range(len(data1_seg)):
            dtw.append([])
            for j in range(len(data2_seg)):
                dtw[-1].append(dtw_metric(data1_seg[i], data2_seg[j])[0])
            est_index_dtw = np.append(est_index_dtw, np.argmin(dtw[-1]))

        # 计算最小距离
        min_min_distance.append(shortest_distance(data1_seg, data1_seg))
        pair_min_min_distance.append(shortest_distance(data1_seg, data2_seg))
        # 计算平均最小距离
        mean_min_distance.append(mean_shortest_distance(data1_seg, data1_seg))
        pair_mean_min_distance.append(mean_shortest_distance(data1_seg, data2_seg))
        # 计算所有的距离
        mean_in_all_distance.append(all_distance(data1_seg, data1_seg))
        pair_mean_in_all_distance.append(all_distance(data1_seg, data2_seg))
        var_in_all_distance.append(all_distance(data1_seg, data1_seg))
        pair_var_in_all_distance.append(all_distance(data1_seg, data2_seg))
        # 计算总距离
        mean_sum_distance.append(total_distance(data1_seg, data1_seg))
        pair_mean_sum_distance.append(total_distance(data1_seg, data2_seg))
        var_sum_distance.append(total_distance(data1_seg, data1_seg))
        pair_var_sum_distance.append(total_distance(data1_seg, data2_seg))

        match_rate.append(np.mean(est_index_dtw == permutation))

    min_min_distances.append(min_min_distance)
    pair_min_min_distances.append(pair_min_min_distance)
    mean_min_distances.append(mean_min_distance)
    pair_mean_min_distances.append(pair_mean_min_distance)
    mean_in_all_distances.append(mean_in_all_distance)
    pair_mean_in_all_distances.append(pair_mean_in_all_distance)
    mean_sum_distances.append(mean_sum_distance)
    pair_mean_sum_distances.append(pair_mean_sum_distance)
    var_in_all_distances.append(var_in_all_distance)
    pair_var_in_all_distances.append(pair_var_in_all_distance)
    var_sum_distances.append(var_sum_distance)
    pair_var_sum_distances.append(pair_var_sum_distance)

    match_rates.append(match_rate)

mean_mean = np.zeros(len(segment_methods))
pair_mean_mean = np.zeros(len(segment_methods))
mean_var = np.zeros(len(segment_methods))
pair_mean_var = np.zeros(len(segment_methods))
for i in range(10000):
    for j in range(len(segment_methods)):
        mean_mean[j] += np.mean(mean_in_all_distances[i][j])
        pair_mean_mean[j] += np.mean(pair_mean_in_all_distances[i][j])
        mean_var[j] += np.var(var_in_all_distances[i][j])
        pair_mean_var[j] += np.var(pair_var_in_all_distances[i][j])
mean_mean /= 10000
pair_mean_mean /= 10000
mean_var /= 10000
pair_mean_var /= 10000
for i in range(len(segment_methods)):
    print(segment_methods[i], np.mean(match_rates, axis=0)[i], np.mean(min_min_distances, axis=0)[i],
          np.mean(mean_min_distances, axis=0)[i], mean_mean[i],
          np.mean(mean_sum_distances, axis=0)[i], mean_var[i], np.var(var_sum_distances, axis=0)[i])
    print(segment_methods[i], np.mean(match_rates, axis=0)[i], np.mean(pair_min_min_distances, axis=0)[i],
            np.mean(pair_mean_min_distances, axis=0)[i], pair_mean_mean[i],
            np.mean(pair_mean_sum_distances, axis=0)[i], pair_mean_var[i], np.var(pair_var_sum_distances, axis=0)[i])

plt.figure()
plt.title('Match Rate')
plt.plot(np.mean(match_rates, axis=0))
plt.xticks(range(len(segment_methods)), segment_methods)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./plot/match_rate' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Min Min Distance')
plt.plot(np.mean(min_min_distances, axis=0))
plt.xticks(range(len(segment_methods)), segment_methods)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./plot/min_min_distance' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Mean Min Distance')
plt.plot(np.mean(mean_min_distances, axis=0))
plt.xticks(range(len(segment_methods)), segment_methods)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./plot/mean_min_distance' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Mean Sum Distance')
plt.plot(np.mean(mean_sum_distances, axis=0))
plt.xticks(range(len(segment_methods)), segment_methods)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./plot/sum_distance' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Mean In All Distance')
plt.plot(mean_mean)
plt.xticks(range(len(segment_methods)), segment_methods)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./plot/in_all_distance' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Var Sum Distance')
plt.plot(np.var(var_sum_distances, axis=0))
plt.xticks(range(len(segment_methods)), segment_methods)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./plot/var_sum_distance' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Var In All Distance')
plt.plot(mean_var)
plt.xticks(range(len(segment_methods)), segment_methods)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./plot/var_in_all_distance' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Pair Min Min Distance')
plt.plot(np.mean(pair_min_min_distances, axis=0))
plt.xticks(range(len(segment_methods)), segment_methods)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./plot/pair_min_min_distance' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Pair Mean Min Distance')
plt.plot(np.mean(pair_mean_min_distances, axis=0))
plt.xticks(range(len(segment_methods)), segment_methods)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./plot/pair_mean_min_distance' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Pair Mean Sum Distance')
plt.plot(np.mean(pair_mean_sum_distances, axis=0))
plt.xticks(range(len(segment_methods)), segment_methods)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./plot/pair_sum_distance' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Pair Mean In All Distance')
plt.plot(pair_mean_mean)
plt.xticks(range(len(segment_methods)), segment_methods)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./plot/pair_in_all_distance' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Pair Var Sum Distance')
plt.plot(np.var(pair_var_sum_distances, axis=0))
plt.xticks(range(len(segment_methods)), segment_methods)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./plot/pair_var_sum_distance' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Pair Var In All Distance')
plt.plot(pair_mean_var)
plt.xticks(range(len(segment_methods)), segment_methods)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./plot/pair_var_in_all_distance' + str(M) + '.png')
plt.show()

# 画出最小距离的直方图
plt.figure()
plt.title('Min Min Distance Histogram')
plt.hist(np.mean(min_min_distances, axis=0), bins=20)
plt.savefig('./plot/min_min_distance_hist' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Pair Min Min Distance Histogram')
plt.hist(np.mean(pair_min_min_distances, axis=0), bins=20)
plt.savefig('./plot/pair_min_min_distance_hist' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Mean Min Distance Histogram')
plt.hist(np.mean(mean_min_distances, axis=0), bins=20)
plt.savefig('./plot/mean_min_distance_hist' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Pair Mean Min Distance Histogram')
plt.hist(np.mean(pair_mean_min_distances, axis=0), bins=20)
plt.savefig('./plot/pair_mean_min_distance_hist' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Mean Sum Distance Histogram')
plt.hist(np.mean(mean_sum_distances, axis=0), bins=20)
plt.savefig('./plot/sum_distance_hist' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Pair Mean Sum Distance Histogram')
plt.hist(np.mean(pair_mean_sum_distances, axis=0), bins=20)
plt.savefig('./plot/pair_sum_distance_hist' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Mean In All Distance Histogram')
plt.hist(mean_mean, bins=20)
plt.savefig('./plot/in_all_distance_hist' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Pair Mean In All Distance Histogram')
plt.hist(pair_mean_mean, bins=20)
plt.savefig('./plot/pair_in_all_distance_hist' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Var Sum Distance Histogram')
plt.hist(np.var(var_sum_distances, axis=0), bins=20)
plt.savefig('./plot/var_sum_distance_hist' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Pair Var Sum Distance Histogram')
plt.hist(np.var(pair_var_sum_distances, axis=0), bins=20)
plt.savefig('./plot/pair_var_sum_distance_hist' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Var In All Distance Histogram')
plt.hist(mean_var, bins=20)
plt.savefig('./plot/var_in_all_distance_hist' + str(M) + '.png')
plt.show()

plt.figure()
plt.title('Pair Var In All Distance Histogram')
plt.hist(pair_mean_var, bins=20)
plt.savefig('./plot/pair_var_in_all_distance_hist' + str(M) + '.png')
plt.show()