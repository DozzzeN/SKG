import numpy as np


def step_corr(data1, data2, step, threshold):
    # 根据step和阈值计算两个序列的相似性
    n, m = len(data1), len(data2)

    corr = np.zeros(n)
    penalty = np.zeros(n)
    for i in range(n):
        distance = np.inf
        for j in range(max(0, i - step), min(m, i + step + 1)):
            distance = min(distance, np.abs(data1[i] - data2[j]))
        if distance > threshold:
            corr[i] = 0
            penalty[i] = distance
        # 如果一个窗内有匹配到的数据，则corr置为1
        else:
            corr[i] = 1
            penalty[i] = 0
    return np.sum(corr), np.sum(penalty)


def step_corr_metric(data1, data2, step, threshold):
    step_corr_res1 = step_corr(data1, data2, step, threshold)
    step_corr_res2 = step_corr(data2, data1, step, threshold)
    return min(step_corr_res1[0], step_corr_res2[0]), min(step_corr_res1[1], step_corr_res2[1])
    # return min(step_corr(data1, data2, step, threshold), step_corr(data2, data1, step, threshold))


def matching(a, b, length):
    matched_indices = []
    matched_single_indices = []
    matched_values = []
    i = 0
    while i < len(b):
        segment_b = b[i:i + length]
        min_corr = 0
        min_penalty = length * length
        match_index = -1
        j = 0
        while j < len(a):
            if len(set(range(j, j + length)) & set(matched_indices)) > 0:
                j += 1
                continue
            segment_a = a[j:j + length]
            corr, penalty = step_corr_metric(segment_a, segment_b, length, 0)
            if corr >= min_corr:
                if penalty < min_penalty:
                    min_corr = corr
                    min_penalty = penalty
                    match_index = j
            j += 1
        if match_index != -1:
            matched_indices.extend([match_index + _ for _ in range(length)])
            matched_single_indices.append(match_index)
            matched_values.append([a[(match_index + _) % len(a)] for _ in range(length)])
            i += length
        else:
            i += 1
    print(matched_values)
    print(matched_single_indices)
    print(np.argsort(matched_single_indices))


ap = [12, 3, 9, 11, 4, 2, 6, 10, 5, 13, 7, 8, 14, 0, 15, 1]
a = [12, 3, 9, 11, 4, 0, 15, 1, 13, 7, 8, 14, 2, 6, 10, 5]
b = [13, 3, 8, 10, 4, 0, 15, 1, 12, 7, 9, 14, 2, 6, 11, 5]

matching(a, ap, 2)
matching(b, ap, 2)
