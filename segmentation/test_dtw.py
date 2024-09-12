import numpy as np
from dtw import accelerated_dtw

# 根据步长step，计算对应data1中每个元素与其最近的data2中的元素的“最小”距离
# 所有距离之和作为两个序列的相似性
def step_dtw_distance(data1, data2, step):
    n, m = len(data1), len(data2)

    dtw_matrix = np.zeros(n)
    for i in range(n):
        distance = np.inf
        for j in range(max(0, i - step), min(m, i + step + 1)):
            distance = min(distance, np.abs(data1[i] - data2[j]))
        dtw_matrix[i] = distance
    return np.sum(dtw_matrix)


def step_dtw_metric(data1, data2, step):
    return min(step_dtw_distance(data1, data2, step), step_dtw_distance(data2, data1, step))

def dtw_metric(data1, data2):
    distance = lambda x, y: np.abs(x - y)
    data1 = np.array(data1)
    data2 = np.array(data2)
    return accelerated_dtw(data1, data2, dist=distance)

# 给出一种新的计算两个序列相似性的方法
# print(dtw_metric([16, 25, 11, 7, 26, 27, 29, 4], [26, 25, 16, 7, 11, 27, 29, 4])[0])
# print(step_dtw_metric([16, 25, 11, 7, 26, 27, 29, 4], [26, 25, 16, 7, 11, 27, 29, 4], 3))
# print(step_dtw_metric([26, 25], [16, 25, 11], 1))
# print(step_dtw_metric([16, 25, 11],[26, 25],  1))
