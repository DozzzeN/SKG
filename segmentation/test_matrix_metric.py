import numpy as np
from dtw import accelerated_dtw

def dtw_metric(data1, data2):
    distance = lambda x, y: np.abs(x - y)
    data1 = np.array(data1)
    data2 = np.array(data2)
    # return dtw(data1, data2, dist=distance)[0]
    return accelerated_dtw(data1, data2, dist=distance)

# 给出一个分段后的序列，计算各个分段之间的DTW距离矩阵
a = np.array([[3, 1], [6], [5, 2], [0, 4, 7]], dtype=object)

dists = []
for i in range(len(a)):
    dist = []
    for j in range(len(a)):
        # dist.append(np.sum(np.abs(a[i] - a[j])))
        dist.append(dtw_metric(a[i], a[j])[0])
    dists.append(dist)
print(np.array(dists))