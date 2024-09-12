import numpy as np
from dtw import accelerated_dtw


def dtw_metric(data1, data2):
    distance = lambda x, y: np.abs(x - y)
    data1 = np.array(data1)
    data2 = np.array(data2)
    # return dtw(data1, data2, dist=distance)[0]
    return accelerated_dtw(data1, data2, dist=distance)

# 尝试将分段填充至等长
arr = [[6, 1], [3, 4, 2], [5]]
# arr = [[6, 1, 6, 1, 6, 1], [3, 4, 2, 3, 4, 2], [5, 5, 5, 5, 5, 5]]
for i in range(len(arr)):
    for j in range(i + 1, len(arr)):
        print(dtw_metric(arr[i], arr[j])[0])
