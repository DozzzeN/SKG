import numpy as np
from scipy.optimize import linear_sum_assignment

def min_total_euclidean_distance(A, B):
    # 找到欧式距离最小的对应数组(匈牙利算法)
    n = len(A)
    cost_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            cost_matrix[i][j] = np.linalg.norm(np.array(A[i]) - np.array(B[j]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    min_distance = cost_matrix[row_ind, col_ind].sum()

    # 生成配对方式
    pairs = list(zip(row_ind, col_ind))

    return pairs, min_distance

def calculate_distances(A, B, pairs):
    distances = []
    for i, j in pairs:
        distance = np.linalg.norm(np.array(A[i]) - np.array(B[j]))
        distances.append(distance)
    return distances

# 示例使用
A = [[1, 2], [3, 4], [5, 6]]
B = [[4, 5], [6, 7], [2, 3]]
pairs, min_distance = min_total_euclidean_distance(A, B)
distances = calculate_distances(A, B, pairs)

print("配对方式:", pairs)
print("最小总欧式距离:", min_distance)
print("每个配对的欧式距离:", sum(distances))
pairs_a = [pair[0] for pair in pairs]
pairs_b = [pair[1] for pair in pairs]
print(np.linalg.norm(np.array(A)[pairs_a] - np.array(B)[pairs_b]))
