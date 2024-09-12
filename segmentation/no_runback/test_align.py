import numpy as np
from dtw import accelerated_dtw

# 示例数据
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])

# 使用accelerated_dtw进行对齐
dist, cost, acc_cost, path = accelerated_dtw(x, y, dist='euclidean')

print(f'Distance: {dist}')
print(f'Cost matrix: {cost}')
print(f'Accumulated cost matrix: {acc_cost}')
print(f'Warping path: {path}')

# 获取变形路径
path_x, path_y = path

# 构建对齐后的序列
aligned_x = x[path_x]
aligned_y = y[path_y]

print(f'原始序列 x: {x}')
print(f'原始序列 y: {y}')
print(f'对齐后的序列 x: {aligned_x}')
print(f'对齐后的序列 y: {aligned_y}')
print(f'对齐后的距离: {np.sum(np.abs(aligned_x - aligned_y))}')
