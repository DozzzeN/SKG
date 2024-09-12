import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

# 原始一维向量
x = np.array([[1], [2], [3], [4]])

# 使用RBF核映射到高维空间
gamma = 0.1
x_high_dim = rbf_kernel(x, gamma=gamma)

print(x_high_dim)

# 原始一维时间序列
x = np.array([1, 2, 3, 4, 5, 6])

# 时间序列展开
window_size = 3
x_expanded = np.lib.stride_tricks.sliding_window_view(x, window_shape=(window_size,))

print(x_expanded)
