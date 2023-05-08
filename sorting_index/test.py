import numpy as np

a1 = np.random.normal(0, 1, 3)  # 未知数
u1 = np.random.normal(0, 1, 3)
u2 = np.random.normal(0, 1, 3)
a2 = [u1, u2, u1 + u2]
a3 = np.matmul(a1, a2)
print(np.linalg.matrix_rank(a2))

ai = np.linalg.pinv(a2)
print(np.matmul(np.matmul(a3, ai), a2), a3)  # 是否有相容解
print(np.matmul(a3, ai))
print(np.matmul(a3, ai) + np.matmul(np.random.normal(0, 1, 3), np.identity(3) - np.matmul(a2, ai)))
print(a1)