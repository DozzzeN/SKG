import numpy as np
import scipy
from sklearn.decomposition import PCA

# A = np.array([[1, 0, 0], [0, 2, 0], [1, 2, 2]])
# # A = np.random.normal(0, 1, (3, 3))
# print(np.linalg.det(A))
# lamda, U = np.linalg.eig(A)
# print(lamda)
# print(np.linalg.det(np.matmul(U, np.matmul(np.diag(lamda), U.T.conjugate()))))
# print(np.linalg.det(np.diag(lamda)))
# print(np.linalg.det(np.matmul(U.T.conjugate(), U)))

# 正交矩阵
Q = scipy.stats.ortho_group.rvs(3, 3)[0]
lambda1 = np.diag(np.random.normal(0, 0.1, 3))
lambda2 = np.eye(3) - lambda1 @ lambda1
lambda2 = np.sqrt(lambda2)

A1 = Q.T @ lambda1 @ Q
B1 = Q.T @ lambda2 @ Q
U = A1 + np.ones((3, 3)) * 1j * B1
# 酉矩阵的共轭转置等于其逆
# print(np.linalg.inv(U))
# print(U.conjugate().T)


A = np.diag(np.random.normal(0, 0.1, 3) ** 2)
B = np.eye(3) * np.random.normal(0, 0.1) ** 2 + A
C = np.eye(3) * np.random.normal(0, 0.1) ** 2 + U.conjugate().T @ A @ U
# 均相等
# print(np.linalg.inv(B))
# print(np.float16(U @ np.linalg.inv(C) @ U.conjugate().T))
# print(np.float16(U @ np.linalg.inv(B) @ U.conjugate().T))

D = np.sqrt(np.linalg.inv(B)) @ B
E = D @ B @ D.T
F = D @ U @ B @ U.conjugate().T @ D.T
# print(np.linalg.matrix_rank(F))
print(np.linalg.eig(F)[0])
print(np.linalg.eig(E)[0])

# D = np.array([[1, 1, 1], [2, 2, 2], [1, 2, 3]])
# print(np.linalg.matrix_rank(D))
# print(np.linalg.eig(D)[0])
