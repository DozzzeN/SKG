import matplotlib.pyplot as plt
import numpy as np


def two_norm(a):
    return np.linalg.svd(a)[1][0]
A = np.random.normal(0, 1, (10, 10))
print(np.linalg.cond(A))
# u, s, v = np.linalg.svd(A)
# print(np.sort(s))
# plt.figure()
# plt.plot(np.sort(s))
# # plt.show()
# # s = np.log(s)
# s = np.convolve(s, np.ones(10) / 10, mode='same')
# plt.figure()
# plt.plot(np.sort(s))
# # plt.show()
# print(np.sort(s))
# s = np.diag(s)
# print(np.linalg.cond(u @ s @ v))

# ruiz equilibration
# print(np.max(A))
print(two_norm(A))
for i in range(len(A)):
    row_norm = []
    col_norm = []
    for j in range(len(A[0])):
        row_norm.append(two_norm(A[j].reshape(len(A), 1)))
        col_norm.append(two_norm(A[:, j].reshape(len(A[0]), 1)))
    A = np.diag(1 / np.sqrt(row_norm)) @ A @ np.diag(1 / np.sqrt(col_norm))
print(np.linalg.cond(A))
print(two_norm(A))
row_norm = []
col_norm = []
for j in range(10):
    row_norm.append(two_norm(A[j].reshape(10, 1)))
    col_norm.append(two_norm(A[:, j].reshape(10, 1)))
print(row_norm)
print(col_norm)
