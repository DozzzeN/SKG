import numpy as np

z_var = []
z_mean = []

a_var = []
a_mean = []

b_var = []
b_mean = []
n = 10
times = 100000
# for i in range(times):
#     x = np.random.normal(0, 1, n)
#     y = np.random.normal(0, 1, n)
#     A = np.random.normal(0, 1, (n, n))
#
#     z = np.sum(np.multiply(x, y))
#     z_var.append(np.var(z))
#     z_mean.append(np.mean(z))
#     a = np.sum(np.random.laplace(0, 1, n))
#     a_var.append(np.var(a))
#     a_mean.append(np.mean(a))
#     b = np.matmul(A, x)
#     b_var = np.var(b)
#     b_mean = np.mean(b)
# print(np.mean(z_var))
# print(np.mean(z_mean))
# print(np.mean(a_var))
# print(np.mean(a_mean))
# print(np.mean(b_var))
# print(np.mean(b_mean))
#
# b_var = []
# b_mean = []
#
# axa_var = []
#
# for i in range(times):
#     x = []
#     A = []
#     AxA = []
#     for j in range(n):
#         x.append(np.random.normal(0, 1))
#         Ai = []
#         for k in range(n):
#             Ai.append(np.random.normal(0, 1))
#         A.append(Ai)
#     AxA.append(np.var(x) * np.array(A @ np.transpose(A)))
#     axa_var.append(np.var(AxA))
#
#     b = np.matmul(A, x)
#     b_var = np.var(b)
#     b_mean = np.mean(b)
# print(np.mean(b_var))
# print(np.mean(b_mean))
# print(np.mean(axa_var))

A = np.random.normal(0, 1, (n, n))
AA = A @ np.transpose(A)
print(np.linalg.svd(A)[1][0])
print(np.linalg.svd(AA)[1][0])
print(np.mean(np.reshape(A, (n * n, 1))))
print(np.mean(np.reshape(AA, (n * n, 1))))
print(np.var(np.reshape(A, (n * n, 1))))
print(np.var(np.reshape(AA, (n * n, 1))))
