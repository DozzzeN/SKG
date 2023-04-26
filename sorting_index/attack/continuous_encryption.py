import numpy as np
import sympy
from toqito.random import random_unitary
from sympy import Matrix, GramSchmidt

# 验证论文里面的中间结果
ukuk = []  # 求uk*ukT的期望
umum = []  # 求um*umT的期望
ukum = []  # 求uk*umT的期望
ukx = []  # 求uk*xT的期望
random_vector = []
k = 100000
for i in range(k):
    n = 5
    x = np.random.uniform(-1, 1, (n, 1))
    x = x / np.linalg.norm(x)
    K = np.matmul(random_unitary(n, True), x)
    # M = np.matmul(scipy.stats.unitary_group.rvs(n), x)  # 生成的酉矩阵带有复数
    M = np.matmul(random_unitary(n, True), x)
    # print(np.linalg.norm(K) ** 2, np.linalg.norm(M) ** 2)
    # print()

    KKT = np.matmul(K, K.T)
    MMT = np.matmul(M, M.T)
    # print(np.trace(KKT), np.trace(MMT))
    # print()

    # 正式求SVD分解
    uk = np.linalg.svd(KKT)[0]  # u矩阵
    um = np.linalg.svd(MMT)[0]
    # print(np.matmul(uk, uk.T))
    # print(np.matmul(um, um.T))
    # print(np.matmul(uk, um.T))
    # print(np.matmul(uk[0], K))
    # print(sum(np.matmul(KKT, uk)))
    ukuk.append(np.matmul(uk, uk.T).flatten())
    umum.append(np.matmul(um, um.T).flatten())
    ukum.append(np.matmul(uk, um.T).flatten())
    ukx.append(np.matmul(uk, KKT.T).flatten())
    random_vector.append(np.dot(np.random.uniform(-1, 1, (n, 1)), np.random.uniform(-1, 1, (1, n))).flatten())

print(sum(np.array(ukuk)) / k)
print(sum(np.array(umum)) / k)
print(sum(np.array(ukum)) / k)
print(sum(np.array(ukx)) / k)

print(sum(sum(np.array(random_vector).T)) / k / k)
print(sum(np.array(random_vector[0]).T) / k)
