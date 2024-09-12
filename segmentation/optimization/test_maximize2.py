import time

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import gurobipy as gp
from gurobipy import GRB
from cvxopt import matrix, solvers


# 使用向量化操作替换循环来计算欧式距离
def sum_distance(solution1, solution2):
    solution1 = np.array(solution1)
    solution2 = np.array(solution2)
    diff = solution1[:, np.newaxis, :] - solution2[np.newaxis, :, :]
    distances = np.sum(diff ** 2, axis=2)
    total = np.sum(distances)
    return total


# def sum_distance(solution1, solution2):
#     total = 0
#     for i in range(len(solution1)):
#         for j in range(len(solution2)):
#             # 计算欧式距离
#             distance = np.square(np.array(solution1[i]) - np.array(solution2[j]))
#             total += distance
#     return total


# 定义目标函数
def objective(K, A, n):
    K = K.reshape(n, n)  # 将 K 重塑为 n x n 矩阵
    B = A @ K  # B = A * K, 这里 B 是一个 1 x n 的向量

    # 对 B 进行均值化和归一化
    B = B - np.mean(B)
    B = (B - np.min(B)) / (np.max(B) - np.min(B))

    sum_bi_squared = np.sum(B ** 2)
    sum_bi = np.sum(B)
    return -(n * sum_bi_squared - sum_bi ** 2)  # 因为我们使用最小化，所以取负数


# 参数设置
n = 4 * 4

# 随机生成向量 A
A_bck = np.random.normal(0, 1, (1, n))  # 生成一个 1 x n 的向量
A = A_bck - np.mean(A_bck)
A = (A - np.min(A)) / (np.max(A) - np.min(A))

# print("Vector A:")
# print(A)

# 初始猜测的矩阵 K
initial_K = np.random.normal(0, 1, size=(n, n)).flatten()  # 生成一个 n x n 的初始 K，并展平成一维数组

# 选择不同的优化算法进行比较
# optimizers = ['l-bfgs-b', 'SLSQP', 'trust-constr', 'Powell']
optimizers = ['l-bfgs-b', 'Powell']

for optimizer in optimizers:
    start_time = time.time_ns()
    result = minimize(objective, initial_K, args=(A, n), method=optimizer)
    elapsed_time = (time.time_ns() - start_time) / 1e6
    print(f"Optimizer: {optimizer}, Time: {elapsed_time} ms")

    # 将最优解重塑为矩阵
    optimal_K = result.x.reshape(n, n)

    # 计算矩阵 B = AK
    B = A @ optimal_K

    # 对 B 进行均值化和归一化
    B = B - np.mean(B)
    B = (B - np.min(B)) / (np.max(B) - np.min(B))

    # 计算最大化后的目标函数值
    objective_value = -result.fun  # 因为我们最小化了负目标函数，所以需要取负
    print(f"Maximized objective value: {sum_distance(B.T, B.T)}\n")

    # 可视化 A 和 B
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.stem(A.flatten(), use_line_collection=True)
    plt.title("Vector A")
    plt.xlabel("Index")
    plt.ylabel("Value")

    plt.subplot(1, 2, 2)
    plt.stem(B.flatten(), use_line_collection=True)
    plt.title("Vector B")
    plt.xlabel("Index")
    plt.ylabel("Value")

    plt.tight_layout()
    # plt.savefig('maximize_method1.svg', dpi=1200, bbox_inches='tight')

    plt.show()

y = [0] * 8 + [1] * 8
y = np.random.permutation(y)
y = y.reshape(1, -1)
y = np.array(y).astype(float)
# y += np.random.normal(0, 1, (1, n))
start_time = time.time_ns()
optimal_K2 = np.linalg.lstsq(A, y, rcond=None)[0]
elapsed_time = (time.time_ns() - start_time) / 1e6
print(f"Time: {elapsed_time} ms")
B2 = A @ optimal_K2

B2 = B2 - np.mean(B2)
B2 = (B2 - np.min(B2)) / (np.max(B2) - np.min(B2))

# print("Matrix B = AK:")
# print(B2)

# 打印最大化后的目标函数值
print(f"Maximized objective value: {sum_distance(np.array(B2).T, np.array(B2).T)}")

# 可视化 A 和 B
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.stem(A.flatten(), use_line_collection=True)
plt.title("Vector A")
plt.xlabel("Index")
plt.ylabel("Value")

plt.subplot(1, 2, 2)
plt.stem(B2.flatten(), use_line_collection=True)
plt.title("Vector B")
plt.xlabel("Index")
plt.ylabel("Value")

plt.tight_layout()
# plt.savefig('maximize_method2.svg', dpi=1200, bbox_inches='tight')

plt.show()

# 可视化 K
optimal_K = optimal_K - np.mean(optimal_K)
optimal_K = (optimal_K - np.min(optimal_K)) / (np.max(optimal_K) - np.min(optimal_K))
plt.figure()
plt.imshow(optimal_K, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Visualization of Matrix K')
plt.show()

# 可视化 K2
optimal_K2 = optimal_K2 - np.mean(optimal_K2)
optimal_K2 = (optimal_K2 - np.min(optimal_K2)) / (np.max(optimal_K2) - np.min(optimal_K2))
plt.figure()
plt.imshow(optimal_K2, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Visualization of Matrix K2')
plt.show()

# 可视化 K2 + 噪音
# optimal_K2n = optimal_K2 + np.random.normal(0, 0.1, (n, n))
optimal_K2n = optimal_K2 + np.random.normal(np.mean(optimal_K2), np.std(optimal_K2), (n, n))
# optimal_K2n = optimal_K2 + np.random.uniform(0, 1, (n, n))
optimal_K2n = optimal_K2n - np.mean(optimal_K2n)
optimal_K2n = (optimal_K2n - np.min(optimal_K2n)) / (np.max(optimal_K2n) - np.min(optimal_K2n))
plt.figure()
plt.imshow(optimal_K2n, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Visualization of Matrix K2 Plus Noise')
plt.show()

Kn = np.random.normal(0, 1, (n, n))
Kn = Kn - np.mean(Kn)
Kn = (Kn - np.min(Kn)) / (np.max(Kn) - np.min(Kn))

# plt.figure()
# plt.imshow(Kn, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.title('Visualization of Random Matrix')
# plt.show()

# PCA分析

def perform_pca(matrix, n_components=None):
    # 创建 PCA 对象
    pca = PCA(n_components=n_components)

    # 拟合模型并转换数据
    principal_components = pca.fit_transform(matrix)

    # 获取 PCA 的结果
    explained_variance = pca.explained_variance_ratio_
    components = pca.components_

    return principal_components, explained_variance, components, pca


# 执行 PCA
n_components = None
principal_components, explained_variance, components, pca = perform_pca(optimal_K2, n_components)
optimal_K2_reconstructed = pca.inverse_transform(principal_components)
# print("\n主成分：")
# print(principal_components)
print("\n优化矩阵特征值比率：")
print(explained_variance)
# print("\n主成分载荷（系数）：")
# print(components)

principal_components, explained_variance, components, pca, = perform_pca(optimal_K2n, n_components)
optimal_K2n_reconstructed = pca.inverse_transform(principal_components)
# print("\n主成分：")
# print(principal_components)
print("\n优化矩阵+噪音矩阵特征值比率：")
print(explained_variance)
# print("\n主成分载荷（系数）：")
# print(components)
print(optimal_K2)
print(optimal_K2n)
print(optimal_K2n_reconstructed)

principal_components, explained_variance, components, pca, = perform_pca(Kn, n_components)
Kn_reconstructed = pca.inverse_transform(principal_components)
# print("\n主成分：")
# print(principal_components)
print("\n噪音矩阵特征值比率：")
print(explained_variance)
# print("\n主成分载荷（系数）：")
# print(components)
