import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


def objective(K_flat, A, alpha):
    # 将展平的 K 恢复为 N x N 矩阵
    N = len(A)
    K = K_flat.reshape(N, N)

    # 计算 B = A @ K
    B = np.dot(A, K)

    # 对 B 进行均值化和归一化
    B = B - np.mean(B)
    B = (B - np.min(B)) / (np.max(B) - np.min(B))

    # 计算最小欧式距离之和
    min_distance_sum = 0
    for i in range(len(B)):
        distances = []
        for j in range(len(B)):
            if i == j:
                continue
            distances.append(np.square(B[i] - B[j]))
        min_distance_sum += 1 / (np.min(distances) + 1e-6)
    # 计算 A-B 欧式距离
    dist_A_B = 0
    for i in range(len(A)):
        dist_A_B += np.square(A[i] - B[i])

    # 我们需要最大化目标函数，因此返回其负值
    return min_distance_sum * alpha * dist_A_B


def constraint(K_flat, A):
    # 将展平的 K 恢复为 N x N 矩阵
    N = len(A)
    K = K_flat.reshape(N, N)

    # 计算 B = A @ K
    B = np.dot(A, K)

    # 约束：B 的所有元素应在 [0, 1] 范围内
    return np.concatenate((B - 0, 1 - B))


def optimize_matrix(A, method, alpha):
    N = len(A)
    # 初始猜测的 K
    initial_K = np.random.rand(N, N)
    initial_K_flat = initial_K.flatten()

    # 定义约束条件
    cons = ({'type': 'ineq', 'fun': constraint, 'args': (A,)})

    # 优化目标函数，包含边界和约束条件
    # if method == 'slsqp' or method == 'trust-constr':
    #     result = minimize(objective, initial_K_flat, args=(A, alpha), method=method, constraints=cons)
    # else:
    #     result = minimize(objective, initial_K_flat, args=(A, alpha), method=method)

    result = minimize(objective, initial_K_flat, args=(A, alpha), method=method)

    # 将优化后的 K 恢复为 N x N 矩阵
    K_optimal = result.x.reshape(N, N)

    return K_optimal


# 选择不同的优化算法进行比较
optimizers = ['l-bfgs-b', 'Powell', 'slsqp', 'trust-constr']
alpha = 1  # 权重参数，用于平衡两个目标

# 对应于method.py的search_random_matrix_uniform函数
np.random.seed(10000)
A = np.random.exponential(1, 16)
# A = np.random.normal(0, 0.1, 16)
# np.random.seed(1000)
# A += np.random.normal(0, 0.1, 16)
A = A - np.mean(A)
A = (A - np.min(A)) / (np.max(A) - np.min(A))
print(A)
for optimizer in optimizers:
    start_time = time.time_ns()
    K_optimal = optimize_matrix(A, optimizer, alpha)
    K_optimal = K_optimal - np.mean(K_optimal)
    K_optimal = (K_optimal - np.min(K_optimal)) / (np.max(K_optimal) - np.min(K_optimal))
    elapsed_time = (time.time_ns() - start_time) / 1e6
    print(f"Optimizer: {optimizer}, Time: {elapsed_time} ms")

    # 使用优化后的 K 计算 B
    B = np.dot(A, K_optimal)
    B = B - np.mean(B)
    B = (B - np.min(B)) / (np.max(B) - np.min(B))
    # 计算最小欧式距离之和
    print(B)
    min_distance_sum = 0
    for i in range(len(B)):
        distances = []
        for j in range(len(B)):
            if i == j:
                continue
            distances.append(np.square(B[i] - B[j]))
        min_distance_sum += 1 / (np.min(distances) + 1e-6)
    print(f"Maximized objective value: {min_distance_sum}")

    min_distance_sum = 0
    for i in range(len(A)):
        distances = []
        for j in range(len(A)):
            if i == j:
                continue
            distances.append(np.square(A[i] - A[j]))
        min_distance_sum += 1 / (np.min(distances) + 1e-6)
    print(f"Original value: {min_distance_sum}")

    norm_A_B = np.linalg.norm(A - B)
    print(f"Norm A-B: {norm_A_B}\n")

    # 可视化 A 和 B
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 2, 1)
    plt.stem(np.sort(A.flatten()), use_line_collection=True)
    plt.title("Vector A")
    plt.xlabel("Index")
    plt.ylabel("Value")

    plt.subplot(2, 2, 2)
    plt.stem(np.sort(B.flatten()), use_line_collection=True)
    plt.title("Vector B")
    plt.xlabel("Index")
    plt.ylabel("Value")

    plt.subplot(2, 2, 3)
    plt.stem(A.flatten(), use_line_collection=True)
    plt.title("Vector A")
    plt.xlabel("Index")
    plt.ylabel("Value")

    plt.subplot(2, 2, 4)
    plt.stem(B.flatten(), use_line_collection=True)
    plt.title("Vector B")
    plt.xlabel("Index")
    plt.ylabel("Value")

    plt.tight_layout()
    # plt.savefig('maximize_method1.svg', dpi=1200, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.imshow(K_optimal, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Visualization of Matrix K')
    plt.show()
