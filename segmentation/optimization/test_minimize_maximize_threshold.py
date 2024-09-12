import numpy as np
from scipy.optimize import minimize
import time

# 示例函数，定义目标向量 A 和阈值 tau
A = np.random.normal(0, 1, 10)
A = A - np.mean(A)
A = (A - np.min(A)) / (np.max(A) - np.min(A))
tau = 10.0


def objective(K_flat):
    # 将展平的 K 恢复为 n x p 矩阵
    K = K_flat.reshape(len(A), -1)

    K = K - np.mean(K)
    K = (K - np.min(K)) / (np.max(K) - np.min(K))

    # 计算 AK - A 的 Frobenius 范数

    norm_AK_A = np.linalg.norm(np.dot(A, K) - A)

    return norm_AK_A


def constraint(K_flat):
    # 将展平的 K 恢复为 n x p 矩阵
    K = K_flat.reshape(len(A), -1)

    # 计算 AK
    AK = np.dot(A, K)

    # 计算每个元素之间的欧式距离
    sum_distances = 0.0
    for i in range(len(AK)):
        for j in range(len(AK)):
            if i != j:
                distance = np.linalg.norm(AK[i] - AK[j])
                sum_distances += distance ** 2

    # 返回约束条件表达式，确保大于阈值 tau
    return tau - sum_distances


def optimize_with_optimizer(optimizer):
    print(f"Using optimizer: {optimizer}")

    # 初始猜测的 K
    initial_K = np.random.rand(len(A), len(A))

    # 初始猜测的展平形式
    initial_K_flat = initial_K.flatten()

    # 定义约束条件
    cons = ({'type': 'ineq', 'fun': constraint})

    # 记录优化开始时间
    start_time = time.time()

    # 使用指定的优化器求解
    result = minimize(objective, initial_K_flat, method=optimizer, constraints=cons)

    # 记录优化结束时间
    end_time = time.time()

    # 打印优化器运行时间
    print(f"Optimization time: {end_time - start_time} seconds")

    # 获取最优解
    K_optimal = result.x.reshape(len(A), -1)

    K_optimal = K_optimal - np.mean(K_optimal)
    K_optimal = (K_optimal - np.min(K_optimal)) / (np.max(K_optimal) - np.min(K_optimal))

    # 计算并打印对应的 AK
    AK_optimal = np.dot(A, K_optimal)

    # 计算 AK 各个元素之间的欧式距离
    distances = []
    for i in range(len(AK_optimal)):
        for j in range(len(AK_optimal)):
            if i != j:
                distance = np.linalg.norm(AK_optimal[i] - AK_optimal[j])
                distances.append(distance)

    # 计算欧式距离之和
    sum_distances = np.sum(distances)

    # 打印结果
    print("AK-A的范数:", np.linalg.norm(AK_optimal - A))
    print("欧式距离之和:", sum_distances)
    print("\n")


# 不同的优化器列表
optimizers = ['SLSQP', 'trust-constr']

# 比较不同优化器的表现
for optimizer in optimizers:
    optimize_with_optimizer(optimizer)
