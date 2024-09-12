import time

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import gurobipy as gp
from gurobipy import GRB
from cvxopt import matrix, solvers
from dtaidistance import dtw_ndim
from dtw import accelerated_dtw

def dtw_metric(data1, data2):
    if np.ndim(data1) == 1:
        distance = lambda x, y: np.abs(x - y)
        data1 = np.array(data1)
        data2 = np.array(data2)
        # return dtw(data1, data2, dist=distance)[0]
        return accelerated_dtw(data1, data2, dist=distance)[0]
    else:
        # 二维DTW距离
        return dtw_ndim.distance(data1, data2)


def compute_min_dtw(data1, data2):
    min_dtw = np.inf
    for i in range(len(data1)):
        for j in range(len(data2)):
            if i == j:
                continue
            min_dtw = min(min_dtw, dtw_metric(data1[i], data2[j]))
    return min_dtw


def segment_sequence(data, segment_lengths):
    segments = []
    start_index = 0
    for length in segment_lengths:
        end_index = start_index + length
        segments.append(data[start_index:end_index])
        start_index = end_index
    return segments


# 定义目标函数
def objective(K, A, segment_method):
    K = np.diag(K)  # 将 K 重塑为 n x n 矩阵
    B = K @ A  # B = K * A, 这里 B 是一个 1 x n 的向量

    B = B - np.mean(B)
    B = (B - np.min(B)) / (np.max(B) - np.min(B))
    B = segment_sequence(B, segment_method)

    return -compute_min_dtw(B, B)  # 因为我们使用最小化，所以取负数


# 参数设置
n = 4 * 4

# 随机生成向量 A
np.random.seed(0)
A_bck = np.random.normal(0, 1, (n, 2))  # 生成一个 1 x n 的向量
A = A_bck - np.mean(A_bck)
A = (A - np.min(A)) / (np.max(A) - np.min(A))
np.random.seed(0)
R = np.random.normal(0, 1, (n, n))
A = R @ A
segment_method = [4, 4, 4, 4]

print("Vector A:")
print(A)

# 初始猜测的矩阵 K
initial_K = np.random.normal(0, 1, n)

# 选择不同的优化算法进行比较
# optimizers = ['l-bfgs-b', 'SLSQP', 'trust-constr', 'Powell']
optimizers = ['l-bfgs-b', 'Powell']
print(f"Original value: {compute_min_dtw(segment_sequence(A, segment_method), segment_sequence(A, segment_method))}\n")

for optimizer in optimizers:
    start_time = time.time_ns()
    result = minimize(objective, initial_K, args=(A, segment_method), method=optimizer)
    elapsed_time = (time.time_ns() - start_time) / 1e6
    print(f"Optimizer: {optimizer}, Time: {elapsed_time} ms")

    # 将最优解重塑为矩阵
    optimal_K = np.diag(result.x)

    # 计算矩阵 B = AK
    B = optimal_K @ A

    # 对 B 进行均值化和归一化
    B = B - np.mean(B)
    B = (B - np.min(B)) / (np.max(B) - np.min(B))
    B = np.array(segment_sequence(B, segment_method))

    # 计算最大化后的目标函数值
    objective_value = -result.fun  # 因为我们最小化了负目标函数，所以需要取负
    print(f"Maximized objective value: {compute_min_dtw(B, B)}\n")

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

    # 可视化 K
    optimal_K = optimal_K - np.mean(optimal_K)
    optimal_K = (optimal_K - np.min(optimal_K)) / (np.max(optimal_K) - np.min(optimal_K))
    plt.figure()
    plt.imshow(optimal_K, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Visualization of Matrix K')
    plt.show()

    # 可视化 噪音 * K
    optimal_Kn = R @ optimal_K
    optimal_Kn = optimal_Kn - np.mean(optimal_Kn)
    optimal_Kn = (optimal_Kn - np.min(optimal_Kn)) / (np.max(optimal_Kn) - np.min(optimal_Kn))
    plt.figure()
    plt.imshow(optimal_Kn, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Visualization of Noise @ Matrix K')
    plt.show()

    Kn = np.random.normal(0, 1, (n, n))
    Kn = Kn - np.mean(Kn)
    Kn = (Kn - np.min(Kn)) / (np.max(Kn) - np.min(Kn))

    plt.figure()
    plt.imshow(Kn, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Visualization of Random Matrix')
    plt.show()
