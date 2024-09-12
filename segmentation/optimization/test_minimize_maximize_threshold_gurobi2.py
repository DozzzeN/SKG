import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

from matplotlib import pyplot as plt

# # 对应于method.py中的optimize_random_matrix_max_min函数
# np.random.seed(10000)
A = np.random.normal(0, 1, 10)
A = A + np.random.normal(0, 0.2, 10)
tau = 10

# 数据预处理：均值化和归一化
# A_mean = np.mean(A)
# A_normalized = (A - A_mean) / np.std(A)
A = A - np.mean(A)
A_normalized = (A - np.min(A)) / (np.max(A) - np.min(A))

# 确定矩阵 K 的维度
n = len(A)

# 创建模型
model = gp.Model()

# 隐藏求解器的输出
# model.Params.OutputFlag = 0

# 设置求解参数以提升速度
model.Params.Threads = 4  # 使用多线程
model.Params.TimeLimit = 3  # 设置时间限制（秒）
model.Params.Presolve = 2  # 开启高级预处理
model.Params.MIPFocus = 1  # 聚焦于寻找可行解
model.Params.Cuts = 2  # 增强切割平面

# 定义变量 K
# K = model.addVars(n, n, lb=-1, ub=1, name="K")
K = model.addVars(n, n, name="K")

# 计算 AK
AK = [[gp.LinExpr(A_normalized[i] * K[i, j]) for j in range(n)] for i in range(n)]
# 均值化和归一化

# 定义辅助变量和约束来表示 AK 中每对元素之间的欧式距离
distances = model.addVars(n, n, lb=0, name="distances")

for i in range(n):
    for j in range(n):
        if i != j:
            model.addConstr(distances[i, j] == gp.quicksum((AK[i][k] - AK[j][k]) ** 2 for k in range(n)),
                            name=f"dist_{i}_{j}")

# 定义辅助变量来表示最小距离
min_distances = model.addVars(n, lb=0, name="min_distances")

for i in range(n):
    model.addConstr(min_distances[i] == gp.min_([distances[i, j] for j in range(n) if i != j]),
                    name=f"min_distance_{i}")

# 定义辅助变量来表示最小距离的倒数
inv_min_distances = model.addVars(n, lb=0, name="inv_min_distances")

# 添加倒数变换约束
for i in range(n):
    model.addConstr(min_distances[i] * inv_min_distances[i] == 1, name=f"inv_min_dist_{i}")

# 设置目标函数：最小化最小距离的倒数之和
model.setObjective(gp.quicksum(inv_min_distances[i] for i in range(n)), GRB.MINIMIZE)

# # 设置目标函数：最大化最小距离之和
# model.setObjective(gp.quicksum(min_distances[i] for i in range(n)), GRB.MAXIMIZE)

# 定义目标函数：最小化 AK - A_normalized 的范数平方和
# objective = gp.quicksum((gp.quicksum(AK[i][j] for j in range(n)) - A_normalized[i]) ** 2 for i in range(n))
# model.setObjective(objective, GRB.MINIMIZE)
# 定义目标函数：最小化 K 的范数
# objective = gp.quicksum(K[i, j] ** 2 for i in range(n) for j in range(n))
# model.setObjective(objective, GRB.MINIMIZE)
# 定义目标函数：最大化 AK 的方差
# AK_mean = gp.quicksum(AK[i][j] for i in range(n) for j in range(n)) / (n * n)
# objective = gp.quicksum((gp.quicksum(AK[i][j] for j in range(n)) - AK_mean) ** 2 for i in range(n))
# model.setObjective(objective, GRB.MAXIMIZE)

# 限制 AK 元素的范围
# for i in range(n):
#     for j in range(n):
#         model.addConstr(K[i, j] >= 0)
#         model.addConstr(K[i, j] <= 1)

# 设置NonConvex参数以允许非凸问题
model.Params.NonConvex = 2

# 开始计时
start_time = time.time()

# 求解优化问题
model.optimize()

# 结束计时
end_time = time.time()

if model.Status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
    print("Optimal value (norm):", model.ObjVal)
else:
    exit()
# 记录结果
results = {
    'status': model.Status,
    'optimal_value': model.ObjVal,
    'run_time': end_time - start_time,
    'K_optimal': np.array([[K[i, j].X for j in range(n)] for i in range(n)]),
    'AK': np.array([A_normalized @ np.array([K[i, j].X for j in range(n)]) for i in range(n)])
}

# 打印结果
print("\nResults:")
print(f"Status: {results['status']}")
print(f"Optimal value (norm): {results['optimal_value']}")
print(f"Run time: {results['run_time']} seconds")
print("Optimal K:")
print(results['K_optimal'])
print("AK:")
print(results['AK'])
print("-" * 30)


def min_distance(solution1, solution2):
    total = 0
    for i in range(len(solution1)):
        min_dist = float("inf")
        for j in range(len(solution2)):
            if i == j:
                continue
            # 计算欧式距离
            min_dist = min(min_dist, np.abs(np.array(solution1[i]) - np.array(solution2[j])))
        total += min_dist
    return total


# 最小欧式距离之和
min_distance_sum = min_distance(A_normalized, A_normalized)
print(f"Min distance sum before: {min_distance_sum}")
B = results['AK']
B = B - np.mean(B)
B = (B - np.min(B)) / (np.max(B) - np.min(B))
min_distance_sum = min_distance(B, B)
print(f"Min distance sum after: {min_distance_sum}")

# 可视化 K
optimal_K = results['K_optimal']
optimal_K = optimal_K - np.min(optimal_K)
optimal_K = optimal_K / np.max(optimal_K)
plt.figure()
plt.imshow(optimal_K, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Visualization of Matrix K')
plt.show()

# 可视化 A 和 B
plt.figure(figsize=(10, 5))

A_normalized = np.sort(A_normalized)
plt.subplot(1, 2, 1)
plt.stem(A_normalized.flatten(), use_line_collection=True)
plt.title("Vector A")
plt.xlabel("Index")
plt.ylabel("Value")

B = results['AK']
B = B - np.min(B)
B_normalized = (B - np.min(B)) / (np.max(B) - np.min(B))
B_normalized = np.sort(B_normalized)
plt.subplot(1, 2, 2)
plt.stem(B_normalized.flatten(), use_line_collection=True)
plt.title("Vector B")
plt.xlabel("Index")
plt.ylabel("Value")

plt.show()
