import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

from matplotlib import pyplot as plt

# 对应于method.py中的optimize_random_matrix_max函数
np.random.seed(10000)
A = np.random.normal(0, 1, 16)
np.random.seed(10000)
A = A + np.random.normal(0, 0.1, 16)
tau = 10

# 数据预处理：均值化和归一化
A_mean = np.mean(A)
A_normalized = (A - A_mean) / np.std(A)
# A = A - np.mean(A)
# A_normalized = (A - np.min(A)) / (np.max(A) - np.min(A))

# 确定矩阵 K 的维度
n = len(A)

# 创建模型
model = gp.Model()

# 隐藏求解器的输出
model.Params.OutputFlag = 0

# 设置求解参数以提升速度
model.Params.Threads = 4  # 使用多线程
model.Params.Presolve = 2  # 开启高级预处理
model.Params.MIPFocus = 1  # 聚焦于寻找可行解
model.Params.Cuts = 2  # 增强切割平面

# 定义变量 K
K = model.addVars(n, n, lb=-GRB.INFINITY, name="K")
# K = model.addVars(n, n, lb=-1, ub=1, name="K")

# 计算 AK
AK = [[gp.LinExpr(A_normalized[i] * K[i, j]) for j in range(n)] for i in range(n)]

# 定义目标函数：最小化 AK - A_normalized 的范数平方和
objective = gp.quicksum((gp.quicksum(AK[i][j] for j in range(n)) - A_normalized[i]) ** 2 for i in range(n))
model.setObjective(objective, GRB.MINIMIZE)

# 定义约束条件：确保 AK 中每个元素之间的欧式距离之和大于 tau
dist_sum = gp.quicksum(gp.quicksum(
    (gp.quicksum(AK[i][k] for k in range(n)) - gp.quicksum(AK[j][k] for k in range(n))) ** 2 for j in range(i + 1, n))
                       for i in range(n))
model.addConstr(dist_sum >= tau, name="distance_constraint")

# 设置NonConvex参数以允许非凸问题
model.Params.NonConvex = 2

# 开始计时
start_time = time.time()

# 求解优化问题
model.optimize()

# 结束计时
end_time = time.time()

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

plt.subplot(1, 2, 1)
plt.stem(A_normalized.flatten(), use_line_collection=True)
plt.title("Vector A")
plt.xlabel("Index")
plt.ylabel("Value")

B = results['AK']
B = B - np.min(B)
B_normalized = (B - np.min(B)) / (np.max(B) - np.min(B))
plt.subplot(1, 2, 2)
plt.stem(B_normalized.flatten(), use_line_collection=True)
plt.title("Vector B")
plt.xlabel("Index")
plt.ylabel("Value")

plt.show()
