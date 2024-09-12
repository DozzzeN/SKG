import numpy as np
import cvxpy as cp
import time

# 示例函数，定义目标向量 A
A = np.array([1, 2, 3, 4, 5, 6])
tau = 10

# 数据预处理：均值化和归一化
A_mean = np.mean(A)
A_normalized = (A - A_mean) / np.std(A)

# 确定矩阵 K 的维度
n = len(A)

# 定义变量 K
K = cp.Variable((n, n))

# 定义目标函数：最小化 AK - A 的范数
objective = cp.Minimize(cp.norm(A_normalized @ K - A_normalized))

# 定义约束条件：确保 AK 中每个元素之间的欧式距离之和大于 tau
AK = A_normalized @ K
dist_sum = cp.sum_squares(cp.vstack([AK[i] - AK[j] for i in range(n) for j in range(i + 1, n)]))
constraints = [dist_sum >= tau]
# 不满足DCP
# constraints = [dist_sum <= tau]

# 构造优化问题
prob = cp.Problem(objective, constraints)

# 比较不同的求解器
solvers = [
    cp.SCS,
    cp.GUROBI,
    cp.ECOS,
    cp.SCIP,
    cp.MOSEK,
    cp.CLARABEL,
    cp.XPRESS
]

results = {}

for solver in solvers:
    print(f"Running solver: {solver}")

    # 求解优化问题
    start_time = time.time()
    prob.solve(solver=solver)
    end_time = time.time()

    # 记录结果
    results[solver] = {
        'status': prob.status,
        'optimal_value': prob.value,
        'run_time': end_time - start_time,
        'K_optimal': K.value,
        'AK': A_normalized @ K.value
    }

# 打印所有求解器的结果
# Problem does not follow DCP rules
# 无法进行优化
print("\nResults:")
for solver_name, result in results.items():
    print(f"Solver: {solver_name}")
    print(f"Status: {result['status']}")
    print(f"Optimal value (norm): {result['optimal_value']}")
    print(f"Run time: {result['run_time']} seconds")
    print("Optimal K:")
    print(result['K_optimal'])
    print("AK:")
    print(result['AK'])
    print("-" * 30)
