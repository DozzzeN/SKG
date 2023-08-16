import pulp
import numpy as np

# 定义矩阵A和向量y
A = np.array([[2, 3], [1, 4], [3, 2]])
y = np.array([5, 6, 7])

# 创建问题实例
problem = pulp.LpProblem("Matrix_Constraint", pulp.LpMinimize)

# 创建决策变量
n = len(A[0])  # 列数
x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat=pulp.LpInteger) for i in range(n)]

# 定义目标函数
objective = pulp.lpSum(x)
problem += objective

# 添加约束条件：Ax = y
for i in range(len(A)):
    constraint = pulp.lpSum(A[i][j] * x[j] for j in range(n)) == y[i]
    problem += constraint

# 求解问题
problem.solve()

# 输出结果
print("Optimal solution:")
for i in range(n):
    print(x[i].name, "=", pulp.value(x[i]))
print("Objective value:", pulp.value(problem.objective))
