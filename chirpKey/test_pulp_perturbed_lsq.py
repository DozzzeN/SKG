import pulp
import numpy as np

# 利用线性规划库pulp求解带扰动的最小二乘法
# 定义矩阵A和向量y
n = 256
np.random.seed(0)
A = np.random.normal(0, 4, (n, n))
np.random.seed(1)
s = np.random.randint(0, 4, n)
y = np.dot(A, s)
# A = np.array([[2, 3], [1, 4], [3, 2]])
# s = np.array([1, 2])
# y = np.array([8, 9, 7])

# 定义扰动矩阵A'
np.random.seed(0)
A_perturbed = A + np.random.normal(0, 1, size=A.shape)  # 这里使用正态分布扰动矩阵A

# 创建问题实例
problem = pulp.LpProblem("Perturbed_Least_Squares", pulp.LpMinimize)

# 创建决策变量
n = len(A[0])  # 列数
x = [pulp.LpVariable(f"x_{i}", lowBound=0) for i in range(n)]
# x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat=pulp.LpInteger) for i in range(n)]

# 定义目标函数
objective = pulp.lpSum(x)
problem += objective

# 添加约束条件：A'x = y + residuals
for i in range(len(A_perturbed)):
    constraint = pulp.lpSum(A_perturbed[i][j] * x[j] for j in range(n)) == y[i]
    problem += constraint

# 求解问题
problem.solve()
# problem.solve(solver=pulp.PULP_CBC_CMD(msg=0))   # 默认的求解器

# 输出结果
print("Optimal solution:")
for i in range(n):
    print(x[i].name, "=", pulp.value(x[i]))
print("Objective value:", pulp.value(problem.objective))
x = np.array([round(pulp.value(x[i])) for i in range(n)])
print(np.sum(np.abs(x - s)))
