import pulp
import numpy as np

# 定义矩阵A和向量y
n = 100
np.random.seed(0)
A = np.random.normal(0, 4, (n, n))
np.random.seed(1)
s = np.random.randint(0, 4, n)
y = np.dot(A, s)

# 定义正则化参数
alpha = 0.8

# 定义扰动矩阵A'
A_perturbed = A + np.random.normal(0, 1, size=A.shape)  # 这里使用正态分布扰动矩阵A

# 创建问题实例
problem = pulp.LpProblem("Regularized_Least_Squares", pulp.LpMinimize)

# 创建决策变量
x = [pulp.LpVariable(f"x_{i}", lowBound=0) for i in range(n)]
# x = [pulp.LpVariable(f"x_{i}", lowBound=0, cat=pulp.LpInteger) for i in range(n)]

# 创建辅助变量
residuals = [pulp.LpVariable(f"residual_{i}", lowBound=0) for i in range(len(A))]

# 定义目标函数
objective = pulp.lpSum(residuals) + alpha * pulp.lpSum(x)
problem += objective

# 添加约束条件：A'x = y + residuals
for i in range(len(A_perturbed)):
    constraint = pulp.lpSum(A_perturbed[i][j] * x[j] for j in range(n)) == y[i] + residuals[i]
    problem += constraint

# 求解问题
problem.solve()

# 输出结果
print("Optimal solution:")
for i in range(n):
    print(x[i].name, "=", int(pulp.value(x[i])))
print("Objective value:", int(pulp.value(problem.objective)))
x = np.array([round(pulp.value(x[i])) for i in range(n)])
print(np.sum(np.abs(x - s)))
