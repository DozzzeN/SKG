import gurobipy as gp
import numpy as np
from gurobipy import GRB

# 创建模型实例
model = gp.Model("Integer Quadratic Programming")

# 创建决策变量
n = 32
np.random.seed(0)
A = np.random.normal(0, 4, (n, n))
np.random.seed(1)
s = np.random.randint(0, 4, n)
y = np.dot(A, s)
# A = np.array([[2, 3], [1, 4], [3, 2]])
# s = np.array([1, 2])
# y = np.array([8, 9, 7])

# 定义正则化参数
alpha = 0.8

# 定义扰动矩阵A'
np.random.seed(0)
A_perturbed = A + np.random.normal(0, 1, size=A.shape)  # 这里使用正态分布扰动矩阵A

# 创建决策变量
inputs = []
for i in range(n):
    # inputs.append(model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"x_{i}"))
    inputs.append(model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{i}"))
obj = sum((np.dot(A_perturbed, inputs) - y) ** 2) + alpha * sum(np.power(inputs, 2))

model.setObjective(obj, sense=gp.GRB.MINIMIZE)

model.setParam("OutputFlag", 0)
model.setParam("LogToConsole", 0)

# 求解问题
model.optimize()

# 输出结果
print("Optimal solution:")
for v in model.getVars():
    print(v.varName, "=", v.x)
print("Objective value:", model.objVal)
x = np.array([round(v.x) for v in model.getVars()])
print(np.sum(np.abs(x - s)))
