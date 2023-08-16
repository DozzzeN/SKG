import gurobipy as gp
import numpy as np
from gurobipy import GRB

# 创建模型实例
model = gp.Model("Integer Quadratic Programming")

# 创建决策变量
n = 100
A = np.random.normal(0, 4, (n, n))
y = np.dot(A, np.random.randint(0, 4, n))
# y = np.random.randint(0, 4, 10)

# 创建决策变量
inputs = []
for i in range(n):
    inputs.append(model.addVar(vtype=GRB.INTEGER, lb=0, name=f'x{i}'))
obj = sum((np.dot(A, inputs) - y) ** 2)

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
