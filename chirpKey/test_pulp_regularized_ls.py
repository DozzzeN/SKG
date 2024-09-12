import numpy as np
from scipy.optimize import minimize

# 定义数据
A = np.array([[2, 3], [1, 4], [3, 2]])
y = np.array([8, 9, 7])

# 目标函数
def loss(x, A, y):
    return np.sum((A.dot(x) - y)**2) + 0.1*np.sum(x**2)

# 初始化
x0 = np.zeros(2)

# 求解
res = minimize(loss, x0, args=(A, y))

# 结果
print(res.x)