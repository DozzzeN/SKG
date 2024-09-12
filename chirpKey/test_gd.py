import numpy as np

# 已知矩阵和向量
m = 256
n = 256
A = np.random.randn(m, n)
s = np.random.randn(n, 1)

y = A @ s

print(np.linalg.inv(A) @ y)

# 初始化x
x = np.zeros((n, 1))

# 梯度下降参数
lr = 0.1
epochs = 100000

for i in range(epochs):
    y_pred = np.dot(A, x)
    dx = (2 / m) * np.dot(A.T, (y_pred - y))

    x = x - lr * dx

print(s)
print(x)