import numpy as np


# 计算相关矩阵 Rxx
def autocorr(x):
    n = len(x)
    Rxx = np.zeros((n, n))  # 初始化相关矩阵为复数类型

    for i in range(n):
        for j in range(n):
            Rxx[i, j] = np.mean(x[i] * np.conj(x[j]))

    return Rxx


# 已知数据
n = 5  # 向量维度
sigma = 0.1  # 噪声方差
h = np.random.normal(0, 1, (n, n))  # 生成随机的 n*n 信道响应矩阵
z = np.random.normal(0, sigma, (n, 1))  # 生成随机的 n*1 噪声
x = np.random.normal(0, 1, (n, 1))  # 生成随机的 n*1 发送信号
y = h @ x + z

# 计算 MMSE 估计的发送信号
h_H = np.conj(h.T)  # h 的共轭转置
Rx = autocorr(x)
# Rx = autocorr(np.random.normal(0, 1, (n, 1)))

# 计算 MMSE 估计
# Rx如何求？ sigma是根据经验给出
# x_mmse = Rx @ np.linalg.inv(h_H @ h @ Rx + sigma * np.eye(n)) @ h_H @ y
x_mmse = np.linalg.pinv(h_H @ h + sigma * np.eye(n)) @ h_H @ y
x_ls = np.linalg.pinv(h_H @ h) @ h_H @ y
x_mmse2 = h_H @ np.linalg.pinv(h @ h_H + sigma * np.eye(n)) @ y

print(x)
print(x_mmse)
print(x_ls)
print(x_mmse2)
