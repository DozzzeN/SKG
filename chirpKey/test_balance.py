import numpy as np
from scipy.signal import medfilt

dataLen = 100
A = np.random.normal(0, 1, (dataLen, dataLen))
U, S, Vt = np.linalg.svd(A)
S = medfilt(S, kernel_size=7)
D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
A_ = U @ D @ Vt
print(np.linalg.cond(A))
print(np.linalg.cond(A_))
print(np.linalg.cond(np.linalg.pinv(A.T @ A)))
print(np.linalg.cond(np.linalg.pinv(A_.T @ A_)))

# 平衡A.T @ A对于ADMM的效果更好，即平衡(A.T @ A)^-1
U, S, Vt = np.linalg.svd(A.T @ A)
S = medfilt(S, kernel_size=7)
D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
ATA = U @ D @ Vt
A = U @ np.sqrt(D) @ Vt
print(np.linalg.cond(A))
print(np.linalg.cond(ATA))
print(np.linalg.cond(A.T @ A))
