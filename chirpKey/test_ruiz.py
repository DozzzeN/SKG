import numpy as np
import scipy.linalg
from scipy.linalg import toeplitz, circulant

A = np.array([[1, 2420], [1, 1.58]])
print(np.linalg.cond(A))

D1 = np.array([[0.0203, 0], [0, 0.8919]])
D2 = np.array([[1.1212, 0], [0, 0.0203]])
print(D1 @ A @ D2)
print(np.linalg.cond(D1 @ A @ D2))
print(np.linalg.svd(D1 @ A @ D2)[1])

# 行列范数为1，但是cond很大
B = np.random.normal(np.sqrt(3) / 3, 0.01, 3)
print(circulant(B))
print(np.linalg.svd(circulant(B))[1])
print(np.linalg.norm(circulant(B)[0]))
print(np.linalg.norm(circulant(B)[1]))
print(np.linalg.norm(circulant(B)[2]))
print(np.linalg.cond(circulant(B)))

print()
# C = toeplitz([1, 2, 3])
C = np.diag([1, 2, 3])
print(C)
print(np.linalg.svd(C)[1])
U, S, V = np.linalg.svd(C)
D = np.diag(1 / np.sqrt(S))
C_prime = U @ D @ np.diag(S) @ D @ V
print(C / S)
print(C_prime)
print(np.linalg.svd(C_prime)[1])
C_prime2 = D @ C @ D
print(C_prime2)
print(np.linalg.svd(C_prime2)[1])

# print(np.linalg.cholesky(C))
U, S, V = np.linalg.svd(C)
D = np.diag(1 / S)
E = U @ D @ U.T
C_prime3 = E @ C
print(np.linalg.svd(C_prime3)[1])
C_prime4 = U @ D @ np.diag(S) @ V
print(np.linalg.svd(C_prime4)[1])
print(E)
print(U @ np.diag(S) @ U.T)
print(U @ np.diag(S) @ V)
