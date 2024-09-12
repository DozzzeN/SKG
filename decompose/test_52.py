import numpy as np

M = 2
V = 2
N = 2
Ns = 2
W = np.random.normal(0, 1, ((M + 1) * V, N, Ns))
P = np.random.normal(0, 1, (N, Ns))
for i in range(Ns):
    P[:, i] = 1
    
Phi = np.random.normal(0, 1, (M + 1, V))

res1 = 0
for m in range(M + 1):
    for t in range(V):
        res1 += np.linalg.norm(W[m * V + t] - P * Phi[m, t], ord='fro') ** 2

A = np.zeros((N, Ns))
B = np.zeros((N, Ns))
for m in range(M + 1):
    for t in range(V):
        A += W[m * V + t] * Phi[m, t] / ((M + 1) * V)

for m in range(M + 1):
    for t in range(V):
        B += W[m * V + t] * Phi[m, t]

res21 = ((M + 1) * V) * np.linalg.norm(A - P, ord='fro') ** 2

res22 = 0
for m in range(M + 1):
    for t in range(V):
        res22 += np.linalg.norm(W[m * V + t], ord='fro') ** 2

res23 = np.linalg.norm(B, ord='fro') ** 2 / ((M + 1) * V)

print(res1, res21 + res22 - res23)
