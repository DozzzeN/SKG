import numpy as np

A = 7
C = np.zeros(2 ** A + 1)
n = 128
for i in range(1, 2 ** A + 1):
    C[i] = int(i * n / (2 ** A)) - sum(C[0:i])
print(C)

c = 0
k = 0
j = 1

U = np.zeros(n + 1)
while j <= n:
    c += int(n / 2 ** A)
    while c >= 1:
       U[j] = k / 2 ** A
       j += 1
       c -= 1
    k += 1
print(U)