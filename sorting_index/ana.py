import numpy as np

x = [3, 0, 5, 1]
n = [[3, 2, 1, 4], [3, 2, 1, 3], [3, 3, 1, 1], [1, 3, 2, 3]]
y = np.matmul(x, n)

p = [24, 25, 10, 20]
nn = np.linalg.inv(n)
print(y)
xp = np.matmul(p, nn)
print(xp)
print(np.matmul(xp, n))
