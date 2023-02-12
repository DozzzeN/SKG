from math import comb, perm

import numpy as np
from matplotlib import pyplot as plt

m = 140
left = np.zeros(m + 1)
dp = np.zeros(m + 1)

point = -1
for i in range(m + 1):
    tmp = 0
    for j in range(i + 1):
        tmp += comb(m, i) * comb(2 * m, i)
    left[i] = tmp
    dp[i] = perm(i, i)
    if dp[i] < left[i]:
        point = i

print(point)
right = comb(3 * m, m)
plt.figure()
plt.yscale('log')
plt.plot(range(len(left)), left, label="ed")
plt.plot(range(len(dp)), dp, label="dp")
plt.legend()
plt.show()
