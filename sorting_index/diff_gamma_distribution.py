import math

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

def distribution(x):
    n = 10
    frac = 0

    for j in range(n):
        frac += math.factorial(n - 1 + j) / math.factorial(n - 1 - j) / math.factorial(j) / math.pow(2, j) * math.pow(
            abs(x), (n - 1 - j))
    res = frac * math.exp(-abs(x)) / math.factorial(n - 1) / math.pow(2, n)
    return res


plt.figure()
r = 50
x = np.array(list(range(r))) - int(r / 2)
y = []
for i in range(r):
    y.append(distribution(i - int(r / 2)))
plt.plot(x, y)
plt.show()

plt.plot(x, stats.norm.pdf(x, 0, 1))
plt.show()

