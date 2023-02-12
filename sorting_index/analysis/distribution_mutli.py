import sys

import matplotlib.pyplot as plt
import numpy as np

n = 100000

data = []
for i in range(n):
    m1 = np.random.rand()
    m2 = np.random.rand()
    y2 = np.random.rand()
    data.append(m1 - m2 - y2)
plt.figure()
plt.hist(data, bins=50)
plt.show()

x1 = np.arange(0, 1, 0.001)
f1 = np.square(x1 - 1) / 2
x2 = np.arange(-2, -1, 0.001)
f2 = np.square(x2 + 2) / 2
x3 = np.arange(-1, 0, 0.001)
f3 = - (2 * x3 * x3 + 2 * x3 - 1) / 2
plt.figure()
plt.plot(x1, f1)
plt.plot(x2, f2)
plt.plot(x3, f3)
plt.show()

x = np.arange(-2, 1, 0.001)
f = -x * x - x + 1 / 2
plt.figure()
plt.plot(f)
plt.show()
