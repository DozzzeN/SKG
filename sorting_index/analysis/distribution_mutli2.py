import sys

import matplotlib.pyplot as plt
import numpy as np

n = 100000

data = []
for i in range(n):
    m1 = np.random.rand()
    m2 = np.random.rand()
    y2 = np.random.rand()
    data.append(2 * m1 - m2 - y2)
plt.figure()
plt.hist(data, bins=50)
plt.show()

