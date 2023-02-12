import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import special

n = 10000

# data = []
# for i in range(n):
#     s1 = np.random.normal()
#     s2 = np.random.normal()
#     data.append(np.abs(s1 - s2))
# plt.figure()
# plt.hist(data, bins=50)
# plt.show()
#
# f = np.arange(0, 10, 0.001)
# f = np.exp(-np.power(f, 2) / 4) / np.sqrt(np.pi)
# plt.figure()
# plt.plot(f)
# plt.show()

data = []
for i in range(n):
    s1 = np.random.normal()
    s2 = np.random.normal()
    s3 = np.random.normal()
    s4 = np.random.normal()
    data.append(np.abs(s1 - s3) - np.abs(s2 - s3))
plt.figure()
plt.hist(data, bins=50)
plt.show()

x1 = np.arange(0, 10, 0.001)
f1 = np.exp(-np.power(x1, 2) / 8) * (1 - special.erf(x1 / 4 * math.sqrt(2))) / math.sqrt(2 * math.pi)
x2 = np.arange(-10, 0, 0.001)
f2 = np.exp(-np.power(x2, 2) / 8) * (1 + special.erf(x2 / 4 * math.sqrt(2))) / math.sqrt(2 * math.pi)
plt.figure()
plt.plot(x1, f1)
plt.plot(x2, f2)
plt.show()
