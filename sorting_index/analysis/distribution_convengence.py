import matplotlib.pyplot as plt
import numpy as np

n = 10000

data = []
for i in range(n):
    s1 = np.random.rand()
    s2 = np.random.rand()
    data.append(np.abs(s1 - s2))
plt.figure()
plt.hist(data, bins=50)
plt.show()

f = np.arange(0, 1, 0.001)
f = 2 * (1 - f)
# plt.figure()
# plt.plot(f)
# plt.show()

data = []
for i in range(n):
    x1 = np.random.rand()
    x2 = np.random.rand()
    y1 = np.random.rand()
    y2 = np.random.rand()
    data.append(np.abs(x1 - y1) + np.abs(x2 - y2))
plt.figure()
plt.hist(data, bins=50)
plt.show()

data = []
for i in range(n):
    x1 = np.random.rand()
    x2 = np.random.rand()
    x3 = np.random.rand()
    y1 = np.random.rand()
    y2 = np.random.rand()
    y3 = np.random.rand()
    data.append(np.abs(x1 - y1) + np.abs(x2 - y2) + np.abs(x3 - y3))
plt.figure()
plt.hist(data, bins=50)
plt.show()
