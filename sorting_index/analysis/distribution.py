import matplotlib.pyplot as plt
import numpy as np

n = 10000

data = []
for i in range(n):
    s1 = np.random.rand()
    s2 = np.random.rand()
    data.append(s1 - s2)
# plt.figure()
# plt.hist(data, bins=50)
# plt.show()

x1 = np.arange(-1, 0, 0.001)
f1 = 1 + x1
x2 = np.arange(0, 1, 0.001)
f2 = 1 - x2
# plt.figure()
# plt.plot(x1, f1)
# plt.plot(x2, f2)
# plt.show()


data = []
for i in range(n):
    s1 = np.random.rand()
    s2 = np.random.rand()
    data.append(np.abs(s1 - s2))
# plt.figure()
# plt.hist(data, bins=50)
# plt.show()

f = np.arange(0, 1, 0.001)
f = 2 * (1 - f)
# plt.figure()
# plt.plot(f)
# plt.show()

data = []
for i in range(n):
    s1 = np.random.rand()
    s2 = np.random.rand()
    s3 = np.random.rand()
    s4 = np.random.rand()
    data.append(np.abs(np.abs(s1 - s2) - np.abs(s3 - s4)))
# plt.figure()
# plt.hist(data, bins=50)
# plt.show()

f = np.arange(0, 1, 0.001)
f = np.power((f - 1), 3) / 3 + np.power((f - 1), 2) * (f + 3)
# plt.figure()
# plt.plot(f)
# plt.show()

data = []
for i in range(n):
    s1 = np.random.rand()
    s2 = np.random.rand()
    s3 = np.random.rand()
    data.append(np.abs(s1 - s3) - np.abs(s2 - s3))
# plt.figure()
# plt.hist(data, bins=50)
# plt.show()

x1 = np.arange(0, 1, 0.001)
f1 = np.power(x1, 3) * 2 / 3 - 2 * x1 + 4 / 3
x2 = np.arange(-1, 0, 0.001)
f2 = - np.power((x2 + 1), 3) / 6 - np.power((x2 + 1), 2) * (x2 - 3) / 2
# plt.figure()
# plt.plot(x1, f1)
# plt.plot(x2, f2)
# plt.show()

data = []
for i in range(n):
    x11 = np.random.rand()
    x12 = np.random.rand()
    y11 = np.random.rand()
    y12 = np.random.rand()
    data.append(np.abs(x11 - y11) + np.abs(x12 - y12))
plt.figure()
plt.hist(data, bins=50)
plt.show()


data = []
for i in range(n):
    x11 = np.random.rand()
    x12 = np.random.rand()
    y11 = np.random.rand()
    y12 = np.random.rand()
    data.append(np.abs(x11 - y11 + x12 - y12))
plt.figure()
plt.hist(data, bins=50)
plt.show()