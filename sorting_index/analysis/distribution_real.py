import matplotlib.pyplot as plt
import numpy as np

n = 100000
a = 0
b = 10
ma = 0
mb = 100
# data = []
# for i in range(n):
#     m1 = np.random.uniform(ma, mb)
#     m2 = np.random.uniform(ma, mb)
#     x1 = np.random.uniform(a, b)
#     x2 = np.random.uniform(a, b)
#     data.append(np.abs(m1 - m2))
# plt.figure()
# plt.hist(data, bins=50)
# plt.show()

# x = np.arange(0, mb - ma, 0.001)
# f = 2 * (mb - ma - x) / np.power(mb - ma, 2)
# plt.figure()
# plt.plot(x, f)
# plt.show()

data = []
for i in range(n):
    m1 = np.random.uniform(ma, mb)
    m2 = np.random.uniform(ma, mb)
    x1 = np.random.uniform(a, b)
    x2 = np.random.uniform(a, b)
    data.append(np.abs(m1 - m2) - np.abs(x1 - x2))
plt.figure()
plt.hist(data, bins=50)
plt.show()

x1 = np.arange(mb - ma - b + a, mb - ma, 0.001)
f1 = (np.square(ma - mb + x1) * (4 * b - 4 * a + 2 * ma - 2 * mb + 2 * x1)) / (
        6 * np.square(a - b) * np.square(ma - mb)) - x1 / np.square(ma - mb) - (2 * ma - 2 * mb + x1) / np.square(
    ma - mb) + ((2 * ma - 2 * mb + 2 * x1) * (
        6 * np.square(a) - 12 * a * b - 4 * a * ma + 4 * a * mb - 4 * a * x1 + 6 * np.square(
    b) + 4 * b * ma - 4 * b * mb + 4 * b * x1 + np.square(ma) - 2 * ma * mb + 2 * ma * x1 + np.square(
    mb) - 2 * mb * x1 + np.square(x1))) / (6 * np.square(a - b) * np.square(ma - mb))
x2 = np.arange(a - b, 0, 0.001)
f2 = - np.power(b - a + x2, 3) / (6 * np.square(a - b) * np.square(ma - mb)) - (
        np.square(b - a + x2) * (b - a + 4 * ma - 4 * mb + x2)) / (2 * np.square(a - b) * np.square(ma - mb))
x3 = np.arange(0, mb - ma - b + a, 0.001)
f3 = -((2 * b) / 3 - (2 * a) / 3 + 2 * ma - 2 * mb + 2 * x3) / np.square(ma - mb)
plt.figure()
plt.plot(x1, f1)
plt.plot(x2, f2)
plt.plot(x3, f3)
plt.show()

p = (3 * np.square(a - b) - 2 * (a - b + 3 * ma - 3 * mb) * (a - b - ma + mb)) / (6 * np.square(ma - mb))
print(p)
