import math
import random


# [0, 1]内随机取出两点，形成一个闭区间，取n次，n个闭区间不出现重叠的概率
# https://blog.csdn.net/qq_31720305/article/details/89078737
import sys

import numpy as np
from matplotlib import pyplot as plt


def selectProb(n):
    res = math.pow(2, n) * math.perm(n) / math.perm(2 * n)
    return res


print(selectProb(4))


# [0, m]中随机选取n个点，任意两个点之间的最小间距
# https://zhuanlan.zhihu.com/p/102503037
def intervalProb(m, n):
    res = m / (math.pow(n, 2) - 1)
    return res


print(intervalProb(100, 10))


def play(m, n):
    samples = np.random.uniform(0, m, n)
    samples.sort()
    min_diff = sys.maxsize
    for i in range(len(samples) - 1):
        min_diff = min(min_diff, abs(samples[i + 1] - samples[i]))
    return min_diff


def theory(m, n):
    x = np.random.uniform(0, m / (n - 1), 1)
    one = n * (n - 1) / math.pow(m, n)
    return one * math.pow(m - (n - 1) * x, n - 1)


min_diffs = []
for i in range(100):
    min_diffs.append(play(10, 10))

pdf = []
for i in range(100):
    pdf.append(theory(10, 10))

plt.figure()
plt.plot(min_diffs, "r")
plt.plot(pdf, "b")
plt.show()