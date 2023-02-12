from itertools import product
import math
import random
from collections import Counter
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import perm

# range of measurements
b = 5
a = 0
# episode length
l = 5
# number of measurements
n = l * 16
numerator = perm(int(pow(b - a, l)), int(n / l))
denominator = int(pow(pow(b - a, l), int(n / l)))

print("rigorous derivation")
print(numerator / denominator)
exit()

samples = np.random.normal(0, 1, n)
step = max(samples) - min(samples)
m = min(samples)
for i in range(len(samples)):
    samples[i] = round((samples[i] - m) / step * (b - 1 - a))

cnt = Counter(samples)
dict_cnt = []
for i in range(len(cnt)):
    dict_cnt.append(cnt[i])
# print(dict_cnt)
perms = [p for p in product(*[list(range(len(cnt)))] * len(cnt))]
# print(len(perms))
count = 0
dict_cnt = np.ones(len(cnt))
for i in range(len(perms)):
    tmp = 1
    for j in range(len(perms[i])):
        tmp *= dict_cnt[perms[i][j]] / n
    count += 1 / tmp
count = round(count)
print(count)
print(pow(b - a, l))

numerator = perm(count, int(n / l))
denominator = int(pow(count, int(n / l)))

print("rigorous derivation")
print(numerator / denominator)

# plt.figure()
# plt.hist(samples, rwidth=0.1)
# plt.show()
# exit()

# numerical simulation
repeated = 0
times = 100000
print(round(times * (1 - numerator / denominator)))

print("numerical simulation", times, "times")
for t in range(times):
    # samples = []
    # for i in range(n):
    #     samples.append(round(random.uniform(a, b)))
    #     samples.append(random.randint(a, b - 1))
    # samples.append(round(round(np.random.normal((a + b - 1) / 2, 1) * 100) / 100))
    samples = np.random.normal(0, 1, n)
    step = max(samples) - min(samples)
    m = min(samples)
    for i in range(len(samples)):
        samples[i] = round((samples[i] - m) / step * (b - 1 - a))

    episodes = np.array(samples).reshape(int(n / l), l)
    episodes = episodes[np.lexsort(episodes[:, ::-1].T)]  # 第一列排序
    flag = False
    for i in range(1, len(episodes)):
        equals = 0
        if flag:
            break
        for j in range(l):
            if episodes[i - 1][j] == episodes[i][j]:
                equals += 1
                if equals == l:
                    repeated += 1
                    flag = True

print(1 - repeated / times)
print(repeated)

# rigorous derivation
# 0.9112263534208994
# 88773.64657910059
# numerical simulation (1000000 times)
# 0.910994
# 89006
