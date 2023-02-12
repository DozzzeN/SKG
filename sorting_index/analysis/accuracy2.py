from itertools import product
import math
import random
from collections import Counter
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import perm

# range of measurements
b = 20
a = 0
# episode length
l = 6
# number of measurements
n = l * 128

print("rigorous derivation")
# numerator = perm(int(pow(b - a, l)), int(n / l))
# denominator = int(pow(pow(b - a, l), int(n / l)))

# print(numerator / denominator)

# numerical simulation
repeated = 0
times = 1000

print("numerical simulation", times, "times")
for t in range(times):
    samples = np.random.normal(0, 1, n)
    step = max(samples) - min(samples)
    m = min(samples)
    for i in range(len(samples)):
        samples[i] = round((samples[i] - m) / step * (b - 1 - a))
    # samples = []
    # for i in range(n):
    #     f = random.randint(1, 3)
    #     if f == 1:
    #         samples.append(0)
    #     elif f == 2:
    #         samples.append(1)
    #     else:
    #         samples.append(2)
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
