import itertools
import math
from collections import Counter

import numpy as np


def search(data, p):
    for i in range(len(data)):
        if p == data[i]:
            return i


# print([p for p in itertools.permutations([1, 2, 3, 4])])
n = 6
a = list(range(n))
print(a)
# b = [p for p in itertools.permutations([3, 1, 2])]
# b = [[4] + list(p) for p in b]
b = [p for p in itertools.permutations(a)]
final_dist = 0
all = []
for j in range(len(b)):
    tmp_dist = 0
    for i in range(len(b[j])):
        real_pos = search(a, b[j][i])
        guess_pos = i
        tmp_dist += abs(real_pos - guess_pos)
    all.append(tmp_dist)
    final_dist += tmp_dist
print(sum(all))
print(sum(all) / math.factorial(n))
print((math.pow(n, 2) - 1) / 3)
print(sorted(Counter(all).items(), key=lambda _: _[0], reverse=False))
print(sum(Counter(all).values()))
