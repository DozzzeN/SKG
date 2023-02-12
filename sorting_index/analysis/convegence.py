import math
import sys

import numpy as np
from scipy.spatial import distance

times = 100000
matches = 0

# l = 2
simulates = 0
secondary = 0
base = 0
cans = []

for t in range(times):
    x11 = np.random.uniform()
    x12 = np.random.uniform()
    x21 = np.random.uniform()
    x22 = np.random.uniform()
    y11 = np.random.uniform()
    y12 = np.random.uniform()
    y21 = np.random.uniform()
    y22 = np.random.uniform()
    z1 = np.abs(x11 - y11)
    z2 = np.abs(x12 - y12)
    z3 = np.abs(x21 - y11)
    z4 = np.abs(x22 - y12)
    z5 = np.abs(x21 - y21)
    z6 = np.abs(x22 - y22)
    z7 = np.abs(x11 - y21)
    z8 = np.abs(x12 - y22)
    a = z1 + z2 < z3 + z4
    b = z5 + z6 < z7 + z8

    a1 = z1 > z3 and z5 > z7
    a2 = z1 < z3 and z5 < z7
    a3 = z1 > z3 and z5 < z7
    a4 = z1 < z3 and z5 > z7
    a1234 = [a1, a2, a3, a4]

    b1 = z2 > z4 and z6 > z8
    b2 = z2 < z4 and z6 < z8
    b3 = z2 > z4 and z6 < z8
    b4 = z2 < z4 and z6 > z8
    b1234 = [b1, b2, b3, b4]

    if a and b:
        matches += 1
    bb = z1 < z3 and z5 < z7
    if bb:
        base += 1

    if bb and z2 < z4 and z6 < z8:
        simulates += 1
    elif bb and z2 > z4 and z6 < z8 and a1:
        simulates += 1
    elif bb and z2 < z4 and z6 > z8 and a2:
        simulates += 1
    elif bb and z2 > z4 and z6 > z8 and a1 and a2:
        simulates += 1

    if z2 < z4 and z6 < z8:
        secondary += 1
    elif z2 > z4 and z6 < z8 and a1:
        secondary += 1
    elif z2 < z4 and z6 > z8 and a2:
        secondary += 1
    elif z2 > z4 and z6 > z8 and a1 and a2:
        secondary += 1

    cans_tmp = []
    if a and b:
        for i in range(len(a1234)):
            if a1234[i]:
                cans_tmp.append(i)
        for i in range(len(b1234)):
            if b1234[i]:
                cans_tmp.append(i)
        cans.append(cans_tmp)
print(base / times)
print(matches / times)
print(secondary / times)
print(simulates * 2 / times)
cans = sorted(cans)
print(len(cans))
i = 1
while i < len(cans):
    if (np.array(cans[i - 1]) == np.array(cans[i])).all():
        del cans[i]
    else:
        i += 1
print(len(cans))
print(cans)

exit()
# l = 1
matches = 0

for t in range(times):
    x1 = np.random.uniform()
    x2 = np.random.uniform()
    y1 = np.random.uniform()
    y2 = np.random.uniform()
    a1 = np.abs(x1 - y1) < np.abs(x2 - y1)
    a2 = np.abs(x2 - y2) < np.abs(x1 - y2)
    if a1 and a2:
        # if a1:
        matches += 1
print(matches / times)

matches = 0

for t in range(times):
    z11 = np.random.uniform()
    z21 = np.random.uniform()
    z12 = np.random.uniform()
    z22 = np.random.uniform()
    # 1 / 2 ^ 2 * 2 !
    if (z11 < z21 and z12 > z22) or \
            (z11 > z21 and z12 < z22):
        matches += 1
print(math.factorial(2) / math.pow(2, 2), matches / times)

matches = 0
for t in range(times):
    z11 = np.random.uniform()
    z21 = np.random.uniform()
    z31 = np.random.uniform()
    z12 = np.random.uniform()
    z22 = np.random.uniform()
    z32 = np.random.uniform()
    z13 = np.random.uniform()
    z23 = np.random.uniform()
    z33 = np.random.uniform()
    # 1 / 3 ^ 3 * 3 !
    if z11 < z21 and z11 < z31 and z22 < z12 and z22 < z32 and z33 < z13 and z33 < z23:
        matches += 1
print(math.factorial(3) / math.pow(3, 3), matches * math.factorial(3) / times)

# real = 0.46246
times = 100000
matches = 0
l = 10
a = 1 / 3
b = 1 / (18 * l)
# approximate
n = 2
l = 1

for t in range(times):
    samples1 = []
    samples2 = []
    for i in range(n):
        # measure = np.random.uniform(a, b)
        measure = 0
        samples1.append(np.random.normal(a, b) + measure)
        samples2.append(np.random.normal(a, b) + measure)

    episodes1 = np.array(samples1).reshape(int(n / l), l)
    episodes2 = np.array(samples2).reshape(int(n / l), l)

    min_dist = np.zeros(int(n / l))
    for i in range(int(n / l)):
        tmp = sys.maxsize
        for j in range(int(n / l)):
            if distance.cityblock(episodes1[i], episodes2[j]) < tmp:
                tmp = distance.cityblock(episodes1[i], episodes2[j])
                min_dist[i] = j
            # 近似：不成立，l增大时p减小
            # if np.abs(np.sum(episodes1[i]) - np.sum(episodes2[j])) < tmp:
            #     tmp = np.abs(np.sum(episodes1[i]) - np.sum(episodes2[j]))
            #     min_dist[i] = j
    min_dist = sorted(min_dist)
    for i in range(1, len(min_dist)):
        if min_dist[i] == min_dist[i - 1]:
            matches += 1
            break
print("real", 1 - matches / times)
