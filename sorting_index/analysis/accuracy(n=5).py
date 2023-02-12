import sys

from matplotlib import pyplot as plt
from scipy.spatial import distance
import numpy as np

# episode length
l = 1
# number of measurements
n = l * 5

matches = 0
equ = 0
times = 1000000

for t in range(times):
    x1 = np.random.rand()
    x2 = np.random.rand()
    x3 = np.random.rand()
    x4 = np.random.rand()
    x5 = np.random.rand()
    y1 = np.random.rand()
    y2 = np.random.rand()
    y3 = np.random.rand()
    y4 = np.random.rand()
    y5 = np.random.rand()
    z11 = np.abs(x1 - y1)
    z12 = np.abs(x1 - y2)
    z13 = np.abs(x1 - y3)
    z14 = np.abs(x1 - y4)
    z15 = np.abs(x1 - y5)
    z21 = np.abs(x2 - y1)
    z22 = np.abs(x2 - y2)
    z23 = np.abs(x2 - y3)
    z24 = np.abs(x2 - y4)
    z25 = np.abs(x2 - y5)
    z31 = np.abs(x3 - y1)
    z32 = np.abs(x3 - y2)
    z33 = np.abs(x3 - y3)
    z34 = np.abs(x3 - y4)
    z35 = np.abs(x3 - y5)
    z41 = np.abs(x4 - y1)
    z42 = np.abs(x4 - y2)
    z43 = np.abs(x4 - y3)
    z44 = np.abs(x4 - y4)
    z45 = np.abs(x4 - y5)
    z51 = np.abs(x5 - y1)
    z52 = np.abs(x5 - y2)
    z53 = np.abs(x5 - y3)
    z54 = np.abs(x5 - y4)
    z55 = np.abs(x5 - y5)

    if z11 < z21 and z11 < z31 and z11 < z41 and z11 < z51:
        if z22 < z12 and z22 < z32 and z22 < z42 and z22 < z52:
            if z33 < z13 and z33 < z23 and z33 < z43 and z33 < z53:
                if z44 < z14 and z44 < z24 and z44 < z34 and z44 < z54:
                    if z55 < z15 and z55 < z25 and z55 < z35 and z55 < z45:
                        matches += 1

print(matches / times)
print(matches / times * 120)

matches = 0
equ = 0
for t in range(times):
    samples1 = []
    samples2 = []
    for i in range(n):
        samples1.append(np.random.rand())
        samples2.append(np.random.rand())

    episodes1 = np.array(samples1).reshape(int(n / l), l)
    episodes2 = np.array(samples2).reshape(int(n / l), l)

    min_dist = np.zeros(int(n / l))
    for i in range(int(n / l)):
        tmp = sys.maxsize
        for j in range(int(n / l)):
            if np.abs(episodes1[i] - episodes2[j]) < tmp:
                tmp = np.abs(episodes1[i] - episodes2[j])
                min_dist[i] = j
    min_dist = sorted(min_dist)
    for i in range(1, len(min_dist)):
        if min_dist[i] == min_dist[i - 1]:
            matches += 1
            break
print(1 - matches / times)
