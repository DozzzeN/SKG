import sys

from matplotlib import pyplot as plt
from scipy.spatial import distance
import numpy as np


distri = "uniform"
# episode length
l = 1
# number of measurements
n = l * 3

matches = 0
equ = 0
times = 100000

for t in range(times):
    if distri == "uniform":
        x1 = np.random.rand()
        x2 = np.random.rand()
        x3 = np.random.rand()
        y1 = np.random.rand()
        y2 = np.random.rand()
        y3 = np.random.rand()
    else:
        x1 = np.random.normal()
        x2 = np.random.normal()
        x3 = np.random.normal()
        y1 = np.random.normal()
        y2 = np.random.normal()
        y3 = np.random.normal()
    z11 = np.abs(x1 - y1)
    z12 = np.abs(x1 - y2)
    z13 = np.abs(x1 - y3)
    z21 = np.abs(x2 - y1)
    z22 = np.abs(x2 - y2)
    z23 = np.abs(x2 - y3)
    z31 = np.abs(x3 - y1)
    z32 = np.abs(x3 - y2)
    z33 = np.abs(x3 - y3)

    if z11 < z21 and z11 < z31:
        matches += 1

print(matches / times)

matches = 0
for t in range(times):
    if distri == "uniform":
        x1 = np.random.rand()
        x2 = np.random.rand()
        x3 = np.random.rand()
        y1 = np.random.rand()
        y2 = np.random.rand()
        y3 = np.random.rand()
    else:
        x1 = np.random.normal()
        x2 = np.random.normal()
        x3 = np.random.normal()
        y1 = np.random.normal()
        y2 = np.random.normal()
        y3 = np.random.normal()
    z11 = np.abs(x1 - y1)
    z12 = np.abs(x1 - y2)
    z13 = np.abs(x1 - y3)
    z21 = np.abs(x2 - y1)
    z22 = np.abs(x2 - y2)
    z23 = np.abs(x2 - y3)
    z31 = np.abs(x3 - y1)
    z32 = np.abs(x3 - y2)
    z33 = np.abs(x3 - y3)

    if z11 < z21 and z11 < z31 and z22 < z12 and z22 < z32 and z33 < z13 and z33 < z23:
        matches += 1

print(matches / times * 6)

matches = 0
for t in range(times):
    if distri == "uniform":
        x1 = np.random.rand()
        x2 = np.random.rand()
        x3 = np.random.rand()
        y1 = np.random.rand()
        y2 = np.random.rand()
        y3 = np.random.rand()
    else:
        x1 = np.random.normal()
        x2 = np.random.normal()
        x3 = np.random.normal()
        y1 = np.random.normal()
        y2 = np.random.normal()
        y3 = np.random.normal()
    z11 = np.abs(x1 - y1)
    z12 = np.abs(x1 - y2)
    z13 = np.abs(x1 - y3)
    z21 = np.abs(x2 - y1)
    z22 = np.abs(x2 - y2)
    z23 = np.abs(x2 - y3)
    z31 = np.abs(x3 - y1)
    z32 = np.abs(x3 - y2)
    z33 = np.abs(x3 - y3)

    if z11 < z21 and z11 < z31 and z22 < z12 and z22 < z32 and z33 < z13 and z33 < z23:
        matches += 1
    if z11 < z21 and z11 < z31 and z32 < z12 and z32 < z22 and z23 < z13 and z23 < z33:
        matches += 1
    if z21 < z11 and z21 < z31 and z12 < z32 and z12 < z22 and z33 < z13 and z33 < z23:
        matches += 1
    if z21 < z11 and z21 < z31 and z32 < z12 and z32 < z22 and z13 < z23 and z13 < z33:
        matches += 1
    if z31 < z21 and z31 < z11 and z12 < z32 and z12 < z22 and z23 < z13 and z23 < z33:
        matches += 1
    if z31 < z21 and z31 < z11 and z22 < z12 and z22 < z32 and z13 < z23 and z13 < z33:
        matches += 1

print(matches / times)

matches = 0
equ = 0
for t in range(times):
    samples1 = []
    samples2 = []
    for i in range(n):
        if distri == "uniform":
            samples1.append(np.random.rand())
            samples2.append(np.random.rand())
        else:
            samples1.append(np.random.normal())
            samples2.append(np.random.normal())

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
