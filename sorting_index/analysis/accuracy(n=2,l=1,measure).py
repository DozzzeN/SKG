import sys

from matplotlib import pyplot as plt
from scipy.spatial import distance
import numpy as np

distri = "uniform"
# episode length
l = 1
# number of measurements
n = l * 2
a = 0
b = 1

matches = 0
times = 100000

x1 = []
x2 = []
y1 = []
y2 = []
z11 = []
z12 = []
z21 = []
z22 = []

for t in range(times):
    measure1 = np.random.rand()
    measure2 = np.random.rand()
    noise1 = np.random.rand()
    noise2 = np.random.rand()
    if distri == "uniform":
        x1.append(noise1 + np.random.uniform(0, 10))
        x2.append(noise2 + np.random.uniform(0, 10))
        y1.append(noise1 + np.random.uniform(0, 10))
        y2.append(noise2 + np.random.uniform(0, 10))
        # x1.append(np.random.rand() + measure1)
        # x2.append(np.random.rand() + measure2)
        # y1.append(np.random.rand() + measure1)
        # y2.append(np.random.rand() + measure2)
        # x1.append(np.random.rand() + np.random.rand())
        # x2.append(np.random.rand() + np.random.rand())
        # y1.append(np.random.rand() + np.random.rand())
        # y2.append(np.random.rand() + np.random.rand())
    else:
        x1.append(np.random.normal(a, b) + measure1)
        x2.append(np.random.normal(a, b) + measure2)
        y1.append(np.random.normal(a, b) + measure1)
        y2.append(np.random.normal(a, b) + measure2)
    z11.append(np.abs(x1[t] - y1[t]))
    z12.append(np.abs(x1[t] - y2[t]))
    z21.append(np.abs(x2[t] - y1[t]))
    z22.append(np.abs(x2[t] - y2[t]))

matches = 0
for t in range(times):
    if z11[t] < z12[t]:
        matches += 1
print("z11 < z12", matches / times)
k1 = matches / times

matches = 0
for t in range(times):
    if z22[t] < z21[t]:
        matches += 1
print("z22 < z21", matches / times)
k2 = matches / times

matches = 0
for t in range(times):
    if z11[t] > z12[t]:
        matches += 1
print("z11 > z12", matches / times)
k3 = matches / times

matches = 0
for t in range(times):
    if z22[t] > z21[t]:
        matches += 1
print("z22 > z21", matches / times)
k4 = matches / times

matches = 0
for t in range(times):
    a = np.abs(x1[t] - y1[t]) > np.abs(x1[t] - y2[t])
    a1 = x1[t] > y1[t] and x1[t] > y2[t]
    if a and a1:
        matches += 1
print("z11 > z12 and x1 > y1 and x1 > y2", matches / times, 7 / 60)

matches = 0
for t in range(times):
    a = np.abs(x1[t] - y1[t]) > np.abs(x1[t] - y2[t])
    a2 = x1[t] < y1[t] and x1[t] < y2[t]
    if a and a2:
        matches += 1
print("z11 > z12 and x1 < y1 and x1 < y2", matches / times, 7 / 60)

matches = 0
for t in range(times):
    a = np.abs(x1[t] - y1[t]) > np.abs(x1[t] - y2[t])
    a3 = x1[t] > y1[t] and x1[t] < y2[t]
    if a and a3:
        matches += 1
print("z11 > z12 and x1 > y1 and x1 < y2", matches / times, 1 / 15)

matches = 0
for t in range(times):
    a = np.abs(x1[t] - y1[t]) > np.abs(x1[t] - y2[t])
    a4 = x1[t] < y1[t] and x1[t] > y2[t]
    if a and a4:
        matches += 1
print("z11 > z12 and x1 < y1 and x1 > y2", matches / times, 1 / 15)

matches = 0
for t in range(times):
    if z11[t] < z12[t] and z22[t] < z21[t]:
        matches += 1
print("z11 < z12 and z22 < z21", matches / times)

matches = 0
for t in range(times):
    if z11[t] > z12[t] and z22[t] > z21[t]:
        matches += 1
print("z11 > z12 and z22 > z21", matches / times)

matches = 0
for t in range(times):
    if z11[t] < z12[t] and z22[t] < z21[t]:
        matches += 1
    elif z11[t] > z12[t] and z22[t] > z21[t]:
        matches += 1
print("z11 < z12 and z22 < z21 or z11 > z12 and z22 > z21", matches / times)

print("upper bound", k1 * k1 + (1 - k1) * (1 - k1))
print("tight upper bound", k1 * k2 + k3 * k4)

matches = 0
for t in range(times):
    samples1 = []
    samples2 = []
    for i in range(n):
        measure = np.random.uniform()
        if distri == "uniform":
            samples1.append(np.random.rand() + measure)
            samples2.append(np.random.rand() + measure)
            # samples1.append(np.random.rand() + np.random.rand())
            # samples2.append(np.random.rand() + np.random.rand())
        else:
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
    min_dist = sorted(min_dist)
    for i in range(1, len(min_dist)):
        if min_dist[i] == min_dist[i - 1]:
            matches += 1
            break
print("real", 1 - matches / times)
