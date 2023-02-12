import sys
from math import factorial

from matplotlib import pyplot as plt
from scipy.spatial import distance
import numpy as np

distri = "uniform"
# episode length
l = 1
# number of measurements
n = l * 4

matches = 0
equ = 0
times = 100000

x1 = []
x2 = []
x3 = []
y1 = []
y2 = []
z11 = []
z12 = []
z21 = []
z22 = []
z31 = []
z32 = []

for t in range(times):
    if distri == "uniform":
        x1.append(np.random.rand())
        x2.append(np.random.rand())
        x3.append(np.random.rand())
        y1.append(np.random.rand())
        y2.append(np.random.rand())
    else:
        x1.append(np.random.normal())
        x2.append(np.random.normal())
        x3.append(np.random.normal())
        y1.append(np.random.normal())
        y2.append(np.random.normal())
    z11.append(np.abs(x1[t] - y1[t]))
    z12.append(np.abs(x1[t] - y2[t]))
    z21.append(np.abs(x2[t] - y1[t]))
    z22.append(np.abs(x2[t] - y2[t]))
    z31.append(np.abs(x3[t] - y1[t]))
    z32.append(np.abs(x3[t] - y2[t]))

for t in range(times):
    if z11[t] < z21[t]:
        matches += 1
print("z11 < z21", matches / times)

matches = 0
for t in range(times):
    if z11[t] < z21[t] and z22[t] < z12[t]:
        matches += 1
print("z11 < z21 and z22 < z12", matches / times)

matches = 0
for t in range(times):
    if z11[t] < z21[t] and z22[t] < z12[t]:
        matches += 1
    elif z11[t] > z21[t] and z22[t] > z12[t]:
        matches += 1
print("z11 < z21 and z22 < z12 or z11 > z21 and z22 > z12", matches / times)

matches = 0
for t in range(times):
    if z11[t] < z21[t] and z22[t] < z12[t] and z11[t] < z31[t] and z22[t] < z32[t]:
        matches += 1
print("z11 < z21 and z22 < z12 and z11 < z31 and z22 < z32", matches / times)

matches = 0
for t in range(times):
    if z11[t] < z21[t] and z11[t] < z31[t]:
        matches += 1
print("z11 < z21 and z11 < z31", matches / times)

matches = 0
equ = 0
for t in range(times):
    samples1 = []
    samples2 = []
    for i in range(n):
        if distri == "uniform":
            samples1.append(np.random.uniform())
            samples2.append(np.random.uniform())
        else:
            samples1.append(np.random.normal())
            samples2.append(np.random.normal())

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
k = int(n / l)
print("real", (1 - matches / times) / factorial(k))
print("n=k", 1 - matches / times)
print(factorial(k) / (np.power(k, k)))
print(factorial(k + 1) / (np.power(k + 1, k + 1)))
print(1 / (np.power(k + 1, k)))
print(factorial(k) * np.power(5 / 24, (k - 1) * (k - 1)))
