import sys

from matplotlib import pyplot as plt
from scipy.spatial import distance
import numpy as np

distri = "uniform"
# episode length
l = 2
# number of measurements
n = l * 2
a = 0
b = 1

matches = 0
times = 100000

x11 = []
x21 = []
x12 = []
x22 = []
y11 = []
y21 = []
y12 = []
y22 = []
z11 = []
z12 = []
z21 = []
z22 = []

for t in range(times):
    if distri == "uniform":
        x11.append(np.random.uniform(a, b))
        x21.append(np.random.uniform(a, b))
        x12.append(np.random.uniform(a, b))
        x22.append(np.random.uniform(a, b))
        y11.append(np.random.uniform(a, b))
        y21.append(np.random.uniform(a, b))
        y12.append(np.random.uniform(a, b))
        y22.append(np.random.uniform(a, b))
    else:
        x11.append(np.random.normal(a, b))
        x21.append(np.random.normal(a, b))
        x12.append(np.random.normal(a, b))
        x22.append(np.random.normal(a, b))
        y11.append(np.random.normal(a, b))
        y21.append(np.random.normal(a, b))
        y12.append(np.random.normal(a, b))
        y22.append(np.random.normal(a, b))
    z11.append(np.abs(x11[t] - y11[t]) + np.abs(x12[t] - y12[t]))
    z12.append(np.abs(x11[t] - y21[t]) + np.abs(x12[t] - y22[t]))
    z21.append(np.abs(x21[t] - y11[t]) + np.abs(x22[t] - y12[t]))
    z22.append(np.abs(x21[t] - y21[t]) + np.abs(x22[t] - y22[t]))

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
    a = np.abs(x11[t] - y11[t]) + np.abs(x12[t] - y12[t]) > np.abs(x11[t] - y21[t]) + np.abs(x12[t] - y22[t])
    a1 = x11[t] > y11[t] and x12[t] > y12[t] and x11[t] > y21[t] and x12[t] > y22[t]
    if a and a1:
        matches += 1
print("z11 > z12 and >>>>", matches / times)

matches = 0
for t in range(times):
    a = np.abs(x11[t] - y11[t]) + np.abs(x12[t] - y12[t]) > np.abs(x11[t] - y21[t]) + np.abs(x12[t] - y22[t])
    a2 = x11[t] > y11[t] and x12[t] < y12[t] and x11[t] > y21[t] and x12[t] > y22[t]
    if a and a2:
        matches += 1
print("z11 > z12 and ><>>", matches / times)

matches = 0
for t in range(times):
    a = np.abs(x11[t] - y11[t]) + np.abs(x12[t] - y12[t]) > np.abs(x11[t] - y21[t]) + np.abs(x12[t] - y22[t])
    a3 = x11[t] < y11[t] and x12[t] > y12[t] and x11[t] > y21[t] and x12[t] > y22[t]
    if a and a3:
        matches += 1
print("z11 > z12 and <>>>", matches / times)

matches = 0
for t in range(times):
    a = np.abs(x11[t] - y11[t]) + np.abs(x12[t] - y12[t]) > np.abs(x11[t] - y21[t]) + np.abs(x12[t] - y22[t])
    a4 = x11[t] < y11[t] and x12[t] < y12[t] and x11[t] > y21[t] and x12[t] > y22[t]
    if a and a4:
        matches += 1
print("z11 > z12 and <<>>", matches / times)

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
        if distri == "uniform":
            samples1.append(np.random.uniform(a, b))
            samples2.append(np.random.uniform(a, b))
        else:
            samples1.append(np.random.normal(a, b))
            samples2.append(np.random.normal(a, b))

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
