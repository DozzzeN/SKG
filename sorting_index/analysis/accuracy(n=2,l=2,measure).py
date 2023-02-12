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
    measure1 = np.random.uniform(a, b)
    measure2 = np.random.uniform(a, b)
    measure3 = np.random.uniform(a, b)
    measure4 = np.random.uniform(a, b)
    if distri == "uniform":
        x11.append(np.random.uniform(a, b) + measure1)
        x21.append(np.random.uniform(a, b) + measure2)
        x12.append(np.random.uniform(a, b) + measure3)
        x22.append(np.random.uniform(a, b) + measure4)
        y11.append(np.random.uniform(a, b) + measure1)
        y21.append(np.random.uniform(a, b) + measure2)
        y12.append(np.random.uniform(a, b) + measure3)
        y22.append(np.random.uniform(a, b) + measure4)
    else:
        x11.append(np.random.normal(a, b) + measure1)
        x21.append(np.random.normal(a, b) + measure2)
        x12.append(np.random.normal(a, b) + measure3)
        x22.append(np.random.normal(a, b) + measure4)
        y11.append(np.random.normal(a, b) + measure1)
        y21.append(np.random.normal(a, b) + measure2)
        y12.append(np.random.normal(a, b) + measure3)
        y22.append(np.random.normal(a, b) + measure4)
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

print("loose upper bound", k1 * k1 + (1 - k1) * (1 - k1))  # 假设k1=k2
print("upper bound", k1 * k2 + (1 - k1) * (1 - k2))
print("equivalent upper bound", k1 * k2 + k3 * k4)

matches = 0
for t in range(times):
    episodes1 = [[x11[t], x12[t]], [x21[t], x22[t]]]
    episodes2 = [[y11[t], y12[t]], [y21[t], y22[t]]]

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
