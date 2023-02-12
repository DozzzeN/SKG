import sys

from matplotlib import pyplot as plt
from scipy.spatial import distance
import numpy as np

distri = "uniform"
# episode length
l = 2
# number of measurements
n = l * 3
a = 0
b = 1
ma = 0
mb = 2

matches = 0
times = 100000

x11 = []
x21 = []
x31 = []
x12 = []
x22 = []
x32 = []
y11 = []
y21 = []
y31 = []
y12 = []
y22 = []
y32 = []
z11 = []
z12 = []
z13 = []
z21 = []
z22 = []
z23 = []
z31 = []
z32 = []
z33 = []

for t in range(times):
    measure1 = np.random.uniform(ma, mb)
    measure2 = np.random.uniform(ma, mb)
    measure3 = np.random.uniform(ma, mb)
    measure4 = np.random.uniform(ma, mb)
    measure5 = np.random.uniform(ma, mb)
    measure6 = np.random.uniform(ma, mb)
    if distri == "uniform":
        x11.append(np.random.uniform(a, b) + measure1)
        x21.append(np.random.uniform(a, b) + measure2)
        x31.append(np.random.uniform(a, b) + measure3)
        x12.append(np.random.uniform(a, b) + measure4)
        x22.append(np.random.uniform(a, b) + measure5)
        x32.append(np.random.uniform(a, b) + measure6)
        y11.append(np.random.uniform(a, b) + measure1)
        y21.append(np.random.uniform(a, b) + measure2)
        y31.append(np.random.uniform(a, b) + measure3)
        y12.append(np.random.uniform(a, b) + measure4)
        y22.append(np.random.uniform(a, b) + measure5)
        y32.append(np.random.uniform(a, b) + measure6)
    else:
        x11.append(np.random.normal(a, b) + measure1)
        x21.append(np.random.normal(a, b) + measure2)
        x31.append(np.random.normal(a, b) + measure3)
        x12.append(np.random.normal(a, b) + measure4)
        x22.append(np.random.normal(a, b) + measure5)
        x32.append(np.random.normal(a, b) + measure6)
        y11.append(np.random.normal(a, b) + measure1)
        y21.append(np.random.normal(a, b) + measure2)
        y31.append(np.random.normal(a, b) + measure3)
        y12.append(np.random.normal(a, b) + measure4)
        y22.append(np.random.normal(a, b) + measure5)
        y32.append(np.random.normal(a, b) + measure6)
    z11.append(np.abs(x11[t] - y11[t]) + np.abs(x12[t] - y12[t]))
    z12.append(np.abs(x11[t] - y21[t]) + np.abs(x12[t] - y22[t]))
    z13.append(np.abs(x11[t] - y31[t]) + np.abs(x12[t] - y32[t]))
    z21.append(np.abs(x21[t] - y11[t]) + np.abs(x22[t] - y12[t]))
    z22.append(np.abs(x21[t] - y21[t]) + np.abs(x22[t] - y22[t]))
    z23.append(np.abs(x21[t] - y31[t]) + np.abs(x22[t] - y32[t]))
    z31.append(np.abs(x31[t] - y11[t]) + np.abs(x32[t] - y12[t]))
    z32.append(np.abs(x31[t] - y21[t]) + np.abs(x32[t] - y22[t]))
    z33.append(np.abs(x31[t] - y31[t]) + np.abs(x32[t] - y32[t]))

matches = 0
for t in range(times):
    if z11[t] < z12[t] and z11[t] < z13[t]:
        matches += 1
print("11 < 12 and 11 < 13", matches / times)
k1 = matches / times

matches = 0
for t in range(times):
    if z22[t] < z21[t] and z22[t] < z23[t]:
        matches += 1
print("22 < 21 and 22 < 23", matches / times)
k2 = matches / times

matches = 0
for t in range(times):
    if z33[t] < z31[t] and z33[t] < z32[t]:
        matches += 1
print("33 < 31 and 33 < 32", matches / times)
k3 = matches / times

matches = 0
for t in range(times):
    if z11[t] < z12[t] and z11[t] < z13[t] and z22[t] < z21[t] and z22[t] < z23[t] \
            and z33[t] < z31[t] and z33[t] < z32[t]:
        matches += 1
print("11 < 12 and 11 < 13 and 22 < 21 and 22 < 23 and 33 < 31 and 33 < 32", matches / times)

print("incorrect upper bound", k1 * k2 * k3 + (1 - k1) * (1 - k2) * (1 - k3))

matches = 0
for t in range(times):
    if z12[t] < z11[t] and z12[t] < z13[t]:
        matches += 1
print("12 < 11 and 12 < 13", matches / times)
l1 = matches / times

matches = 0
for t in range(times):
    if z13[t] < z11[t] and z13[t] < z12[t]:
        matches += 1
print("13 < 11 and 13 < 12", matches / times)
l2 = matches / times

matches = 0
for t in range(times):
    if z21[t] < z22[t] and z21[t] < z23[t]:
        matches += 1
print("21 < 22 and 21 < 23", matches / times)
l3 = matches / times

matches = 0
for t in range(times):
    if z23[t] < z22[t] and z23[t] < z21[t]:
        matches += 1
print("23 < 22 and 23 < 21", matches / times)
l4 = matches / times

matches = 0
for t in range(times):
    if z31[t] < z33[t] and z31[t] < z32[t]:
        matches += 1
print("31 < 33 and 31 < 32", matches / times)
l5 = matches / times

matches = 0
for t in range(times):
    if z32[t] < z33[t] and z32[t] < z31[t]:
        matches += 1
print("32 < 33 and 32 < 31", matches / times)
l6 = matches / times

p1 = max(k1, k2, k3)
p2 = max(l1, l2, l3, l4, l5, l6)
print(p1, p2)
print("loose upper bound", np.power(p1, 3) + 3 * p1 * np.power(p2, 2) + 2 * np.power(p2, 3))

# k1 l1 l2
# l3 k2 l4
# l5 l6 k3
print("upper bound", k1 * k2 * k3 + k1 * l4 * l6 + l1 * l3 * k3 + l1 * l4 * l5 + l2 * l3 * l6 + l2 * k2 * l5)

real = 0
for t in range(times):
    episodes1 = [[x11[t], x12[t]], [x21[t], x22[t]], [x31[t], x32[t]]]
    episodes2 = [[y11[t], y12[t]], [y21[t], y22[t]], [y31[t], y32[t]]]

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
            real += 1
            break
print("real", 1 - real / times)
