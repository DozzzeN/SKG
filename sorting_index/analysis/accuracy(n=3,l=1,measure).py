import sys

from matplotlib import pyplot as plt
from scipy.spatial import distance
import numpy as np

distri = "uniform"
# episode length
l = 1
# number of measurements
n = l * 3
a = 0
b = 1
ma = 0
mb = 2

matches = 0
times = 100000

x1 = []
x2 = []
x3 = []
y1 = []
y2 = []
y3 = []
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
    if distri == "uniform":
        x1.append(np.random.uniform(a, b) + measure1)
        x2.append(np.random.uniform(a, b) + measure2)
        x3.append(np.random.uniform(a, b) + measure3)
        y1.append(np.random.uniform(a, b) + measure1)
        y2.append(np.random.uniform(a, b) + measure2)
        y3.append(np.random.uniform(a, b) + measure3)
    else:
        x1.append(np.random.normal(a, b) + measure1)
        x2.append(np.random.normal(a, b) + measure2)
        x3.append(np.random.normal(a, b) + measure3)
        y1.append(np.random.normal(a, b) + measure1)
        y2.append(np.random.normal(a, b) + measure2)
        y3.append(np.random.normal(a, b) + measure3)
    z11.append(np.abs(x1[t] - y1[t]))
    z12.append(np.abs(x1[t] - y2[t]))
    z13.append(np.abs(x1[t] - y3[t]))
    z21.append(np.abs(x2[t] - y1[t]))
    z22.append(np.abs(x2[t] - y2[t]))
    z23.append(np.abs(x2[t] - y3[t]))
    z31.append(np.abs(x3[t] - y1[t]))
    z32.append(np.abs(x3[t] - y2[t]))
    z33.append(np.abs(x3[t] - y3[t]))

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
    episodes1 = [[x1[t]], [x2[t]], [x3[t]]]
    episodes2 = [[y1[t]], [y2[t]], [y3[t]]]

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
