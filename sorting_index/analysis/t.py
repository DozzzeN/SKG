import sys

from matplotlib import pyplot as plt
from scipy.spatial import distance
import numpy as np

distri = "uniform"
# episode length
l = 1
# number of measurements
n = l * 2

matches = 0
equ = 0
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
    if distri == "uniform":
        x1.append(np.random.rand())
        x2.append(np.random.rand())
        y1.append(np.random.rand())
        y2.append(np.random.rand())
    else:
        x1.append(np.random.normal())
        x2.append(np.random.normal())
        y1.append(np.random.normal())
        y2.append(np.random.normal())
    z11.append(np.abs(x1[t] - y1[t]))
    z12.append(np.abs(x1[t] - y2[t]))
    z21.append(np.abs(x2[t] - y1[t]))
    z22.append(np.abs(x2[t] - y2[t]))

for t in range(times):
    if x1[t] < x2[t]:
        matches += 1
print(matches / times)

matches = 0
for t in range(times):
    if x1[t] < x2[t] and x1[t] < y1[t]:
        matches += 1
print(matches / times)

matches = 0
for t in range(times):
    if x1[t] < x2[t] and x1[t] < y1[t] and x1[t] < y2[t]:
        matches += 1
print(matches / times)
