import math
import sys

from scipy.spatial import distance
import numpy as np

distri = "uniform"

l = 1
n = l * 3
a = 0
b = 10
ma = 0
mb = 100

times = 100000
matches = 0
simulates = 0
simulates1 = 0
simulates2 = 0
simulates3 = 0

for t in range(times):
    noise1 = []
    noise2 = []
    measure = []
    samples1 = []
    samples2 = []
    for i in range(n):
        measure.append(np.random.uniform(ma, mb))
        if distri == "uniform":
            noise1.append(np.random.uniform(a, b))
            noise2.append(np.random.uniform(a, b))
        else:
            noise1.append(np.random.normal(a, b))
            noise2.append(np.random.normal(a, b))

    samples1 = np.array(measure) + np.array(noise1)
    samples2 = np.array(measure) + np.array(noise2)

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

    if (np.abs(noise1[0] - noise2[0]) < np.abs(measure[0] - measure[1])
            and np.abs(noise1[0] - noise2[0]) < np.abs(measure[0] - measure[2])
            and np.abs(noise1[1] - noise2[1]) < np.abs(measure[0] - measure[1])
            and np.abs(noise1[1] - noise2[1]) < np.abs(measure[1] - measure[2])
            and np.abs(noise1[2] - noise2[2]) < np.abs(measure[0] - measure[2])
            and np.abs(noise1[2] - noise2[2]) < np.abs(measure[1] - measure[2])):
        simulates += 1
    if np.abs(noise1[0] - noise2[0]) < np.abs(measure[0] - measure[1]) \
            and np.abs(noise1[1] - noise2[1]) < np.abs(measure[0] - measure[1]) \
            and np.abs(noise1[2] - noise2[2]) < np.abs(measure[1] - measure[2]):
        simulates1 += 1
    if np.abs(noise1[0] - noise2[0]) < np.abs(measure[0] - measure[1]) \
            and np.abs(noise1[0] - noise2[0]) < np.abs(measure[0] - measure[2]):
        simulates2 += 1
    if np.abs(noise1[0] - noise2[0]) < np.abs(measure[0] - measure[1]):
        simulates3 += 1
print("real", 1 - matches / times)
print("simulate", simulates / times)
print("simulate1", simulates1 / times)
print("simulate2", np.power(simulates2 / times, 3))
print("simulate3", np.power(simulates3 / times, 3))
p = (3 * np.square(a - b) - 2 * (a - b + 3 * ma - 3 * mb) * (a - b - ma + mb)) / (6 * np.square(ma - mb))
print("probability", np.power(p, 3) + np.power(1 - p, 3))
print("probability", np.power(p, 3))
print("probability", p)
