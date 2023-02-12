import math
import sys

from scipy.spatial import distance
import numpy as np

distri = "normal"
p2 = 0
pk = []

for k in range(2, 10):
    l = 2
    n = l * k

    print("n", k)
    a = 0
    b = 1

    times = 1000000
    matches = 0

    for t in range(times):
        samples1 = []
        samples2 = []
        for i in range(n):
            measure = np.random.uniform(a, b)
            # measure = 0
            if distri == "uniform":
                samples1.append(np.random.uniform(a, b) + measure)
                samples2.append(np.random.uniform(a, b) + measure)
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