import math
import sys

from scipy.spatial import distance
import numpy as np

distri = "normal"

probs = []
for l in range(1, 10):
    n = l * 2
    a = 0
    b = 5
    ma = 0
    mb = 1

    times = 100000
    matches = 0

    for t in range(times):
        noise1 = []
        noise2 = []
        measure = []
        samples1 = []
        samples2 = []
        for i in range(n):
            if distri == "uniform":
                measure.append(np.random.uniform(ma, mb))
                noise1.append(np.random.uniform(a, b))
                noise2.append(np.random.uniform(a, b))
            else:
                measure.append(np.random.normal(ma, mb))
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
    print("l", l,  round(1 - matches / times, 5))
    probs.append(round(1 - matches / times, 5))
print(probs)