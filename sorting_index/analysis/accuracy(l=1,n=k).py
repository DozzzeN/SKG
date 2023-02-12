import math
import sys

from scipy.spatial import distance
import numpy as np

distri = "uniform"
p2 = 0
pk = []

for k in range(2, 10):
    l = 1
    n = l * k
    print("n", k)
    a = 0
    b = 1

    times = 100000
    matches = 0

    for t in range(times):
        samples1 = []
        samples2 = []
        for i in range(n):
            measure = np.random.uniform(a, b)
            if distri == "uniform":
                samples1.append(np.random.uniform(a, b) + measure)
                samples2.append(np.random.uniform(a, b) + measure)
            else:
                samples1.append(np.random.normal(a, b) + measure)
                samples2.append(np.random.normal(a, b) + measure)
            # if distri == "uniform":
                # samples1.append(np.random.uniform(a, b))
                # samples2.append(np.random.uniform(a, b))
            # else:
            #     samples1.append(np.random.normal(a, b))
            #     samples2.append(np.random.normal(a, b))

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
    if k == 2:
        p2 = (1 - matches / times) / 2
    print("tight lower bound", math.factorial(k) / np.power(k + 1, k))
    print("tight lower bound", math.factorial(k) / np.power(k + 1, k + 1))
    print("upper bound", math.factorial(k) / np.power(k, k))
    if k > 2:
        print("recursive upper bound", pk[len(pk) - 1] * k * np.power(p2, (k - 1)))
        print("recursive upper bound", np.power(pk[len(pk) - 1] / math.factorial(k - 1),
                                                np.floor(k / (k - 2))) * math.factorial(k))
    pk.append(1 - matches / times)

    print("lower bound", math.factorial(k) * np.power(p2, k * (k - 1) / 2))
    print("lower bound", math.factorial(k) * np.power(p2, (k - 1) * (k - 1)))
    print("upper bound", math.factorial(k) * np.power(p2, k - 1))

    print("tight lower bound", math.factorial(k) / np.power(k + 1, k))
