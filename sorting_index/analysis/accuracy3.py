import sys
from scipy.spatial import distance
import numpy as np

# range of measurements
b = 5
a = 0
# episode length
l = 4
# number of measurements
n = l * 4

mismatches = 0
equ = 0
times = 1000

for t in range(times):
    measures = np.random.randint(a, b, n)

    samples1 = np.random.randint(0, 2, n)
    samples2 = np.random.randint(0, 2, n)

    # samples1 = np.random.normal(0, 1, n)
    # samples2 = np.random.normal(0, 1, n)
    # samples1 = (samples1 - np.min(samples1)) / (np.max(samples1) - np.min(samples1))
    # samples2 = (samples2 - np.min(samples2)) / (np.max(samples2) - np.min(samples2))

    episodes1 = np.array(samples1 + measures).reshape(int(n / l), l)
    episodes2 = np.array(samples2 + measures).reshape(int(n / l), l)

    dist = []
    for i in range(int(n / l)):
        dist.append(distance.cityblock(episodes1[i], episodes2[i]))

    flag = False
    for i in range(int(n / l)):
        if flag:
            break
        for j in range(0, int(n / l) - 1):
            if flag:
                break
            for k in range(j + 1, int(n / l)):
                if flag:
                    break
                if distance.cityblock(episodes1[i], episodes2[j]) == distance.cityblock(episodes1[i], episodes2[k]):
                    equ += 1
                    flag = True
                    # print(t, j, k, episodes2[j], episodes2[k], distance.cityblock(episodes1[i], episodes2[j]),
                    #       distance.cityblock(episodes1[i], episodes2[k]))

    matrix = []
    for i in range(int(n / l)):
        tmp = []
        for j in range(int(n / l)):
            tmp.append(distance.cityblock(episodes1[i], episodes2[j]))
        matrix.append(tmp)
    # print("---------------------")
    # print(np.mean(dist))
    # print(np.array(matrix))

    min_dist = np.zeros(int(n / l))
    for i in range(int(n / l)):
        tmp = sys.maxsize
        for j in range(int(n / l)):
            if distance.cityblock(episodes1[i], episodes2[j]) < tmp:
                tmp = distance.cityblock(episodes1[i], episodes2[j])
                min_dist[i] = j
    min_dist = sorted(min_dist)
    # print(min_dist)
    for i in range(1, len(min_dist)):
        if min_dist[i] == min_dist[i - 1]:
            mismatches += 1
            print(t)
            break
print(1 - mismatches / times)
print(mismatches)
print(equ)
