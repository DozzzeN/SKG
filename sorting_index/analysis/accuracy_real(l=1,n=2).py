import math
import sys

from scipy.spatial import distance
import numpy as np
import heartrate
heartrate.trace(browser=True)

distri = "uniform"

l = 1
n = l * 2
a = 0
b = 10
ma = 0
mb = 100

times = 100000
matches = 0
simulates = 0

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

    if np.abs(noise1[0] - noise2[0]) < np.abs(measure[0] - measure[1]) \
            and np.abs(noise1[1] - noise2[1]) < np.abs(measure[0] - measure[1]):
        simulates += 1
    # 可忽略
    # if np.abs(noise1[0] - noise2[0]) > np.abs(measure[0] - measure[1]) \
    #         and np.abs(noise1[1] - noise2[1]) > np.abs(measure[0] - measure[1]):
    #     simulates += 1
print("real", 1 - matches / times)
print("simulate", simulates / times)
p = (3 * np.square(a - b) - 2 * (a - b + 3 * ma - 3 * mb) * (a - b - ma + mb)) / (6 * np.square(ma - mb))
print("probability", p * p + (1 - p) * (1 - p))
print("probability", p * p)
print("probability", p)
# 最终结果，是忽略次对角线概率的推导值
prob = np.square(a - b) / (24 * np.square(ma - mb)) + np.square(a - b - ma + mb) / (8 * np.square(ma - mb)) - 2 * (
        (a - b) * (a - b - ma + mb)) / (15 * np.square(ma - mb))
print("prob", prob * 8)
