import random
import sys
import time

import numpy as np
from pyLSHash import LSHash
from scipy.spatial import distance

max_len = 256
max_dim = 7
repeat = 100
start1 = time.time()
lsh = LSHash(hash_size=max_len, input_dim=max_dim)
ranArray = np.random.random([max_len, max_dim])
for i in range(max_len):
    lsh.index(ranArray[i], extra_data=i)

# q = np.random.random(size=(1, max_dim))
# q = ranArray[np.random.randint(0, len(ranArray))]
# q = q + np.random.randint(0, max_dim * max_len, size=len(q))
# res = lsh.query(q, dist_func=distance.euclidean)
# min_index = -1
# min_dist = sys.maxsize
# can = []
# for i in range(len(res)):
#     can.append(res[i][0][1])
#     if res[i][1] < min_dist:
#         min_dist = res[i][1]
#         min_index = res[i][0][1]
# print(min_index)
# print(sorted(can))

for i in range(repeat):
    q = np.random.random([max_dim])
    res = lsh.query(q, dist_func=lambda x, y: distance.cityblock(x, y))
end1 = time.time()

start2 = time.time()
# min_dist = sys.maxsize
# min_index = -1
# for i in range(max_len):
#     dist = distance.euclidean(q, ranArray[i])
#     if dist < min_dist:
#         min_dist = dist
#         min_index = i
# print(min_index)
for j in range(repeat):
    min_dist = sys.maxsize
    q = np.random.random([max_dim])
    for i in range(max_len):
        dist = distance.cityblock(q, ranArray[i])
        if dist < min_dist:
            min_dist = dist
end2 = time.time()

print(end1 - start1)
print(end2 - start2)

# quick start
# lsh = LSHash(hash_size=6, input_dim=8)
# lsh.index([1, 2, 3, 4, 5, 6, 7, 8])
# lsh.index([2, 3, 4, 5, 6, 7, 8, 9])
# # attach extra_data
# lsh.index([2, 3, 4, 5, 6, 7, 8, 9], extra_data="some vector info")
# lsh.index([10, 12, 99, 1, 5, 31, 2, 3])
#
# res = lsh.query([1, 2, 3, 4, 5, 6, 7, 7], dist_func=lambda x, y: distance.cityblock(x, y))
# min_index = -1
# min_dist = sys.maxsize
# for i in range(len(res)):
#     if res[i][1] < min_dist:
#         min_dist = res[i][1]
#         min_index = i
# print(min_index)
