import sys
import time

import numpy as np
from lshashing import LSHRandom
from pyLSHash import LSHash
from scipy.spatial import distance

from EuclideanLSH import EuclideanLSH

max_dim = 7
max_len = 256
# data = np.random.random([max_len, max_dim])
# query = np.random.random([max_dim])
data = np.random.randint(size=(max_len, max_dim), low=0, high=10)
query = np.random.randint(size=max_dim, low=0, high=10)

# 自己实现的，准确但速度慢
lsh = EuclideanLSH(10, 1, max_dim)
lsh.insert(data)

start1 = time.time()
res = lsh.query(query, 1)
res = np.array(res)
top = -1
for i in range(len(data)):
    if np.sum(np.power(data[i] - query, 2), axis=-1) == np.sum(np.power(res[0] - query, 2), axis=-1):
        top = i
print(top)
end1 = time.time()

# LSHash库，速度快，但不准确
lsh = LSHash(hash_size=10, input_dim=max_dim)
for i in range(len(data)):
    lsh.index(data[i], extra_data=i)
start2 = time.time()
res = lsh.query(query)
min_index = -1
min_dist = sys.maxsize
can = []
for i in range(len(res)):
    can.append(res[i][0][1])
    if res[i][1] < min_dist:
        min_dist = res[i][1]
        min_index = res[i][0][1]
print(min_index)
end2 = time.time()

# LSHRandom库
lsh = LSHRandom(data, hash_len=max_dim, num_tables=10)
start3 = time.time()
search = lsh.knn_search(data, query, k=5, buckets=3)
print(search)
end3 = time.time()

# ground truth
start4 = time.time()
min_dist = sys.maxsize
min_index = -1
for i in range(max_len):
    dist = distance.euclidean(query, data[i])
    if dist < min_dist:
        min_dist = dist
        min_index = i
print(min_index)
end4 = time.time()

print(end1 - start1)
print(end2 - start2)
print(end3 - start3)
print(end4 - start4)