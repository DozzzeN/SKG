import math
import sys
import time

import numpy as np
from shapely.geometry import Polygon
from scipy.spatial.distance import directed_hausdorff


def standard_hd(x, y):
    h1 = 0
    for xi in x:
        shortest = sys.maxsize
        for yi in y:
            d = round(math.pow(xi[0] - yi[0], 2) + math.pow(xi[1] - yi[1], 2), 10)
            if d < shortest:
                shortest = d
        if shortest > h1:
            h1 = shortest

    h2 = 0
    for xi in y:
        shortest = sys.maxsize
        for yi in x:
            d = round(math.pow(xi[0] - yi[0], 2) + math.pow(xi[1] - yi[1], 2), 10)
            if d < shortest:
                shortest = d
        if shortest > h2:
            h2 = shortest
    return max(h1, h2)


def hd(x, y):
    h = Polygon(x).hausdorff_distance(Polygon(y))
    return h


# p1 = [(1, 0), (0, 0), (0, 1)]
# p2 = [(2, 0), (0.5, 0.5), (2, 1)]

p1 = np.random.normal(0, 1, (100, 2))
p2 = np.random.normal(0, 1, (100, 2))
# start_time = time.time()
# for i in range(1000):
#     standard_hd(p1, p2)
# end_time = time.time()
# print(end_time - start_time)
#
# start_time = time.time()
# for i in range(1000):
#     hd(p1, p2)
# end_time = time.time()
# print(end_time - start_time)
#
# start_time = time.time()
# for i in range(1000):
#     max(math.pow(directed_hausdorff(p1, p2)[0], 2), math.pow(directed_hausdorff(p2, p1)[0], 2))
# end_time = time.time()
# print(end_time - start_time)

# O(n^2)
print(standard_hd(p1, p2))
# O(n)
print(math.pow(hd(p1, p2), 2))
# O(n)
print(max(math.pow(directed_hausdorff(p1, p2)[0], 2), math.pow(directed_hausdorff(p2, p1)[0], 2)))
