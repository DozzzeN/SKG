import math
import sys

from shapely.geometry import Polygon


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


p1 = [(1, 0), (0, 0), (0, 1)]
p2 = [(2, 0), (0.5, 0.5), (2, 1)]
print(standard_hd(p1, p2))
print(math.pow(hd(p1, p2), 2))
