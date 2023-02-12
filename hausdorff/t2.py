import sys

from shapely.geometry import Polygon

# 如果点在边上，这个点也不在凸包上
p = Polygon([[1, 1], [0, 0], [0, 1], [1, 0], [0, 1 / 2]])
q = Polygon([[1, 1], [0, 0], [0, 1], [1, 0], [0, 1 / 2]])
r = Polygon([[sys.maxsize, sys.maxsize], [sys.maxsize, sys.maxsize], [sys.maxsize, sys.maxsize]])
print(list(p.convex_hull.exterior.coords)[0:-1])
print(p.hausdorff_distance(q))
print(p.hausdorff_distance(r))
