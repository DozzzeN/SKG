import numpy as np

from ddtw import dtw_metric_restraint, pathRegion

distance = lambda x, y: np.abs(x - y)

print("distance =", dtw_metric_restraint([6, 3, 1, 5], [1, 4, 6, 7], pathRegion(4, 2)))

# x = [1, 1, 3, 5]
# y = [1, 3, 5, 7]
# print(dtw(x, y, dist=distance)[0])
# print(dtw(x, y, dist=distance)[1])
# print(dtw(x, y, dist=distance)[2])
# print(dtw(x, y, dist=distance)[3])
#
# x = [1, 1]
# y = [1, 3]
# print(dtw(x, y, dist=distance)[0])
#
# x = [3, 5]
# y = [5, 7]
# print(dtw(x, y, dist=distance)[0])
# print()
# x = [3, 1, 3, 5]
# y = [1, 3, 5, 7]
# print(dtw(x, y, dist=distance)[0])
# print(dtw(x, y, dist=distance)[1])
# print(dtw(x, y, dist=distance)[2])
# print(dtw(x, y, dist=distance)[3])
#
# x = [3, 1]
# y = [1, 3]
# print(dtw(x, y, dist=distance)[0])
#
# x = [3, 5]
# y = [5, 7]
# print(dtw(x, y, dist=distance)[0])
