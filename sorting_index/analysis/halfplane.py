# import itertools
#
# import numpy as np
#
# times = 1000000
#
# # a and b为真的所有情况
# matches = 0
# for t in range(times):
#     x1 = np.random.normal()
#     x2 = np.random.normal()
#     y1 = np.random.normal()
#     y2 = np.random.normal()
#
#     if (x1 + x2) / 2 < y2 and y2 < x2 and y1 < x1:
#         matches += 1
# print(matches / times)


import math

import scipy.integrate

inf = 1000


def func(x2, y2, x1, y1):
    return math.exp(-1 / 2 * (x2 ** 2 + y2 ** 2 + x1 ** 2 + y1 ** 2)) / (4 * math.pi ** 2)


def lim1(y2, x1):
    return [y2, 2 * y2 - x1]


def lim2(x1):
    return [x1, inf]


def lim3(y1):
    return [y1, inf]


i = scipy.integrate.nquad(func, [lim1, lim2, lim3, [-inf, inf]])
print(i)
