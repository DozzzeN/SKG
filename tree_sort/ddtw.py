from __future__ import absolute_import, division

from collections import defaultdict

import numpy as np


def ddtw(x, y, dist=None):
    ''' return the distance between 2 time series without approximation
        Parameters
        ----------
        x : array_like
            input array 1
        y : array_like
            input array 2
        dist : function or int
            The method for calculating the distance between x[i] and y[j]. If
            dist is an int of value p > 0, then the p-norm will be used. If
            dist is a function then dist(x[i], y[j]) will be used. If dist is
            None then abs(x[i] - y[j]) will be used.
        Returns
        -------
        distance : float
            the approximate distance between the 2 time series
        path : list
            list of indexes for the inputs x and y

    '''
    return __dtw(x, y, None, dist)


def __dtw(x, y, window, dist):
    len_x, len_y = len(x), len(y)
    if window is None:
        window = [(i, j) for i in range(1, len_x - 1) for j in range(1, len_y - 1)]
    window = [(i + 1, j + 1) for i, j in window]
    D = defaultdict(lambda: (float('inf'),))
    D[1, 1] = (0, 0, 0)
    for i, j in window:
        dt = dist(x, y, i - 1, j - 1)
        D[i, j] = min((D[i - 1, j][0] + dt, i - 1, j), (D[i, j - 1][0] + dt, i, j - 1),
                      (D[i - 1, j - 1][0] + dt, i - 1, j - 1), key=lambda a: a[0])
    path = []
    i, j = len_x - 1, len_y - 1
    while not (i == j == 1):
        try:
            path.append((i - 1, j - 1))
            i, j = D[i, j][1], D[i, j][2]
        except IndexError:
            print("Getting IndexError here", D[i, j])
    path.reverse()
    return (D[len_x - 1, len_y - 1][0], path)


def derivative(x, index):
    if len(x) == 0:
        raise Exception("Incorrect input. Must be an array with more than 1 element.")
    elif index == len(x) - 1:
        print("problem")
        return 0
    return ((x[index] - x[index - 1]) + ((x[index + 1] - x[index - 1]) / 2)) / 2


def derivative_metric(x, y, x_index, y_index):
    if x_index == 0 or y_index == 0:
        print("problem")
    elif x_index == len(x) or y_index == len(y):
        print("problem")
    else:
        return (derivative(x, x_index) - derivative(y, y_index)) ** 2


def dtw_metric(s1, s2):
    dtw = {}
    for i in range(len(s1)):
        dtw[(i, -1)] = float('inf')
    for i in range(len(s2)):
        dtw[(-1, i)] = float('inf')
    dtw[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = abs(s1[i] - s2[j])
            dtw[(i, j)] = dist + min(dtw[(i - 1, j)], dtw[(i, j - 1)], dtw[i - 1, j - 1])

    return dtw[len(s1) - 1, len(s2) - 1]


def dtw_metric_restraint(s1, s2, region):
    dtw = {}
    for i in range(len(s1)):
        dtw[(-1, i)] = float('inf')
    for i in range(len(s2)):
        dtw[(i, -1)] = float('inf')
    dtw[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            if region[i][j] != 0:
                dist = abs(s1[i] - s2[j])
            else:
                dist = float('inf')
            dtw[(i, j)] = dist + min(dtw[(i - 1, j)], dtw[(i, j - 1)], dtw[i - 1, j - 1])

    i = len(s1) - 1
    j = len(s2) - 1

    path = []
    path.append([i, j])
    while i > 0 or j > 0:
        previous = min(dtw[(i - 1, j)], dtw[(i, j - 1)], dtw[(i - 1, j - 1)])
        if previous == dtw[(i - 1, j - 1)]:
            i -= 1
            j -= 1
        elif previous == dtw[(i, j - 1)]:
            j -= 1
        elif previous == dtw[(i - 1, j)]:
            i -= 1
        path.append([i, j])

    # return dtw[len(s1) - 1, len(s2) - 1], path
    return dtw[len(s1) - 1, len(s2) - 1]


def pathRegion(dimension, width=1):
    res = np.zeros(shape=(dimension, dimension), dtype=np.int8)
    for i in range(dimension):
        for j in range(dimension):
            # 路径可达位置
            if i == j:
                res[i][j] = 1
                for k in range(width + 1):
                    if j - k >= 0:
                        res[i][j - k] = 1
                    if j + k <= dimension - 1:
                        res[i][j + k] = 1
    return res


def dtw_metric_slope_weighting(s1, s2, X1=2, X2=2, X3=1):
    dtw = {}
    for i in range(len(s1)):
        dtw[(i, -1)] = float('inf')
        dtw[(i, -2)] = float('inf')
        dtw[(i, -3)] = float('inf')
    for i in range(len(s2)):
        dtw[(-1, i)] = float('inf')
        dtw[(-2, i)] = float('inf')
        dtw[(-3, i)] = float('inf')
    dtw[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = abs(s1[i] - s2[j])
            dtw[(i, j)] = dist + min(X1 * dtw[(i - 1, j)], X2 * dtw[(i, j - 1)], X3 * dtw[i - 1, j - 1])

    return dtw[len(s1) - 1, len(s2) - 1]


def dtw_metric_step_pattern(s1, s2):
    dtw = {}
    for i in range(len(s1)):
        dtw[(i, -1)] = float('inf')
        dtw[(i, -2)] = 0
    for i in range(len(s2)):
        dtw[(-1, i)] = float('inf')
        dtw[(-2, i)] = 0
    dtw[(-1, -1)] = 0
    dtw[(-1, -2)] = 0
    dtw[(-2, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = abs(s1[i] - s2[j])
            dtw[(i, j)] = dist + min(dtw[(i - 1, j - 2)], dtw[(i - 2, j - 1)], dtw[i - 1, j - 1])

    return dtw[len(s1) - 1, len(s2) - 1]


# s1 = [1, 2, 2, 3, 2, -1, 4, 6, 2, -5, 6, -3]
# s2 = [1, 2, 3, 2, 2, -1, 5, 4, 2, -6, 6, -3]
# s0 = [3, 4, 4, 5, 3, 4, 2, 5, 2, -2, -3, -3]
# print(dtw_metric(s1, s0))
# print(dtw_metric(s2, s0))
#
# print(dtw_metric_slope_weighting(s1, s0))
# print(dtw_metric_slope_weighting(s2, s0))
# print(dtw_metric_slope_weighting(s1, s0) - dtw_metric_slope_weighting(s2, s0))
#
# print(dtw_metric_step_pattern(s1, s0))
# print(dtw_metric_step_pattern(s2, s0))
# print(dtw_metric_step_pattern(s1, s0) - dtw_metric_step_pattern(s2, s0))

s1 = [2, 3, 1, 5]
s01 = [3, 6, 4, 7]

# s2 = [2.5, 3, 1.5]
# s02 = [3.5, 5.5, 4.5]
s2 = [2, 3.5, 1, 5]
s02 = [3, 6.5, 4, 7]

# s1 = np.array(s1) * 2
# s01 = np.array(s01) * 1
# s2 = np.array(s2) * 2
# s02 = np.array(s02) * 1

index = list(range(len(s2)))
combine = list(zip(s1, s2, index))
np.random.shuffle(combine)
s1, s2, index = zip(*combine)
print(s1)
print(s2)
print(index)

print(dtw_metric(s1, s01))
print(dtw_metric(s2, s02))
print(dtw_metric(s1, s01) - dtw_metric(s2, s02))

# print(dtw_metric_slope_weighting(s1, s01))
# print(dtw_metric_slope_weighting(s2, s02))
print(dtw_metric_slope_weighting(s1, s01) - dtw_metric_slope_weighting(s2, s02))

# print(dtw_metric_step_pattern(s1, s01))
# print(dtw_metric_step_pattern(s2, s02))
print(dtw_metric_step_pattern(s1, s01) - dtw_metric_step_pattern(s2, s02))
