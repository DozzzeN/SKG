import numpy as np
from dtaidistance import dtw_ndim

series1 = np.array([
    [0, 0],
    [0, 1],
    [2, 1],
    [0, 1],
    [0, 0]], dtype=np.double)
series2 = np.array([
    [0, 0],
    [2, 1],
    [0, 1],
    [0, .5],
    [0, 0]], dtype=np.double)
d = dtw_ndim.distance(series1, series2)

ndim = series1.shape[1]
dtw_i = 0
print(d)
for dim in range(ndim):
    dtw_dist = dtw_ndim.distance(series1[:, dim], series2[:, dim])
    dtw_i += dtw_dist
    print(dtw_dist)
