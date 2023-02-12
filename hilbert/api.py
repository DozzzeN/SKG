import math

import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

p = 2
n = 2
hilbert_curve = HilbertCurve(p, n)
# distances = list(range(int(math.pow(2, n * p))))  # 2 ^ (np)
# points = hilbert_curve.points_from_distances(distances)
# for point, dist in zip(points, distances):
#     print(f'point(h={dist}) = {point}')

# num_points = 10_000
# points = np.random.randint(
#         low=0,
#         high=hilbert_curve.max_x + 1,
#         size=(num_points, hilbert_curve.n)
#     )
# distances = hilbert_curve.distances_from_points(points)
# for point, dist in zip(points, distances):
#     print(f'distance(x={point}) = {dist}')

points = [[0, 0], [1, 1], [1, 0]]
distances = hilbert_curve.distances_from_points(points, match_type=True)
for point, dist in zip(points, distances):
    print(f'distance(x={point}) = {dist}')
