import itertools

import numpy as np

times = 1000000

matches = 0

for t in range(times):
    x11 = np.random.rand()
    x21 = np.random.rand()
    x12 = np.random.rand()
    x22 = np.random.rand()
    y11 = np.random.rand()
    y21 = np.random.rand()
    y12 = np.random.rand()
    y22 = np.random.rand()

    # 0.00086470
    # if y11 < x11 and x11 < y21 and y21 < x21 and y12 < x12 and x12 < y22 and y22 < x22 \
    #         and y21 + y22 > (x21 + x22 + x11 + x12) / 2 and (x21 + x22 + x11 + x12) / 2 - y22 > x11:
    #     matches += 1

    # 0.00270940
    if x11 > y11 and x21 > y11 and x21 > y21 \
            and y12 < x12 and x12 < y22 and y22 < x22 \
            and x11 > y21 and x12 < y22 \
            and y22 > (x21 + x22 + x12 - x11) / 2 \
            and x11 + x12 < x21 + x22:
        matches += 1
print(matches / times)
