import itertools

import numpy as np

times = 100000

matches1 = 0
matches2 = 0
for t in range(times):
    x1 = np.random.normal()
    x2 = np.random.normal()
    y1 = np.random.normal()
    y2 = np.random.normal()
    z11 = np.abs(x1 - y1)
    z12 = np.abs(x1 - y2)
    z21 = np.abs(x2 - y1)
    z22 = np.abs(x2 - y2)

    a = z11 < z21
    b = z22 < z12

    a1 = x1 > y1 and x2 > y1
    a2 = x1 < y1 and x2 < y1
    a3 = x1 > y1 and x2 < y1
    a4 = x1 < y1 and x2 > y1
    a1234 = [a1, a2, a3, a4]

    b1 = x2 > y2 and x1 > y2
    b2 = x2 < y2 and x1 < y2
    b3 = x2 > y2 and x1 < y2
    b4 = x2 < y2 and x1 > y2
    b1234 = [b1, b2, b3, b4]

    cans_tmp = []
    if a and b:
        if a1 and b2:
            matches1 += 1
        if a1 and b3:
            matches2 += 1
print(matches1 / times)
print(matches2 / times)
# 0 [0, 1] 0.04218
# 1 [0, 2] 0.0176
# 2 [1, 0] 0.04098
# 3 [1, 3] 0.01816
# 4 [2, 0] 0.01799
# 5 [2, 3] 0.01881
# 6 [3, 1] 0.01796
# 7 [3, 2] 0.01856
