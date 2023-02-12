import itertools

import numpy as np

times = 100000

matches = 0
cans = []
for t in range(times):
    n1 = np.random.rand()
    n2 = np.random.rand()
    x1 = np.random.rand() + n1
    x2 = np.random.rand() + n2
    y1 = np.random.rand() + n1
    y2 = np.random.rand() + n2
    z11 = np.abs(x1 - y1)
    z12 = np.abs(x1 - y2)
    z21 = np.abs(x2 - y1)
    z22 = np.abs(x2 - y2)

    a = z11 < z21
    b = z22 < z12

    c = z11 > z21
    d = z22 < z12

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
    if c and d:
        for i in range(len(a1234)):
            if a1234[i]:
                cans_tmp.append(i)
        for i in range(len(b1234)):
            if b1234[i]:
                cans_tmp.append(i)
        matches += 1
        cans.append(cans_tmp)
print(matches / times)
cans = sorted(cans)
print(len(cans))
i = 1
while i < len(cans):
    if (np.array(cans[i - 1]) == np.array(cans[i])).all():
        del cans[i]
    else:
        i += 1
print(len(cans))
print(cans)
