import itertools

import numpy as np

times = 100000

matches = 0
cans = []
for t in range(times):
    x1 = np.random.rand()
    x2 = np.random.rand()
    x3 = np.random.rand()
    y1 = np.random.rand()
    y2 = np.random.rand()
    y3 = np.random.rand()
    z11 = np.abs(x1 - y1)
    z12 = np.abs(x1 - y2)
    z13 = np.abs(x1 - y3)
    z21 = np.abs(x2 - y1)
    z22 = np.abs(x2 - y2)
    z23 = np.abs(x2 - y3)
    z31 = np.abs(x3 - y1)
    z32 = np.abs(x3 - y2)
    z33 = np.abs(x3 - y3)

    a = z11 < z21
    b = z11 < z31
    c = z22 < z12
    d = z22 < z32
    e = z33 < z13
    f = z33 < z23

    a1 = x1 > y1 and x2 > y1
    a2 = x1 < y1 and x2 < y1
    a3 = x1 > y1 and x2 < y1
    a4 = x1 < y1 and x2 > y1
    a1234 = [a1, a2, a3, a4]

    b1 = x1 > y1 and x3 > y1
    b2 = x1 < y1 and x3 < y1
    b3 = x1 > y1 and x3 < y1
    b4 = x1 < y1 and x3 > y1
    b1234 = [b1, b2, b3, b4]

    c1 = x2 > y2 and x1 > y2
    c2 = x2 < y2 and x1 < y2
    c3 = x2 > y2 and x1 < y2
    c4 = x2 < y2 and x1 > y2
    c1234 = [c1, c2, c3, c4]

    d1 = x2 > y2 and x3 > y2
    d2 = x2 < y2 and x3 < y2
    d3 = x2 > y2 and x3 < y2
    d4 = x2 < y2 and x3 > y2
    d1234 = [d1, d2, d3, d4]

    e1 = x3 > y3 and x1 > y3
    e2 = x3 < y3 and x1 < y3
    e3 = x3 > y3 and x1 < y3
    e4 = x3 < y3 and x1 > y3
    e1234 = [e1, e2, e3, e4]

    f1 = x3 > y3 and x2 > y3
    f2 = x3 < y3 and x2 < y3
    f3 = x3 > y3 and x2 < y3
    f4 = x3 < y3 and x2 > y3
    f1234 = [f1, f2, f3, f4]

    cans_tmp = []
    if a and b and c and d and e and f:
        for i in range(len(a1234)):
            if a1234[i]:
                cans_tmp.append(i)
        for i in range(len(b1234)):
            if b1234[i]:
                cans_tmp.append(i)
        for i in range(len(c1234)):
            if c1234[i]:
                cans_tmp.append(i)
        for i in range(len(d1234)):
            if d1234[i]:
                cans_tmp.append(i)
        for i in range(len(e1234)):
            if e1234[i]:
                cans_tmp.append(i)
        for i in range(len(f1234)):
            if f1234[i]:
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
