import itertools

import numpy as np

times = 10000000

matches = 0
cans = []
for t in range(times):
    x1 = np.random.rand()
    x2 = np.random.rand()
    x3 = np.random.rand()
    x4 = np.random.rand()
    y1 = np.random.rand()
    y2 = np.random.rand()
    y3 = np.random.rand()
    y4 = np.random.rand()
    z11 = np.abs(x1 - y1)
    z12 = np.abs(x1 - y2)
    z13 = np.abs(x1 - y3)
    z14 = np.abs(x1 - y4)
    z21 = np.abs(x2 - y1)
    z22 = np.abs(x2 - y2)
    z23 = np.abs(x2 - y3)
    z24 = np.abs(x2 - y4)
    z31 = np.abs(x3 - y1)
    z32 = np.abs(x3 - y2)
    z33 = np.abs(x3 - y3)
    z34 = np.abs(x3 - y4)
    z41 = np.abs(x4 - y1)
    z42 = np.abs(x4 - y2)
    z43 = np.abs(x4 - y3)
    z44 = np.abs(x4 - y4)

    a = z11 < z21
    b = z11 < z31
    c = z11 < z41

    d = z22 < z12
    e = z22 < z32
    f = z22 < z42

    g = z33 < z13
    h = z33 < z23
    i = z33 < z43

    j = z44 < z14
    k = z44 < z24
    l = z44 < z34

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

    c1 = x1 > y1 and x4 > y1
    c2 = x1 < y1 and x4 < y1
    c3 = x1 > y1 and x4 < y1
    c4 = x1 < y1 and x4 > y1
    c1234 = [c1, c2, c3, c4]

    d1 = x2 > y2 and x1 > y2
    d2 = x2 < y2 and x1 < y2
    d3 = x2 > y2 and x1 < y2
    d4 = x2 < y2 and x1 > y2
    d1234 = [d1, d2, d3, d4]

    e1 = x2 > y2 and x3 > y2
    e2 = x2 < y2 and x3 < y2
    e3 = x2 > y2 and x3 < y2
    e4 = x2 < y2 and x3 > y2
    e1234 = [e1, e2, e3, e4]

    f1 = x2 > y2 and x4 > y2
    f2 = x2 < y2 and x4 < y2
    f3 = x2 > y2 and x4 < y2
    f4 = x2 < y2 and x4 > y2
    f1234 = [f1, f2, f3, f4]

    g1 = x3 > y3 and x1 > y3
    g2 = x3 < y3 and x1 < y3
    g3 = x3 > y3 and x1 < y3
    g4 = x3 < y3 and x1 > y3
    g1234 = [g1, g2, g3, g4]

    h1 = x3 > y3 and x2 > y3
    h2 = x3 < y3 and x2 < y3
    h3 = x3 > y3 and x2 < y3
    h4 = x3 < y3 and x2 > y3
    h1234 = [h1, h2, h3, h4]

    i1 = x3 > y3 and x4 > y3
    i2 = x3 < y3 and x4 < y3
    i3 = x3 > y3 and x4 < y3
    i4 = x3 < y3 and x4 > y3
    i1234 = [i1, i2, i3, i4]

    j1 = x4 > y4 and x1 > y4
    j2 = x4 < y4 and x1 < y4
    j3 = x4 > y4 and x1 < y4
    j4 = x4 < y4 and x1 > y4
    j1234 = [j1, j2, j3, j4]

    k1 = x4 > y4 and x2 > y4
    k2 = x4 < y4 and x2 < y4
    k3 = x4 > y4 and x2 < y4
    k4 = x4 < y4 and x2 > y4
    k1234 = [k1, k2, k3, k4]

    l1 = x4 > y4 and x3 > y4
    l2 = x4 < y4 and x3 < y4
    l3 = x4 > y4 and x3 < y4
    l4 = x4 < y4 and x3 > y4
    l1234 = [l1, l2, l3, l4]

    cans_tmp = []
    if a and b and c and d and e and f and g and h and i and j and k and l:
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
        for i in range(len(g1234)):
            if g1234[i]:
                cans_tmp.append(i)
        for i in range(len(h1234)):
            if h1234[i]:
                cans_tmp.append(i)
        for i in range(len(i1234)):
            if i1234[i]:
                cans_tmp.append(i)
        for i in range(len(j1234)):
            if j1234[i]:
                cans_tmp.append(i)
        for i in range(len(k1234)):
            if k1234[i]:
                cans_tmp.append(i)
        for i in range(len(l1234)):
            if l1234[i]:
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
