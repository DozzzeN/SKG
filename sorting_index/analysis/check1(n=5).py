import numpy as np

times = 100000000

matches = 0
cans = []
for ti in range(times):
    x1 = np.random.rand()
    x2 = np.random.rand()
    x3 = np.random.rand()
    x4 = np.random.rand()
    x5 = np.random.rand()
    y1 = np.random.rand()
    y2 = np.random.rand()
    y3 = np.random.rand()
    y4 = np.random.rand()
    y5 = np.random.rand()
    z11 = np.abs(x1 - y1)
    z12 = np.abs(x1 - y2)
    z13 = np.abs(x1 - y3)
    z14 = np.abs(x1 - y4)
    z15 = np.abs(x1 - y5)
    z21 = np.abs(x2 - y1)
    z22 = np.abs(x2 - y2)
    z23 = np.abs(x2 - y3)
    z24 = np.abs(x2 - y4)
    z25 = np.abs(x2 - y5)
    z31 = np.abs(x3 - y1)
    z32 = np.abs(x3 - y2)
    z33 = np.abs(x3 - y3)
    z34 = np.abs(x3 - y4)
    z35 = np.abs(x3 - y5)
    z41 = np.abs(x4 - y1)
    z42 = np.abs(x4 - y2)
    z43 = np.abs(x4 - y3)
    z44 = np.abs(x4 - y4)
    z45 = np.abs(x4 - y5)
    z51 = np.abs(x5 - y1)
    z52 = np.abs(x5 - y2)
    z53 = np.abs(x5 - y3)
    z54 = np.abs(x5 - y4)
    z55 = np.abs(x5 - y5)

    a = z11 < z21
    b = z11 < z31
    c = z11 < z41
    d = z11 < z51
    e = z22 < z12
    f = z22 < z32
    g = z22 < z42
    h = z22 < z52
    i = z33 < z13
    j = z33 < z23
    k = z33 < z43
    l = z33 < z53
    m = z44 < z14
    n = z44 < z24
    o = z44 < z34
    p = z44 < z54
    q = z55 < z15
    r = z55 < z25
    s = z55 < z35
    t = z55 < z45

    a1 = x1 > y1 and x2 > y1
    a2 = x1 < y1 and x2 < y1
    a3 = x1 > y1 and x2 < y1
    a4 = x1 < y1 and x2 > y1
    b1 = x1 > y1 and x3 > y1
    b2 = x1 < y1 and x3 < y1
    b3 = x1 > y1 and x3 < y1
    b4 = x1 < y1 and x3 > y1
    c1 = x1 > y1 and x4 > y1
    c2 = x1 < y1 and x4 < y1
    c3 = x1 > y1 and x4 < y1
    c4 = x1 < y1 and x4 > y1
    d1 = x1 > y1 and x5 > y1
    d2 = x1 < y1 and x5 < y1
    d3 = x1 > y1 and x5 < y1
    d4 = x1 < y1 and x5 > y1

    e1 = x2 > y2 and x1 > y2
    e2 = x2 < y2 and x1 < y2
    e3 = x2 > y2 and x1 < y2
    e4 = x2 < y2 and x1 > y2
    f1 = x2 > y2 and x3 > y2
    f2 = x2 < y2 and x3 < y2
    f3 = x2 > y2 and x3 < y2
    f4 = x2 < y2 and x3 > y2
    g1 = x2 > y2 and x4 > y2
    g2 = x2 < y2 and x4 < y2
    g3 = x2 > y2 and x4 < y2
    g4 = x2 < y2 and x4 > y2
    h1 = x2 > y2 and x5 > y2
    h2 = x2 < y2 and x5 < y2
    h3 = x2 > y2 and x5 < y2
    h4 = x2 < y2 and x5 > y2

    i1 = x3 > y3 and x1 > y3
    i2 = x3 < y3 and x1 < y3
    i3 = x3 > y3 and x1 < y3
    i4 = x3 < y3 and x1 > y3
    j1 = x3 > y3 and x2 > y3
    j2 = x3 < y3 and x2 < y3
    j3 = x3 > y3 and x2 < y3
    j4 = x3 < y3 and x2 > y3
    k1 = x3 > y3 and x4 > y3
    k2 = x3 < y3 and x4 < y3
    k3 = x3 > y3 and x4 < y3
    k4 = x3 < y3 and x4 > y3
    l1 = x3 > y3 and x5 > y3
    l2 = x3 < y3 and x5 < y3
    l3 = x3 > y3 and x5 < y3
    l4 = x3 < y3 and x5 > y3

    m1 = x4 > y4 and x1 > y4
    m2 = x4 < y4 and x1 < y4
    m3 = x4 > y4 and x1 < y4
    m4 = x4 < y4 and x1 > y4
    n1 = x4 > y4 and x2 > y4
    n2 = x4 < y4 and x2 < y4
    n3 = x4 > y4 and x2 < y4
    n4 = x4 < y4 and x2 > y4
    o1 = x4 > y4 and x3 > y4
    o2 = x4 < y4 and x3 < y4
    o3 = x4 > y4 and x3 < y4
    o4 = x4 < y4 and x3 > y4
    p1 = x4 > y4 and x5 > y4
    p2 = x4 < y4 and x5 < y4
    p3 = x4 > y4 and x5 < y4
    p4 = x4 < y4 and x5 > y4

    q1 = x5 > y5 and x1 > y5
    q2 = x5 < y5 and x1 < y5
    q3 = x5 > y5 and x1 < y5
    q4 = x5 < y5 and x1 > y5
    r1 = x5 > y5 and x2 > y5
    r2 = x5 < y5 and x2 < y5
    r3 = x5 > y5 and x2 < y5
    r4 = x5 < y5 and x2 > y5
    s1 = x5 > y5 and x3 > y5
    s2 = x5 < y5 and x3 < y5
    s3 = x5 > y5 and x3 < y5
    s4 = x5 < y5 and x3 > y5
    t1 = x5 > y5 and x4 > y5
    t2 = x5 < y5 and x4 < y5
    t3 = x5 > y5 and x4 < y5
    t4 = x5 < y5 and x4 > y5

    a1234 = [a1, a2, a3, a4]
    b1234 = [b1, b2, b3, b4]
    c1234 = [c1, c2, c3, c4]
    d1234 = [d1, d2, d3, d4]
    e1234 = [e1, e2, e3, e4]
    f1234 = [f1, f2, f3, f4]
    g1234 = [g1, g2, g3, g4]
    h1234 = [h1, h2, h3, h4]
    i1234 = [i1, i2, i3, i4]
    j1234 = [j1, j2, j3, j4]
    k1234 = [k1, k2, k3, k4]
    l1234 = [l1, l2, l3, l4]
    m1234 = [m1, m2, m3, m4]
    n1234 = [n1, n2, n3, n4]
    o1234 = [o1, o2, o3, o4]
    p1234 = [p1, p2, p3, p4]
    q1234 = [q1, q2, q3, q4]
    r1234 = [r1, r2, r3, r4]
    s1234 = [s1, s2, s3, s4]
    t1234 = [t1, t2, t3, t4]

    cans_tmp = []
    if a and b and c and d and e and f and g and h and i and j and k and l and m and n and o and p and q and r and s and t:
        for ii in range(len(a1234)):
            if a1234[ii]:
                cans_tmp.append(ii)
        for ii in range(len(b1234)):
            if b1234[ii]:
                cans_tmp.append(ii)
        for ii in range(len(c1234)):
            if c1234[ii]:
                cans_tmp.append(ii)
        for ii in range(len(d1234)):
            if d1234[ii]:
                cans_tmp.append(ii)
        for ii in range(len(e1234)):
            if e1234[ii]:
                cans_tmp.append(ii)
        for ii in range(len(f1234)):
            if f1234[ii]:
                cans_tmp.append(ii)
        for ii in range(len(g1234)):
            if g1234[ii]:
                cans_tmp.append(ii)
        for ii in range(len(h1234)):
            if h1234[ii]:
                cans_tmp.append(ii)
        for ii in range(len(i1234)):
            if i1234[ii]:
                cans_tmp.append(ii)
        for ii in range(len(j1234)):
            if j1234[ii]:
                cans_tmp.append(ii)
        for ii in range(len(k1234)):
            if k1234[ii]:
                cans_tmp.append(ii)
        for ii in range(len(l1234)):
            if l1234[ii]:
                cans_tmp.append(ii)
        for ii in range(len(m1234)):
            if m1234[ii]:
                cans_tmp.append(ii)
        for ii in range(len(n1234)):
            if n1234[ii]:
                cans_tmp.append(ii)
        for ii in range(len(o1234)):
            if o1234[ii]:
                cans_tmp.append(ii)
        for ii in range(len(p1234)):
            if p1234[ii]:
                cans_tmp.append(ii)
        for ii in range(len(q1234)):
            if q1234[ii]:
                cans_tmp.append(ii)
        for ii in range(len(r1234)):
            if r1234[ii]:
                cans_tmp.append(ii)
        for ii in range(len(s1234)):
            if s1234[ii]:
                cans_tmp.append(ii)
        for ii in range(len(t1234)):
            if t1234[ii]:
                cans_tmp.append(ii)

        matches += 1
        cans.append(cans_tmp)
print(matches / times)
cans = sorted(cans)
print(len(cans))
ii = 1
while ii < len(cans):
    if (np.array(cans[ii - 1]) == np.array(cans[ii])).all():
        del cans[ii]
    else:
        ii += 1
print(len(cans))
print(cans)
