import numpy as np

matches = 0
times = 100000
a = 0
b = 10
ma = 0
mb = 100

cans = []
for t in range(times):
    x1 = np.random.uniform(a, b)
    x2 = np.random.uniform(a, b)
    x3 = np.random.uniform(a, b)
    y1 = np.random.uniform(a, b)
    y2 = np.random.uniform(a, b)
    y3 = np.random.uniform(a, b)
    m1 = np.random.uniform(ma, mb)
    m2 = np.random.uniform(ma, mb)
    m3 = np.random.uniform(ma, mb)

    z11 = np.abs(x1 - y1)
    z21 = np.abs(m1 - m2)
    z31 = np.abs(m1 - m3)
    z12 = np.abs(m1 - m2)
    z22 = np.abs(x2 - y2)
    z32 = np.abs(m2 - m3)
    z13 = np.abs(m1 - m3)
    z23 = np.abs(m2 - m3)
    z33 = np.abs(x3 - y3)

    _a = z11 < z21
    _b = z11 < z31
    _c = z22 < z12
    _d = z22 < z32
    _e = z33 < z13
    _f = z33 < z23

    a1 = x1 > y1 and m1 > m2
    a2 = x1 < y1 and m1 < m2
    a3 = x1 > y1 and m1 < m2
    a4 = x1 < y1 and m1 > m2
    a1234 = [a1, a2, a3, a4]

    b1 = x1 > y1 and m1 > m3
    b2 = x1 < y1 and m1 < m3
    b3 = x1 > y1 and m1 < m3
    b4 = x1 < y1 and m1 > m3
    b1234 = [b1, b2, b3, b4]

    c1 = x2 > y2 and m1 > m2
    c2 = x2 < y2 and m1 < m2
    c3 = x2 > y2 and m1 < m2
    c4 = x2 < y2 and m1 > m2
    c1234 = [c1, c2, c3, c4]

    d1 = x2 > y2 and m2 > m3
    d2 = x2 < y2 and m2 < m3
    d3 = x2 > y2 and m2 < m3
    d4 = x2 < y2 and m2 > m3
    d1234 = [d1, d2, d3, d4]

    e1 = x3 > y3 and m1 > m3
    e2 = x3 < y3 and m1 < m3
    e3 = x3 > y3 and m1 < m3
    e4 = x3 < y3 and m1 > m3
    e1234 = [e1, e2, e3, e4]

    f1 = x3 > y3 and m2 > m3
    f2 = x3 < y3 and m2 < m3
    f3 = x3 > y3 and m2 < m3
    f4 = x3 < y3 and m2 > m3
    f1234 = [f1, f2, f3, f4]

    cans_tmp = []
    if _a and _b and _c and _d and _e and _f:
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
cans_dict = {}
for i in range(len(cans)):
    s = "".join(str(j) for j in cans[i])
    if cans_dict.get(s) is None:
        cans_dict[s] = 1
    else:
        cans_dict[s] += 1
key = list(cans_dict.keys())
cans = []
for i in range(len(key)):
    tmp = []
    for j in range(len(key[i])):
        tmp.append(int(key[i][j]))
    cans.append(tmp)
print(cans)
print(len(cans))
