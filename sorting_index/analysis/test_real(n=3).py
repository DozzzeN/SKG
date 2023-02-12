import itertools

import numpy as np

times = 100000

matches = 0
matches1 = 0
matches11 = 0
matches111 = 0
matches112 = 0
matches12 = 0
matches2 = 0
matches21 = 0
matches22 = 0
matches3 = 0
matches31 = 0
matches311 = 0
matches312 = 0
matches32 = 0
matches4 = 0
matches41 = 0
matches411 = 0
matches412 = 0
matches42 = 0
matches421 = 0
matches422 = 0
a = 0
b = 10
ma = 0
mb = 100

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

    if x1 > y1 and x2 > y2 and x3 > y3 and m1 > m2 and m1 > m3 and m2 > m3 \
            and x1 - y1 < m1 - m2 and x1 - y1 < m1 - m3 \
            and x2 - y2 < m1 - m2 and x2 - y2 < m2 - m3 \
            and x3 - y3 < m1 - m3 and x3 - y3 < m2 - m3:
        matches += 1

print("matches", matches / times, (matches1 + matches2 + matches3 + matches4) / times)
print("matches1", matches1 / times, (matches11 + matches12) / times)
print("matches11", matches11 / times)
print("matches111", matches111 / times)
print("matches112", matches112 / times)
print("matches12", matches12 / times)
print("matches2", matches2 / times, (matches21 + matches22) / times)
print("matches21", matches21 / times)
print("matches22", matches22 / times)
print("matches3", matches3 / times, (matches31 + matches32) / times)
print("matches31", matches31 / times)
print("matches311", matches311 / times)
print("matches312", matches312 / times)
print("matches32", matches32 / times)
print("matches4", matches4 / times, (matches41 + matches42) / times)
print("matches41", matches41 / times)
print("matches411", matches411 / times)
print("matches412", matches412 / times)
print("matches42", matches42 / times)
print("matches421", matches421 / times)
print("matches422", matches422 / times)
