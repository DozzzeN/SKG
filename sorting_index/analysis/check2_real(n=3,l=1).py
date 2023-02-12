from collections import Counter

import numpy as np

times = 1000000

# a and b and c and d and e and f为真的所有情况
bools = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 3, 3], [0, 0, 0, 2, 0, 2], [0, 0, 0, 2, 3, 1], [0, 0, 3, 1, 0, 2],
         [0, 0, 3, 1, 3, 1], [0, 0, 3, 3, 0, 0], [0, 0, 3, 3, 3, 3], [0, 2, 0, 2, 1, 1], [0, 2, 0, 2, 2, 2],
         [0, 2, 3, 1, 1, 1], [0, 2, 3, 1, 2, 2], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 2, 2], [1, 1, 1, 3, 1, 3],
         [1, 1, 1, 3, 2, 0], [1, 1, 2, 0, 1, 3], [1, 1, 2, 0, 2, 0], [1, 1, 2, 2, 1, 1], [1, 1, 2, 2, 2, 2],
         [1, 3, 1, 3, 0, 0], [1, 3, 1, 3, 3, 3], [1, 3, 2, 0, 0, 0], [1, 3, 2, 0, 3, 3], [2, 0, 1, 3, 0, 0],
         [2, 0, 1, 3, 3, 3], [2, 0, 2, 0, 0, 0], [2, 0, 2, 0, 3, 3], [2, 2, 1, 1, 1, 1], [2, 2, 1, 1, 2, 2],
         [2, 2, 1, 3, 1, 3], [2, 2, 1, 3, 2, 0], [2, 2, 2, 0, 1, 3], [2, 2, 2, 0, 2, 0], [2, 2, 2, 2, 1, 1],
         [2, 2, 2, 2, 2, 2], [3, 1, 0, 2, 1, 1], [3, 1, 0, 2, 2, 2], [3, 1, 3, 1, 1, 1], [3, 1, 3, 1, 2, 2],
         [3, 3, 0, 0, 0, 0], [3, 3, 0, 0, 3, 3], [3, 3, 0, 2, 0, 2], [3, 3, 0, 2, 3, 1], [3, 3, 3, 1, 0, 2],
         [3, 3, 3, 1, 3, 1], [3, 3, 3, 3, 0, 0], [3, 3, 3, 3, 3, 3]]

prob = 0

a = 0
b = 10
ma = 0
mb = 100

x1 = []
x2 = []
x3 = []
y1 = []
y2 = []
y3 = []
m1 = []
m2 = []
m3 = []
z11 = []
z12 = []
z13 = []
z21 = []
z22 = []
z23 = []
z31 = []
z32 = []
z33 = []
_a = []
_b = []
_c = []
_d = []
_e = []
_f = []
a1 = []
a2 = []
a3 = []
a4 = []
a1234 = []

b1 = []
b2 = []
b3 = []
b4 = []
b1234 = []

c1 = []
c2 = []
c3 = []
c4 = []
c1234 = []

d1 = []
d2 = []
d3 = []
d4 = []
d1234 = []

e1 = []
e2 = []
e3 = []
e4 = []
e1234 = []

f1 = []
f2 = []
f3 = []
f4 = []
f1234 = []

for t in range(times):
    x1.append(np.random.uniform(a, b))
    x2.append(np.random.uniform(a, b))
    x3.append(np.random.uniform(a, b))
    y1.append(np.random.uniform(a, b))
    y2.append(np.random.uniform(a, b))
    y3.append(np.random.uniform(a, b))
    m1.append(np.random.uniform(ma, mb))
    m2.append(np.random.uniform(ma, mb))
    m3.append(np.random.uniform(ma, mb))
    z11.append(np.abs(x1[t] - y1[t]))
    z21.append(np.abs(m1[t] - m2[t]))
    z31.append(np.abs(m1[t] - m3[t]))
    z12.append(np.abs(m1[t] - m2[t]))
    z22.append(np.abs(x2[t] - y2[t]))
    z32.append(np.abs(m2[t] - m3[t]))
    z13.append(np.abs(m1[t] - m3[t]))
    z23.append(np.abs(m2[t] - m3[t]))
    z33.append(np.abs(x3[t] - y3[t]))

    _a.append(z11[t] < z21[t])
    _b.append(z11[t] < z31[t])
    _c.append(z22[t] < z12[t])
    _d.append(z22[t] < z32[t])
    _e.append(z33[t] < z13[t])
    _f.append(z33[t] < z23[t])

    a1.append(x1[t] > y1[t] and m1[t] > m2[t])
    a2.append(x1[t] < y1[t] and m1[t] < m2[t])
    a3.append(x1[t] > y1[t] and m1[t] < m2[t])
    a4.append(x1[t] < y1[t] and m1[t] > m2[t])
    a1234.append([a1[t], a2[t], a3[t], a4[t]])

    b1.append(x1[t] > y1[t] and m1[t] > m3[t])
    b2.append(x1[t] < y1[t] and m1[t] < m3[t])
    b3.append(x1[t] > y1[t] and m1[t] < m3[t])
    b4.append(x1[t] < y1[t] and m1[t] > m3[t])
    b1234.append([b1[t], b2[t], b3[t], b4[t]])

    c1.append(x2[t] > y2[t] and m1[t] > m2[t])
    c2.append(x2[t] < y2[t] and m1[t] < m2[t])
    c3.append(x2[t] > y2[t] and m1[t] < m2[t])
    c4.append(x2[t] < y2[t] and m1[t] > m2[t])
    c1234.append([c1[t], c2[t], c3[t], c4[t]])

    d1.append(x2[t] > y2[t] and m2[t] > m3[t])
    d2.append(x2[t] < y2[t] and m2[t] < m3[t])
    d3.append(x2[t] > y2[t] and m2[t] < m3[t])
    d4.append(x2[t] < y2[t] and m2[t] > m3[t])
    d1234.append([d1[t], d2[t], d3[t], d4[t]])

    e1.append(x3[t] > y3[t] and m1[t] > m3[t])
    e2.append(x3[t] < y3[t] and m1[t] < m3[t])
    e3.append(x3[t] > y3[t] and m1[t] < m3[t])
    e4.append(x3[t] < y3[t] and m1[t] > m3[t])
    e1234.append([e1[t], e2[t], e3[t], e4[t]])

    f1.append(x3[t] > y3[t] and m2[t] > m3[t])
    f2.append(x3[t] < y3[t] and m2[t] < m3[t])
    f3.append(x3[t] > y3[t] and m2[t] < m3[t])
    f4.append(x3[t] < y3[t] and m2[t] > m3[t])
    f1234.append([f1[t], f2[t], f3[t], f4[t]])

for i in range(len(bools)):
    matches = 0

    for t in range(times):
        flag = True
        if _a[t] and _b[t] and _c[t] and _d[t] and _e[t] and _f[t]:
            for j in range(len(bools[i])):
                if j == 0:
                    flag = flag and a1234[t][bools[i][j]]
                if j == 1:
                    flag = flag and b1234[t][bools[i][j]]
                if j == 2:
                    flag = flag and c1234[t][bools[i][j]]
                if j == 3:
                    flag = flag and d1234[t][bools[i][j]]
                if j == 4:
                    flag = flag and e1234[t][bools[i][j]]
                if j == 5:
                    flag = flag and f1234[t][bools[i][j]]
            if flag:
                matches += 1
    if matches != 0:
        print(i, bools[i], "%.8f" % (matches / times))
        prob += matches / times
print(prob)