from collections import Counter

import numpy as np

times = 1000000

# bools = list(itertools.product([0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]))

# a and b and c and d and e and f为真的所有情况
bools = [[0, 0, 1, 1, 1, 3], [0, 0, 1, 1, 2, 0], [0, 0, 1, 3, 1, 1], [0, 0, 1, 3, 2, 2], [0, 0, 2, 0, 1, 1],
         [0, 0, 2, 0, 2, 2], [0, 0, 2, 2, 1, 3], [0, 0, 2, 2, 2, 0], [0, 2, 1, 1, 0, 0], [0, 2, 1, 1, 3, 3],
         [0, 2, 2, 2, 0, 0], [0, 2, 2, 2, 3, 3], [1, 1, 0, 0, 0, 2], [1, 1, 0, 0, 3, 1], [1, 1, 0, 2, 0, 0],
         [1, 1, 0, 2, 3, 3], [1, 1, 3, 1, 0, 0], [1, 1, 3, 1, 3, 3], [1, 1, 3, 3, 0, 2], [1, 1, 3, 3, 3, 1],
         [1, 3, 0, 0, 1, 1], [1, 3, 0, 0, 2, 2], [1, 3, 3, 3, 1, 1], [1, 3, 3, 3, 2, 2], [2, 0, 0, 0, 1, 1],
         [2, 0, 0, 0, 2, 2], [2, 0, 3, 3, 1, 1], [2, 0, 3, 3, 2, 2], [2, 2, 0, 0, 0, 2], [2, 2, 0, 0, 3, 1],
         [2, 2, 0, 2, 0, 0], [2, 2, 0, 2, 3, 3], [2, 2, 3, 1, 0, 0], [2, 2, 3, 1, 3, 3], [2, 2, 3, 3, 0, 2],
         [2, 2, 3, 3, 3, 1], [3, 1, 1, 1, 0, 0], [3, 1, 1, 1, 3, 3], [3, 1, 2, 2, 0, 0], [3, 1, 2, 2, 3, 3],
         [3, 3, 1, 1, 1, 3], [3, 3, 1, 1, 2, 0], [3, 3, 1, 3, 1, 1], [3, 3, 1, 3, 2, 2], [3, 3, 2, 0, 1, 1],
         [3, 3, 2, 0, 2, 2], [3, 3, 2, 2, 1, 3], [3, 3, 2, 2, 2, 0]]
prob = 0

x1 = []
x2 = []
x3 = []
y1 = []
y2 = []
y3 = []
z11 = []
z12 = []
z13 = []
z21 = []
z22 = []
z23 = []
z31 = []
z32 = []
z33 = []
a = []
b = []
c = []
d = []
e = []
f = []
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
    x1.append(np.random.rand())
    x2.append(np.random.rand())
    x3.append(np.random.rand())
    y1.append(np.random.rand())
    y2.append(np.random.rand())
    y3.append(np.random.rand())
    z11.append(np.abs(x1[t] - y1[t]))
    z12.append(np.abs(x1[t] - y2[t]))
    z13.append(np.abs(x1[t] - y3[t]))
    z21.append(np.abs(x2[t] - y1[t]))
    z22.append(np.abs(x2[t] - y2[t]))
    z23.append(np.abs(x2[t] - y3[t]))
    z31.append(np.abs(x3[t] - y1[t]))
    z32.append(np.abs(x3[t] - y2[t]))
    z33.append(np.abs(x3[t] - y3[t]))

    a.append(z11[t] < z21[t])
    b.append(z11[t] < z31[t])
    c.append(z22[t] < z12[t])
    d.append(z22[t] < z32[t])
    e.append(z33[t] < z13[t])
    f.append(z33[t] < z23[t])

    a1.append(x1[t] > y1[t] and x2[t] > y1[t])
    a2.append(x1[t] < y1[t] and x2[t] < y1[t])
    a3.append(x1[t] > y1[t] and x2[t] < y1[t])
    a4.append(x1[t] < y1[t] and x2[t] > y1[t])
    a1234.append([a1[t], a2[t], a3[t], a4[t]])

    b1.append(x1[t] > y1[t] and x3[t] > y1[t])
    b2.append(x1[t] < y1[t] and x3[t] < y1[t])
    b3.append(x1[t] > y1[t] and x3[t] < y1[t])
    b4.append(x1[t] < y1[t] and x3[t] > y1[t])
    b1234.append([b1[t], b2[t], b3[t], b4[t]])

    c1.append(x2[t] > y2[t] and x1[t] > y2[t])
    c2.append(x2[t] < y2[t] and x1[t] < y2[t])
    c3.append(x2[t] > y2[t] and x1[t] < y2[t])
    c4.append(x2[t] < y2[t] and x1[t] > y2[t])
    c1234.append([c1[t], c2[t], c3[t], c4[t]])

    d1.append(x2[t] > y2[t] and x3[t] > y2[t])
    d2.append(x2[t] < y2[t] and x3[t] < y2[t])
    d3.append(x2[t] > y2[t] and x3[t] < y2[t])
    d4.append(x2[t] < y2[t] and x3[t] > y2[t])
    d1234.append([d1[t], d2[t], d3[t], d4[t]])

    e1.append(x3[t] > y3[t] and x1[t] > y3[t])
    e2.append(x3[t] < y3[t] and x1[t] < y3[t])
    e3.append(x3[t] > y3[t] and x1[t] < y3[t])
    e4.append(x3[t] < y3[t] and x1[t] > y3[t])
    e1234.append([e1[t], e2[t], e3[t], e4[t]])

    f1.append(x3[t] > y3[t] and x2[t] > y3[t])
    f2.append(x3[t] < y3[t] and x2[t] < y3[t])
    f3.append(x3[t] > y3[t] and x2[t] < y3[t])
    f4.append(x3[t] < y3[t] and x2[t] > y3[t])
    f1234.append([f1[t], f2[t], f3[t], f4[t]])

for i in range(len(bools)):
    matches = 0

    for t in range(times):
        flag = True
        if a[t] and b[t] and c[t] and d[t] and e[t] and f[t]:
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
        print(i, bools[i], "%.8f" % (matches / times), "%.8f" % (1 / 1440), "%.8f" % (1 / 2880))
        prob += matches / times
print(prob * 6)

# 统计每位之和，每位元素出现次数
w = np.zeros(len(bools[0]))
for i in range(len(bools)):
    for j in range(len(bools[i])):
        w[j] += bools[i][j]
print(w)
print(len(bools))
ws = []
for i in range(len(bools[0])):
    ws.append([])
for i in range(len(bools)):
    for j in range(len(bools[i])):
        ws[j].append(bools[i][j])
for i in range(len(ws)):
    print(Counter(ws[i]))
