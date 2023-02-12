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
    y1 = np.random.uniform(a, b)
    y2 = np.random.uniform(a, b)
    m1 = np.random.uniform(ma, mb)
    m2 = np.random.uniform(ma, mb)

    z11 = np.abs(x1 - y1)
    z21 = np.abs(m1 - m2)
    z12 = np.abs(x2 - y2)
    z22 = np.abs(m1 - m2)

    _a = z11 < z21
    _b = z12 < z22

    a1 = x1 > y1 and m1 > m2
    a2 = x1 < y1 and m1 < m2
    a3 = x1 > y1 and m1 < m2
    a4 = x1 < y1 and m1 > m2
    a1234 = [a1, a2, a3, a4]

    b1 = x2 > y2 and m1 > m2
    b2 = x2 < y2 and m1 < m2
    b3 = x2 > y2 and m1 < m2
    b4 = x2 < y2 and m1 > m2
    b1234 = [b1, b2, b3, b4]

    cans_tmp = []
    if _a and _b:
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
