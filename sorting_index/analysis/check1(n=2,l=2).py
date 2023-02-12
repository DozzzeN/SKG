import numpy as np

matches = 0
times = 1000000

cans = []
for t in range(times):
    x11 = np.random.rand()
    x12 = np.random.rand()
    x21 = np.random.rand()
    x22 = np.random.rand()
    y11 = np.random.rand()
    y12 = np.random.rand()
    y21 = np.random.rand()
    y22 = np.random.rand()

    z11 = np.abs(x11 - y11) + np.abs(x12 - y12)
    z12 = np.abs(x11 - y21) + np.abs(x12 - y22)
    z21 = np.abs(x21 - y11) + np.abs(x22 - y12)
    z22 = np.abs(x21 - y21) + np.abs(x22 - y22)

    a = z11 < z21
    b = z22 < z12

    a111 = x11 > y11 and x12 > y12
    a112 = x11 > y11 and x12 < y12
    a113 = x11 < y11 and x12 > y12
    a114 = x11 < y11 and x12 < y12

    a211 = x11 > y21 and x12 > y22
    a212 = x11 > y21 and x12 < y22
    a213 = x11 < y21 and x12 > y22
    a214 = x11 < y21 and x12 < y22

    b121 = x21 > y11 and x22 > y12
    b122 = x21 > y11 and x22 < y12
    b123 = x21 < y11 and x22 > y12
    b124 = x21 < y11 and x22 < y12

    b221 = x21 > y21 and x22 > y22
    b222 = x21 > y21 and x22 < y22
    b223 = x21 < y21 and x22 > y22
    b224 = x21 < y21 and x22 < y22

    a11 = [a111, a112, a113, a114]
    a12 = [a211, a212, a213, a214]
    b21 = [b121, b122, b123, b124]
    b22 = [b221, b222, b223, b224]

    cans_tmp = []
    if a and b:
        for i in range(len(a11)):
            if a11[i]:
                cans_tmp.append(i)
        for i in range(len(a12)):
            if a12[i]:
                cans_tmp.append(i)
        for i in range(len(b21)):
            if b21[i]:
                cans_tmp.append(i)
        for i in range(len(b22)):
            if b22[i]:
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
