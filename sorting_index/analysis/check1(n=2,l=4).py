import numpy as np

# episode length
l = 3
# number of measurements
n = l * 2
a = 0
b = 1

matches = 0
times = 200000000
# 34989个

cans = []
for t in range(times):
    x11 = np.random.rand()
    x12 = np.random.rand()
    x13 = np.random.rand()
    x14 = np.random.rand()
    x21 = np.random.rand()
    x22 = np.random.rand()
    x23 = np.random.rand()
    x24 = np.random.rand()
    y11 = np.random.rand()
    y12 = np.random.rand()
    y13 = np.random.rand()
    y14 = np.random.rand()
    y21 = np.random.rand()
    y22 = np.random.rand()
    y23 = np.random.rand()
    y24 = np.random.rand()

    z11 = np.abs(x11 - y11) + np.abs(x12 - y12) + np.abs(x13 - y13) + np.abs(x14 - y14)
    z12 = np.abs(x11 - y21) + np.abs(x12 - y22) + np.abs(x13 - y23) + np.abs(x14 - y24)
    z21 = np.abs(x21 - y11) + np.abs(x22 - y12) + np.abs(x23 - y13) + np.abs(x24 - y14)
    z22 = np.abs(x21 - y21) + np.abs(x22 - y22) + np.abs(x23 - y23) + np.abs(x24 - y24)

    a = z11 < z21
    b = z22 < z12

    a1101 = x11 > y11 and x12 > y12 and x13 > y13 and x14 > y14
    a1102 = x11 > y11 and x12 > y12 and x13 > y13 and x14 < y14
    a1103 = x11 > y11 and x12 > y12 and x13 < y13 and x14 > y14
    a1104 = x11 > y11 and x12 > y12 and x13 < y13 and x14 < y14
    a1105 = x11 > y11 and x12 < y12 and x13 > y13 and x14 > y14
    a1106 = x11 > y11 and x12 < y12 and x13 > y13 and x14 < y14
    a1107 = x11 > y11 and x12 < y12 and x13 < y13 and x14 > y14
    a1108 = x11 > y11 and x12 < y12 and x13 < y13 and x14 < y14
    a1109 = x11 < y11 and x12 > y12 and x13 > y13 and x14 > y14
    a1110 = x11 < y11 and x12 > y12 and x13 > y13 and x14 < y14
    a1111 = x11 < y11 and x12 > y12 and x13 < y13 and x14 > y14
    a1112 = x11 < y11 and x12 > y12 and x13 < y13 and x14 < y14
    a1113 = x11 < y11 and x12 < y12 and x13 > y13 and x14 > y14
    a1114 = x11 < y11 and x12 < y12 and x13 > y13 and x14 < y14
    a1115 = x11 < y11 and x12 < y12 and x13 < y13 and x14 > y14
    a1116 = x11 < y11 and x12 < y12 and x13 < y13 and x14 < y14

    a2101 = x11 > y21 and x12 > y22 and x13 > y23 and x14 > y24
    a2102 = x11 > y21 and x12 > y22 and x13 > y23 and x14 < y24
    a2103 = x11 > y21 and x12 > y22 and x13 < y23 and x14 > y24
    a2104 = x11 > y21 and x12 > y22 and x13 < y23 and x14 < y24
    a2105 = x11 > y21 and x12 < y22 and x13 > y23 and x14 > y24
    a2106 = x11 > y21 and x12 < y22 and x13 > y23 and x14 < y24
    a2107 = x11 > y21 and x12 < y22 and x13 < y23 and x14 > y24
    a2108 = x11 > y21 and x12 < y22 and x13 < y23 and x14 < y24
    a2109 = x11 < y21 and x12 > y22 and x13 > y23 and x14 > y24
    a2110 = x11 < y21 and x12 > y22 and x13 > y23 and x14 < y24
    a2111 = x11 < y21 and x12 > y22 and x13 < y23 and x14 > y24
    a2112 = x11 < y21 and x12 > y22 and x13 < y23 and x14 < y24
    a2113 = x11 < y21 and x12 < y22 and x13 > y23 and x14 > y24
    a2114 = x11 < y21 and x12 < y22 and x13 > y23 and x14 < y24
    a2115 = x11 < y21 and x12 < y22 and x13 < y23 and x14 > y24
    a2116 = x11 < y21 and x12 < y22 and x13 < y23 and x14 < y24

    b1201 = x21 > y11 and x22 > y12 and x23 > y13 and x24 > y14
    b1202 = x21 > y11 and x22 > y12 and x23 > y13 and x24 < y14
    b1203 = x21 > y11 and x22 > y12 and x23 < y13 and x24 > y14
    b1204 = x21 > y11 and x22 > y12 and x23 < y13 and x24 < y14
    b1205 = x21 > y11 and x22 < y12 and x23 > y13 and x24 > y14
    b1206 = x21 > y11 and x22 < y12 and x23 > y13 and x24 < y14
    b1207 = x21 > y11 and x22 < y12 and x23 < y13 and x24 > y14
    b1208 = x21 > y11 and x22 < y12 and x23 < y13 and x24 < y14
    b1209 = x21 < y11 and x22 > y12 and x23 > y13 and x24 > y14
    b1210 = x21 < y11 and x22 > y12 and x23 > y13 and x24 < y14
    b1211 = x21 < y11 and x22 > y12 and x23 < y13 and x24 > y14
    b1212 = x21 < y11 and x22 > y12 and x23 < y13 and x24 < y14
    b1213 = x21 < y11 and x22 < y12 and x23 > y13 and x24 > y14
    b1214 = x21 < y11 and x22 < y12 and x23 > y13 and x24 < y14
    b1215 = x21 < y11 and x22 < y12 and x23 < y13 and x24 > y14
    b1216 = x21 < y11 and x22 < y12 and x23 < y13 and x24 < y14

    b2201 = x21 > y21 and x22 > y22 and x23 > y23 and x24 > y24
    b2202 = x21 > y21 and x22 > y22 and x23 > y23 and x24 < y24
    b2203 = x21 > y21 and x22 > y22 and x23 < y23 and x24 > y24
    b2204 = x21 > y21 and x22 > y22 and x23 < y23 and x24 < y24
    b2205 = x21 > y21 and x22 < y22 and x23 > y23 and x24 > y24
    b2206 = x21 > y21 and x22 < y22 and x23 > y23 and x24 < y24
    b2207 = x21 > y21 and x22 < y22 and x23 < y23 and x24 > y24
    b2208 = x21 > y21 and x22 < y22 and x23 < y23 and x24 < y24
    b2209 = x21 < y21 and x22 > y22 and x23 > y23 and x24 > y24
    b2210 = x21 < y21 and x22 > y22 and x23 > y23 and x24 < y24
    b2211 = x21 < y21 and x22 > y22 and x23 < y23 and x24 > y24
    b2212 = x21 < y21 and x22 > y22 and x23 < y23 and x24 < y24
    b2213 = x21 < y21 and x22 < y22 and x23 > y23 and x24 > y24
    b2214 = x21 < y21 and x22 < y22 and x23 > y23 and x24 < y24
    b2215 = x21 < y21 and x22 < y22 and x23 < y23 and x24 > y24
    b2216 = x21 < y21 and x22 < y22 and x23 < y23 and x24 < y24

    a11 = [a1101, a1102, a1103, a1104, a1105, a1106, a1107, a1108, a1109, a1110, a1111, a1112, a1113, a1114, a1115, a1116]
    a12 = [a2101, a2102, a2103, a2104, a2105, a2106, a2107, a2108, a2109, a2110, a2111, a2112, a2113, a2114, a2115, a2116]
    b21 = [b1201, b1202, b1203, b1204, b1205, b1206, b1207, b1208, b1209, b1210, b1211, b1212, b1213, b1214, b1215, b1216]
    b22 = [b2201, b2202, b2203, b2204, b2205, b2206, b2207, b2208, b2209, b2210, b2211, b2212, b2213, b2214, b2215, b2216]

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
