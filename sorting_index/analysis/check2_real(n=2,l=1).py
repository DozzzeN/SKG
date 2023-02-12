import itertools

import numpy as np

times = 1000000

# a and b为真的所有情况
bools = [[0, 0], [0, 3], [1, 1], [1, 2], [2, 1], [2, 2], [3, 0], [3, 3]]
prob = 0
for i in range(len(bools)):
    matches = 0
    a = 0
    b = 10
    ma = 0
    mb = 100

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

        flag = True
        if _a and _b:
            for j in range(len(bools[i])):
                if j == 0:
                    flag = flag and a1234[bools[i][j]]
                if j == 1:
                    flag = flag and b1234[bools[i][j]]
            if flag:
                matches += 1
    if matches != 0:
        print(i, bools[i], matches / times)
        prob += matches / times
print(prob)
