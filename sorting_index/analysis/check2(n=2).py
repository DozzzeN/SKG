import itertools

import numpy as np

times = 100000

# a and b为真的所有情况
# bools = [[0, 1], [0, 2], [1, 0], [1, 3], [2, 0], [2, 3], [3, 1], [3, 2]]

# c and d为真的所有情况
bools = [[0, 1], [0, 3], [1, 0], [1, 2], [2, 1], [2, 3], [3, 0], [3, 2]]
prob = 0

for i in range(len(bools)):
    matches = 0
    for t in range(times):
        # n1 = np.random.rand()
        # n2 = np.random.rand()
        # x1 = np.random.rand() + n1
        # x2 = np.random.rand() + n2
        # y1 = np.random.rand() + n1
        # y2 = np.random.rand() + n2
        x1 = np.random.normal()
        x2 = np.random.normal()
        y1 = np.random.normal()
        y2 = np.random.normal()
        z21 = np.abs(x2 - y1)
        z11 = np.abs(x1 - y1)
        z12 = np.abs(x1 - y2)
        z22 = np.abs(x2 - y2)

        a = z11 < z21
        b = z22 < z12

        c = z11 > z21
        d = z22 > z12

        a1 = x1 > y1 and x2 > y1
        a2 = x1 < y1 and x2 < y1
        a3 = x1 > y1 and x2 < y1
        a4 = x1 < y1 and x2 > y1
        a1234 = [a1, a2, a3, a4]

        b1 = x2 > y2 and x1 > y2
        b2 = x2 < y2 and x1 < y2
        b3 = x2 > y2 and x1 < y2
        b4 = x2 < y2 and x1 > y2
        b1234 = [b1, b2, b3, b4]

        flag = True
        if c and d:
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
print(prob * 2)
# rand
# 0 [0, 1] 0.06667785
# 1 [0, 2] 0.036796866666666664
# 2 [1, 0] 0.0666407
# 3 [1, 3] 0.0367458
# 4 [2, 0] 0.036851
# 5 [2, 3] 0.0389095
# 6 [3, 1] 0.0367946
# 7 [3, 2] 0.0389195
# 0.7169306

# normal
# a and b
# 0 [0, 1] 0.04213
# 1 [0, 2] 0.01868
# 2 [1, 0] 0.04158
# 3 [1, 3] 0.01881
# 4 [2, 0] 0.01861
# 5 [2, 3] 0.01895
# 6 [3, 1] 0.01871
# 7 [3, 2] 0.02009
# 0.39511999999999997

# c and d
# 0 [0, 1] 0.04106
# 1 [0, 3] 0.01782
# 2 [1, 0] 0.04075
# 3 [1, 2] 0.01846
# 4 [2, 1] 0.01829
# 5 [2, 3] 0.01889
# 6 [3, 0] 0.01831
# 7 [3, 2] 0.01938
# 0.38592

