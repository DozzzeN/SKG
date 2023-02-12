import itertools

# pool = list(itertools.product(["x1", "x2", "y1", "y2", "n1", "n2"], repeat=2))
# bools = list(itertools.product(pool, repeat=4))
# matches = 0
# times = 0
# for i in range(len(bools)):
#     p = []
#     for j in range(len(bools[i])):
#         p += list(bools[i][j])
#     flag = True
#     for j in range(1, len(p)):
#         flag = (flag and p[j - 1] == p[j])
#     # if flag is False:
#     #     times += 1
#     if p[1] == p[3] or p[5] == p[7]:
#         times += 1
#     if p[1] == p[3] and p[5] == p[7]:
#         times -= 1
#     if p[1] == p[3] and p[5] == p[7]:
#         if p[0] != p[2] and p[2] != p[4] and p[4] != p[6] and p[6] != p[0]:
#             matches += 1
# print(matches / len(bools) / 2)
# print(matches, len(bools))
# print(times)

bools = list(itertools.product(["a", "b", "c", "d", "e", "f"], repeat=8))
matches = 0
times = 0
for i in range(len(bools)):
    p = []
    for j in range(len(bools[i])):
        p += list(bools[i][j])
    flag = True
    pb = p[0:4] + p[5:6] + p[7:8]
    pp = sorted(pb)
    for j in range(1, len(pp)):
        flag = (flag and pp[j - 1] != pp[j])
    if flag is True:
        times += 1
        # print(pb, p[4], p[6], pp)
    p4 = -1
    p6 = -1
    for j in range(len(pb)):
        if pb[j] == p[4]:
            p4 = j
        if pb[j] == p[6]:
            p6 = j
    if p4 != p6 and p4 != -1 and p6 != -1 and p[4] != p[6] and flag:
        matches += 1
print(matches / len(bools))
print(matches, len(bools))
print(times)
