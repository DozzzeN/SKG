import random

import numpy as np
from matplotlib import pyplot as plt

from alignment import alignFloatInsDel, genAlignInsDel

keyLens = []
times = 1000000
insLens = []
delLens = []
nrLens = []
# for i in range(times):
#     a = 128
#     insNum = random.randint(0, 128)
#     insLens.append(insNum)
#     delNum = random.randint(0, 128)
#     while insNum - delNum + a <= 40:
#         delNum = random.randint(0, 128)
#     delLens.append(delNum)
#     for _ in range(insNum):
#         a += 1
#     for _ in range(delNum):
#         a -= 1
#     keyLens.append(a)
# # print(keyLens)
# print("keyLens", np.mean(keyLens), np.std(keyLens), min(keyLens), max(keyLens))
# # print(insLens)
# print("insLens", np.mean(insLens), min(insLens), max(insLens))
# # print(delLens)
# print("delLens", np.mean(delLens), min(delLens), max(delLens))

for i in range(times):
    a = 128
    insNum = random.randint(0, 128)
    insLens.append(insNum)
    delNum = random.randint(0, insNum)
    delLens.append(delNum)
    keyLens.append(a + insNum - delNum)
    nrLens.append(a - delNum)
# print(keyLens)
print("keyLens", np.mean(keyLens), np.std(keyLens), min(keyLens), max(keyLens))
# print(insLens)
print("insLens", np.mean(insLens), min(insLens), max(insLens))
# print(delLens)
print("delLens", np.mean(delLens), min(delLens), max(delLens))
print("nrLens", np.mean(nrLens), min(nrLens), max(nrLens))

bins_interval = 1
margin = 1
bins = range(min(nrLens), max(nrLens) + bins_interval - 1, bins_interval)
plt.xlim(min(nrLens) - margin, max(nrLens) + margin)
plt.title("probability-distribution")
plt.xlabel('Interval')
plt.ylabel('Probability')
plt.hist(x=nrLens, bins=bins, histtype='bar', color=['r'])
plt.show()

# times = 10
# sortCSIa1 = list(range(200))
# print("sortCSIa1", len(sortCSIa1), sortCSIa1)
# opNums = int(len(sortCSIa1) / 2)
# keyLens = []
# for i in range(times):
#     index = random.sample(range(opNums), opNums)
#     editOps = random.sample(range(int(len(sortCSIa1))), opNums)
#     # editOps = np.random.randint(0, int(len(sortCSIa1)), opNums)  # 编辑的位置如果相同那么编辑后的长度也不固定
#     editOps.sort()
#     sortCSIa1P = sortCSIa1.copy()
#     insNum = 0
#     for i in range(opNums - 1, -1, -1):
#         flag = random.randint(0, 1)
#         if flag == 0:
#             sortCSIa1P.insert(editOps[i], -2)
#             insNum += 1
#         elif flag == 1:
#             sortCSIa1P.remove(sortCSIa1P[editOps[i]])
#     print("sortCSIa1P", len(sortCSIa1P), sortCSIa1P)
#     rule = alignFloatInsDel({'=': 0, '+': 1, '-': 1}, sortCSIa1P, sortCSIa1, 0.1)
#     # 编辑操作的个数位64个，编辑前为128位，编辑后不一定是确定的某一位
#     # key = genAlignInsDel(rule)
#     # print(len(key), key)
#     key = genAlignInsDel(rule)
#     # keyInsDel = genAlignInsDel(rule)
#     # print(len(keyInsDel), keyInsDel)
#     print("insNum", insNum)
#     print("key", len(key), key)
#     keyLens.append(len(key))
#     # print(len(sortCSIa1P), sortCSIa1P)
#     print("rule", len(rule), rule)
#     print("editOps", len(editOps), list(editOps))
# print(keyLens)
# print(min(keyLens), max(keyLens), np.mean(keyLens), np.std(keyLens))
