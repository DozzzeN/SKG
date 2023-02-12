import random

import numpy as np
from matplotlib import pyplot as plt
from alignment import alignFloatInsDel, genAlign

times = 1000
sortCSIa1 = list(range(128))
npLens = []
keyLens = []
insLens = []
delLens = []
for i in range(times):
    insIndex = random.sample(range(len(sortCSIa1)), random.randint(0, len(sortCSIa1)))
    delIndex = random.sample(range(len(sortCSIa1)), random.randint(0, len(sortCSIa1)))
    if len(sortCSIa1) + len(insIndex) - len(delIndex) <= 40:
        delIndex = random.sample(range(len(sortCSIa1)), random.randint(0, len(sortCSIa1)))
    insIndex.sort()
    delIndex.sort()
    sortCSIa1P = sortCSIa1.copy()
    for i in range(len(insIndex)):
        sortCSIa1P.insert(insIndex[i], -2)
    for i in range(len(delIndex) - 1, 0, -1):
        sortCSIa1P.remove(sortCSIa1P[delIndex[i]])
    npLens.append(len(sortCSIa1P))
    rule = alignFloatInsDel({'=': 0, '+': 1, '-': 1}, sortCSIa1P, sortCSIa1, 0.1)
    key = genAlign(rule)
    keyLens.append(len(key))
    insLens.append(len(insIndex))
    delLens.append(len(delIndex))
print(min(npLens), max(npLens), np.mean(npLens), np.std(npLens))
# print(min(keyLens), max(keyLens), np.mean(keyLens), np.std(keyLens))
print(min(insLens), max(insLens), np.mean(insLens), np.std(insLens))
print(min(delLens), max(delLens), np.mean(delLens), np.std(delLens))

# bins_interval = 1
# margin = 1
# bins = range(min(insLens), max(insLens) + bins_interval - 1, bins_interval)
# plt.xlim(min(insLens) - margin, max(insLens) + margin)
# plt.title("probability-distribution")
# plt.xlabel('Interval')
# plt.ylabel('Probability')
# plt.hist(x=insLens, bins=bins, histtype='bar', color=['r'])
# plt.show()

# bins_interval = 1
# margin = 1
# bins = range(min(delLens), max(delLens) + bins_interval - 1, bins_interval)
# plt.xlim(min(delLens) - margin, max(delLens) + margin)
# plt.title("probability-distribution")
# plt.xlabel('Interval')
# plt.ylabel('Probability')
# plt.hist(x=delLens, bins=bins, histtype='bar', color=['r'])
# plt.show()
