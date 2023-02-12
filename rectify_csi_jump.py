import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr

import entropy_estimators

CSIa1Orig = list(loadmat("testdata(2).mat")['testdata'][:, 0])
CSIb1Orig = list(loadmat("testdata(2).mat")['testdata'][:, 1])

CSIa = []
CSIb = []
print(len(CSIa1Orig), len(CSIb1Orig))

jumps = []
jumps.append(0)
for i in range(1, min(len(CSIa1Orig), len(CSIb1Orig))):
    if abs(CSIa1Orig[i] - CSIa1Orig[i - 1]) > 0.5:
        jumps.append(i - 1)

for i in range(1, len(jumps)):
    for j in range(jumps[i] - jumps[i - 1]):
        CSIa.append(CSIa1Orig[j] - np.mean(CSIa1Orig[jumps[i - 1]:jumps[i]]))
        CSIb.append(CSIb1Orig[j] - np.mean(CSIb1Orig[jumps[i - 1]:jumps[i]]))

CSIa1 = []
CSIb1 = []
for i in range(1, min(len(CSIa), len(CSIb))):
    if abs(CSIa[i] - CSIb[i]) <= 0.8:
        CSIa1.append(CSIa[i])
        CSIb1.append(CSIb[i])

print(len(CSIa), len(CSIb))
# print(len(CSIa1), len(CSIb1))

# print(pearsonr(CSIa1, CSIb1)[0])
print(pearsonr(CSIa, CSIb)[0])
print(pearsonr(CSIa1Orig, CSIb1Orig)[0])

# savemat('testdata_r.mat', {"testdata": np.array([CSIa, CSIb]).T})

# plt.figure()
# plt.plot(CSIa)
# plt.show()
#
# plt.figure()
# plt.plot(CSIb)
# plt.show()
