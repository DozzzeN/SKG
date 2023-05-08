import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr

fileName = "csi_static_outdoor"
rawData = loadmat(fileName + ".mat")
if fileName in rawData.keys():
    CSIa1Orig = rawData[fileName][:, 0]
    CSIb1Orig = rawData[fileName][:, 1]
else:
    CSIa1Orig = rawData['testdata'][:, 0]
    CSIb1Orig = rawData['testdata'][:, 1]
dataLen = len(CSIa1Orig)
print(dataLen)

print(pearsonr(CSIa1Orig, CSIb1Orig)[0])
CSIa1Orig = CSIa1Orig - (np.mean(CSIa1Orig) - np.mean(CSIb1Orig))

# plt.figure()
# plt.plot(CSIa1Orig)
# plt.plot(CSIb1Orig)
# plt.show()

CSIa1Origb = []
CSIb1Origb = []

window = 2
for i in range(0, dataLen):
    CSIa1Origb.append(CSIa1Orig[i] - (np.mean(CSIa1Orig[i:i + window]) - np.mean(CSIb1Orig[i:i + window])))
    CSIb1Origb.append(CSIb1Orig[i])

print(len(CSIa1Origb))
print(pearsonr(CSIa1Origb, CSIb1Origb)[0])

CSIa1Origbb = []
CSIb1Origbb = []
indices = []
for i in range(0, len(CSIa1Origb)):
    if abs(CSIa1Origb[i] - CSIb1Origb[i]) < 0.1:
        indices.append(i)
        CSIa1Origbb.append(CSIa1Origb[i])
        CSIb1Origbb.append(CSIb1Origb[i])

# plt.figure()
# plt.plot(CSIa1Origbb)
# plt.plot(CSIb1Origbb)
# plt.show()

print(len(CSIa1Origbb))
print(pearsonr(CSIa1Origbb, CSIb1Origbb)[0])

savemat(fileName + '_r.mat', {"testdata": np.array([CSIa1Origbb, CSIb1Origbb]).T})
