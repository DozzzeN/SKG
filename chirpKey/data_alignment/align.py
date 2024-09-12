import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr
from dtw import dtw, accelerated_dtw


def dtw_metric(data1, data2):
    distance = lambda x, y: np.abs(x - y)
    data1 = np.array(data1)
    data2 = np.array(data2)
    # return dtw(data1, data2, dist=distance)[0]
    return accelerated_dtw(data1, data2, dist=distance)


fileNameA = "./csi_ea.mat"
fileNameB = "./csi_eb.mat"
fileNameE = "./csi_ee.mat"

fileNameR = fileNameA[:fileNameA.find('.mat')] + 'r.mat'

csiA = loadmat(fileNameA)['csi_value'].T[0]
csiB = loadmat(fileNameB)['csi_value'].T[0]
csiE = loadmat(fileNameE)['csi_value'].T[0]

print(len(csiA))
print(len(csiB))
print(len(csiE))
dataLen = min(min((len(csiA), len(csiB))), len(csiE))
csiA = csiA - np.mean(csiA)
csiB = csiB - np.mean(csiB)
csiE = csiE - np.mean(csiE)
# csiA = (csiA - np.min(csiA)) / (np.max(csiA) - np.min(csiA))
# csiB = (csiB - np.min(csiB)) / (np.max(csiB) - np.min(csiB))
print(pearsonr(csiA[:dataLen], csiB[:dataLen])[0])

# matrix = np.random.normal(0, 1, (dataLen, dataLen))
# csiA = np.matmul(matrix, csiA[:dataLen])
# csiB = np.matmul(matrix, csiB[:dataLen])
# csiE = np.matmul(matrix, csiE[:dataLen])

# shuffleInd = np.random.permutation(dataLen)
# csiA = csiA[:dataLen][shuffleInd]
# csiB = csiB[:dataLen][shuffleInd]
# csiE = csiE[:dataLen][shuffleInd]
# print(pearsonr(csiA, csiB)[0])

# plt.figure()
# plt.plot(csiA[0:200])
# plt.show()
#
# plt.figure()
# plt.plot(csiB[0:200])
# plt.show()

# splitLen越大相关性越好，数据量也越多
splitLen = 200
csiAAlign2 = []
csiBAlign2 = []
csiEAlign2 = []
for i in range(1, dataLen, splitLen):
    if i + splitLen >= dataLen:
        break
    csiAPart = csiA[i: i + splitLen] - np.mean(csiA[i: i + splitLen])
    csiBPart = csiB[i: i + splitLen] - np.mean(csiB[i: i + splitLen])
    csiEPart = csiE[i: i + splitLen] - np.mean(csiE[i: i + splitLen])
    # print(i)
    # print(pearsonr(csiAPart, csiBPart)[0])

    dtw_computation = dtw_metric(csiAPart, csiBPart)
    pathA = dtw_computation[3][0]
    pathB = dtw_computation[3][1]

    csiAPart = csiAPart[pathA]
    csiBPart = csiBPart[pathB]
    csiEPart = csiEPart[pathA]
    # for ear.mat
    # csiEPart = csiEPart[pathB]
    # print(pearsonr(csiAPart, csiBPart)[0])

    csiAAlign1 = []
    csiBAlign1 = []
    csiEAlign1 = []
    for j in range(1, splitLen):
        # csiAAlign1.append(csiAPart[j])
        # csiBAlign1.append(csiBPart[j])

        # 相邻点相等则丢弃：进一步提升相关性，但降低数据量
        if csiAPart[j - 1] == csiAPart[j] or csiBPart[j - 1] == csiBPart[j]:
            continue
        else:
            csiAAlign1.append(csiAPart[j])
            csiBAlign1.append(csiBPart[j])
            csiEAlign1.append(csiEPart[j])

    for j in range(len(csiAAlign1)):
        if abs(csiAAlign1[j] - csiBAlign1[j]) < \
                (max(max(csiAAlign1), max(csiBAlign1)) - min(min(csiAAlign1), min(csiBAlign1))) / 10:
            csiAAlign2.append(csiAAlign1[j])
            csiBAlign2.append(csiBAlign1[j])
            csiEAlign2.append(csiEAlign1[j])

print(pearsonr(csiAAlign2, csiBAlign2)[0])
print(pearsonr(csiAAlign2, csiEAlign2)[0])
print(pearsonr(csiBAlign2, csiEAlign2)[0])
print(len(csiAAlign2))

savemat(fileNameR, {"csi": np.array([csiAAlign2, csiBAlign2, csiEAlign2]).T})
