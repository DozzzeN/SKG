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


fileNameA = "./ZengKai/mo/bit2/csiA.mat"
fileNameB = "./ZengKai/mo/bit2/csiB.mat"
# fileNameA = "./ZengKai/so/bit1/csiA.mat"
# fileNameB = "./ZengKai/so/bit1/csiB.mat"
fileNameR = fileNameA[:fileNameA.find('csi')] + 'r.mat'

csiA = loadmat(fileNameA)['csiA'].T[0]
csiB = loadmat(fileNameB)['csiB'].T[0]

print(len(csiA))
print(len(csiB))
dataLen = min((len(csiA), len(csiB)))
csiA = csiA - np.mean(csiA)
csiB = csiB - np.mean(csiB)
print(pearsonr(csiA[:dataLen], csiB[:dataLen])[0])

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
for i in range(1, dataLen, splitLen):
    if i + splitLen >= dataLen:
        break
    csiAPart = csiA[i: i + splitLen] - np.mean(csiA[i: i + splitLen])
    csiBPart = csiB[i: i + splitLen] - np.mean(csiB[i: i + splitLen])
    # print(i)
    # print(pearsonr(csiAPart, csiBPart)[0])

    dtw_computation = dtw_metric(csiAPart, csiBPart)
    pathA = dtw_computation[3][0]
    pathB = dtw_computation[3][1]

    csiAPart = csiAPart[pathA]
    csiBPart = csiBPart[pathB]
    # print(pearsonr(csiAPart, csiBPart)[0])

    csiAAlign1 = []
    csiBAlign1 = []
    for j in range(2, splitLen):
        # csiAAlign1.append(csiAPart[j])
        # csiBAlign1.append(csiBPart[j])

        # 相邻点相等则丢弃：进一步提升相关性，但降低数据量
        # if csiAPart[j - 1] == csiAPart[j] or csiBPart[j - 1] == csiBPart[j]:
        #     continue
        if (csiAPart[j - 1] == csiAPart[j] and csiAPart[j - 2] == csiAPart[j - 1]) \
                or (csiBPart[j - 1] == csiBPart[j] and csiBPart[j - 2] == csiBPart[j - 1]):
            continue
        else:
            csiAAlign1.append(csiAPart[j])
            csiBAlign1.append(csiBPart[j])

    for j in range(len(csiAAlign1)):
        if abs(csiAAlign1[j] - csiBAlign1[j]) < \
                (max(max(csiAAlign1), max(csiBAlign1)) - min(min(csiAAlign1), min(csiBAlign1))) / 8:
            csiAAlign2.append(csiAAlign1[j])
            csiBAlign2.append(csiBAlign1[j])

print(pearsonr(csiAAlign2, csiBAlign2)[0])
print(len(csiAAlign2))
print()

savemat(fileNameR, {"csi": np.array([csiAAlign2, csiBAlign2]).T})
