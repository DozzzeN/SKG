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


for f in range(3, 8):
    fileNameA = "./Jana/si/静态-无遮挡[1," + str(f) +"]csiA.mat"
    fileNameB = "./Jana/si/静态-无遮挡[1," + str(f) +"]csiB.mat"

    fileNameR = fileNameA[:fileNameA.find('csi')] + 'r.mat'

    csiA = loadmat(fileNameA)['csiA'].T[0]
    csiB = loadmat(fileNameB)['csiB'].T[0]

    print(len(csiA))
    print(len(csiB))
    dataLen = min((len(csiA), len(csiB)))
    csiA = csiA - np.mean(csiA)
    csiB = csiB - np.mean(csiB)
    # csiA = (csiA - np.min(csiA)) / (np.max(csiA) - np.min(csiA))
    # csiB = (csiB - np.min(csiB)) / (np.max(csiB) - np.min(csiB))
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
    csiAAlign = []
    csiBAlign = []
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

        for j in range(splitLen):
            if abs(csiAPart[j] - csiBPart[j]) < (max(max(csiAPart), max(csiBPart)) - min(min(csiAPart), min(csiBPart))) / 2:
                csiAAlign.append(csiAPart[j])
                csiBAlign.append(csiBPart[j])

        # packetLen = 10
        # for j in range(1, splitLen, packetLen):
        #     if abs(np.var(csiAPart[j:j + packetLen]) - np.var(csiBPart[j:j + packetLen])) < \
        #             abs(np.var(csiAPart) - np.var(csiBPart)):
        #         csiAAlign.extend(csiAPart[j:j + packetLen])
        #         csiBAlign.extend(csiBPart[j:j + packetLen])
        # csiAAlign.extend(csiAPart)
        # csiBAlign.extend(csiBPart)

    print(pearsonr(csiAAlign, csiBAlign)[0])
    print(len(csiAAlign))

    savemat(fileNameR, {"csi": np.array([csiAAlign, csiBAlign]).T})
