import sys
import time
from tkinter import messagebox

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import exponential as Exp
from pyentrp import entropy as ent
from scipy.io import loadmat, savemat
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

from mwmatching import maxWeightMatching


def entropyPerm(CSIa1Orig, CSIb1Orig, dataLen, entropyThres):
    ts = CSIa1Orig / np.max(CSIa1Orig)
    # shanon_entropy = ent.shannon_entropy(ts)
    # perm_entropy = ent.permutation_entropy(ts, order=3, delay=1, normalize=True)
    # mulperm_entropy = ent.multiscale_permutation_entropy(ts, 3, 1, 1)
    mul_entropy = ent.multiscale_entropy(ts, 3, maxscale=1)
    # print(mul_entropy)

    cnts = 0
    while mul_entropy < entropyThres and cnts < 10:
        # while mul_entropy < 2.510
        shuffleInd = np.random.permutation(dataLen)
        CSIa1Orig = CSIa1Orig[shuffleInd]
        CSIb1Orig = CSIb1Orig[shuffleInd]
        # CSIa2Orig = CSIa2Orig[shuffleInd]
        # CSIb2Orig = CSIb2Orig[shuffleInd]

        ts = CSIa1Orig / np.max(CSIa1Orig)
        mul_entropy = ent.multiscale_entropy(ts, 4, maxscale=1)
        cnts += 1
        # print(mul_entropy[0])

    return CSIa1Orig, CSIb1Orig


def splitEntropyPerm(CSIa1Orig, CSIb1Orig, segLen, dataLen, entropyThres):
    # 先整体shuffle一次
    np.random.seed(0)
    shuffleInd = np.random.permutation(dataLen)
    CSIa1Orig = CSIa1Orig[shuffleInd]
    CSIb1Orig = CSIb1Orig[shuffleInd]

    sortCSIa1Reshape = CSIa1Orig[0:segLen * int(dataLen / segLen)]
    sortCSIb1Reshape = CSIb1Orig[0:segLen * int(dataLen / segLen)]

    sortCSIa1Reshape = sortCSIa1Reshape.reshape(int(len(sortCSIa1Reshape) / segLen), segLen)
    sortCSIb1Reshape = sortCSIb1Reshape.reshape(int(len(sortCSIb1Reshape) / segLen), segLen)
    n = len(sortCSIa1Reshape)

    for i in range(n):
        a_mul_entropy = ent.multiscale_entropy(sortCSIa1Reshape[i], 3, maxscale=1)

        cnts = 0
        while a_mul_entropy < entropyThres and cnts < 10:
            np.random.seed(cnts)
            shuffleInd = np.random.permutation(len(sortCSIa1Reshape[i]))
            sortCSIa1Reshape[i] = sortCSIa1Reshape[i][shuffleInd]
            sortCSIb1Reshape[i] = sortCSIb1Reshape[i][shuffleInd]

            a_mul_entropy = ent.multiscale_entropy(sortCSIa1Reshape[i], 3, maxscale=1)
            cnts += 1

    _CSIa1Orig = []
    _CSIb1Orig = []

    for i in range(len(sortCSIa1Reshape)):
        for j in range(len(sortCSIa1Reshape[i])):
            _CSIa1Orig.append(sortCSIa1Reshape[i][j])
            _CSIb1Orig.append(sortCSIb1Reshape[i][j])

    return np.array(_CSIa1Orig), np.array(_CSIb1Orig)


def addNoise(origin, SNR, seed):
    dataLen = len(origin)
    np.random.seed(seed)
    noise = np.random.normal(0, 1, size=dataLen)
    signal_power = np.sum(origin ** 2) / dataLen
    noise_power = np.sum(noise ** 2) / dataLen
    noise_variance = signal_power / (10 ** (SNR / 10))
    noise = noise * np.sqrt(noise_variance / noise_power)
    return origin + noise


# 方法二

# SNR = 10
# topNum = 16 keyLen = 128
# 0.8519345238 0.0 1.4 1.1927083333333335
# threshold
# 0.8683035714 0.0 1.4 1.215625

# SNR = 10
# topNum = 16 keyLen = 256
# 0.7576729911 0.0 1.6 1.2122767857142858
# threshold
# 0.8383789062 0.0 1.6 1.34140625

dataLen = 10000
SNR = 10
np.random.seed(0)
channel = np.random.normal(0, 1, size=dataLen)
CSIa1Orig = addNoise(channel, SNR, 1)

CSIa1Orig = np.array(CSIa1Orig)

CSIa1OrigBack = CSIa1Orig.copy()

intvl = 5
keyLen = 256

times = 0

originSum = 0
correctSum = 0
originWholeSum = 0
correctWholeSum = 0
topNum = 16
overhead = 0

threshold = 10

print("topNum", topNum)
print("sample number", len(CSIa1Orig))

modifiedCSIa1 = []
modifiedCSIb1 = []

for staInd in range(0, len(CSIa1Orig), intvl * keyLen):
    endInd = staInd + keyLen * intvl
    if endInd >= len(CSIa1Orig):
        break

    tmpCSIa1 = CSIa1OrigBack[range(staInd, endInd, 1)]

    allDists = []

    lts = []

    thresholds = []
    lts.append(np.ones(intvl))
    changed = 0
    for i in range(1, keyLen):
        noise = np.ones(intvl)
        cnt = 0
        while True:
            rowDists = []
            epiInda1 = tmpCSIa1[i * intvl: (i + 1) * intvl]
            for j in range(0, i):
                epiInda2 = tmpCSIa1[j * intvl: (j + 1) * intvl]
                rowDists.append(sum(abs(epiInda1 - epiInda2)))
            minDist = min(rowDists)

            if i == 1:
                lts.append(noise)
                thresholds.append(minDist)
                break

            # if minDist / max(thresholds) > threshold:
            if minDist > max(thresholds):
                lts.append(noise)
                thresholds.append(minDist)
                break
            else:
                #  记录修改的次数
                if cnt == 0:
                    changed += 1
                cnt += 1
                # print("minDist", minDist)
                np.random.seed(cnt * (staInd + 1))
                noise = np.random.normal(0, 10, size=intvl)
                tmpCSIa1[i * intvl: (i + 1) * intvl] *= noise

    # print("changed", changed / (keyLen - 1))
    minDist = []
    for i in range(1, keyLen):
        rowDists = []
        epiInda1 = tmpCSIa1[i * intvl: (i + 1) * intvl]
        for j in range(0, i):
            epiInda2 = tmpCSIa1[j * intvl: (j + 1) * intvl]
            rowDists.append(sum(abs(epiInda1 - epiInda2)))

        allDists.append(rowDists)
        minDist.append(min(rowDists))
    print("minDist", minDist)

    for i in range(keyLen):
        # Alice: (h + noise) * lts
        modifiedCSIa1.extend(CSIa1OrigBack[i * intvl: (i + 1) * intvl] * lts[i])
        # 效果较好，但不符合实际
        # Bob: lts * h
        # modifiedCSIb1.extend(channel[i * intvl: (i + 1) * intvl] * lts[i])
        # 和直接在Alice数据上加噪音差不多
        # Bob: lts * h + noise
        # modifiedCSIb1.extend(addNoise(channel[i * intvl: (i + 1) * intvl] * lts[i], SNR))

    # tmpCSIa1b = modifiedCSIa1
    # for i in range(keyLen):
    #     if np.isclose(tmpCSIa1b[i * intvl: (i + 1) * intvl], tmpCSIa1[i * intvl: (i + 1) * intvl]).all() is False:
    #         print(np.isclose(tmpCSIa1b[i * intvl: (i + 1) * intvl], tmpCSIa1[i * intvl: (i + 1) * intvl]))

# Bob: Alice + noise
modifiedCSIb1 = addNoise(np.array(modifiedCSIa1), SNR, 1)
savemat("csi_opt2_" + str(keyLen) + "_" + str(threshold) + ".mat", {'A': np.array([modifiedCSIa1, modifiedCSIb1]).T})
print(pearsonr(modifiedCSIa1, modifiedCSIb1)[0])