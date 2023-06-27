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
    return origin + noise, noise

# 方法三：只满足第一个条件

# SNR = 10
# topNum = 16 keyLen = 128
# 0.8519345238 0.0 1.4 1.1927083333333335
# threshold = 10
# 0.9556642347 0.2857142857 1.3920758928571428 1.3303571428571428

# topNum = 16 keyLen = 256
# 0.7576729911 0.0 1.6 1.2122767857142858
# threshold = 10
# 0.9031514852 0.0 1.582421875 1.4291666666666667

dataLen = 10000
SNR = 10
np.random.seed(0)
channel = np.random.normal(0, 1, size=dataLen)

CSIa1Orig = np.array(channel)

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

    lts.append(np.ones(intvl))
    changed = 0

    newCSIa1 = []
    newCSIa1.extend(addNoise(tmpCSIa1[0: intvl], SNR, 1)[0])
    newCSIb1 = []
    newCSIb1.extend(addNoise(tmpCSIa1[0: intvl], SNR, 2)[0])
    for i in range(1, keyLen):
        lts_noise = np.ones(intvl)
        cnt = 0
        while True:
            rowDists = []
            epiInda1 = tmpCSIa1[i * intvl: (i + 1) * intvl].copy()
            epiInda1 *= lts_noise
            epiInda1, channel_noise_2 = addNoise(epiInda1, SNR, 3)
            for j in range(0, i):
                epiInda2 = tmpCSIa1[j * intvl: (j + 1) * intvl].copy()
                epiInda2, channel_noise_1 = addNoise(epiInda2, SNR, 4)
                epiInda2 *= lts[j]
                rowDists.append(sum(abs(epiInda1 - epiInda2)))

            minDist = min(rowDists)
            if minDist > threshold:
                # 不改变噪音
                # newCSIa1.extend(epiInda1)
                # newCSIb1.extend(addNoise(tmpCSIa1[i * intvl: (i + 1) * intvl].copy(), SNR)[0] * lts_noise)
                newCSIa1.extend(addNoise(epiInda1 - channel_noise_2, SNR, 5)[0])
                newCSIb1.extend(addNoise(tmpCSIa1[i * intvl: (i + 1) * intvl].copy(), SNR, 6)[0] * lts_noise)
                lts.append(lts_noise)
                # print("lts_noise", lts_noise)
                # print("minDist", minDist)
                break
            else:
                #  记录修改的次数
                if cnt == 0:
                    changed += 1
                cnt += 1
                np.random.seed(cnt * (staInd + 1))
                lts_noise = np.random.normal(0, 10, size=intvl)

    # print("changed", changed / (keyLen - 1))
    minDist = []
    for i in range(1, keyLen):
        rowDists = []
        epiInda1 = np.array(newCSIa1[i * intvl: (i + 1) * intvl])
        for j in range(0, i):
            epiInda2 = np.array(newCSIb1[j * intvl: (j + 1) * intvl])
            rowDists.append(sum(abs(epiInda1 - epiInda2)))

        allDists.append(rowDists)
        minDist.append(min(rowDists))
    print("minDist", minDist)

    modifiedCSIa1.extend(newCSIa1)
    modifiedCSIb1.extend(newCSIb1)

savemat("csi_opt31_" + str(keyLen) + "_" + str(threshold) + ".mat", {'A': np.array([modifiedCSIa1, modifiedCSIb1]).T})
print(pearsonr(modifiedCSIa1, modifiedCSIb1)[0])