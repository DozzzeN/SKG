import time
from tkinter import messagebox

import numpy as np
from numpy.random import exponential as Exp
from pyentrp import entropy as ent
from scipy.io import loadmat
from scipy.spatial.distance import pdist

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
    shuffleInd = np.random.permutation(dataLen)
    CSIa1Orig = CSIa1Orig[shuffleInd]
    CSIb1Orig = CSIb1Orig[shuffleInd]

    sortCSIa1Reshape = CSIa1Orig[0:segLen * int(dataLen / segLen)]
    sortCSIb1Reshape = CSIb1Orig[0:segLen * int(dataLen / segLen)]

    sortCSIa1Reshape = sortCSIa1Reshape.reshape(int(len(sortCSIa1Reshape) / segLen), segLen)
    sortCSIb1Reshape = sortCSIb1Reshape.reshape(int(len(sortCSIb1Reshape) / segLen), segLen)

    for i in range(segLen):
        a_mul_entropy = ent.multiscale_entropy(sortCSIa1Reshape[i], 3, maxscale=1)

        cnts = 0
        while a_mul_entropy < entropyThres and cnts < 10:
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


fileName = "../data/data_static_outdoor_1.mat"
rawData = loadmat(fileName)
CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
dataLen = len(CSIa1Orig)

entropyThres = 2
CSIa1Orig, CSIb1Orig = splitEntropyPerm(CSIa1Orig, CSIb1Orig, 64, dataLen, entropyThres)

for rep in range(1500):
    print("rep", rep)
    CSIa1OrigBack = CSIa1Orig.copy()
    CSIb1OrigBack = CSIb1Orig.copy()
    noiseOrig = np.random.uniform(5, 6, size=dataLen)  ## Multiplication item normal distribution
    noiseOrigBack = noiseOrig.copy()

    # sft = 1
    intvl = 5
    keyLen = 64
    times = 0

    originSum = 0
    correctSum = 0
    originWholeSum = 0
    correctWholeSum = 0
    topNum = 32

    codings = ""
    for staInd in range(0, len(CSIa1Orig), intvl * keyLen):
        endInd = staInd + keyLen * intvl
        if endInd >= len(CSIa1Orig):
            break
        times += 1

        CSIa1Orig = CSIa1OrigBack.copy()
        CSIb1Orig = CSIb1OrigBack.copy()
        noiseOrig = noiseOrigBack.copy()

        permLen = len(range(staInd, endInd, intvl))
        permInd = np.random.permutation(permLen)

        binary = ""
        for i in range(len(permInd)):
            binary += bin(permInd[i])[2:]
        codings = ""
        for i in range(0, len(binary), 25):
            codings += binary[i:i + 25] + "\n"

    with open('key_data_static_outdoor_1.txt', 'a', ) as f:
        f.write(codings)
messagebox.showinfo("提示", "测试结束")
