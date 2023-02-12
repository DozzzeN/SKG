import csv
import time

import numpy as np
from scipy.io import loadmat

fileName = "../data/data_static_indoor_1.mat"
rawData = loadmat(fileName)

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
dataLen = len(CSIa1Orig)

segLen = 7
keyLen = 256 * segLen

times = 0
overhead = 0

for staInd in range(0, dataLen, keyLen):
    endInd = staInd + keyLen
    print("range:", staInd, endInd)
    if endInd >= len(CSIa1Orig):
        break
    times += 1

    CSIa1Orig = rawData['A'][:, 0]
    CSIb1Orig = rawData['A'][:, 1]
    CSIn1Orig = np.random.normal(loc=1, scale=2, size=dataLen)

    seed = np.random.randint(10000)
    np.random.seed(seed)
    noiseOrig = np.random.uniform(0, np.std(CSIa1Orig) * 3, size=dataLen)
    CSIa1Orig = (CSIa1Orig - np.mean(CSIa1Orig)) + noiseOrig
    CSIb1Orig = (CSIb1Orig - np.mean(CSIb1Orig)) + noiseOrig
    CSIn1Orig = (CSIn1Orig - np.mean(CSIn1Orig)) + noiseOrig

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]

    tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()

    start = time.time()
    tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()
    minEpiIndClosenessLsb = np.zeros(int(keyLen / segLen), dtype=int)

    for i in range(int(keyLen / segLen)):
        epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]

        epiIndClosenessLsb = np.zeros(int(keyLen / segLen))

        for j in range(int(keyLen / segLen)):
            epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]

            epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))

        minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)
    print(minEpiIndClosenessLsb)

    end = time.time()
    overhead += end - start
    print("time:", end - start)

print(overhead / times)
