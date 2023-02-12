import csv
import itertools
import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.stats import pearsonr


def search(data, p):
    for i in range(len(data)):
        if p == data[i]:
            return i


n = 8
b = list(range(0, n))
r = [p for p in itertools.permutations(list(range(n)))]
# r = b
# r = r[1:] + r[0:1]
a = r[1:] + r[0:1]
final_dist = 0
times = 100000
for j in range(times):
    b = r[np.random.randint(0, len(r))]
    a = r[np.random.randint(0, len(r))]
    tmp_dist = 0
    for i in range(len(b)):
        real_pos = search(a, b[i])
        guess_pos = i
        tmp_dist += abs(real_pos - guess_pos)
    final_dist += tmp_dist
print(final_dist / times / n)
# print(final_dist, len(r), final_dist / len(r))
# 119750400 / 3628800
# exit()

rawData = loadmat("../skyglow/Scenario3-Mobile/data_mobile_1.mat")
# rawData = loadmat("../skyglow/Scenario2-Office-LoS/data3_upto5.mat")
# rawData = loadmat("../skyglow/Scenario5-Outdoors-Mobile-OutsideSphere1/data_mobile_1.mat")
# rawData = loadmat("../skyglow/Scenario6-Outdoors-Stationary-OutsideSphere1/data_static_1.mat")
CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
dataLen = len(CSIa1Orig)

keyLens = [8, 16, 32, 64, 128, 256]
segLen = 7

for keyLen in keyLens:
    print("keyLen", keyLen)
    keyLen = keyLen * segLen

    originSum = 0
    correctSum = 0
    randomSum = 0
    noiseSum = 0

    originDecSum = 0
    correctDecSum = 0
    randomDecSum = 0
    noiseDecSum = 0

    originWholeSum = 0
    correctWholeSum = 0
    randomWholeSum = 0
    noiseWholeSum = 0

    times = 0

    addNoise = "mul"

    distance = 0
    real_dist = 0

    mhd_dist = 0
    for staInd in range(0, int(dataLen), keyLen):
        endInd = staInd + keyLen
        # print("range:", staInd, endInd)
        if endInd >= len(CSIa1Orig):
            break
        times += 1

        # np.random.seed(1)
        CSIa1Orig = rawData['A'][:, 0]
        CSIb1Orig = rawData['A'][:, 1]

        # imitation attack
        # CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=len(CSIa1Orig))
        # stalking attack
        CSIe1Orig = loadmat("../skyglow/Scenario3-Mobile/data_eave_mobile_1.mat")['A'][:, 0]
        # CSIe1Orig = loadmat("../skyglow/Scenario2-Office-LoS/data3_eave_upto5.mat")['A'][:, 0]
        # CSIe1Orig = loadmat("../skyglow/Scenario5-Outdoors-Mobile-OutsideSphere1/data_eave_mobile_1.mat")['A'][:, 0]
        # CSIe1Orig = loadmat("../skyglow/Scenario6-Outdoors-Stationary-OutsideSphere1/data_eave_static_1.mat")['A'][:, 0]

        min_length = min(len(CSIa1Orig), len(CSIb1Orig), len(CSIe1Orig))
        CSIa1Orig = CSIa1Orig[0:min_length]
        CSIb1Orig = CSIb1Orig[0:min_length]
        CSIe1Orig = CSIe1Orig[0:min_length]
        # noiseOrig = np.random.normal(np.mean(CSIa1Orig), np.std(CSIa1Orig), size=len(CSIa1Orig))
        # noiseOrig = np.random.normal(0, np.std(CSIa1Orig), size=len(CSIa1Orig))
        # np.random.seed(int(seeds[times - 1][0]))
        seed = np.random.randint(100000)
        np.random.seed(seed)

        if addNoise == "add":
            noiseOrig = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=len(CSIa1Orig))
            CSIa1Orig = (CSIa1Orig - np.mean(CSIa1Orig)) + noiseOrig
            CSIb1Orig = (CSIb1Orig - np.mean(CSIb1Orig)) + noiseOrig
            CSIe1Orig = (CSIe1Orig - np.mean(CSIe1Orig)) + noiseOrig
            CSIn1Orig = noiseOrig

            tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
            tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
            tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
            tmpNoise = CSIn1Orig[range(staInd, endInd, 1)]
        else:
            tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
            tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
            tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]

            # randomMatrix = np.random.normal(np.mean(tmpCSIa1), np.std(tmpCSIa1), size=(keyLen, keyLen))
            randomMatrix = np.random.uniform(0, np.abs(np.std(CSIa1Orig)), size=(keyLen, keyLen))
            tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
            tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
            tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
            tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
            tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)
            tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix)
            # inference attack
            # tmpCSIe1 = np.matmul(np.ones(keyLen), randomMatrix)
            tmpNoise = np.matmul(np.ones(keyLen), randomMatrix)

        # 最后各自的密钥
        a_list = []
        b_list = []
        e_list = []
        n_list = []

        tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
        tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()
        tmpCSIe1Ind = np.array(tmpCSIe1).argsort().argsort()
        tmpCSIn1Ind = np.array(tmpNoise).argsort().argsort()

        minEpiIndClosenessLsb = np.zeros(int(keyLen / segLen), dtype=int)
        minEpiIndClosenessLse = np.zeros(int(keyLen / segLen), dtype=int)
        minEpiIndClosenessLsn = np.zeros(int(keyLen / segLen), dtype=int)

        tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)
        permutation = list(range(int(keyLen / segLen)))
        combineMetric = list(zip(tmpCSIa1IndReshape, permutation))
        np.random.seed(staInd)
        np.random.shuffle(combineMetric)
        tmpCSIa1IndReshape, permutation = zip(*combineMetric)
        tmpCSIa1Ind = np.hstack((tmpCSIa1IndReshape))

        for i in range(int(keyLen / segLen)):
            epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]

            epiIndClosenessLsb = np.zeros(int(keyLen / segLen))
            epiIndClosenessLse = np.zeros(int(keyLen / segLen))
            epiIndClosenessLsn = np.zeros(int(keyLen / segLen))

            for j in range(int(keyLen / segLen)):
                epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
                epiInde1 = tmpCSIe1Ind[j * segLen: (j + 1) * segLen]
                epiIndn1 = tmpCSIn1Ind[j * segLen: (j + 1) * segLen]

                epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))
                epiIndClosenessLse[j] = sum(abs(epiInde1 - np.array(epiInda1)))
                epiIndClosenessLsn[j] = sum(abs(epiIndn1 - np.array(epiInda1)))

            minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)
            minEpiIndClosenessLse[i] = np.argmin(epiIndClosenessLse)
            minEpiIndClosenessLsn[i] = np.argmin(epiIndClosenessLsn)

        # a_list_number = list(range(int(keyLen / segLen)))
        a_list_number = list(permutation)
        b_list_number = list(minEpiIndClosenessLsb)
        e_list_number = list(minEpiIndClosenessLse)
        # eavesdropping attack
        # e_list_number = list(range(len(a_list_number)))
        # np.random.shuffle(e_list_number)
        n_list_number = list(minEpiIndClosenessLsn)

        # mhd：绝对值距离之和除以密钥长度
        mhd_dist += sum(abs(np.array(a_list_number) - np.array(e_list_number))) / int(keyLen / segLen)
        tmp_dist = 0
        for i in range(len(e_list_number)):
            real_pos = search(a_list_number, e_list_number[i])
            guess_pos = i
            tmp_dist += abs(real_pos - guess_pos)
        # distance：与真实值索引偏移距离之和除以密钥长度
        distance += (tmp_dist / int(keyLen / segLen))

        # real_dist += sum(abs(np.array(a_list_number) - np.array(b_list_number))) / int(keyLen / segLen)
        tmp_dist = 0
        for i in range(len(b_list_number)):
            real_pos = search(a_list_number, b_list_number[i])
            guess_pos = i
            tmp_dist += abs(real_pos - guess_pos)
        # real：真实值之间索引偏移距离之和除以密钥长度
        real_dist += (tmp_dist / int(keyLen / segLen))

    print(round(distance / times, 8))
    print(round(mhd_dist / times, 8))
    print(round(real_dist / times, 8))
