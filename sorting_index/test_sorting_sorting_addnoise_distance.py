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

# keyLens = [8, 16, 32, 64, 128, 256]
# segLen = 7
keyLens = [256, 512, 768, 1024]
segLen = 5

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

    distance_e1 = 0
    distance_e2 = 0
    distance_n1 = 0
    distance_n2 = 0
    distance_n3 = 0
    distance_n4 = 0
    real_dist_e1 = 0
    real_dist_e2 = 0
    real_dist_n1 = 0
    real_dist_n2 = 0
    real_dist_n3 = 0
    real_dist_n4 = 0

    mhd_e1_dist = 0
    mhd_e2_dist = 0
    mhd_n1_dist = 0
    mhd_n2_dist = 0
    mhd_n3_dist = 0
    mhd_n4_dist = 0
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
        CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=len(CSIa1Orig))
        # stalking attack
        CSIe2Orig = loadmat("../skyglow/Scenario2-Office-LoS-eve_NLoS/data_eave_LOS_EVE_NLOS.mat")['A'][:, 0]

        tmpNoise1 = []
        tmpNoise2 = []
        tmpNoise3 = []
        tmpNoise4 = []

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
            CSIe2Orig = (CSIe2Orig - np.mean(CSIe2Orig)) + noiseOrig
            CSIn1Orig = noiseOrig

            tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
            tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
            tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
            tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]
        else:
            tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
            tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
            tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
            tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]

            # randomMatrix = np.random.normal(np.mean(tmpCSIa1), np.std(tmpCSIa1), size=(keyLen, keyLen))
            randomMatrix = np.random.uniform(0, np.abs(np.std(CSIa1Orig)), size=(keyLen, keyLen))
            tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
            tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
            tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
            tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
            tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)
            tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix)
            tmpCSIe2 = np.matmul(tmpCSIe2, randomMatrix)
            tmpNoise1 = randomMatrix.mean(axis=0)  # 按列求均值
            tmpNoise2 = randomMatrix.mean(axis=1)  # 按行求均值
            tmpNoise3 = np.matmul(np.ones(keyLen), randomMatrix)
            tmpNoise4 = np.random.normal(loc=np.mean(tmpCSIa1), scale=np.std(tmpCSIa1, ddof=1), size=len(tmpCSIa1))

        # 最后各自的密钥
        a_list = []
        b_list = []
        e1_list = []
        e2_list = []
        n1_list = []
        n2_list = []
        n3_list = []
        n4_list = []

        tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
        tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()
        tmpCSIe1Ind = np.array(tmpCSIe1).argsort().argsort()
        tmpCSIe2Ind = np.array(tmpCSIe2).argsort().argsort()
        tmpCSIn1Ind = np.array(tmpNoise1).argsort().argsort()
        tmpCSIn2Ind = np.array(tmpNoise2).argsort().argsort()
        tmpCSIn3Ind = np.array(tmpNoise3).argsort().argsort()
        tmpCSIn4Ind = np.array(tmpNoise4).argsort().argsort()

        minEpiIndClosenessLsb = np.zeros(int(keyLen / segLen), dtype=int)
        minEpiIndClosenessLse1 = np.zeros(int(keyLen / segLen), dtype=int)
        minEpiIndClosenessLse2 = np.zeros(int(keyLen / segLen), dtype=int)
        minEpiIndClosenessLsn = np.zeros(int(keyLen / segLen), dtype=int)
        minEpiIndClosenessLsn1 = np.zeros(int(keyLen / segLen), dtype=int)
        minEpiIndClosenessLsn2 = np.zeros(int(keyLen / segLen), dtype=int)
        minEpiIndClosenessLsn3 = np.zeros(int(keyLen / segLen), dtype=int)
        minEpiIndClosenessLsn4 = np.zeros(int(keyLen / segLen), dtype=int)

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
            epiIndClosenessLse1 = np.zeros(int(keyLen / segLen))
            epiIndClosenessLse2 = np.zeros(int(keyLen / segLen))
            epiIndClosenessLsn1 = np.zeros(int(keyLen / segLen))
            epiIndClosenessLsn2 = np.zeros(int(keyLen / segLen))
            epiIndClosenessLsn3 = np.zeros(int(keyLen / segLen))
            epiIndClosenessLsn4 = np.zeros(int(keyLen / segLen))

            for j in range(int(keyLen / segLen)):
                epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
                epiInde1 = tmpCSIe1Ind[j * segLen: (j + 1) * segLen]
                epiInde2 = tmpCSIe2Ind[j * segLen: (j + 1) * segLen]
                epiIndn1 = tmpCSIn1Ind[j * segLen: (j + 1) * segLen]
                epiIndn2 = tmpCSIn2Ind[j * segLen: (j + 1) * segLen]
                epiIndn3 = tmpCSIn3Ind[j * segLen: (j + 1) * segLen]
                epiIndn4 = tmpCSIn4Ind[j * segLen: (j + 1) * segLen]

                epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))
                epiIndClosenessLse1[j] = sum(abs(epiInde1 - np.array(epiInda1)))
                epiIndClosenessLse2[j] = sum(abs(epiInde2 - np.array(epiInda1)))
                epiIndClosenessLsn1[j] = sum(abs(epiIndn1 - np.array(epiInda1)))
                epiIndClosenessLsn2[j] = sum(abs(epiIndn2 - np.array(epiInda1)))
                epiIndClosenessLsn3[j] = sum(abs(epiIndn3 - np.array(epiInda1)))
                epiIndClosenessLsn4[j] = sum(abs(epiIndn4 - np.array(epiInda1)))

            minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)
            minEpiIndClosenessLse1[i] = np.argmin(epiIndClosenessLse1)
            minEpiIndClosenessLse2[i] = np.argmin(epiIndClosenessLse2)
            minEpiIndClosenessLsn1[i] = np.argmin(epiIndClosenessLsn1)
            minEpiIndClosenessLsn2[i] = np.argmin(epiIndClosenessLsn2)
            minEpiIndClosenessLsn3[i] = np.argmin(epiIndClosenessLsn3)
            minEpiIndClosenessLsn4[i] = np.argmin(epiIndClosenessLsn4)

        # a_list_number = list(range(int(keyLen / segLen)))
        a_list_number = list(permutation)
        b_list_number = list(minEpiIndClosenessLsb)
        e1_list_number = list(minEpiIndClosenessLse1)
        e2_list_number = list(minEpiIndClosenessLse2)
        n1_list_number = list(minEpiIndClosenessLsn1)
        n2_list_number = list(minEpiIndClosenessLsn2)
        n3_list_number = list(minEpiIndClosenessLsn3)
        # eavesdropping attack
        n4_list_number = list(range(len(a_list_number)))
        np.random.shuffle(n4_list_number)

        # mhd：绝对值距离之和除以密钥长度
        mhd_e1_dist += sum(abs(np.array(a_list_number) - np.array(e1_list_number))) / int(keyLen / segLen)
        mhd_e2_dist += sum(abs(np.array(a_list_number) - np.array(e2_list_number))) / int(keyLen / segLen)
        mhd_n1_dist += sum(abs(np.array(a_list_number) - np.array(n1_list_number))) / int(keyLen / segLen)
        mhd_n2_dist += sum(abs(np.array(a_list_number) - np.array(n2_list_number))) / int(keyLen / segLen)
        mhd_n3_dist += sum(abs(np.array(a_list_number) - np.array(n3_list_number))) / int(keyLen / segLen)
        mhd_n4_dist += sum(abs(np.array(a_list_number) - np.array(n4_list_number))) / int(keyLen / segLen)

        # distance：与真实值索引偏移距离之和除以密钥长度
        tmp_e1_dist = 0
        for i in range(len(e1_list_number)):
            real_pos = search(a_list_number, e1_list_number[i])
            guess_pos = i
            tmp_e1_dist += abs(real_pos - guess_pos)
        distance_e1 += (tmp_e1_dist / int(keyLen / segLen))

        tmp_e2_dist = 0
        for i in range(len(e2_list_number)):
            real_pos = search(a_list_number, e2_list_number[i])
            guess_pos = i
            tmp_e2_dist += abs(real_pos - guess_pos)
        distance_e2 += (tmp_e2_dist / int(keyLen / segLen))

        tmp_n1_dist = 0
        for i in range(len(n1_list_number)):
            real_pos = search(a_list_number, n1_list_number[i])
            guess_pos = i
            tmp_n1_dist += abs(real_pos - guess_pos)
        distance_n1 += (tmp_n1_dist / int(keyLen / segLen))

        tmp_n2_dist = 0
        for i in range(len(n2_list_number)):
            real_pos = search(a_list_number, n2_list_number[i])
            guess_pos = i
            tmp_n2_dist += abs(real_pos - guess_pos)
        distance_n2 += (tmp_n2_dist / int(keyLen / segLen))

        tmp_n3_dist = 0
        for i in range(len(n3_list_number)):
            real_pos = search(a_list_number, n3_list_number[i])
            guess_pos = i
            tmp_n3_dist += abs(real_pos - guess_pos)
        distance_n3 += (tmp_n3_dist / int(keyLen / segLen))

        tmp_n4_dist = 0
        for i in range(len(n4_list_number)):
            real_pos = search(a_list_number, n4_list_number[i])
            guess_pos = i
            tmp_n4_dist += abs(real_pos - guess_pos)
        distance_n4 += (tmp_n4_dist / int(keyLen / segLen))

    print("与真实值索引偏移距离之和除以密钥长度")
    print("e1", round(distance_e1 / times, 8))
    print("e2", round(distance_e2 / times, 8))
    print("n1", round(distance_n1 / times, 8))
    print("n2", round(distance_n2 / times, 8))
    print("n3", round(distance_n3 / times, 8))
    print("n4", round(distance_n4 / times, 8))

    print("绝对值距离之和除以密钥长度")
    print("e1", round(mhd_e1_dist / times, 8))
    print("e2", round(mhd_e2_dist / times, 8))
    print("n1", round(mhd_n1_dist / times, 8))
    print("n2", round(mhd_n2_dist / times, 8))
    print("n3", round(mhd_n3_dist / times, 8))
    print("n4", round(mhd_n4_dist / times, 8))