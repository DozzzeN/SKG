import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import exponential as Exp
from scipy.io import loadmat
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

from mwmatching import maxWeightMatching
from pyentrp import entropy as ent
import EntropyHub as eh


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
    n = len(sortCSIa1Reshape)

    for i in range(n):
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


def addNoise(origin, SNR):
    dataLen = len(origin)
    noise = np.random.normal(0, 1, size=dataLen)
    signal_power = np.sum(np.power(origin, 2)) / dataLen
    noise_power = np.sum(noise ** 2) / dataLen
    noise_variance = signal_power / (10 ** (SNR / 10))
    noise = noise * np.sqrt(noise_variance / noise_power)
    return origin + noise, noise

N = 6
intvl = N * 3 * 3
keyLen = 256

fileNames = ["../original/CSI_3_sir_allr.mat",
             "../original/CSI_5_sor(1).mat",
             "../original/CSI_5_sor.mat",
             "../original/CSI_7_mor.mat",
             "../original/CSI_7_mo.mat",
             "../original/CSI_6_mir.mat",
             "../original/CSI_6_mi.mat"]

for fileName in fileNames:
    CSIa1Orig = loadmat(fileName)['csi'][:, 0]
    CSIb1Orig = loadmat(fileName)['csi'][:, 1]
    if fileName == "../original/CSI_5_sor(1).mat" or fileName == "../original/CSI_5_sor.mat":
        # for CSI_5_sor(1).mat and CSI_5_sor.mat
        CSIa1Orig = np.tile(CSIa1Orig, 20)
        CSIb1Orig = np.tile(CSIb1Orig, 20)

    # for i in range(int(len(CSIa1Orig) / keyLen / intvl)):
    #     print(pearsonr(CSIa1Orig[i * keyLen * intvl:(i + 1) * keyLen * intvl],
    #                    CSIb1Orig[i * keyLen * intvl:(i + 1) * keyLen * intvl])[0])
    dataLen = len(CSIa1Orig)
    print(dataLen)
    CSIb1Orig = CSIb1Orig - (np.mean(CSIb1Orig) - np.mean(CSIa1Orig))

    # 固定随机置换的种子
    np.random.seed(1)  # 8 1024 8; 4 128 4
    combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig))
    np.random.shuffle(combineCSIx1Orig)
    CSIa1Orig, CSIb1Orig = zip(*combineCSIx1Orig)

    CSIa1Orig = np.array(CSIa1Orig)
    CSIb1Orig = np.array(CSIb1Orig)

    # entropyThres = 2
    # CSIa1Orig, CSIb1Orig = splitEntropyPerm(CSIa1Orig, CSIb1Orig, 6, dataLen, entropyThres)

    CSIa1OrigBack = CSIa1Orig.copy()
    CSIb1OrigBack = CSIb1Orig.copy()

    times = 0

    originSum = 0
    correctSum = 0
    originWholeSum = 0
    correctWholeSum = 0
    overhead = 0

    print("sample number", len(CSIa1Orig))
    corrs = []
    for staInd in range(0, len(CSIa1Orig), intvl * keyLen):
        processTime = time.time()

        endInd = staInd + keyLen * intvl
        # print("range:", staInd, endInd)
        if endInd >= len(CSIa1Orig):
            break
        times += 1

        CSIa1Orig = CSIa1OrigBack.copy()
        CSIb1Orig = CSIb1OrigBack.copy()

        permLen = len(range(staInd, endInd, intvl))
        origInd = np.array([xx for xx in range(staInd, endInd, 1)])

        CSIa1Epi = CSIa1Orig[origInd]
        CSIb1Epi = CSIb1Orig[origInd]

        CSIa1Orig[origInd] = CSIa1Epi
        CSIb1Orig[origInd] = CSIb1Epi

        CSIa1OrigReshape = np.array(CSIa1Orig)[staInd: endInd].reshape(3, keyLen, int(intvl / 3 / N), N)
        CSIb1OrigReshape = np.array(CSIb1Orig)[staInd: endInd].reshape(3, keyLen, int(intvl / 3 / N), N)

        CSIa1Ga = CSIa1OrigReshape[0]
        CSIa1Gb = CSIa1OrigReshape[1]
        CSIa1Gc = CSIa1OrigReshape[2]

        CSIb1Ga = CSIb1OrigReshape[0]
        CSIb1Gb = CSIb1OrigReshape[1]
        CSIb1Gc = CSIb1OrigReshape[2]

        # SVD decomposition
        # 选第二第三个特征值效果差，故选第一和第二个
        CSIa1SigmaA2 = []
        CSIa1SigmaA3 = []
        for i in range(len(CSIa1Ga)):
            CSIa1SigmaA2.append(np.linalg.svd(CSIa1Ga[i])[1][0])
            CSIa1SigmaA3.append(np.linalg.svd(CSIa1Ga[i])[1][1])
        CSIa1SigmaB2 = []
        CSIa1SigmaB3 = []
        for i in range(len(CSIa1Gb)):
            CSIa1SigmaB2.append(np.linalg.svd(CSIa1Gb[i])[1][0])
            CSIa1SigmaB3.append(np.linalg.svd(CSIa1Gb[i])[1][1])
        CSIa1SigmaC2 = []
        CSIa1SigmaC3 = []
        for i in range(len(CSIa1Gc)):
            CSIa1SigmaC2.append(np.linalg.svd(CSIa1Gc[i])[1][0])
            CSIa1SigmaC3.append(np.linalg.svd(CSIa1Gc[i])[1][1])
        CSIb1SigmaA2 = []
        CSIb1SigmaA3 = []
        for i in range(len(CSIb1Ga)):
            CSIb1SigmaA2.append(np.linalg.svd(CSIb1Ga[i])[1][0])
            CSIb1SigmaA3.append(np.linalg.svd(CSIb1Ga[i])[1][1])

        CSIb1SigmaB2 = []
        CSIb1SigmaB3 = []
        for i in range(len(CSIb1Gb)):
            CSIb1SigmaB2.append(np.linalg.svd(CSIb1Gb[i])[1][0])
            CSIb1SigmaB3.append(np.linalg.svd(CSIb1Gb[i])[1][1])
        CSIb1SigmaC2 = []
        CSIb1SigmaC3 = []
        for i in range(len(CSIb1Gc)):
            CSIb1SigmaC2.append(np.linalg.svd(CSIb1Gc[i])[1][0])
            CSIb1SigmaC3.append(np.linalg.svd(CSIb1Gc[i])[1][1])

        # feature extraction with length of permLen
        # A's bit of 0
        CSIa1DeltaSigma02 = np.array(CSIa1SigmaB2) - np.array(CSIa1SigmaA2)
        CSIa1DeltaSigma03 = np.array(CSIa1SigmaB3) - np.array(CSIa1SigmaA3)
        # A's bit of 1
        CSIa1DeltaSigma12 = np.array(CSIa1SigmaC2) - np.array(CSIa1SigmaB2)
        CSIa1DeltaSigma13 = np.array(CSIa1SigmaC3) - np.array(CSIa1SigmaB3)
        # B's bit of 0
        CSIb1DeltaSigma02 = np.array(CSIb1SigmaB2) - np.array(CSIb1SigmaA2)
        CSIb1DeltaSigma03 = np.array(CSIb1SigmaB3) - np.array(CSIb1SigmaA3)
        # B's bit of 1
        CSIb1DeltaSigma12 = np.array(CSIb1SigmaC2) - np.array(CSIb1SigmaB2)
        CSIb1DeltaSigma13 = np.array(CSIb1SigmaC3) - np.array(CSIb1SigmaB3)

        corrs.append(pearsonr(CSIa1DeltaSigma02, CSIb1DeltaSigma02)[0])
        corrs.append(pearsonr(CSIa1DeltaSigma03, CSIb1DeltaSigma03)[0])
        corrs.append(pearsonr(CSIa1DeltaSigma12, CSIb1DeltaSigma12)[0])
        corrs.append(pearsonr(CSIa1DeltaSigma13, CSIb1DeltaSigma13)[0])

        # feature stack with length of 2 * permLen
        CSIa1G0 = list(zip(CSIa1DeltaSigma02, CSIa1DeltaSigma03))
        CSIa1G1 = list(zip(CSIa1DeltaSigma12, CSIa1DeltaSigma13))
        CSIb1G0 = list(zip(CSIb1DeltaSigma02, CSIb1DeltaSigma03))
        CSIb1G1 = list(zip(CSIb1DeltaSigma12, CSIb1DeltaSigma13))

        start = time.time()
        matchTime = time.time()

        # feature pairing
        # matchA = []
        # matchB = []

        # for i in range(permLen):
        #     if np.sum(np.abs(CSIa1G0[i][1] - CSIa1G1[i][1])) >= np.sum(np.abs(CSIa1G0[i][0] - CSIa1G1[i][0])):
        #         matchA.append([0, 1])
        #     else:
        #         matchA.append([1, 0])
        #
        #     if np.sum(np.abs(CSIb1G0[i][1] - CSIb1G1[i][1])) >= np.sum(np.abs(CSIb1G0[i][0] - CSIb1G1[i][0])):
        #         matchB.append([0, 1])
        #     else:
        #         matchB.append([1, 0])

        # print("--- matchTime %s seconds ---" % (time.time() - matchTime))
        overhead += time.time() - start

        # random permutation
        keyA = np.random.binomial(1, 0.5, permLen)
        featureA = []
        # 效果差
        # match = np.random.binomial(1, 0.5, (permLen, 2))
        # for i in range(permLen):
        #     if keyA[i] == 0:
        #         featureA.append(np.array([CSIa1G0[i][match[i][0]], CSIa1G1[i][match[i][1]]]))
        #     else:
        #         featureA.append(np.array([CSIa1G0[i][match[i][1]], CSIa1G1[i][match[i][0]]]))
        #
        # keyB = []
        # for i in range(permLen):
        #     featureB0 = np.array([CSIb1G0[i][match[i][0]], CSIb1G1[i][match[i][1]]])
        #     featureB1 = np.array([CSIb1G0[i][match[i][1]], CSIb1G1[i][match[i][0]]])
        #     if np.sum(np.abs(featureB0 - featureA[i])) < np.sum(np.abs(featureB1 - featureA[i])):
        #         keyB.append(0)
        #     else:
        #         keyB.append(1)

        for i in range(permLen):
            if keyA[i] == 0:
                featureA.append(np.array([CSIa1G0[i][0], CSIa1G1[i][1]]))
            else:
                featureA.append(np.array([CSIa1G0[i][1], CSIa1G1[i][0]]))

        keyB = []
        for i in range(permLen):
            featureB0 = np.array([CSIb1G0[i][0], CSIb1G1[i][1]])
            featureB1 = np.array([CSIb1G0[i][1], CSIb1G1[i][0]])
            if np.sum(np.abs(featureB0 - featureA[i])) < np.sum(np.abs(featureB1 - featureA[i])):
                keyB.append(0)
            else:
                keyB.append(1)

        # print(len(keyA), list(keyA))
        # print(len(keyB), list(keyB))

        sum1 = min(len(keyA), len(keyB))
        sum2 = 0
        for i in range(0, sum1):
            sum2 += (keyA[i] == keyB[i])
        # if sum2 == sum1:
        #     print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
        # else:
        #     print("\033[0;31;40ma-b", sum2, sum2 / sum1, "\033[0m")

        originSum += sum1
        correctSum += sum2

        originWholeSum += 1
        correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum

    print("\033[0;34;40ma-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10), "\033[0m")
    print("\033[0;34;40ma-b whole match", correctWholeSum, "/", originWholeSum, "=",
          round(correctWholeSum / originWholeSum, 10), "\033[0m")
    print("times", times)
    # 考虑到重用，BGR会double
    print(round(correctSum / originSum, 10), round(correctWholeSum / originWholeSum, 10),
          2 * round(originSum / times / keyLen / intvl, 10),
          2 * round(correctSum / times / keyLen / intvl, 10))
    print("fileName", fileName)
    # messagebox.showinfo("提示", "测试结束")
    print("corr", np.mean(corrs))
    # keyListA = []
    # keyListB = []
    # for i in range(len(keyA)):
    #     keyListA.append(int(keyA[i]))

    # for i in range(len(keyB)):
    #     keyListB.append(int(keyB[i]))
    # print("entropy of a:", ent.shannon_entropy(keyListA))
    # print("entropy of b:", ent.shannon_entropy(keyListB))
    #
    # print("approx. entropy of a:", eh.ApEn(keyListA)[0][0])
    # print("approx. entropy of b:", eh.ApEn(keyListB)[0][0])
