import time
from tkinter import messagebox

import numpy as np
from numpy.random import exponential as Exp
from pyentrp import entropy as ent
from scipy.io import loadmat
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


# fileNames = ["../data/data_mobile_indoor_1.mat",
#              "../data/data_static_indoor_1.mat",
#              "../data/data_mobile_outdoor_1.mat",
#              "../data/data_static_outdoor_1.mat"
#              ]

fileNames = ["../csi/csi_mobile_indoor_1_r",
            "../csi/csi_static_indoor_1_r",
            "../csi/csi_mobile_outdoor_r",
            "../csi/csi_static_outdoor_r"]


for fileName in fileNames:
    rawData = loadmat(fileName)
    print(fileName)

    if fileName.find("csi") != -1:
        CSIa1Orig = rawData['testdata'][:, 0]
        CSIb1Orig = rawData['testdata'][:, 1]
    else:
        CSIa1Orig = rawData['A'][:, 0]
        CSIb1Orig = rawData['A'][:, 1]

    dataLen = len(CSIa1Orig)
    CSIb1Orig = CSIb1Orig - (np.mean(CSIb1Orig) - np.mean(CSIa1Orig))

    # 固定随机置换的种子
    np.random.seed(1)  # 8 1024 8; 4 128 4
    combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig))
    np.random.shuffle(combineCSIx1Orig)
    CSIa1Orig, CSIb1Orig = zip(*combineCSIx1Orig)

    CSIa1Orig = np.array(CSIa1Orig)
    CSIb1Orig = np.array(CSIb1Orig)

    entropyThres = 2
    CSIa1Orig, CSIb1Orig = splitEntropyPerm(CSIa1Orig, CSIb1Orig, 6, dataLen, entropyThres)
    # CSIa1Orig, CSIb1Orig = entropyPerm(CSIa1Orig, CSIb1Orig, dataLen, entropyThres)

    CSIa1OrigBack = CSIa1Orig.copy()
    CSIb1OrigBack = CSIb1Orig.copy()

    intvl = 1
    keyLen = 128

    times = 0

    originSum = 0
    correctSum = 0
    originWholeSum = 0
    correctWholeSum = 0
    topNum = 128
    overhead = 0

    print("topNum", topNum)
    print("sample number", len(CSIa1Orig))

    for staInd in range(0, int(len(CSIa1Orig) / 2), intvl * keyLen):
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

        # Random permutation
        newOrigInd = np.array([xx for xx in range(staInd, endInd, intvl)])
        permInd = np.random.permutation(permLen)  ## KEY
        permOrigInd = newOrigInd[permInd]

        ## Main: Weighted biparitite maximum matching
        edges = []
        edgesn = []
        matchSort = []

        start = time.time()
        for ii in range(permLen):
            coefLs = []
            aIndVec = np.array([aa for aa in range(permOrigInd[ii], permOrigInd[ii] + intvl, 1)])  ## for permuted CSIa1

            distLs = []
            edgesTmp = []

            for jj in range(permLen, permLen * 2):
                bIndVec = np.array([bb for bb in range(newOrigInd[jj - permLen], newOrigInd[jj - permLen] + intvl, 1)])

                CSIa1Tmp = CSIa1Orig[aIndVec]
                CSIb1Tmp = CSIb1Orig[bIndVec]

                distpara = 'cityblock'
                X = np.vstack((CSIa1Tmp, CSIb1Tmp))
                dist = pdist(X, distpara)

                distLs.append(dist[0])
                edgesTmp.append([ii, jj, dist[0]])

            sortInd = np.argsort(distLs)  ## Increasing order
            topInd = sortInd[0:topNum]
            for kk in topInd:
                edges.append(edgesTmp[kk])

            if topNum == 1:
                matchSort.append(topInd[0])
        # print("--- processTime %s seconds ---" % (time.time() - processTime))

        matchTime = time.time()

        # key agreement
        neg_edges = [(i, j, -wt) for i, j, wt in edges]
        match = maxWeightMatching(neg_edges, maxcardinality=True)

        matchb = [j - permLen for (i, j, wt) in neg_edges if match[i] == j]
        # print("--- matchTime %s seconds ---" % (time.time() - matchTime))

        overhead += time.time() - start

        print(pearsonr(CSIa1Epi[permInd], CSIb1Epi[matchb])[0],
              pearsonr(CSIa1Epi, CSIb1Epi)[0], pearsonr(permInd, matchb)[0])

        a_list = CSIa1Epi[permInd]
        b_list = CSIb1Epi[matchb]
        a_list_number = []
        b_list_number = []

        segLen = 2
        tmpCSIa1Reshape = np.array(a_list).reshape(int(keyLen / segLen), segLen)
        tmpCSIb1Reshape = np.array(b_list).reshape(int(keyLen / segLen), segLen)

        for i in range(int(keyLen / segLen)):
            a_list_number.extend(np.array(np.argsort(tmpCSIa1Reshape[i])).tolist())
            b_list_number.extend(np.array(np.argsort(tmpCSIb1Reshape[i])).tolist())

        a_list_number_back = a_list_number.copy()
        b_list_number_back = b_list_number.copy()
        a_list_number = ""
        b_list_number = ""
        for i in range(keyLen):
            # 转为二进制
            a_list_number += np.binary_repr(int(a_list_number_back[i]), width=int(np.log2(segLen)))
            b_list_number += np.binary_repr(int(b_list_number_back[i]), width=int(np.log2(segLen)))

        # # parameters setting in section of comparison
        # alpha = 0.2
        # for i in range(int(keyLen / segLen)):
        #     dropTmp = []
        #     # rangeA1 = max(tmpCSIa1Reshape[i]) - min(tmpCSIa1Reshape[i])
        #     # nA1 = int(math.log2(rangeA1))
        #     # single bit quantization
        #     q1A1 = np.mean(tmpCSIa1Reshape[i]) + alpha * np.std(tmpCSIa1Reshape[i])
        #     q2A1 = np.mean(tmpCSIa1Reshape[i]) - alpha * np.std(tmpCSIa1Reshape[i])
        #     q1B1 = np.mean(tmpCSIb1Reshape[i]) + alpha * np.std(tmpCSIb1Reshape[i])
        #     q2B1 = np.mean(tmpCSIb1Reshape[i]) - alpha * np.std(tmpCSIb1Reshape[i])
        #     for j in range(segLen):
        #         if tmpCSIa1Reshape[i][j] < q1A1 and tmpCSIa1Reshape[i][j] > q2A1:
        #             dropTmp.append(j)
        #         if tmpCSIb1Reshape[i][j] < q1B1 and tmpCSIb1Reshape[i][j] > q2B1:
        #             dropTmp.append(j)
        #
        #     for j in range(segLen):
        #         if j in dropTmp:
        #             continue
        #         if tmpCSIa1Reshape[i][j] > q1A1:
        #             a_list_number.append(1)
        #         if tmpCSIa1Reshape[i][j] < q2A1:
        #             a_list_number.append(0)
        #         if tmpCSIb1Reshape[i][j] > q1B1:
        #             b_list_number.append(1)
        #         if tmpCSIb1Reshape[i][j] < q2B1:
        #             b_list_number.append(0)

        sum1 = min(len(a_list_number), len(b_list_number))
        sum2 = 0
        for i in range(0, sum1):
            sum2 += (a_list_number[i] == b_list_number[i])
        if sum2 == sum1:
            print(a_list_number)
            print(b_list_number)
            print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
        else:
            print("\033[0;31;40ma-b", sum2, sum2 / sum1, "\033[0m")

        originSum += sum1
        correctSum += sum2

        originWholeSum += 1
        correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum

    print("a-b all", correctSum, "/", originSum, "=", correctSum / originSum)
    print("a-b whole match", correctWholeSum, "/", originWholeSum, "=", correctWholeSum / originWholeSum)
    print(times)
    print(overhead / times)
    print(originSum / len(CSIa1Orig))
    print(correctSum / len(CSIa1Orig))
    print(round(correctSum / originSum, 10), round(correctWholeSum / originWholeSum, 10),
          originSum / times / keyLen / intvl,
          correctSum / times / keyLen / intvl)
messagebox.showinfo("提示", "测试结束")
