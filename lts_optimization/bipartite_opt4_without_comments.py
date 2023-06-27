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


def addNoise(origin, SNR):
    dataLen = len(origin)
    noise = np.random.normal(0, 1, size=dataLen)
    signal_power = np.sum(origin ** 2) / dataLen
    noise_power = np.sum(noise ** 2) / dataLen
    noise_variance = signal_power / (10 ** (SNR / 10))
    noise = noise * np.sqrt(noise_variance / noise_power)
    return origin + noise, noise


# 方法三：不固定噪音生成的种子，改变噪音（final algorithm）

dataLen = 10000
SNR = 10
channel = np.random.normal(0, 1, size=dataLen)

CSIa1Orig = np.array(channel)

CSIa1OrigBack = CSIa1Orig.copy()

intvl = 7
keyLen = 128

overhead = 0

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
    newCSIa1.extend(addNoise(tmpCSIa1[0: intvl], SNR)[0])
    newCSIb1 = []
    newCSIb1.extend(addNoise(tmpCSIa1[0: intvl], SNR)[0])
    for i in range(1, keyLen):
        lts_noise = np.ones(intvl)
        cnt = 0
        while True:
            rowDists = []
            epiInda1 = tmpCSIa1[i * intvl: (i + 1) * intvl].copy()
            epiInda1 *= lts_noise
            epiInda1, channel_noise_2 = addNoise(epiInda1, SNR)
            for j in range(0, i):
                epiInda2 = tmpCSIa1[j * intvl: (j + 1) * intvl].copy()
                epiInda2, channel_noise_1 = addNoise(epiInda2, SNR)
                epiInda2 *= lts[j]
                rowDists.append(sum(abs(epiInda1 - epiInda2)))

            minDist = min(rowDists)

            epiIndb1 = tmpCSIa1[i * intvl: (i + 1) * intvl].copy()
            epiIndb1, channel_noise_1 = addNoise(epiIndb1, SNR)
            epiIndb1 *= lts_noise
            threshold = sum(abs(epiInda1 - epiIndb1))

            if minDist > 2 * threshold:
                newCSIa1.extend(addNoise(epiInda1 - channel_noise_2, SNR)[0])
                newCSIb1.extend(epiIndb1)
                lts.append(lts_noise)
                break
            else:
                #  记录修改的次数
                if cnt == 0:
                    changed += 1
                cnt += 1
                lts_noise = np.random.normal(0, 10, size=intvl)

    modifiedCSIa1.extend(newCSIa1)
    modifiedCSIb1.extend(newCSIb1)

########################################################################################################################

CSIa1Orig = np.array(modifiedCSIa1)
CSIb1Orig = np.array(modifiedCSIb1)

CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()

times = 0

originSum = 0
correctSum = 0
originWholeSum = 0
correctWholeSum = 0
topNum = 16
overhead = 0

print("topNum", topNum)
print("sample number", len(CSIa1Orig))

for staInd in range(0, len(CSIa1Orig), intvl * keyLen):
    processTime = time.time()

    endInd = staInd + keyLen * intvl
    print("range:", staInd, endInd)
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
    np.random.seed(staInd)
    permInd = np.random.permutation(permLen)
    permOrigInd = newOrigInd[permInd]

    # Main: Weighted bipartite maximum matching
    edges = []
    edgesn = []
    matchSort = []

    start = time.time()
    for ii in range(permLen):
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
    print("--- processTime %s seconds ---" % (time.time() - processTime))

    matchTime = time.time()

    # key agreement
    neg_edges = [(i, j, -wt) for i, j, wt in edges]
    match = maxWeightMatching(neg_edges, maxcardinality=True)

    matchb = [j - permLen for (i, j, wt) in neg_edges if match[i] == j]
    print("--- matchTime %s seconds ---" % (time.time() - matchTime))

    overhead += time.time() - start

    a_list = permInd
    b_list = matchb

    # 转化为keyLen长的bits
    a_bits = ""
    b_bits = ""

    # 转成二进制，0填充成0000
    for i in range(len(permInd)):
        number = bin(permInd[i])[2:].zfill(int(np.log2(len(permInd))))
        a_bits += number
    for i in range(len(matchb)):
        number = bin(matchb[i])[2:].zfill(int(np.log2(len(matchb))))
        b_bits += number

    # print("keys of a:", len(a_list), a_list)
    # print("keys of a:", len(a_bits), a_bits)
    # print("keys of b:", len(b_list), b_list)
    # print("keys of b:", len(b_bits), b_bits)

    sum1 = min(len(a_bits), len(b_bits))
    sum2 = 0
    for i in range(0, sum1):
        sum2 += (a_bits[i] == b_bits[i])
    if sum2 == sum1:
        print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
    else:
        print("\033[0;31;40ma-b", sum2, sum2 / sum1, "\033[0m")

    originSum += sum1
    correctSum += sum2

    originWholeSum += 1
    correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum

print("\033[0;34;40ma-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10), "\033[0m")
print("\033[0;34;40ma-b whole match", correctWholeSum, "/", originWholeSum, "=",
      round(correctWholeSum / originWholeSum, 10), "\033[0m")
print("times", times)
print(round(correctSum / originSum, 10), round(correctWholeSum / originWholeSum, 10),
      originSum / times / keyLen / intvl,
      correctSum / times / keyLen / intvl)
# messagebox.showinfo("提示", "测试结束")