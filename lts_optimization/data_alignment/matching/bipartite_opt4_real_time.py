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
    signal_power = np.sum(origin ** 2) / dataLen
    noise_power = np.sum(noise ** 2) / dataLen
    noise_variance = signal_power / (10 ** (SNR / 10))
    noise = noise * np.sqrt(noise_variance / noise_power)
    return origin + noise, noise


intvl = 7
keyLen = 128

# 原始数据
CSIa1Orig = loadmat('../original/mo_r.mat')['A'][:, 0]
CSIb1Orig = loadmat('../original/mo_r.mat')['CSI'][:, 0]
# 扰动数据对齐
# CSIa1Orig = loadmat('CSI_5_sir.mat')['csi'][:, 0]
# CSIb1Orig = loadmat('CSI_5_sir.mat')['csi'][:, 1]
# 扰动数据手动对齐
# CSIa1Orig = loadmat('CSI_5_si_r_hand.mat')['A'][:, 0]
# CSIb1Orig = loadmat('CSI_5_si_r_hand.mat')['CSI'][:, 0]

for i in range(int(len(CSIa1Orig) / 128 / 7)):
    print(pearsonr(CSIa1Orig[i * 128 * 7:(i + 1) * 128 * 7], CSIb1Orig[i * 128 * 7:(i + 1) * 128 * 7])[0])

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

entropyThres = 2
CSIa1Orig, CSIb1Orig = splitEntropyPerm(CSIa1Orig, CSIb1Orig, 6, dataLen, entropyThres)

CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()

times = 0

originSum = 0
correctSum = 0
originWholeSum = 0
correctWholeSum = 0
topNum = keyLen
overhead = 0

print("topNum", topNum)
print("sample number", len(CSIa1Orig))

keyA = []
keyB = []

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

    keyA.extend(a_bits)
    keyB.extend(b_bits)

print("\033[0;34;40ma-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10), "\033[0m")
print("\033[0;34;40ma-b whole match", correctWholeSum, "/", originWholeSum, "=",
      round(correctWholeSum / originWholeSum, 10), "\033[0m")
print("times", times)
print(round(correctSum / originSum, 10), round(correctWholeSum / originWholeSum, 10),
      originSum / times / keyLen / intvl,
      correctSum / times / keyLen / intvl)
# messagebox.showinfo("提示", "测试结束")

keyListA = []
keyListB = []
for i in range(len(keyA)):
    keyListA.append(int(keyA[i]))

for i in range(len(keyB)):
    keyListB.append(int(keyB[i]))
print("entropy of a:", ent.shannon_entropy(keyListA))
print("entropy of b:", ent.shannon_entropy(keyListB))

print("approx. entropy of a:", eh.ApEn(keyListA)[0][0])
print("approx. entropy of b:", eh.ApEn(keyListB)[0][0])
