import time

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


def splitEntropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, segLen, dataLen, entropyThres):
    # 先整体shuffle一次
    shuffleInd = np.random.permutation(dataLen)
    CSIa1Orig = CSIa1Orig[shuffleInd]
    CSIb1Orig = CSIb1Orig[shuffleInd]
    CSIe1Orig = CSIe1Orig[shuffleInd]

    sortCSIa1Reshape = CSIa1Orig[0:segLen * int(dataLen / segLen)]
    sortCSIb1Reshape = CSIb1Orig[0:segLen * int(dataLen / segLen)]
    sortCSIe1Reshape = CSIe1Orig[0:segLen * int(dataLen / segLen)]

    sortCSIa1Reshape = sortCSIa1Reshape.reshape(int(len(sortCSIa1Reshape) / segLen), segLen)
    sortCSIb1Reshape = sortCSIb1Reshape.reshape(int(len(sortCSIb1Reshape) / segLen), segLen)
    sortCSIe1Reshape = sortCSIe1Reshape.reshape(int(len(sortCSIe1Reshape) / segLen), segLen)
    n = len(sortCSIa1Reshape)

    for i in range(n):
        a_mul_entropy = ent.multiscale_entropy(sortCSIa1Reshape[i], 3, maxscale=1)

        cnts = 0
        while a_mul_entropy < entropyThres and cnts < 10:
            shuffleInd = np.random.permutation(len(sortCSIa1Reshape[i]))
            sortCSIa1Reshape[i] = sortCSIa1Reshape[i][shuffleInd]
            sortCSIb1Reshape[i] = sortCSIb1Reshape[i][shuffleInd]
            sortCSIe1Reshape[i] = sortCSIe1Reshape[i][shuffleInd]

            a_mul_entropy = ent.multiscale_entropy(sortCSIa1Reshape[i], 3, maxscale=1)
            cnts += 1

    _CSIa1Orig = []
    _CSIb1Orig = []
    _CSIe1Orig = []

    for i in range(len(sortCSIa1Reshape)):
        for j in range(len(sortCSIa1Reshape[i])):
            _CSIa1Orig.append(sortCSIa1Reshape[i][j])
            _CSIb1Orig.append(sortCSIb1Reshape[i][j])
            _CSIe1Orig.append(sortCSIe1Reshape[i][j])

    return np.array(_CSIa1Orig), np.array(_CSIb1Orig), np.array(_CSIe1Orig)


# fileName = "../data/data_static_indoor_1_r.mat"
# rawData = loadmat(fileName)
# csv = open("../edit_distance/evaluations/bipartite.csv", "a+")

CSIa1Orig = loadmat("predictable/CSI1.mat")['CSI1'][:, 0]
CSIb1Orig = CSIa1Orig + np.random.normal(0, 0.01, len(CSIa1Orig))
dataLen = len(CSIa1Orig)
CSIb1Orig = CSIb1Orig - (np.mean(CSIb1Orig) - np.mean(CSIa1Orig))

CSIn1Orig = np.random.normal(loc=-1, scale=1, size=dataLen)  ## Multiplication item normal distribution
CSIe1Orig = loadmat("predictable/CSI2.mat")['CSI2'][:, 0]
dataLen = min(dataLen, len(CSIe1Orig))
CSIa1Orig = CSIa1Orig[0:dataLen]
CSIb1Orig = CSIb1Orig[0:dataLen]
CSIe1Orig = CSIe1Orig[0:dataLen]
CSIn1Orig = CSIn1Orig[0:dataLen]

entropyThres = 2
CSIa1Orig, CSIb1Orig, CSIe1Orig = splitEntropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, 5, dataLen, entropyThres)
# CSIa1Orig, CSIb1Orig = entropyPerm(CSIa1Orig, CSIb1Orig, dataLen, entropyThres)

CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
CSIe1OrigBack = CSIe1Orig.copy()
CSIn1OrigBack = CSIn1Orig.copy()

intvl = 6
keyLen = 128
times = 0

originSum = 0
correctSum = 0
randomSum = 0
noiseSum = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum = 0
noiseWholeSum = 0

topNum = keyLen

for staInd in range(0, len(CSIa1Orig), intvl * keyLen):
    endInd = staInd + keyLen * intvl
    print("range:", staInd, endInd)
    if endInd >= len(CSIa1Orig):
        break
    times += 1

    CSIa1Orig = CSIa1OrigBack.copy()
    CSIb1Orig = CSIb1OrigBack.copy()
    CSIe1Orig = CSIe1OrigBack.copy()
    CSIn1Orig = CSIn1OrigBack.copy()

    permLen = len(range(staInd, endInd, intvl))
    origInd = np.array([xx for xx in range(staInd, endInd, 1)])

    CSIa1Epi = CSIa1Orig[origInd]
    CSIb1Epi = CSIb1Orig[origInd]
    CSIe1Epi = CSIe1Orig[origInd]
    CSIn1Epi = CSIn1Orig[origInd]

    CSIa1Orig[origInd] = CSIa1Epi
    CSIb1Orig[origInd] = CSIb1Epi
    CSIe1Orig[origInd] = CSIe1Epi
    CSIn1Orig[origInd] = CSIn1Epi

    # Random permutation
    newOrigInd = np.array([xx for xx in range(staInd, endInd, intvl)])
    permInd = np.random.permutation(permLen)  ## KEY
    permOrigInd = newOrigInd[permInd]

    ## Main: Weighted biparitite maximum matching
    edgesb = []
    edgese = []
    edgesn = []
    matchSort = []

    for ii in range(permLen):
        coefLs = []
        aIndVec = np.array([aa for aa in range(permOrigInd[ii], permOrigInd[ii] + intvl, 1)])  ## for permuted CSIa1

        distLsb = []
        distLse = []
        distLsn = []
        edgesTmpb = []
        edgesTmpe = []
        edgesTmpn = []

        for jj in range(permLen, permLen * 2):
            bIndVec = np.array([bb for bb in range(newOrigInd[jj - permLen], newOrigInd[jj - permLen] + intvl, 1)])

            CSIa1Tmp = CSIa1Orig[aIndVec]
            CSIb1Tmp = CSIb1Orig[bIndVec]
            CSIe1Tmp = CSIe1Orig[bIndVec]
            CSIn1Tmp = CSIn1Orig[bIndVec]

            distpara = 'cityblock'
            distb = pdist(np.vstack((CSIa1Tmp, CSIb1Tmp)), distpara)
            diste = pdist(np.vstack((CSIa1Tmp, CSIe1Tmp)), distpara)
            distn = pdist(np.vstack((CSIa1Tmp, CSIn1Tmp)), distpara)

            distLsb.append(distb[0])
            edgesTmpb.append([ii, jj, distb[0]])

            distLse.append(diste[0])
            edgesTmpe.append([ii, jj, diste[0]])

            distLsn.append(distn[0])
            edgesTmpn.append([ii, jj, distn[0]])

        sortIndb = np.argsort(distLsb)  ## Increasing order
        topIndb = sortIndb[0:topNum]
        for kk in topIndb:
            edgesb.append(edgesTmpb[kk])

        sortInde = np.argsort(distLse)  ## Increasing order
        topInde = sortInde[0:topNum]
        for kk in topInde:
            edgese.append(edgesTmpe[kk])

        sortIndn = np.argsort(distLsn)  ## Increasing order
        topIndn = sortIndn[0:topNum]
        for kk in topIndn:
            edgesn.append(edgesTmpn[kk])

        if topNum == 1:
            matchSort.append(topIndb[0])

    # key agreement
    neg_edgesb = [(i, j, -wt) for i, j, wt in edgesb]
    match_b = maxWeightMatching(neg_edgesb, maxcardinality=True)

    neg_edgese = [(i, j, -wt) for i, j, wt in edgese]
    match_e = maxWeightMatching(neg_edgese, maxcardinality=True)

    neg_edgesn = [(i, j, -wt) for i, j, wt in edgesn]
    match_n = maxWeightMatching(neg_edgesn, maxcardinality=True)

    matchb = [j - permLen for (i, j, wt) in neg_edgesb if match_b[i] == j]
    matche = [j - permLen for (i, j, wt) in neg_edgese if match_e[i] == j]
    matchn = [j - permLen for (i, j, wt) in neg_edgesn if match_n[i] == j]

    a_list = ""
    b_list = ""
    e_list = ""
    n_list = ""

    # 转为二进制
    for i in range(len(permInd)):
        a_list += bin(permInd[i])[2:]
    for i in range(len(matchb)):
        b_list += bin(matchb[i])[2:]
    for i in range(len(matche)):
        e_list += bin(matche[i])[2:]
    for i in range(len(matchn)):
        n_list += bin(matchn[i])[2:]

    # 转化为keyLen长的bits
    a_bits = ""
    b_bits = ""
    e_bits = ""
    n_bits = ""

    rounda = int(len(a_list) / keyLen)
    remaina = len(a_list) - rounda * keyLen
    if remaina >= 0:
        rounda += 1

    for i in range(keyLen):
        tmp = 0
        for j in range(rounda):
            if j * keyLen + i >= len(a_list):
                continue
            tmp += int(a_list[j * keyLen + i])
        a_bits += str(tmp % 2)

    roundb = int(len(b_list) / keyLen)
    remainb = len(b_list) - roundb * keyLen
    if remainb >= 0:
        roundb += 1

    for i in range(keyLen):
        tmp = 0
        for j in range(roundb):
            if j * keyLen + i >= len(b_list):
                continue
            tmp += int(b_list[j * keyLen + i])
        b_bits += str(tmp % 2)

    rounde = int(len(e_list) / keyLen)
    remaine = len(e_list) - rounde * keyLen
    if remaine >= 0:
        rounde += 1

    for i in range(keyLen):
        tmp = 0
        for j in range(rounde):
            if j * keyLen + i >= len(e_list):
                continue
            tmp += int(e_list[j * keyLen + i])
        e_bits += str(tmp % 2)

    roundn = int(len(n_list) / keyLen)
    remainn = len(n_list) - roundn * keyLen
    if remainn >= 0:
        roundn += 1

    for i in range(keyLen):
        tmp = 0
        for j in range(roundn):
            if j * keyLen + i >= len(n_list):
                continue
            tmp += int(n_list[j * keyLen + i])
        n_bits += str(tmp % 2)

    print("keys of a:", len(a_list), a_list)
    print("keys of a:", len(a_bits), a_bits)
    print("keys of b:", len(b_list), b_list)
    print("keys of b:", len(b_bits), b_bits)
    print("keys of e:", len(e_list), e_list)
    print("keys of e:", len(e_bits), e_bits)
    print("keys of n:", len(n_list), n_list)
    print("keys of n:", len(n_bits), n_bits)

    sum1 = min(len(a_bits), len(b_bits))
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for i in range(0, sum1):
        sum2 += (a_bits[i] == b_bits[i])
    for i in range(min(len(a_bits), len(e_bits))):
        sum3 += (a_bits[i] == e_bits[i])
    for i in range(min(len(a_bits), len(n_bits))):
        sum4 += (a_bits[i] == n_bits[i])
    if sum2 == sum1:
        print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
    else:
        print("\033[0;31;40ma-b", sum2, sum2 / sum1, "\033[0m")
    print("a-e", sum3, sum3 / sum1)
    print("a-n", sum4, sum4 / sum1)
    print("----------------------")

    originSum += sum1
    correctSum += sum2
    randomSum += sum3
    noiseSum += sum4

    originWholeSum += 1
    correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
    randomWholeSum = randomWholeSum + 1 if sum3 == sum1 else randomWholeSum
    noiseWholeSum = noiseWholeSum + 1 if sum4 == sum1 else noiseWholeSum

print("a-b all", correctSum, "/", originSum, "=", correctSum / originSum)
print("a-e all", randomSum, "/", originSum, "=", randomSum / originSum)
print("a-n all", noiseSum, "/", originSum, "=", noiseSum / originSum)
print("a-b whole match", correctWholeSum, "/", originWholeSum, "=", correctWholeSum / originWholeSum)
print("a-e whole match", randomWholeSum, "/", originWholeSum, "=", randomWholeSum / originWholeSum)
print("a-n whole match", noiseWholeSum, "/", originWholeSum, "=", noiseWholeSum / originWholeSum)
print(times)