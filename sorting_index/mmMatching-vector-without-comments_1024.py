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


# fileName = "../data/data_mobile_indoor_1.mat"
fileName = "../csi/csi_static_indoor_1_r.mat"
rawData = loadmat(fileName)
print(fileName)

# CSIa1Orig = rawData['testdata'][:, 0]
# CSIb1Orig = rawData['testdata'][:, 1]
CSIa1Orig = np.tile(rawData['testdata'][:, 0], 1)
CSIb1Orig = np.tile(rawData['testdata'][:, 1], 1)
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

intvl = 5
keyLen = 1024

# intvl = 7
# keyLen = 512
# 1.0 1.0 1.0 1.0 (0.923235445646574 0.923235445646574)
# 1.0 1.0 1.0 1.0 (0.5314353499406881 0.5314353499406881)
# 0.4995814732 0.0 1.0 0.4995814732142857 (0.9985273631840796 0.49884577114427864)
# 0.9700520833 0.5 1.0 0.9700520833333334 (0.9588014981273407 0.930087390761548)

# intvl = 6
# keyLen = 512
# 1.0 1.0 1.0 1.0 (0.9496136012364761 0.9496136012364761)
# 0.525390625 0.0 1.0 0.525390625 (0.911032028469751 0.4786476868327403)

# intvl = 5
# keyLen = 512
# topNum = 64
# 0.9977058532 0.4285714286 1.8 1.7958705357142857

# intvl = 5
# keyLen = 1024
# mi (topNum=16) 0.9927734375 0.0 2.0 1.985546875 (1.5826893353941268 1.5712519319938175)
# mi (topNum=32) 0.9938802083 0.0 2.0 1.9877604166666667 (1.5826893353941267 1.5730036063884596)
# mi (topNum=8) 0.9904296875 0.0 2.0 1.980859375 (1.5826893353941267 1.5675425038639876)
# csi mi1r (topNum=8) 0.9689453125 0.0 2.0 1.937890625 (1.0660004164064127 1.0328961066000417)
# csi mi1r (topNum=16) 0.975 0.0 2.0 1.95 (1.0660004164064127 1.0393504059962524)
# csi mi1r (topNum=32) 0.975 0.0 2.0 1.95 (1.0660004164064127 1.0393504059962524)
# csi mi1r (topNum=1024) 0.9756510417 0.0 2.0 1.9513020833333332 (1.918800749531543 1.8720799500312304)

# si (topNum=16) 0.805651738 0.0 1.9890830592105264 1.6025082236842105 (1.9253532338308457 1.5511641791044777)
# si (topNum=32) 0.8414370888 0.0 2.0 1.6828741776315792 (1.9359203980099502 1.628955223880597)
# si (topNum=64) 0.8419202303 0.0 2.0 1.6838404605263158 (1.9359203980099502 1.6298905472636815)
# csi sir (topNum=64) 0.979296875 0.0 2.0 1.95859375 (1.3796820264079763 1.3511182969549986)
# csi si1r (topNum=64) 0.7447265625 0.0 2.0 1.489453125 (1.6976127320954908 1.2642572944297081)
# csi si1r (topNum=256) 0.7447265625 0.0 2.0 1.489453125 (1.6976127320954908 1.2642572944297081)
# csi si1r (topNum=1024) 0.8271949405 0.0 1.8 1.4889508928571429 (1.7824933687002653 1.4744694960212201)

# mo (topNum=1024) 0.9537109375 0.0 2.0 1.907421875
# mo (topNum=16) 0.9509765625 0.0 2.0 1.901953125 (1.5183867141162515 1.4439501779359432)
# csi mor (topNum=16) 0.9542480469 0.0 2.0 1.90849609375 (1.6876802637000412 1.6104655953852494)
# csi mor (topNum=64) 0.9547851563 0.0 2.0 1.9095703125 (1.6876802637000412 1.611372064276885)
# csi mor (topNum=1024) 0.9798900463 0.0 1.8 1.7638020833333332 (1.7087762669962918 1.6744128553770086)

# so (topNum=16) 0.7171380615 0.0 1.949072265625 1.39775390625
# so (topNum=1024) 0.9076660156 0.0 2.0 1.81533203125 (1.8262885678616017 1.6576600677724274)
# csi sor (topNum=16) 0.9103515625 0.0 2.0 1.820703125 (1.0368570271364925 0.9439044147428108)
# csi sor (topNum=64) 0.9015407986 0.0 2.0 1.803081597222222 (1.8663426488456865 1.6825840421223168)
# csi sor (topNum=256) 0.9109375 0.0 2.0 1.821875 (1.0368570271364925 0.9445119481571487)
# csi sor (topNum=1024) 0.9454495614 0.0 1.8 1.7018092105263158 (1.7730255164034021 1.6763061968408262)

times = 0

originSum = 0
correctSum = 0
originWholeSum = 0
correctWholeSum = 0
topNum = 64
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
    # a_list = ""
    # b_list = ""

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

    # 转为二进制
    # for i in range(len(permInd)):
    #     a_list += bin(permInd[i])[2:]
    # for i in range(len(matchb)):
    #     b_list += bin(matchb[i])[2:]
    #
    # rounda = int(len(a_list) / keyLen)
    # remaina = len(a_list) - rounda * keyLen
    # if remaina >= 0:
    #     rounda += 1
    #
    # for i in range(keyLen):
    #     tmp = 0
    #     for j in range(rounda):
    #         if j * keyLen + i >= len(a_list):
    #             continue
    #         tmp += int(a_list[j * keyLen + i])
    #     a_bits += str(tmp % 2)
    #
    # roundb = int(len(b_list) / keyLen)
    # remainb = len(b_list) - roundb * keyLen
    # if remainb >= 0:
    #     roundb += 1
    #
    # for i in range(keyLen):
    #     tmp = 0
    #     for j in range(roundb):
    #         if j * keyLen + i >= len(b_list):
    #             continue
    #         tmp += int(b_list[j * keyLen + i])
    #     b_bits += str(tmp % 2)

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