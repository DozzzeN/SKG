import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

rawData = loadmat('../data/data_mobile_indoor_1.mat')

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

dataLen = len(CSIa1Orig)

CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()

paraLs = [8, 9]

for para in paraLs:
    intvl = para
    keyLen = 100

    for staInd in range(0, dataLen - keyLen * intvl - 1, intvl):
        print(staInd)
        endInd = staInd + keyLen * intvl

        # --------------------------------------------
        # BEGIN: Ranking SKG
        # --------------------------------------------
        tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
        tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]

        epiLen = len(range(staInd, endInd, 1))

        ## For indics closeness;
        tmpCSIa1Ind = tmpCSIa1.argsort().argsort()
        tmpCSIb1Ind = tmpCSIb1.argsort().argsort()

        tmpCSIa1Sort = np.sort(tmpCSIa1)
        tmpCSIb1Sort = np.sort(tmpCSIb1)

        ## For value closeness;
        tmpCSIa1IndP = tmpCSIa1.argsort()
        tmpCSIb1IndP = tmpCSIb1.argsort()

        tmpCSIa1SortA = tmpCSIa1[tmpCSIa1IndP]
        tmpCSIb1SortB = tmpCSIb1[tmpCSIa1IndP]

        minEpiIndClosenessLs = np.zeros(keyLen)
        minEpiValClosenessLs = np.zeros(keyLen)
        for pp in range(0, keyLen, 1):
            epiInda1 = tmpCSIa1Ind[pp * intvl:(pp + 1) * intvl]

            epiIndClosenessLs = np.zeros(keyLen)
            epiValClosenessLs = np.zeros(keyLen)

            for qq in range(0, keyLen, 1):
                epiIndb1 = tmpCSIb1Ind[qq * intvl:(qq + 1) * intvl]

                # 对于分段后的索引，每个分段之间对应位置绝对值距离
                epiIndClosenessLs[qq] = sum(abs(epiInda1 - epiIndb1))
                # epiIndClosenessLs[qq] = max(abs(epiInda1 - epiIndb1))

                # 对于分段后的排序值，每个分段之间对应位置绝对值距离
                epiValClosenessLs[qq] = sum(abs(tmpCSIa1SortA[epiInda1] - tmpCSIb1SortB[epiIndb1]))

                # epiClosenessLs[qq] = abs(sum(epiInda1) - sum(epiIndb1))

                # print(epiInda1)
                # print(epiIndb1)

            minEpiIndClosenessLs[pp] = np.argmin(epiIndClosenessLs)
            minEpiValClosenessLs[pp] = np.argmin(epiValClosenessLs)

            print(epiInda1)
            qq = int(minEpiIndClosenessLs[pp])
            print(tmpCSIb1Ind[qq * intvl:(qq + 1) * intvl])
            print(tmpCSIb1Ind[qq * intvl:(qq + 1) * intvl] - epiInda1)
            print('-----------------------')

        print(minEpiIndClosenessLs - np.array(range(0, keyLen, 1)))
        print(minEpiValClosenessLs - np.array(range(0, keyLen, 1)))

        plt.plot(tmpCSIa1Sort)
        plt.plot(tmpCSIb1Sort, 'r--')
        # plt.show()
        exit()
