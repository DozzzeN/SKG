import time

import numpy as np
from scipy import signal
from scipy.io import loadmat
from sklearn import preprocessing

from algorithm import smooth, binaryTreeSort, quadTreeSort, genSample, insertNoise, binaryTreeSortPerm, binaryLevelSort, \
    binaryTreeMetricSort, singleMetricSort, levelMetricSortPerm, levelMetricSort, lossyLevelMetricSortOfB, \
    lossyLevelMetricSortOfA, simpleLevelMetricSort, levelNoiseMetricSortPerm, splitEntropyPerm

fileName = "../data/data_static_indoor_1_r_m.mat"
rawData = loadmat(fileName)

CSIa1Orig = rawData['A'][:, 0]

CSIa1Orig = np.array(CSIa1Orig)

# CSIa1Orig, _, _ = splitEntropyPerm(CSIa1Orig, CSIa1Orig, CSIa1Orig, 5, len(CSIa1Orig))

CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')

CSIa1OrigBack = CSIa1Orig.copy()

sortMethods = [[binaryTreeSortPerm, binaryTreeSort], quadTreeSort, binaryLevelSort, binaryTreeMetricSort,
               singleMetricSort,
               [levelMetricSortPerm, levelMetricSort, lossyLevelMetricSortOfB, lossyLevelMetricSortOfA,
                simpleLevelMetricSort, levelNoiseMetricSortPerm]]
sortMethod = sortMethods[5]

intvl = 1
keyLen = 512
interval_length = 2
addNoise = False
insertRatio = 1
ratio = 1
closeness = 5
metric = "dtw"

for loop in range(30):
    for segLen in range(4, 16):
        print("segLen", segLen)

        codings = ""
        times = 0

        for staInd in range(0, len(CSIa1Orig), intvl * keyLen):
            endInd = staInd + keyLen * intvl
            if endInd >= len(CSIa1Orig):
                break
            times += 1

            CSIa1Orig = CSIa1OrigBack.copy()

            tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]

            # 去除直流分量
            tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)

            sortCSIa1 = tmpCSIa1

            # 取原数据的一部分来reshape
            sortCSIa1Reshape = sortCSIa1[0:segLen * int(len(sortCSIa1) / segLen)]

            sortCSIa1Reshape = sortCSIa1Reshape.reshape(int(len(sortCSIa1Reshape) / segLen), segLen)

            sortCSIa1 = np.array(genSample(sortCSIa1Reshape, ratio))

            # 最后各自的密钥
            a_list = []

            sortCSIa1 = np.array(sortCSIa1)

            scale = 2
            offset = 100
            sortInsrt = list(np.random.uniform(
                np.min(sortCSIa1) * scale - offset,
                np.max(sortCSIa1) * scale + offset,
                len(sortCSIa1) * 100))

            a_list_number = sortMethod[4](list(sortCSIa1), interval_length, list(sortInsrt), metric)

            # 转成层序密钥
            a_level_number = []

            i = 0
            step = 1
            while i < len(a_list_number):
                a_level_number.append(list(a_list_number[i: i + 2 ** step]))
                i = i + 2 ** step
                step += 1
            i = 0
            step = 1

            # 转成二进制
            for i in range(len(a_level_number)):
                for j in range(len(a_level_number[i])):
                    number = bin(int(a_level_number[i][j]))[2:].zfill(i + 1)
                    a_list += number

            # print("keys of a:", len(a_list), a_list)
            # print("keys of a:", len(a_list_number), a_list_number)

            coding = ""
            for i in range(len(a_list)):
                coding += a_list[i]
            codings += coding + "\n"

        with open('./full_key_static_3.txt', 'a', ) as f:
            f.write(codings)