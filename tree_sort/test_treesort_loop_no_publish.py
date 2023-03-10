import time
from tkinter import messagebox

import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing

from algorithm import smooth, binaryTreeSort, quadTreeSort, genSample, binaryTreeSortPerm, binaryLevelSort, \
    binaryTreeMetricSort, singleMetricSort, levelMetricSortPerm, levelMetricSort, lossyLevelMetricSortOfB, \
    lossyLevelMetricSortOfA, simpleLevelMetricSort, levelNoiseMetricSortPerm, difference, derivative_sq_integral_smth, \
    diff_sq_integral_stationary, diff_sq_integral_rough, integral_sq_derivative, integral_sq_derivative_increment

start_time = time.time()
fileName = "../data/data_static_indoor_1_r_m.mat"
rawData = loadmat(fileName)
csv = open("./no_publish.csv", "a+")
csv.write("\n")
# csv.write("filename," + "times," + "segLen," +
#           "correctBitRate," + "randomBitRate," + "noiseBitRate," +
#           "correctWholeRate," + "randomWholeRate," + "noiseWholeRate\n")

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

dataLen = len(CSIa1Orig)

CSIa1Orig = np.array(CSIa1Orig)
CSIb1Orig = np.array(CSIb1Orig)
CSIn1Orig = np.random.normal(loc=-100, scale=1000, size=dataLen)
CSIn2Orig = np.random.normal(loc=-100, scale=1000, size=dataLen)

CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)
CSIe2Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)

CSIi1Orig = rawData['A'][:, 0]
CSIi2Orig = rawData['A'][:, 1]

combineCSIiXOrig = list(zip(CSIi1Orig, CSIi2Orig))
np.random.shuffle(combineCSIiXOrig)
CSIi1Orig, CSIi2Orig = zip(*combineCSIiXOrig)

CSIi1Orig = np.array(CSIi1Orig)
CSIi2Orig = np.array(CSIi2Orig)

# entropyTime = time.time()
# CSIa1Orig, CSIb1Orig, CSIe1Orig = splitEntropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, 5, dataLen)
# print("--- entropyTime %s seconds ---" % (time.time() - entropyTime))

CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')
CSIb1Orig = smooth(CSIb1Orig, window_len=15, window='flat')
CSIe1Orig = smooth(CSIe1Orig, window_len=15, window="flat")
CSIe2Orig = smooth(CSIe2Orig, window_len=15, window="flat")
CSIn1Orig = smooth(CSIn1Orig, window_len=15, window="flat")
CSIn2Orig = smooth(CSIn2Orig, window_len=15, window="flat")
CSIi1Orig = smooth(CSIi1Orig, window_len=15, window="flat")
CSIi2Orig = smooth(CSIi2Orig, window_len=15, window="flat")

sortMethods = [[binaryTreeSortPerm, binaryTreeSort], quadTreeSort, binaryLevelSort, binaryTreeMetricSort,
               singleMetricSort,
               [levelMetricSortPerm, levelMetricSort, lossyLevelMetricSortOfB, lossyLevelMetricSortOfA,
                simpleLevelMetricSort, levelNoiseMetricSortPerm]]
sortMethod = sortMethods[5]

intvl = 1
keyLen = 128
interval_length = 2
addNoise = False
insertRatio = 1
ratio = 1
rawOp = "none"
insertOp = "none"
metric = "dtw"

for segLen in range(4, 16):
    print("segLen", segLen)
    originSum = 0
    correctSum = 0
    randomSum = 0
    noiseSum = 0

    originWholeSum = 0
    correctWholeSum = 0
    randomWholeSum = 0
    noiseWholeSum = 0

    codings = ""
    times = 0

    for staInd in range(0, len(CSIa1Orig), intvl * keyLen):

        endInd = staInd + keyLen * intvl
        if endInd >= len(CSIa1Orig):
            break
        times += 1

        tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
        tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
        tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
        tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]
        tmpNoise = CSIn1Orig[range(staInd, endInd, 1)]
        tmpNoisy = CSIn2Orig[range(staInd, endInd, 1)]
        tmpCSIi1 = CSIi1Orig[range(staInd, endInd, 1)]
        tmpCSIi2 = CSIi2Orig[range(staInd, endInd, 1)]

        # ??????????????????
        # tmpCSIa1 = tmpCSIa1 - (np.mean(tmpCSIa1) - np.mean(tmpCSIb1))  # Mean value consistency
        tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
        tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
        tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
        tmpNoise = tmpNoise - np.mean(tmpNoise)

        # threshold 1 15
        scale = 10
        offset = -200
        if rawOp == "fft":
            sortCSIa1 = np.abs(np.fft.fft(tmpCSIa1))
            sortCSIb1 = np.abs(np.fft.fft(tmpCSIb1))
            sortCSIe1 = np.abs(np.fft.fft(tmpCSIe1))
            sortNoise = np.abs(np.fft.fft(tmpNoise))

            sortCSIa1 = sortCSIa1 - np.mean(sortCSIa1)
            sortCSIb1 = sortCSIb1 - np.mean(sortCSIb1)
            sortCSIe1 = sortCSIe1 - np.mean(sortCSIe1)
            sortNoise = sortNoise - np.mean(sortNoise)
        else:
            sortCSIa1 = tmpCSIa1
            sortCSIb1 = tmpCSIb1
            sortCSIe1 = tmpCSIe1
            sortNoise = tmpNoise

        if insertOp == "fft":
            tmpCSIe2 = tmpCSIe2 - np.mean(tmpCSIe2)
            tmpCSIi1 = tmpCSIi1 - np.mean(tmpCSIi1)
            tmpCSIi2 = tmpCSIi2 - np.mean(tmpCSIi2)
            tmpNoisy = tmpNoisy - np.mean(tmpNoisy)

            sortCSIe2 = np.abs(np.fft.fft(tmpCSIe2))
            sortCSIi1 = np.abs(np.fft.fft(tmpCSIi1))
            sortCSIi2 = np.abs(np.fft.fft(tmpCSIi2))
            sortNoisy = np.abs(np.fft.fft(tmpNoisy))

            sortCSIe2 = sortCSIe2 - np.mean(sortCSIe2)
            sortCSIi1 = sortCSIi1 - np.mean(sortCSIi1)
            sortCSIi2 = sortCSIi2 - np.mean(sortCSIi2)
            sortNoisy = sortNoisy - np.mean(sortNoisy)
        else:
            sortCSIe2 = tmpCSIe2
            sortCSIi1 = tmpCSIi1
            sortCSIi2 = tmpCSIi2
            sortNoisy = tmpNoisy

        # ?????????????????????
        transform = "diff_sq_integral_rough"
        if transform == "linear":
            sortCSIe2 = sortCSIe2 * scale + offset
            sortCSIi1 = sortCSIi1 * scale + offset
            sortCSIi2 = sortCSIi2 * scale + offset
            sortNoisy = sortNoisy * scale + offset
        elif transform == "square":
            sortCSIe2 = sortCSIe2 * sortCSIe2
            sortCSIi1 = sortCSIi1 * sortCSIi1
            sortCSIi2 = sortCSIi2 * sortCSIi2
            sortNoisy = sortNoisy * sortNoisy
        elif transform == "cosine":
            sortCSIe2 = np.cos(sortCSIe2)
            sortCSIi1 = np.cos(sortCSIi1)
            sortCSIi2 = np.cos(sortCSIi2)
            sortNoisy = np.cos(sortNoisy)
        elif transform == "exponent":
            sortCSIe2 = np.power(1.1, np.abs(sortCSIe2))
            sortCSIi1 = np.power(1.1, np.abs(sortCSIi1))
            sortCSIi2 = np.power(1.1, np.abs(sortCSIi2))
            sortNoisy = np.power(1.1, np.abs(sortNoisy))
        elif transform == "base":
            sortCSIe2 = np.power(np.abs(sortCSIe2), 1.5)
            sortCSIi1 = np.power(np.abs(sortCSIi1), 1.5)
            sortCSIi2 = np.power(np.abs(sortCSIi2), 1.5)
            sortNoisy = np.power(np.abs(sortNoisy), 1.5)
        elif transform == "logarithm":
            sortCSIe2 = np.log2(np.abs(sortCSIe2)) * 10
            sortCSIi1 = np.log2(np.abs(sortCSIi1)) * 10
            sortCSIi2 = np.log2(np.abs(sortCSIi2)) * 10
            sortNoisy = np.log2(np.abs(sortNoisy)) * 10
        elif transform == "box-cox":
            pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
            sortCSIe22D = sortCSIe2.reshape(2, int(len(sortCSIe2) / 2))
            sortCSIe2 = pt.fit_transform(np.abs(sortCSIe22D)).reshape(1, -1)[0]
            sortCSIi12D = sortCSIi1.reshape(2, int(len(sortCSIi1) / 2))
            sortCSIi1 = pt.fit_transform(np.abs(sortCSIi12D)).reshape(1, -1)[0]
            sortCSIi22D = sortCSIi2.reshape(2, int(len(sortCSIi2) / 2))
            sortCSIi2 = pt.fit_transform(np.abs(sortCSIi22D)).reshape(1, -1)[0]
            sortNoisy2D = sortNoisy.reshape(2, int(len(sortNoisy) / 2))
            sortNoisy = pt.fit_transform(np.abs(sortNoisy2D)).reshape(1, -1)[0]
            sortCSIe2 = sortCSIe2 * 10
            sortCSIi1 = sortCSIi1 * 10
            sortCSIi2 = sortCSIi2 * 10
            sortNoisy = sortNoisy * 10
        elif transform == "reciprocal":
            sortCSIe2 = 5000 / sortCSIe2 + sortCSIe2
            sortCSIi1 = 5000 / sortCSIi1 + sortCSIi1
            sortCSIi2 = 5000 / sortCSIi2 + sortCSIi2
            sortNoisy = 5000 / sortNoisy + sortNoisy
        elif transform == "tangent":
            sortCSIe2 = np.tan(np.abs(sortCSIe2) / np.pi)
            sortCSIi1 = np.tan(np.abs(sortCSIi1) / np.pi)
            sortCSIi2 = np.tan(np.abs(sortCSIi2) / np.pi)
            sortNoisy = np.tan(np.abs(sortNoisy) / np.pi)
        elif transform == "remainder":
            sortCSIe2 = sortCSIe2 - np.mean(sortCSIe2) * (np.round(sortCSIe2 / np.mean(sortCSIe2)) - 1)
            sortCSIi1 = sortCSIi1 - np.mean(sortCSIi1) * (np.round(sortCSIi1 / np.mean(sortCSIi1)) - 1)
            sortCSIi2 = sortCSIi2 - np.mean(sortCSIi2) * (np.round(sortCSIi2 / np.mean(sortCSIi2)) - 1)
            sortNoisy = sortNoisy - np.mean(sortNoisy) * (np.round(sortNoisy / np.mean(sortNoisy)) - 1)
        elif transform == "quotient":
            sortCSIe2 = sortCSIe2 / np.mean(sortCSIe2) * 100
            sortCSIi1 = sortCSIi1 / np.mean(sortCSIi1) * 100
            sortCSIi2 = sortCSIi2 / np.mean(sortCSIi2) * 100
            sortNoisy = sortNoisy / np.mean(sortNoisy) * 100
        elif transform == "difference":
            sortCSIe2 = difference(sortCSIe2)
            sortCSIi1 = difference(sortCSIi1)
            sortCSIi2 = difference(sortCSIi2)
            sortNoisy = difference(sortNoisy)
        elif transform == "integral_square_derivative":
            tmpCSIa1 = np.array(integral_sq_derivative(tmpCSIa1))
            tmpCSIb1 = np.array(integral_sq_derivative(tmpCSIb1))
            tmpCSIe1 = np.array(integral_sq_derivative(tmpCSIe1))
            tmpNoise = np.array(integral_sq_derivative(tmpNoise))
        elif transform == "diff_sq_integral_rough":
            tmpCSIa1 = np.array(diff_sq_integral_rough(tmpCSIa1))
            tmpCSIb1 = np.array(diff_sq_integral_rough(tmpCSIb1))
            tmpCSIe1 = np.array(diff_sq_integral_rough(tmpCSIe1))
            tmpNoise = np.array(diff_sq_integral_rough(tmpNoise))

        # disperse
        # sortCSIi1 = np.array(level_disperse(sortCSIi1, perm_threshold, 4))
        # sortCSIi2 = np.array(level_disperse(sortCSIi2, perm_threshold, 4))
        # sortCSIa1 = disperse(sortCSIa1, perm_threshold)
        # sortCSIb1 = disperse(sortCSIb1, perm_threshold)

        # combineSortCSIiX = list(zip(sortCSIi1, sortCSIi2))
        # np.random.shuffle(combineSortCSIiX)
        # sortCSIi1, sortCSIi2 = zip(*combineSortCSIiX)
        # sortCSIi1 = np.array(sortCSIi1)
        # sortCSIi2 = np.array(sortCSIi2)

        # ???????????????????????????reshape
        sortCSIa1Reshape = sortCSIa1[0:segLen * int(len(sortCSIa1) / segLen)]
        sortCSIb1Reshape = sortCSIb1[0:segLen * int(len(sortCSIb1) / segLen)]
        sortCSIe1Reshape = sortCSIe1[0:segLen * int(len(sortCSIe1) / segLen)]
        sortCSIe2Reshape = sortCSIe2[0:segLen * int(len(sortCSIe2) / segLen)]
        sortNoiseReshape = sortNoise[0:segLen * int(len(sortNoise) / segLen)]
        sortNoisyReshape = sortNoisy[0:segLen * int(len(sortNoisy) / segLen)]
        sortCSIi1Reshape = sortCSIi1[0:segLen * int(len(sortCSIi1) / segLen)]
        sortCSIi2Reshape = sortCSIi2[0:segLen * int(len(sortCSIi2) / segLen)]

        sortCSIa1Reshape = sortCSIa1Reshape.reshape(int(len(sortCSIa1Reshape) / segLen), segLen)
        sortCSIb1Reshape = sortCSIb1Reshape.reshape(int(len(sortCSIb1Reshape) / segLen), segLen)
        sortCSIe1Reshape = sortCSIe1Reshape.reshape(int(len(sortCSIe1Reshape) / segLen), segLen)
        sortCSIe2Reshape = sortCSIe2Reshape.reshape(int(len(sortCSIe2Reshape) / segLen), segLen)
        sortNoiseReshape = sortNoiseReshape.reshape(int(len(sortNoiseReshape) / segLen), segLen)
        sortNoisyReshape = sortNoisyReshape.reshape(int(len(sortNoisyReshape) / segLen), segLen)
        sortCSIi1Reshape = sortCSIi1Reshape.reshape(int(len(sortCSIi1Reshape) / segLen), segLen)
        sortCSIi2Reshape = sortCSIi2Reshape.reshape(int(len(sortCSIi2Reshape) / segLen), segLen)

        sortCSIa1 = np.array(genSample(sortCSIa1Reshape, ratio))
        sortCSIb1 = np.array(genSample(sortCSIb1Reshape, ratio))
        sortCSIe1 = np.array(genSample(sortCSIe1Reshape, ratio))
        sortCSIe2 = np.array(genSample(sortCSIe2Reshape, ratio))
        sortNoise = np.array(genSample(sortNoiseReshape, ratio))
        sortNoisy = np.array(genSample(sortNoisyReshape, ratio))
        sortCSIi1 = np.array(genSample(sortCSIi1Reshape, ratio))
        sortCSIi2 = np.array(genSample(sortCSIi2Reshape, ratio))

        # ?????????????????????
        a_list = []
        b_list = []
        e_list = []
        n_list = []

        sortCSIa1 = np.array(sortCSIa1)
        sortCSIb1 = np.array(sortCSIb1)
        sortCSIe1 = np.array(sortCSIe1)
        sortCSIe2 = np.array(sortCSIe2)
        sortNoise = np.array(sortNoise)
        sortNoisy = np.array(sortNoisy)

        # b?????????????????????blots?????????a???a????????????
        _, perm = sortMethod[0](list(sortCSIa1), interval_length, list(sortCSIi1), metric)

        exchange_blots = True
        if exchange_blots is False:
            _, blots = sortMethod[2](list(sortCSIb1), interval_length, list(sortCSIi2), perm, metric)
        else:
            _, blotsOfB = sortMethod[2](list(sortCSIb1), interval_length, list(sortCSIi2), perm, metric)
            _, blotsOfA = sortMethod[2](list(sortCSIa1), interval_length, list(sortCSIi1), perm, metric)
            linear_blotOfB = []
            linear_blotOfA = []

            # ??????
            for i in range(len(blotsOfB)):
                cur_blot = []
                for j in range(len(blotsOfB[i])):
                    cur_blot.append(blotsOfB[i][j][1])
                linear_blotOfB.append(cur_blot)
            for i in range(len(blotsOfA)):
                cur_blot = []
                for j in range(len(blotsOfA[i])):
                    cur_blot.append(blotsOfA[i][j][1])
                linear_blotOfA.append(cur_blot)

            # ???????????????
            linear_blot = linear_blotOfB.copy()
            for i in range(len(linear_blotOfA)):
                for j in range(len(linear_blotOfA[i])):
                    linear_blot[i].append(linear_blotOfA[i][j])
            for i in range(len(linear_blot)):
                for j in range(len(linear_blot[i])):
                    linear_blot[i] = list(set(linear_blot[i]))
                    linear_blot[i].sort()

            # ??????
            blots = []
            for i in range(len(linear_blot)):
                cur_blot = []
                for j in range(len(linear_blot[i])):
                    cur_blot.append([i, linear_blot[i][j]])
                blots.append(cur_blot)

        a_list_number = sortMethod[3](list(sortCSIa1), interval_length, list(sortCSIi1), perm, blots, metric)
        b_list_number = sortMethod[3](list(sortCSIb1), interval_length, list(sortCSIi2), perm, blots, metric)
        e_list_number = sortMethod[3](list(sortCSIe1), interval_length, list(sortCSIe2), perm, blots, metric)
        n_list_number = sortMethod[3](list(sortNoise), interval_length, list(sortNoisy), perm, blots, metric)

        # ????????????????????????blots
        # _, perm = sortMethod[0](list(sortCSIa1), interval_length, list(sortNoise), metric)
        # b_list_number, _ = sortMethod[2](list(sortCSIb1), interval_length, list(sortNoise), perm, metric)
        # a_list_number, _ = sortMethod[2](list(sortCSIa1), interval_length, list(sortNoise), perm, metric)
        # e_list_number, _ = sortMethod[2](list(sortCSIe1), interval_length, list(sortNoise), perm, metric)
        # n_list_number, _ = sortMethod[2](list(sortNoise), interval_length, list(sortNoise), perm, metric)

        # ????????????????????????
        # _, perm = sortMethod[0](list(sortCSIa1), interval_length, list(sortCSIi1), metric)
        # b_list_number = sortMethod[1](list(sortCSIb1), interval_length, list(sortCSIi2), perm, metric)
        # a_list_number = sortMethod[1](list(sortCSIa1), interval_length, list(sortCSIi1), perm, metric)
        # e_list_number = sortMethod[1](list(sortCSIe1), interval_length, list(sortCSIe2), perm, metric)
        # n_list_number = sortMethod[1](list(sortNoise), interval_length, list(sortNoisy), perm, metric)

        # ????????????????????????
        # a_list_number, designed_noise = sortMethod[5](list(sortCSIa1), interval_length, list(sortCSIi1), metric)
        # a_list_number, designed_noise = sortMethod[5](list(sortCSIa1), interval_length, list(sortNoise), metric)
        # sortInsrt = list(np.random.normal(loc=np.mean(sortCSIa1) * 2,
        #                                scale=np.std(sortCSIa1, ddof=1) * 10,
        #                                size=len(sortCSIa1) * 10))
        # sortInsrt = list(np.random.laplace(loc=np.mean(sortCSIa1),
        #                                 scale=10,
        #                                 size=len(sortCSIa1) * 10))
        # scale = 2
        # offset = 100
        # sortInsrt = list(np.random.uniform(
        #     np.min(sortCSIa1) * scale - offset,
        #     np.max(sortCSIa1) * scale + offset,
        #     len(sortCSIa1) * 100))
        # sortEaves = list(np.random.uniform(
        #     np.min(sortCSIa1) * scale - offset,
        #     np.max(sortCSIa1) * scale + offset,
        #     len(sortCSIa1)))
        # sortEaves = list(np.random.uniform(
        #     np.mean(sortCSIa1) - scale * np.std(sortCSIa1, ddof=1),
        #     np.mean(sortCSIa1) + scale * np.std(sortCSIa1, ddof=1),
        #     len(sortCSIa1)))
        # sortEaves = list(np.random.normal(np.mean(sortCSIa1), np.std(sortCSIa1, ddof=1) * 10, len(sortCSIa1)))

        # plt.figure()
        # plt.plot(sortCSIa1, 'r')
        # plt.plot(sortCSIb1, 'b')
        # plt.plot(sortCSIe1, 'k')
        # plt.plot(sortEaves, 'g')
        # plt.show()

        # a_list_number, _ = sortMethod[5](list(sortCSIa1), interval_length, list(sortCSIi1), metric)
        # b_list_number, _ = sortMethod[5](list(sortCSIb1), interval_length, list(sortCSIi2), metric)
        # e_list_number, _ = sortMethod[5](list(sortEaves), interval_length, list(sortEaves), metric)
        # n_list_number, _ = sortMethod[5](list(sortNoise), interval_length, list(sortNoise), metric)

        # ?????????????????????
        # a_list_number = sortMethod[4](list(sortCSIa1), interval_length, list(sortCSIi1), metric)
        # b_list_number = sortMethod[4](list(sortCSIb1), interval_length, list(sortCSIi2), metric)
        # e_list_number = sortMethod[4](list(sortCSIe1), interval_length, list(sortCSIi1), metric)
        # n_list_number = sortMethod[4](list(sortNoise), interval_length, list(sortCSIi1), metric)

        # ??????????????????
        # a_list_number = sortMethod[4](list(sortCSIa1), interval_length, list(sortNoise), metric)
        # b_list_number = sortMethod[4](list(sortCSIb1), interval_length, list(sortNoise), metric)
        # e_list_number = sortMethod[4](list(sortCSIe1), interval_length, list(sortNoise), metric)
        # n_list_number = sortMethod[4](list(sortNoise), interval_length, list(sortNoise), metric)

        # simpleLevelMetricSort
        # a_list_number = sortMethod(list(sortCSIa1), interval_length)
        # b_list_number = sortMethod(list(sortCSIb1), interval_length)
        # e_list_number = sortMethod(list(sortCSIe1), interval_length)
        # n_list_number = sortMethod(list(sortNoise), interval_length)

        # ??????????????????
        a_level_number = []
        b_level_number = []
        e_level_number = []
        n_level_number = []
        i = 0
        step = 1
        while i < len(a_list_number):
            a_level_number.append(list(a_list_number[i: i + 2 ** step]))
            i = i + 2 ** step
            step += 1
        i = 0
        step = 1
        while i < len(b_list_number):
            b_level_number.append(list(b_list_number[i: i + 2 ** step]))
            i = i + 2 ** step
            step += 1
        i = 0
        step = 1
        while i < len(e_list_number):
            e_level_number.append(list(e_list_number[i: i + 2 ** step]))
            i = i + 2 ** step
            step += 1
        i = 0
        step = 1
        while i < len(n_list_number):
            n_level_number.append(list(n_list_number[i: i + 2 ** step]))
            i = i + 2 ** step
            step += 1

        # ???????????????
        for i in range(len(a_level_number)):
            for j in range(len(a_level_number[i])):
                number = bin(int(a_level_number[i][j]))[2:].zfill(i + 1)
                a_list += number
        for i in range(len(b_level_number)):
            for j in range(len(b_level_number[i])):
                number = bin(int(b_level_number[i][j]))[2:].zfill(i + 1)
                b_list += number
        for i in range(len(e_level_number)):
            for j in range(len(e_level_number[i])):
                number = bin(int(e_level_number[i][j]))[2:].zfill(i + 1)
                e_list += number
        for i in range(len(n_level_number)):
            for j in range(len(n_level_number[i])):
                number = bin(int(n_level_number[i][j]))[2:].zfill(i + 1)
                n_list += number

        # ???????????????????????????
        for i in range(len(a_list) - len(e_list)):
            e_list += str(np.random.randint(0, 2))
        for i in range(len(a_list) - len(n_list)):
            n_list += str(np.random.randint(0, 2))

        # print("keys of a:", len(a_list), a_list)
        # print("keys of a:", len(a_list_number), a_list_number)
        # print("keys of b:", len(b_list), b_list)
        # print("keys of b:", len(b_list_number), b_list_number)
        # print("keys of e:", len(e_list), e_list)
        # print("keys of e:", len(e_list_number), e_list_number)
        # print("keys of n:", len(n_list), n_list)
        # print("keys of n:", len(n_list_number), n_list_number)

        sum1 = min(len(a_list), len(b_list))
        sum2 = 0
        sum3 = 0
        sum4 = 0
        for i in range(0, sum1):
            sum2 += (a_list[i] == b_list[i])
        for i in range(min(len(a_list), len(e_list))):
            sum3 += (a_list[i] == e_list[i])
        for i in range(min(len(a_list), len(n_list))):
            sum4 += (a_list[i] == n_list[i])

        if sum1 == 0:
            continue
        if sum2 == sum1:
            print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
        else:
            print("\033[0;31;40ma-b", "bad", sum2, sum2 / sum1, "\033[0m")
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

        coding = ""
        for i in range(len(a_list)):
            coding += a_list[i]
        codings += coding + "\n"

    # with open('./full_key_mobile_3.txt', 'a', ) as f:
    #     f.write(codings)

    print("a-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10))
    print("a-e all", randomSum, "/", originSum, "=", round(randomSum / originSum, 10))
    print("a-n all", noiseSum, "/", originSum, "=", round(noiseSum / originSum, 10))
    print("a-b whole match", correctWholeSum, "/", originWholeSum, "=", round(correctWholeSum / originWholeSum, 10))
    print("a-e whole match", randomWholeSum, "/", originWholeSum, "=", round(randomWholeSum / originWholeSum, 10))
    print("a-n whole match", noiseWholeSum, "/", originWholeSum, "=", round(noiseWholeSum / originWholeSum, 10))
    print("times", times)
    csv.write(fileName + ',' + str(times) + ',' + str(segLen)
              + ',' + str(correctSum) + " / " + str(originSum) + " = " + str(round(correctSum / originSum, 10))
              + ',' + str(randomSum) + " / " + str(originSum) + " = " + str(round(randomSum / originSum, 10))
              + ',' + str(noiseSum) + " / " + str(originSum) + " = " + str(round(noiseSum / originSum, 10))
              + ',' + str(correctWholeSum) + " / " + str(originWholeSum) + " = " + str(
        round(correctWholeSum / originWholeSum, 10))
              + ',' + str(randomWholeSum) + " / " + str(originWholeSum) + " = " + str(
        round(randomWholeSum / originWholeSum, 10))
              + ',' + str(noiseWholeSum) + " / " + str(originWholeSum) + " = " + str(
        round(noiseWholeSum / originWholeSum, 10)) + '\n')

messagebox.showinfo("??????", "?????????????????????" + str(time.time() - start_time))