import time

import numpy as np
from pyentrp import entropy as ent
from scipy import signal
from scipy.io import loadmat

from algorithm import smooth, binaryTreeSort, quadTreeSort, genSample, binaryTreeSortPerm, binaryLevelSort, \
    binaryTreeMetricSort, singleMetricSort, levelMetricSortPerm, levelMetricSort, lossyLevelMetricSortOfB, \
    lossyLevelMetricSortOfA, simpleLevelMetricSort, shannonEntropy

fileName = "../data/data_static_indoor_1_r_m.mat"
rawData = loadmat(fileName)
csv = open("./entropy.csv", "a+")
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
# CSIn1Orig = np.random.normal(loc=-1, scale=1, size=dataLen)
# CSIa1OrigMean = np.mean(CSIa1Orig)
# CSIa1OrigStd = np.std(CSIa1Orig, ddof=1)
# randomLoc = np.random.randint(CSIa1OrigMean - CSIa1OrigMean / 2, - CSIa1OrigMean / 2) if CSIa1OrigMean < 0 \
#     else np.random.randint(- CSIa1OrigMean / 2, CSIa1OrigMean - CSIa1OrigMean / 2)
# randomScale = np.random.randint(CSIa1OrigStd / 2, CSIa1OrigStd + CSIa1OrigStd / 2)
# CSIn1Orig = np.random.normal(loc=randomLoc, scale=randomScale, size=dataLen)
# CSIn1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)
CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)

CSIi1Orig = loadmat('../data/data_static_indoor_1_r_m.mat')['A'][:, 0]
# CSIi1Orig = CSIi1Orig + np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)
# CSIi1Orig = CSIi1Orig + np.random.normal(loc=0, scale=np.std(CSIi1Orig, ddof=1), size=dataLen)
np.random.shuffle(CSIi1Orig)

# entropyTime = time.time()
# CSIa1Orig, CSIb1Orig, CSIe1Orig = splitEntropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, 5, dataLen)
# print("--- entropyTime %s seconds ---" % (time.time() - entropyTime))

CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')
CSIb1Orig = smooth(CSIb1Orig, window_len=15, window='flat')
CSIe1Orig = smooth(CSIe1Orig, window_len=15, window="flat")
CSIn1Orig = smooth(CSIn1Orig, window_len=15, window="flat")
CSIi1Orig = smooth(CSIi1Orig, window_len=15, window="flat")

CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
CSIe1OrigBack = CSIe1Orig.copy()
CSIn1OrigBack = CSIn1Orig.copy()
CSIi1OrigBack = CSIi1Orig.copy()

sortMethods = [[binaryTreeSortPerm, binaryTreeSort], quadTreeSort, binaryLevelSort, binaryTreeMetricSort,
               singleMetricSort,
               [levelMetricSortPerm, levelMetricSort, lossyLevelMetricSortOfB, lossyLevelMetricSortOfA,
                simpleLevelMetricSort]]
sortMethod = sortMethods[5]
# sortMethod = sortMethods[5][4]

intvl = 5
keyLen = 128
interval_length = 2
addNoise = False
insertRatio = 1
ratio = 1
closeness = 5
metric = "dtw"

for segLen in range(15, 3, -1):
    # for segLen in range(2, 11):
    # segLen = int(math.pow(2, segLen))
    print("segLen", segLen)
    originSum = 0
    correctSum = 0
    randomSum = 0
    noiseSum = 0

    originWholeSum = 0
    correctWholeSum = 0
    randomWholeSum = 0
    noiseWholeSum = 0

    entropy_a_sum = 0
    entropy_b_sum = 0
    entropy_e_sum = 0
    entropy_n_sum = 0

    sample_entropy_a_sum = 0
    sample_entropy_b_sum = 0
    sample_entropy_e_sum = 0
    sample_entropy_n_sum = 0

    codings = ""
    times = 0

    for staInd in range(0, len(CSIa1Orig), intvl * keyLen):
        processTime = time.time()

        endInd = staInd + keyLen * intvl
        # print("range:", staInd, endInd)
        if endInd >= len(CSIa1Orig):
            break
        times += 1

        CSIa1Orig = CSIa1OrigBack.copy()
        CSIb1Orig = CSIb1OrigBack.copy()
        CSIe1Orig = CSIe1OrigBack.copy()
        CSIn1Orig = CSIn1OrigBack.copy()
        CSIi1Orig = CSIi1OrigBack.copy()

        tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
        tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
        tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
        tmpNoise = CSIn1Orig[range(staInd, endInd, 1)]
        tmpCSIi1 = CSIi1Orig[range(staInd, endInd, 1)]

        # ??????????????????
        # tmpCSIa1 = tmpCSIa1 - (np.mean(tmpCSIa1) - np.mean(tmpCSIb1))  # Mean value consistency
        tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
        tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
        # tmpCSIi1 = tmpCSIi1 - np.mean(tmpCSIi1)
        # tmpNoise = tmpNoise - np.mean(tmpNoise)

        # linspace?????????????????????50???????????????????????????????????????????????????????????????
        # signal.square??????????????????????????????
        tmpPulse = signal.square(
            2 * np.pi * 1 / intvl * np.linspace(0, np.pi * 0.5 * keyLen, keyLen * intvl))  ## Rectangular pulse

        if addNoise:
            # tmpCSIa1 = tmpPulse * (np.float_power(np.abs(tmpCSIa1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIa1))
            # tmpCSIb1 = tmpPulse * (np.float_power(np.abs(tmpCSIb1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIb1))
            # tmpCSIe1 = tmpPulse * (np.float_power(np.abs(tmpCSIe1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIe1))
            # tmpCSIa1 = np.float_power(np.abs(tmpCSIa1), tmpNoise)
            # tmpCSIb1 = np.float_power(np.abs(tmpCSIb1), tmpNoise)
            # tmpCSIe1 = np.float_power(np.abs(tmpCSIe1), tmpNoise)
            tmpCSIa1 = tmpCSIa1 * tmpNoise
            tmpCSIb1 = tmpCSIb1 * tmpNoise
            tmpCSIe1 = tmpCSIe1 * tmpNoise
            # tmpCSIa1 = tmpCSIa1 + tmpNoise
            # tmpCSIb1 = tmpCSIb1 + tmpNoise
            # tmpCSIe1 = tmpCSIe1 + tmpNoise
            # tmpCSI = list(zip(tmpCSIa1, tmpCSIb1, tmpCSIe1))
            # np.random.shuffle(tmpCSI)
            # tmpCSIa1N, tmpCSIb1N, tmpCSIe1N = zip(*tmpCSI)
            # tmpCSIa1N, tmpCSIb1N, tmpCSIe1N = splitEntropyPerm(tmpCSIa1, tmpCSIb1, tmpCSIe1, 5, len(tmpCSIa1))
            # tmpCSIa1 = tmpCSIa1 + tmpCSIa1N
            # tmpCSIb1 = tmpCSIb1 + tmpCSIb1N
            # tmpCSIe1 = tmpCSIe1 + tmpCSIe1N
            # tmpCSIa1 = tmpCSIa1 * np.fft.fft(tmpCSIa1N)
            # tmpCSIb1 = tmpCSIb1 * np.fft.fft(tmpCSIb1N)
            # tmpCSIe1 = tmpCSIe1 * np.fft.fft(tmpCSIe1N)
        else:
            # tmpCSIa1 = tmpPulse * tmpCSIa1
            # tmpCSIb1 = tmpPulse * tmpCSIb1
            # tmpCSIe1 = tmpPulse * tmpCSIe1
            tmpCSIa1 = tmpCSIa1
            tmpCSIb1 = tmpCSIb1
            tmpCSIe1 = tmpCSIe1
            tmpNoise = tmpNoise

        CSIa1Orig[range(staInd, endInd, 1)] = tmpCSIa1
        CSIb1Orig[range(staInd, endInd, 1)] = tmpCSIb1
        CSIe1Orig[range(staInd, endInd, 1)] = tmpCSIe1
        CSIn1Orig[range(staInd, endInd, 1)] = tmpNoise
        CSIi1Orig[range(staInd, endInd, 1)] = tmpCSIi1

        permLen = len(range(staInd, endInd, intvl))
        origInd = np.array([xx for xx in range(staInd, endInd, intvl)])

        sortCSIa1 = np.zeros(permLen)
        sortCSIb1 = np.zeros(permLen)
        sortCSIe1 = np.zeros(permLen)
        sortNoise = np.zeros(permLen)
        sortCSIi1 = np.zeros(permLen)

        for ii in range(permLen):
            aIndVec = np.array([aa for aa in range(origInd[ii], origInd[ii] + intvl, 1)])  ## for non-permuted CSIa1

            for jj in range(permLen, permLen * 2):
                bIndVec = np.array([bb for bb in range(origInd[jj - permLen], origInd[jj - permLen] + intvl, 1)])

                CSIa1Tmp = CSIa1Orig[aIndVec]
                CSIb1Tmp = CSIb1Orig[bIndVec]
                CSIe1Tmp = CSIe1Orig[bIndVec]
                CSIn1Tmp = CSIn1Orig[aIndVec]
                CSIi1Tmp = CSIi1Orig[aIndVec]

                sortCSIa1[ii] = np.mean(CSIa1Tmp)  ## Metric 1: Mean
                sortCSIb1[jj - permLen] = np.mean(CSIb1Tmp)  # ???????????????
                sortCSIe1[jj - permLen] = np.mean(CSIe1Tmp)
                sortNoise[ii - permLen] = np.mean(CSIn1Tmp)
                sortCSIi1[ii - permLen] = np.mean(CSIi1Tmp)

        # sortCSIa1????????????????????????????????????
        # sortCSIa1 = np.log10(np.abs(sortCSIa1))
        # sortCSIb1 = np.log10(np.abs(sortCSIb1))
        # sortCSIe1 = np.log10(np.abs(sortCSIe1))
        # sortNoise = np.log10(np.abs(sortNoise))

        # ???????????????????????????reshape
        sortCSIa1Reshape = sortCSIa1[0:segLen * int(len(sortCSIa1) / segLen)]
        sortCSIb1Reshape = sortCSIb1[0:segLen * int(len(sortCSIb1) / segLen)]
        sortCSIe1Reshape = sortCSIe1[0:segLen * int(len(sortCSIe1) / segLen)]
        sortNoiseReshape = sortNoise[0:segLen * int(len(sortNoise) / segLen)]
        sortCSIi1Reshape = sortCSIi1[0:segLen * int(len(sortCSIi1) / segLen)]

        sortCSIa1Reshape = sortCSIa1Reshape.reshape(int(len(sortCSIa1Reshape) / segLen), segLen)
        sortCSIb1Reshape = sortCSIb1Reshape.reshape(int(len(sortCSIb1Reshape) / segLen), segLen)
        sortCSIe1Reshape = sortCSIe1Reshape.reshape(int(len(sortCSIe1Reshape) / segLen), segLen)
        sortNoiseReshape = sortNoiseReshape.reshape(int(len(sortNoiseReshape) / segLen), segLen)
        sortCSIi1Reshape = sortCSIi1Reshape.reshape(int(len(sortCSIi1Reshape) / segLen), segLen)

        # ?????????
        # for i in range(len(sortCSIa1Reshape)):
        #     # sklearn????????????????????????????????????????????????????????????
        #     sortCSIa1.append(preprocessing.MinMaxScaler().fit_transform(
        #         np.array(sortCSIa1Reshape[i]).reshape(-1, 1)).reshape(1, -1).tolist()[0])
        #     sortCSIb1.append(preprocessing.MinMaxScaler().fit_transform(
        #         np.array(sortCSIb1Reshape[i]).reshape(-1, 1)).reshape(1, -1).tolist()[0])
        #     sortCSIe1.append(preprocessing.MinMaxScaler().fit_transform(
        #         np.array(sortCSIe1Reshape[i]).reshape(-1, 1)).reshape(1, -1).tolist()[0])
        #     sortNoise.append(preprocessing.MinMaxScaler().fit_transform(
        #         np.array(sortNoiseReshape[i]).reshape(-1, 1)).reshape(1, -1).tolist()[0])

        # sortCSIa1 = np.array(genArray(sortCSIa1Reshape))
        # sortCSIb1 = np.array(genArray(sortCSIb1Reshape))
        # sortCSIe1 = np.array(genArray(sortCSIe1Reshape))
        # sortNoise = np.array(genArray(sortNoiseReshape))

        sortCSIa1 = np.array(genSample(sortCSIa1Reshape, ratio))
        sortCSIb1 = np.array(genSample(sortCSIb1Reshape, ratio))
        sortCSIe1 = np.array(genSample(sortCSIe1Reshape, ratio))
        sortNoise = np.array(genSample(sortNoiseReshape, ratio))
        sortCSIi1 = np.array(genSample(sortCSIi1Reshape, ratio))

        # ???????????????
        # sortCSIi1 = np.random.normal(loc=np.mean(sortCSIa1), scale=np.std(sortCSIa1, ddof=1), size=len(sortCSIa1))
        # np.random.shuffle(sortCSIi1)
        # insertIndex = np.random.permutation(len(sortCSIa1))
        # sortCSIa1 = insertNoise(sortCSIa1, sortCSIi1, insertIndex, insertRatio)
        # sortCSIb1 = insertNoise(sortCSIb1, sortCSIi1, insertIndex, insertRatio)
        # sortCSIe1 = insertNoise(sortCSIe1, sortCSIi1, insertIndex, insertRatio)
        # sortNoise = insertNoise(sortNoise, sortCSIi1, insertIndex, insertRatio)

        # ?????????????????????
        a_list = []
        b_list = []
        e_list = []
        n_list = []

        sortCSIa1 = np.array(sortCSIa1)
        sortCSIb1 = np.array(sortCSIb1)
        sortCSIe1 = np.array(sortCSIe1)
        sortNoise = np.array(sortNoise)

        # a_list_number = sortMethod(list(sortCSIa1), interval_length, list(sortNoise))
        # b_list_number = sortMethod(list(sortCSIb1), interval_length, list(sortNoise))
        # e_list_number = sortMethod(list(sortCSIe1), interval_length, list(sortNoise))
        # n_list_number = sortMethod(list(sortNoise), interval_length, list(sortNoise))

        # if metric == "kl":
        #    # ??????KL???????????????????????????????????????
        #    # sortCSIa1 = sortCSIa1 - np.min(sortCSIa1) + 0.1
        #    # sortCSIb1 = sortCSIb1 - np.min(sortCSIb1) + 0.1
        #    # sortCSIe1 = sortCSIe1 - np.min(sortCSIe1) + 0.1
        #    # sortNoise = sortNoise - np.min(sortNoise) + 0.1

        # b?????????????????????blots?????????a???a????????????
        # _, perm = sortMethod[0](list(sortCSIa1), interval_length, list(sortNoise), metric)
        # b_list_number, blots = sortMethod[2](list(sortCSIb1), interval_length, list(sortNoise), perm, metric)
        # a_list_number = sortMethod[3](list(sortCSIa1), interval_length, list(sortNoise), perm, blots, metric)
        # e_list_number = sortMethod[3](list(sortCSIe1), interval_length, list(sortNoise), perm, blots, metric)
        # n_list_number = sortMethod[3](list(sortNoise), interval_length, list(sortNoise), perm, blots, metric)

        # ????????????????????????blots
        # _, perm = sortMethod[0](list(sortCSIa1), interval_length, list(sortNoise), metric)
        # b_list_number, _ = sortMethod[2](list(sortCSIb1), interval_length, list(sortNoise), perm, metric)
        # a_list_number, _ = sortMethod[2](list(sortCSIa1), interval_length, list(sortNoise), perm, metric)
        # e_list_number, _ = sortMethod[2](list(sortCSIe1), interval_length, list(sortNoise), perm, metric)
        # n_list_number, _ = sortMethod[2](list(sortNoise), interval_length, list(sortNoise), perm, metric)

        # ????????????????????????
        _, perm = sortMethod[0](list(sortCSIa1), interval_length, list(sortNoise), metric)
        b_list_number = sortMethod[1](list(sortCSIb1), interval_length, list(sortNoise), perm, metric)
        a_list_number = sortMethod[1](list(sortCSIa1), interval_length, list(sortNoise), perm, metric)
        e_list_number = sortMethod[1](list(sortCSIe1), interval_length, list(sortNoise), perm, metric)
        n_list_number = sortMethod[1](list(sortNoise), interval_length, list(sortNoise), perm, metric)

        # ??????????????????
        # a_list_number = sortMethod[4](list(sortCSIa1), interval_length, list(sortNoise), metric)
        # b_list_number = sortMethod[4](list(sortCSIb1), interval_length, list(sortNoise), metric)
        # e_list_number = sortMethod[4](list(sortCSIe1), interval_length, list(sortNoise), metric)
        # n_list_number = sortMethod[4](list(sortNoise), interval_length, list(sortNoise), metric)

        # a_list_number, perm, = sortMethod[0](list(sortCSIa1), interval_length, closeness)
        # sortCSIb1 = sortCSIb1[perm]
        # sortCSIe1 = sortCSIe1[perm]
        # sortNoise = sortNoise[perm]
        # b_list_number = sortMethod[1](list(sortCSIb1), interval_length)
        # e_list_number = sortMethod[1](list(sortCSIe1), interval_length)
        # n_list_number = sortMethod[1](list(sortNoise), interval_length)

        # scale = 1
        # for i in range(len(a_list_number)):
        #     a_list.append(int(a_list_number[i] / scale))
        # for i in range(len(b_list_number)):
        #     b_list.append(int(b_list_number[i] / scale))
        # for i in range(len(e_list_number)):
        #     e_list.append(int(e_list_number[i] / scale))
        # for i in range(len(n_list_number)):
        #     n_list.append(int(n_list_number[i] / scale))

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

        # ???????????????
        for i in range(len(a_list_number)):
            a_list += bin(int(a_list_number[i]))[2:]
        for i in range(len(b_list_number)):
            b_list += bin(int(b_list_number[i]))[2:]
        for i in range(len(e_list_number)):
            e_list += bin(int(e_list_number[i]))[2:]
        for i in range(len(n_list_number)):
            n_list += bin(int(n_list_number[i]))[2:]

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

        # ?????????
        entropy_a_sum += shannonEntropy(a_list)
        entropy_b_sum += shannonEntropy(b_list)
        entropy_e_sum += shannonEntropy(e_list)
        entropy_n_sum += shannonEntropy(n_list)

        print("entropy of a:", shannonEntropy(a_list))
        print("entropy of b:", shannonEntropy(b_list))
        print("entropy of e:", shannonEntropy(e_list))
        print("entropy of n:", shannonEntropy(n_list))

        # ?????????
        sample_entropy_a_sum += ent.multiscale_entropy(list(map(int, a_list)), 3, maxscale=1)[0]
        sample_entropy_b_sum += ent.multiscale_entropy(list(map(int, b_list)), 3, maxscale=1)[0]
        sample_entropy_e_sum += ent.multiscale_entropy(list(map(int, e_list)), 3, maxscale=1)[0]
        sample_entropy_n_sum += ent.multiscale_entropy(list(map(int, n_list)), 3, maxscale=1)[0]

        print("sample entropy of a:", ent.multiscale_entropy(list(map(int, a_list)), 3, maxscale=1))
        print("sample entropy of b:", ent.multiscale_entropy(list(map(int, b_list)), 3, maxscale=1))
        print("sample entropy of e:", ent.multiscale_entropy(list(map(int, e_list)), 3, maxscale=1))
        print("sample entropy of n:", ent.multiscale_entropy(list(map(int, n_list)), 3, maxscale=1))

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

        # if sum2 == sum1:
        #     print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
        # else:
        #     print("\033[0;31;40ma-b", "bad", sum2, sum2 / sum1, "\033[0m")
        # print("a-e", sum3, sum3 / sum1)
        # print("a-n", sum4, sum4 / sum1)
        # print("----------------------")
        originSum += sum1
        correctSum += sum2
        randomSum += sum3
        noiseSum += sum4

        originWholeSum += 1
        correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
        randomWholeSum = randomWholeSum + 1 if sum3 == sum1 else randomWholeSum
        noiseWholeSum = noiseWholeSum + 1 if sum4 == sum1 else noiseWholeSum

        # ????????????
        # char_weights = []
        # weights = Counter(a_list)  # ??????list?????????????????????
        # for i in range(len(a_list)):
        #     char_weights.append((a_list[i], weights[a_list[i]]))
        # tree = HuffmanTree(char_weights)
        # tree.get_code()
        # HuffmanTree.codings += "\n"

        # coding = ""
        # for i in range(len(a_list)):
        #     coding += a_list[i]
        # codings += coding + "\n"

    # with open('./full_key_mobile.txt', 'a', ) as f:
    #     f.write(codings)

    # print("a-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10))
    # print("a-e all", randomSum, "/", originSum, "=", round(randomSum / originSum, 10))
    # print("a-n all", noiseSum, "/", originSum, "=", round(noiseSum / originSum, 10))
    # print("a-b whole match", correctWholeSum, "/", originWholeSum, "=", round(correctWholeSum / originWholeSum, 10))
    # print("a-e whole match", randomWholeSum, "/", originWholeSum, "=", round(randomWholeSum / originWholeSum, 10))
    # print("a-n whole match", noiseWholeSum, "/", originWholeSum, "=", round(noiseWholeSum / originWholeSum, 10))
    # print("times", times)
    # csv.write(fileName + ',' + str(times) + ',' + str(segLen)
    #           + ',' + str(correctSum) + " / " + str(originSum) + " = " + str(round(correctSum / originSum, 10))
    #           + ',' + str(randomSum) + " / " + str(originSum) + " = " + str(round(randomSum / originSum, 10))
    #           + ',' + str(noiseSum) + " / " + str(originSum) + " = " + str(round(noiseSum / originSum, 10))
    #           + ',' + str(correctWholeSum) + " / " + str(originWholeSum) + " = " + str(
    #     round(correctWholeSum / originWholeSum, 10))
    #           + ',' + str(randomWholeSum) + " / " + str(originWholeSum) + " = " + str(
    #     round(randomWholeSum / originWholeSum, 10))
    #           + ',' + str(noiseWholeSum) + " / " + str(originWholeSum) + " = " + str(
    #     round(noiseWholeSum / originWholeSum, 10)) + '\n')

    print("entropy")
    print("\033[0;31;40ma", entropy_a_sum / times, "\033[0m")
    print("\033[0;31;40mb", entropy_b_sum / times, "\033[0m")
    print("\033[0;31;40me", entropy_e_sum / times, "\033[0m")
    print("\033[0;31;40mn", entropy_n_sum / times, "\033[0m")
    print("sample entropy")

    print("\033[0;31;40ma", sample_entropy_a_sum / times, "\033[0m")
    print("\033[0;31;40mb", sample_entropy_b_sum / times, "\033[0m")
    print("\033[0;31;40me", sample_entropy_e_sum / times, "\033[0m")
    print("\033[0;31;40mn", sample_entropy_n_sum / times, "\033[0m")

    csv.write(fileName + ',' + str(times) + ',' + str(segLen) + ',' +
              str(round(entropy_a_sum / times, 10)) + ',' + str(round(entropy_b_sum / times, 10)) + ',' +
              str(round(entropy_e_sum / times, 10)) + ',' + str(round(entropy_n_sum / times, 10)) + ',' +
              str(round(sample_entropy_a_sum / times, 10)) + ',' +
              str(round(sample_entropy_b_sum / times, 10)) + ',' +
              str(round(sample_entropy_e_sum / times, 10)) + ',' +
              str(round(sample_entropy_n_sum / times, 10)) + ',' + '\n')
