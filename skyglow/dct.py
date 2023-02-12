import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct
from scipy.io import loadmat

fileName = "../data/data_mobile_indoor_1.mat"
rawData = loadmat(fileName)
csv = open("../edit_distance/evaluations/comparison_dct.csv", "a+")
CSIa1OrigRaw = rawData['A'][:, 0]
CSIb1OrigRaw = rawData['A'][:, 1]
minLen = min(len(CSIa1OrigRaw), len(CSIb1OrigRaw))
CSIa1Orig = CSIa1OrigRaw[:minLen]
CSIb1Orig = CSIb1OrigRaw[:minLen]
dataLen = len(CSIa1Orig)
CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)
CSIn1Orig = np.random.normal(loc=-1, scale=1, size=dataLen)  ## Multiplication item normal distribution

intvl = 5
keyLen = 64
times = 0

originSum = 0
correctSum = 0
randomSum = 0
noiseSum = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum = 0
noiseWholeSum = 0

codings = ""

# CSIa1Orig = CSIa1Orig - np.mean(CSIa1Orig)
# CSIb1Orig = CSIb1Orig - np.mean(CSIb1Orig)
# CSIe1Orig = CSIe1Orig - np.mean(CSIe1Orig)
# CSIn1Orig = CSIn1Orig - np.mean(CSIn1Orig)

# CSIa1Orig = dct(CSIa1Orig)
# CSIb1Orig = dct(CSIb1Orig)
# CSIe1Orig = dct(CSIe1Orig)
# CSIn1Orig = dct(CSIn1Orig)

for staInd in range(0, len(CSIa1Orig), intvl * keyLen):
    endInd = staInd + keyLen * intvl
    print("range:", staInd, endInd)
    if endInd >= len(CSIa1Orig):
        break
    times += 1

    permLen = len(range(staInd, endInd, intvl))
    origInd = np.array([xx for xx in range(staInd, endInd, 1)])

    CSIa1Epi = CSIa1Orig[origInd]
    CSIb1Epi = CSIb1Orig[origInd]

    CSIa1Orig[origInd] = CSIa1Epi
    CSIb1Orig[origInd] = CSIb1Epi

    CSIa1Orig[range(staInd, endInd, 1)] = CSIa1Orig[range(staInd, endInd, 1)] - np.mean(CSIa1Orig[range(staInd, endInd, 1)])
    CSIb1Orig[range(staInd, endInd, 1)] = CSIb1Orig[range(staInd, endInd, 1)] - np.mean(CSIb1Orig[range(staInd, endInd, 1)])
    CSIe1Orig[range(staInd, endInd, 1)] = CSIe1Orig[range(staInd, endInd, 1)] - np.mean(CSIe1Orig[range(staInd, endInd, 1)])
    CSIn1Orig[range(staInd, endInd, 1)] = CSIn1Orig[range(staInd, endInd, 1)] - np.mean(CSIn1Orig[range(staInd, endInd, 1)])

    dctCSIa1 = dct(CSIa1Orig[range(staInd, endInd, 1)])
    dctCSIb1 = dct(CSIb1Orig[range(staInd, endInd, 1)])
    dctCSIe1 = dct(CSIe1Orig[range(staInd, endInd, 1)])
    dctCSIn1 = dct(CSIn1Orig[range(staInd, endInd, 1)])

    # dctCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    # dctCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
    # dctCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
    # dctCSIn1 = CSIn1Orig[range(staInd, endInd, 1)]

    mean_a = np.mean(dctCSIa1)
    mean_b = np.mean(dctCSIb1)
    mean_e = np.mean(dctCSIe1)
    mean_n = np.mean(dctCSIn1)

    std_a = np.std(dctCSIa1)
    std_b = np.std(dctCSIb1)
    std_e = np.std(dctCSIe1)
    std_n = np.std(dctCSIn1)

    a_list = []
    b_list = []
    e_list = []
    n_list = []

    for i in range(len(dctCSIe1)):
        if dctCSIa1[i] > mean_a + std_a:
            a_list.append("11")
        elif dctCSIa1[i] <= mean_a + std_a and dctCSIa1[i] > mean_a:
            a_list.append("10")
        elif dctCSIa1[i] <= mean_a and dctCSIa1[i] > mean_a - std_a:
            a_list.append("01")
        elif dctCSIa1[i] <= mean_a - std_a:
            a_list.append("00")

    for i in range(len(dctCSIb1)):
        if dctCSIb1[i] > mean_b + std_b:
            b_list.append("11")
        elif dctCSIb1[i] <= mean_b + std_b and dctCSIb1[i] > mean_b:
            b_list.append("10")
        elif dctCSIb1[i] <= mean_b and dctCSIb1[i] > mean_b - std_b:
            b_list.append("01")
        elif dctCSIb1[i] <= mean_b - std_b:
            b_list.append("00")

    for i in range(len(dctCSIe1)):
        if dctCSIe1[i] > mean_e + std_e:
            e_list.append("11")
        elif dctCSIe1[i] <= mean_e + std_e and dctCSIe1[i] > mean_e:
            e_list.append("10")
        elif dctCSIe1[i] <= mean_e and dctCSIe1[i] > mean_e - std_e:
            e_list.append("01")
        elif dctCSIe1[i] <= mean_e - std_e:
            e_list.append("00")

    for i in range(len(dctCSIn1)):
        if dctCSIn1[i] >= mean_n + std_n:
            n_list.append("11")
        elif dctCSIn1[i] < mean_n + std_n and dctCSIn1[i] >= mean_n:
            n_list.append("10")
        elif dctCSIn1[i] < mean_n and dctCSIn1[i] > mean_n - std_n:
            n_list.append("01")
        elif dctCSIn1[i] <= mean_n - std_n:
            n_list.append("00")

    # print(a_list)
    # print(b_list)
    # print(e_list)
    # print(n_list)

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

    if sum2 == sum1:
        print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
    else:
        print("\033[0;31;40ma-b", sum2, sum2 / sum1, "\033[0m")
    print("a-e", sum3, sum3 / sum1)
    print("----------------------")
    originSum += sum1
    correctSum += sum2
    randomSum += sum3
    noiseSum += sum4

    originWholeSum += 1
    correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
    randomWholeSum = randomWholeSum + 1 if sum3 == sum1 else randomWholeSum
    noiseWholeSum = noiseWholeSum + 1 if sum4 == sum1 else noiseWholeSum

    # for i in range(len(a_list)):
    #     codings += str(a_list[i])
    # codings += "\n"

    # binary_sequence = np.array(a_list)
    # for i in range(len(binary_sequence)):
    #     binary_sequence[i] = int(binary_sequence[i], 2)
    # eligible_battery: dict = check_eligibility_all_battery(binary_sequence, SP800_22R1A_BATTERY)
    # results = run_all_battery(binary_sequence, eligible_battery, False)
    # for result, elapsed_time in results:
    #     if result.passed:
    #         print("- PASSED - score: " + str(
    #             np.round(result.score, 3)) + " - " + result.name + " - elapsed time: " + str(elapsed_time) + " ms")
    #     else:
    #         print("- FAILED - score: " + str(
    #             np.round(result.score, 3)) + " - " + result.name + " - elapsed time: " + str(elapsed_time) + " ms")

# with open('./dct.txt', 'a', ) as f:
#     f.write(codings)
print("a-b all", correctSum, "/", originSum, "=", correctSum / originSum)
print("a-e all", randomSum, "/", originSum, "=", randomSum / originSum)
print("a-n all", noiseSum, "/", originSum, "=", noiseSum / originSum)
print("a-b whole match", correctWholeSum, "/", originWholeSum, "=", correctWholeSum / originWholeSum)
print("a-e whole match", randomWholeSum, "/", originWholeSum, "=", randomWholeSum / originWholeSum)
print("a-n whole match", noiseWholeSum, "/", originWholeSum, "=", noiseWholeSum / originWholeSum)
print("times", times)
# csv.write(fileName + ',' + str(times) + ',' + '' + ',' + '' + ',' + ''
#           + ',' + str(correctSum / originSum) + ',' + str(randomSum / originSum)
#           + ',' + str(noiseSum / originSum) + ',' + str(correctWholeSum / originWholeSum)
#           + ',' + str(randomWholeSum / originWholeSum) + ',' + str(noiseWholeSum / originWholeSum) + '\n')
# csv.close()
