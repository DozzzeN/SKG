import csv
import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.stats import pearsonr


def smooth(x, window_len=11, window='hanning'):
    # ndim返回数组的维度
    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser'")

    # np.r_拼接多个数组，要求待拼接的多个数组的列数必须相同
    # 切片[开始索引:结束索引:步进长度]
    # 使用算术平均矩阵来平滑数据
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        # 元素为float，返回window_len个1.的数组
        w = np.ones(window_len, 'd')
    elif window == 'kaiser':
        beta = 5
        w = eval('np.' + window + '(window_len, beta)')
    else:
        w = eval('np.' + window + '(window_len)')

    # 进行卷积操作
    y = np.convolve(w / w.sum(), s, mode='valid')  # 6759
    return y


rawData = loadmat("../data/data_static_indoor_1.mat")

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
# CSIa1Orig = rawData['A'][:, 0][9000:20000]
# CSIb1Orig = rawData['A'][:, 1][9000:20000]
# stalking attack
CSIe2Orig = loadmat("../data/data_static_indoor_1.mat")['A'][:, 0]
dataLen = min(len(CSIe2Orig), len(CSIa1Orig))

segLen = 5
keyLen = 256 * segLen
rec = False  # 效果有限
originSum = 0
correctSum = 0
randomSum = 0
noiseSum = 0

originDecSum = 0
correctDecSum = 0
randomDecSum = 0
noiseDecSum = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum = 0
noiseWholeSum = 0

times = 0
overhead = 0

addNoise = "mul"
codings = ""

for staInd in range(0, int(dataLen / 1), int(keyLen)):
    start = time.time()
    endInd = staInd + keyLen
    print("range:", staInd, endInd)
    if endInd >= len(CSIa1Orig):
        break
    times += 1

    np.random.seed(0)
    CSIa1Orig = rawData['A'][:, 0]
    CSIb1Orig = rawData['A'][:, 1]

    CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=len(CSIa1Orig))
    # CSIe1Orig = loadmat("../data/data_eave_mobile_1.mat")['A'][:, 0]

    # noiseOrig = np.random.normal(np.mean(CSIa1Orig), np.std(CSIa1Orig), size=len(CSIa1Orig))
    # noiseOrig = np.random.normal(0, np.std(CSIa1Orig), size=len(CSIa1Orig))
    # np.random.seed(int(seeds[times - 1][0]))
    seed = np.random.randint(100000)
    np.random.seed(seed)

    if addNoise == "add":
        noiseOrig = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=len(CSIa1Orig))
        CSIa1Orig = (CSIa1Orig - np.mean(CSIa1Orig)) + noiseOrig
        CSIb1Orig = (CSIb1Orig - np.mean(CSIb1Orig)) + noiseOrig
        CSIe1Orig = (CSIe1Orig - np.mean(CSIe1Orig)) + noiseOrig
        CSIn1Orig = noiseOrig

        tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
        tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
        tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
        tmpNoise = CSIn1Orig[range(staInd, endInd, 1)]
    elif addNoise == "mul":
        # 固定随机置换的种子
        # np.random.seed(0)
        # combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig))
        # np.random.shuffle(combineCSIx1Orig)
        # CSIa1Orig, CSIb1Orig = zip(*combineCSIx1Orig)

        CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
        CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')

        tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
        tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
        tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]

        # randomMatrix = np.random.normal(np.mean(tmpCSIa1), np.std(tmpCSIa1), size=(keyLen, keyLen))
        randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
        tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
        tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
        tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
        tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
        tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)
        tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix)
        tmpNoise = np.matmul(np.ones(keyLen), randomMatrix)
    else:
        CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
        CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')

        tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
        tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
        tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]

        tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
        tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
        tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
        tmpNoise = np.random.normal(np.mean(tmpCSIa1), np.std(tmpCSIa1), size=keyLen)

    # 最后各自的密钥
    a_list = []
    b_list = []
    e_list = []
    n_list = []

    # without sorting
    # tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
    # tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()
    # tmpCSIe1Ind = np.array(tmpCSIe1).argsort().argsort()
    # tmpCSIn1Ind = np.array(tmpNoise).argsort().argsort()

    tmpCSIa1Ind = np.array(tmpCSIa1)
    tmpCSIb1Ind = np.array(tmpCSIb1)
    tmpCSIe1Ind = np.array(tmpCSIe1)
    tmpCSIn1Ind = np.array(tmpNoise)

    minEpiIndClosenessLsb = np.zeros(int(keyLen / segLen), dtype=int)
    minEpiIndClosenessLse = np.zeros(int(keyLen / segLen), dtype=int)
    minEpiIndClosenessLsn = np.zeros(int(keyLen / segLen), dtype=int)

    tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)
    permutation = list(range(int(keyLen / segLen)))
    combineMetric = list(zip(tmpCSIa1IndReshape, permutation))
    np.random.seed(staInd)
    np.random.shuffle(combineMetric)
    tmpCSIa1IndReshape, permutation = zip(*combineMetric)
    tmpCSIa1Ind = np.hstack((tmpCSIa1IndReshape))

    for i in range(int(keyLen / segLen)):
        epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]

        epiIndClosenessLsb = np.zeros(int(keyLen / segLen))
        epiIndClosenessLse = np.zeros(int(keyLen / segLen))
        epiIndClosenessLsn = np.zeros(int(keyLen / segLen))

        for j in range(int(keyLen / segLen)):
            epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
            epiInde1 = tmpCSIe1Ind[j * segLen: (j + 1) * segLen]
            epiIndn1 = tmpCSIn1Ind[j * segLen: (j + 1) * segLen]

            epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))
            epiIndClosenessLse[j] = sum(abs(epiInde1 - np.array(epiInda1)))
            epiIndClosenessLsn[j] = sum(abs(epiIndn1 - np.array(epiInda1)))

        minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)
        minEpiIndClosenessLse[i] = np.argmin(epiIndClosenessLse)
        minEpiIndClosenessLsn[i] = np.argmin(epiIndClosenessLsn)

    # a_list_number = list(range(int(keyLen / segLen)))
    a_list_number = list(permutation)
    b_list_number = list(minEpiIndClosenessLsb)
    e_list_number = list(minEpiIndClosenessLse)
    n_list_number = list(minEpiIndClosenessLsn)

    # 转成二进制，0填充成0000
    for i in range(len(a_list_number)):
        number = bin(a_list_number[i])[2:].zfill(int(np.log2(len(a_list_number))))
        a_list += number
    for i in range(len(b_list_number)):
        number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
        b_list += number
    for i in range(len(e_list_number)):
        number = bin(e_list_number[i])[2:].zfill(int(np.log2(len(e_list_number))))
        e_list += number
    for i in range(len(n_list_number)):
        number = bin(n_list_number[i])[2:].zfill(int(np.log2(len(n_list_number))))
        n_list += number

    # 对齐密钥，随机补全
    for i in range(len(a_list) - len(e_list)):
        e_list += str(np.random.randint(0, 2))
    for i in range(len(a_list) - len(n_list)):
        n_list += str(np.random.randint(0, 2))

    # print("keys of a:", len(a_list), a_list)
    print("keys of a:", len(a_list_number), a_list_number)
    # print("keys of b:", len(b_list), b_list)
    print("keys of b:", len(b_list_number), b_list_number)
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

        # 自适应纠错
        if sum1 != sum2 and rec:
            # a告诉b哪些位置出错，b对其纠错
            # for i in range(len(a_list_number)):
            #     if a_list_number[i] != b_list_number[i]:
            #         epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]
            #
            #         epiIndClosenessLsb = np.zeros(int(keyLen / segLen))
            #
            #         for j in range(int(keyLen / segLen)):
            #             epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
            #             epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))
            #
            #         min_b = np.argmin(epiIndClosenessLsb)
            #         epiIndClosenessLsb[min_b] = keyLen * segLen
            #         b_list_number[i] = np.argmin(epiIndClosenessLsb)
            #
            #         b_list = []
            #
            #         for i in range(len(b_list_number)):
            #             number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
            #             b_list += number
            #
            #         # print("keys of b:", len(b_list_number), b_list_number)
            #
            #         sum2 = 0
            #         for i in range(0, min(len(a_list), len(b_list))):
            #             sum2 += (a_list[i] == b_list[i])

            trueError = []
            for i in range(len(a_list_number)):
                if a_list_number[i] != b_list_number[i]:
                    trueError.append(i)
            print("true error", trueError)
            print("a-b", sum2, sum2 / sum1)
            reconciliation = b_list_number.copy()
            reconciliation.sort()

            repeatInd = []
            # 检查两个候选
            closeness = []
            for i in range(len(reconciliation) - 1):
                # 相等的索引就是密钥出错的地方
                if reconciliation[i] == reconciliation[i + 1]:
                    repeatInd.append(reconciliation[i])
            repeatNumber = []
            for i in range(len(repeatInd)):
                tmp = []
                for j in range(len(b_list_number)):
                    if repeatInd[i] == b_list_number[j]:
                        tmp.append(j)
                repeatNumber.append(tmp)
            for i in range(len(repeatNumber)):
                tmp = []
                for j in range(len(repeatNumber[i])):
                    epiInda1 = tmpCSIa1Ind[repeatNumber[i][j] * segLen:(repeatNumber[i][j] + 1) * segLen]

                    epiIndClosenessLsb = np.zeros(int(keyLen / segLen))

                    for k in range(int(keyLen / segLen)):
                        epiIndb1 = tmpCSIb1Ind[k * segLen: (k + 1) * segLen]

                        epiIndClosenessLsb[k] = sum(abs(epiIndb1 - np.array(epiInda1)))

                    min_b = np.argmin(epiIndClosenessLsb)
                    tmp.append(epiIndClosenessLsb[min_b])

                closeness.append(tmp)

            errorInd = []

            for i in range(len(closeness)):
                for j in range(len(closeness[i]) - 1):
                    if closeness[i][j] < closeness[i][j + 1]:
                        errorInd.append(repeatNumber[i][j + 1])
                    else:
                        errorInd.append(repeatNumber[i][j])
            print(errorInd)
            b_list_number1 = b_list_number.copy()
            for i in errorInd:
                epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]
                epiIndClosenessLsb = np.zeros(int(keyLen / segLen))

                for j in range(int(keyLen / segLen)):
                    epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
                    epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))

                min_b = np.argmin(epiIndClosenessLsb)
                while min_b in b_list_number:
                    epiIndClosenessLsb[min_b] = keyLen * segLen
                    min_b = np.argmin(epiIndClosenessLsb)
                b_list_number1[i] = min_b

            b_list = []

            for i in range(len(b_list_number1)):
                number = bin(b_list_number1[i])[2:].zfill(int(np.log2(len(b_list_number1))))
                b_list += number

            print("keys of b:", len(b_list_number1), b_list_number1)

            sum2 = 0
            for i in range(0, min(len(a_list), len(b_list))):
                sum2 += (a_list[i] == b_list[i])

            if sum1 == sum2:
                b_list_number = b_list_number1

            # 二次纠错
            if sum1 != sum2:
                for r in range(len(repeatNumber)):
                    tmp = list(set(repeatNumber[r]) - set(errorInd))
                    errorInd = tmp
                    print(errorInd)
                    b_list_number2 = b_list_number.copy()
                    for i in errorInd:
                        epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]
                        epiIndClosenessLsb = np.zeros(int(keyLen / segLen))

                        for j in range(int(keyLen / segLen)):
                            epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
                            epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))

                        min_b = np.argmin(epiIndClosenessLsb)
                        while min_b in b_list_number:
                            epiIndClosenessLsb[min_b] = keyLen * segLen
                            min_b = np.argmin(epiIndClosenessLsb)
                        b_list_number2[i] = min_b

                    b_list = []

                    for i in range(len(b_list_number2)):
                        number = bin(b_list_number2[i])[2:].zfill(int(np.log2(len(b_list_number2))))
                        b_list += number

                    # print("keys of b:", len(b_list_number2), b_list_number2)

                    sum2 = 0
                    for i in range(0, min(len(a_list), len(b_list))):
                        sum2 += (a_list[i] == b_list[i])

                    if sum1 == sum2:
                        b_list_number = b_list_number2

    end = time.time()
    overhead += end - start
    print("time:", end - start)

    print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
    print("a-e", sum3, sum3 / sum1)
    print("a-n", sum4, sum4 / sum1)
    originSum += sum1
    correctSum += sum2
    randomSum += sum3
    noiseSum += sum4

    decSum1 = min(len(a_list_number), len(b_list_number))
    decSum2 = 0
    decSum3 = 0
    decSum4 = 0
    for i in range(0, decSum1):
        decSum2 += (a_list_number[i] == b_list_number[i])
    for i in range(min(len(a_list_number), len(e_list_number))):
        decSum3 += (a_list_number[i] == e_list_number[i])
    for i in range(min(len(a_list_number), len(n_list_number))):
        decSum4 += (a_list_number[i] == n_list_number[i])
    if decSum1 == 0:
        continue
    if decSum2 == decSum1:
        print("\033[0;32;40ma-b dec", decSum2, decSum2 / decSum1, "\033[0m")
    else:
        print("\033[0;31;40ma-b dec", "bad", decSum2, decSum2 / decSum1, "\033[0m")
    print("a-e", decSum3, decSum3 / decSum1)
    print("a-n", decSum4, decSum4 / decSum1)
    print("----------------------")
    originDecSum += decSum1
    correctDecSum += decSum2
    randomDecSum += decSum3
    noiseDecSum += decSum4

    originWholeSum += 1
    correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
    randomWholeSum = randomWholeSum + 1 if sum3 == sum1 else randomWholeSum
    noiseWholeSum = noiseWholeSum + 1 if sum4 == sum1 else noiseWholeSum

print("a-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10))
print("a-e all", randomSum, "/", originSum, "=", round(randomSum / originSum, 10))
print("a-n all", noiseSum, "/", originSum, "=", round(noiseSum / originSum, 10))
print("a-b dec", correctDecSum, "/", originDecSum, "=", round(correctDecSum / originDecSum, 10))
print("a-e dec", randomDecSum, "/", originDecSum, "=", round(randomDecSum / originDecSum, 10))
print("a-n dec", noiseDecSum, "/", originDecSum, "=", round(noiseDecSum / originDecSum, 10))
print("a-b whole match", correctWholeSum, "/", originWholeSum, "=", round(correctWholeSum / originWholeSum, 10))
print("a-e whole match", randomWholeSum, "/", originWholeSum, "=", round(randomWholeSum / originWholeSum, 10))
print("a-n whole match", noiseWholeSum, "/", originWholeSum, "=", round(noiseWholeSum / originWholeSum, 10))
print("times", times)
print(overhead / originWholeSum)
