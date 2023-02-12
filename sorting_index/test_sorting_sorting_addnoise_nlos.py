import csv
import math
import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import dct
from scipy.io import loadmat
from scipy.stats import pearsonr, boxcox

from zca import ZCA


# 定义计算离散点积分的函数
def integral_from_start(x, y):
    import scipy
    from scipy.integrate import simps  # 用于计算积分
    integrals = []
    for i in range(len(y)):  # 计算梯形的面积，由于是累加，所以是切片"i+1"
        integrals.append(scipy.integrate.trapz(y[:i + 1], x[:i + 1]))
    return integrals


# 定义计算离散点导数的函数
def derivative(x, y):  # x, y的类型均为列表
    diff_x = []  # 用来存储x列表中的两数之差
    for i, j in zip(x[0::], x[1::]):
        diff_x.append(j - i)

    diff_y = []  # 用来存储y列表中的两数之差
    for i, j in zip(y[0::], y[1::]):
        diff_y.append(j - i)

    slopes = []  # 用来存储斜率
    for i in range(len(diff_y)):
        slopes.append(diff_y[i] / diff_x[i])

    deriv = []  # 用来存储一阶导数
    for i, j in zip(slopes[0::], slopes[1::]):
        deriv.append((0.5 * (i + j)))  # 根据离散点导数的定义，计算并存储结果
    deriv.insert(0, slopes[0])  # (左)端点的导数即为与其最近点的斜率
    deriv.append(slopes[-1])  # (右)端点的导数即为与其最近点的斜率
    return deriv  # 返回存储一阶导数结果的列表


def integral_sq_derivative_increment(data, noise):
    index = list(range(len(data)))
    intgrl = integral_from_start(index, data)
    # square = np.power(intgrl, 2)
    square = intgrl + noise
    diff = derivative(index, square)
    return diff


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


def normal2uniform(data):
    data1 = data[:int(len(data) / 2)]
    data2 = data[int(len(data) / 2):]
    data_reshape = np.array(data[0: 2 * int(len(data) / 2)])
    data_reshape = data_reshape.reshape(int(len(data_reshape) / 2), 2)
    x_list = []
    for i in range(len(data_reshape)):
        # r = np.sum(np.square(data_reshape[i]))
        r = np.sum(data1[i] * data1[i] + data2[i] * data2[i])
        x_list.append(np.exp(-0.5 * r))
        x_list.append(np.exp(-0.5 * r))
        # r = data2[i] / data1[i]
        # x_list.append(math.atan(r) / math.pi + 0.5)
        # x_list.append(math.atan(r) / math.pi + 0.5)

    return x_list


rawData = loadmat("../data/data_NLOS.mat")

# CSIa1Orig = rawData['A'][:, 0][0: 20000]
# CSIb1Orig = rawData['A'][:, 1][0: 20000]
CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
dataLen = len(CSIa1Orig)

segLen = 5
# 256的性能差，用128代替
keyLen = 128 * segLen
rec = True
tell = True

originSum = 0
correctSum = 0
randomSum1 = 0
randomSum2 = 0
noiseSum = 0

originDecSum = 0
correctDecSum = 0
randomDecSum1 = 0
randomDecSum2 = 0
noiseDecSum = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum1 = 0
randomWholeSum2 = 0
noiseWholeSum = 0

times = 0
overhead = 0

addNoise = "mul"
codings = ""

# static indoor
# for staInd in range(0, int(dataLen / 5.5), int(keyLen / 10)):
# for staInd in range(0, int(dataLen / 5), int(keyLen / 5)):
for staInd in range(0, dataLen, keyLen):
    cnt = 0
    maxSum2 = -1
    while True:
        start = time.time()
        cnt += 1
        endInd = staInd + keyLen
        print("range:", staInd, endInd)
        if endInd >= len(CSIa1Orig):
            break
        times += 1

        # np.random.seed(1)
        CSIa1Orig = rawData['A'][:, 0]
        CSIb1Orig = rawData['A'][:, 1]

        # imitation attack
        CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=len(CSIa1Orig))
        # stalking attack
        CSIe2Orig = loadmat("../skyglow/Scenario2-Office-LoS-eve_NLoS/data_eave_LOS_EVE_NLOS.mat")['A'][:, 0]

        # noiseOrig = np.random.normal(np.mean(CSIa1Orig), np.std(CSIa1Orig), size=len(CSIa1Orig))
        # noiseOrig = np.random.normal(0, np.std(CSIa1Orig), size=len(CSIa1Orig))
        # np.random.seed(int(seeds[times - 1][0]))
        seed = np.random.randint(100000)
        np.random.seed(seed)

        if addNoise == "add":
            # noiseOrig = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=len(CSIa1Orig))
            # CSIa1Orig = (CSIa1Orig - np.mean(CSIa1Orig)) + noiseOrig
            # CSIb1Orig = (CSIb1Orig - np.mean(CSIb1Orig)) + noiseOrig
            # CSIe1Orig = (CSIe1Orig - np.mean(CSIe1Orig)) + noiseOrig
            # CSIn1Orig = noiseOrig

            noiseOrig = np.random.normal(0, np.std(CSIa1Orig), size=len(CSIa1Orig))
            CSIa1Orig = CSIa1Orig + noiseOrig
            CSIb1Orig = CSIb1Orig + noiseOrig
            CSIe1Orig = CSIe1Orig + noiseOrig
            CSIe2Orig = CSIe2Orig + noiseOrig
            CSIn1Orig = noiseOrig

            tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
            tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
            tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
            tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]
            tmpNoise = CSIn1Orig[range(staInd, endInd, 1)]
        elif addNoise == "mul":
            # 静态数据需要置换
            # 固定随机置换的种子
            np.random.seed(0)
            combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig))
            np.random.shuffle(combineCSIx1Orig)
            CSIa1Orig, CSIb1Orig = zip(*combineCSIx1Orig)

            CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
            CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')

            tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
            tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
            tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
            tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]

            randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
            tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
            tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
            tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
            tmpCSIe2 = tmpCSIe2 - np.mean(tmpCSIe2)
            tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
            tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)
            tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix)
            tmpCSIe2 = np.matmul(tmpCSIe2, randomMatrix)
            # inference attack
            tmpNoise = np.matmul(np.ones(keyLen), randomMatrix)
        else:
            tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
            tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
            tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
            tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]
            tmpNoise = np.random.normal(0, np.std(CSIa1Orig), size=keyLen)

        # tmpCSIa1 = np.array(integral_sq_derivative_increment(tmpCSIa1, tmpNoise)) * tmpCSIa1
        # tmpCSIb1 = np.array(integral_sq_derivative_increment(tmpCSIb1, tmpNoise)) * tmpCSIb1
        # tmpCSIe1 = np.array(integral_sq_derivative_increment(tmpCSIe1, tmpNoise)) * tmpCSIe1
        print("correlation a-e1", pearsonr(tmpCSIa1, tmpCSIe1)[0])
        print("correlation a-e2", pearsonr(tmpCSIa1, tmpCSIe2)[0])
        print("correlation a-n", pearsonr(tmpCSIa1, tmpNoise)[0])
        # tmpNoise = np.array(integral_sq_derivative_increment(np.ones(keyLen), tmpNoise)) * np.ones(keyLen)
        # print("correlation a-n'", pearsonr(tmpCSIa1, tmpNoise)[0])

        # tmpCSIe1 = np.random.normal(loc=np.mean(tmpCSIe1), scale=np.std(tmpCSIe1, ddof=1), size=len(tmpCSIe1))

        # noise = np.random.uniform(0, 1, keyLen)
        # tmpCSIa1 = np.power(np.abs(tmpCSIa1), noise)
        # tmpCSIb1 = np.power(np.abs(tmpCSIb1), noise)
        # tmpCSIe1 = np.power(np.abs(tmpCSIe1), noise)
        # tmpNoise = np.power(np.abs(tmpNoise), noise)

        # box-muller
        # tmpCSIa1 = boxcox(np.abs(tmpCSIa1))[0]
        # tmpCSIb1 = boxcox(np.abs(tmpCSIb1))[0]
        # tmpCSIe1 = boxcox(np.abs(tmpCSIe1))[0]
        # tmpNoise = boxcox(np.abs(tmpNoise))[0]
        # tmpCSIa1 = normal2uniform(tmpCSIa1)
        # tmpCSIb1 = normal2uniform(tmpCSIb1)
        # tmpCSIe1 = normal2uniform(tmpCSIe1)
        # tmpNoise = normal2uniform(tmpNoise)

        # plt.figure()
        # plt.plot(np.sort(tmpCSIa1))
        # plt.show()

        # 最后各自的密钥
        a_list = []
        b_list = []
        e1_list = []
        e2_list = []
        n_list = []

        tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
        tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()
        tmpCSIe1Ind = np.array(tmpCSIe1).argsort().argsort()
        tmpCSIe2Ind = np.array(tmpCSIe2).argsort().argsort()
        tmpCSIn1Ind = np.array(tmpNoise).argsort().argsort()

        minEpiIndClosenessLsb = np.zeros(int(keyLen / segLen), dtype=int)
        minEpiIndClosenessLse1 = np.zeros(int(keyLen / segLen), dtype=int)
        minEpiIndClosenessLse2 = np.zeros(int(keyLen / segLen), dtype=int)
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
            epiIndClosenessLse1 = np.zeros(int(keyLen / segLen))
            epiIndClosenessLse2 = np.zeros(int(keyLen / segLen))
            epiIndClosenessLsn = np.zeros(int(keyLen / segLen))

            for j in range(int(keyLen / segLen)):
                epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
                epiInde1 = tmpCSIe1Ind[j * segLen: (j + 1) * segLen]
                epiInde2 = tmpCSIe2Ind[j * segLen: (j + 1) * segLen]
                epiIndn1 = tmpCSIn1Ind[j * segLen: (j + 1) * segLen]

                epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))
                epiIndClosenessLse1[j] = sum(abs(epiInde1 - np.array(epiInda1)))
                epiIndClosenessLse2[j] = sum(abs(epiInde2 - np.array(epiInda1)))
                epiIndClosenessLsn[j] = sum(abs(epiIndn1 - np.array(epiInda1)))

            minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)
            minEpiIndClosenessLse1[i] = np.argmin(epiIndClosenessLse1)
            minEpiIndClosenessLse2[i] = np.argmin(epiIndClosenessLse2)
            minEpiIndClosenessLsn[i] = np.argmin(epiIndClosenessLsn)

        # a_list_number = list(range(int(keyLen / segLen)))
        a_list_number = list(permutation)
        b_list_number = list(minEpiIndClosenessLsb)
        e1_list_number = list(minEpiIndClosenessLse1)
        e2_list_number = list(minEpiIndClosenessLse2)
        n_list_number = list(minEpiIndClosenessLsn)

        # 转成二进制，0填充成0000
        for i in range(len(a_list_number)):
            number = bin(a_list_number[i])[2:].zfill(int(np.log2(len(a_list_number))))
            a_list += number
        for i in range(len(b_list_number)):
            number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
            b_list += number
        for i in range(len(e1_list_number)):
            number = bin(e1_list_number[i])[2:].zfill(int(np.log2(len(e1_list_number))))
            e1_list += number
        for i in range(len(e2_list_number)):
            number = bin(e2_list_number[i])[2:].zfill(int(np.log2(len(e2_list_number))))
            e2_list += number
        for i in range(len(n_list_number)):
            number = bin(n_list_number[i])[2:].zfill(int(np.log2(len(n_list_number))))
            n_list += number

        # 对齐密钥，随机补全
        for i in range(len(a_list) - len(e1_list)):
            e1_list += str(np.random.randint(0, 2))
        for i in range(len(a_list) - len(e2_list)):
            e2_list += str(np.random.randint(0, 2))
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
        sum31 = 0
        sum32 = 0
        sum4 = 0
        for i in range(0, sum1):
            sum2 += (a_list[i] == b_list[i])
        for i in range(min(len(a_list), len(e1_list))):
            sum31 += (a_list[i] == e1_list[i])
        for i in range(min(len(a_list), len(e2_list))):
            sum32 += (a_list[i] == e2_list[i])
        for i in range(min(len(a_list), len(n_list))):
            sum4 += (a_list[i] == n_list[i])

        end = time.time()
        overhead += end - start
        print("time:", end - start)

        # 自适应纠错
        if sum1 != sum2 and rec:
            if tell:
                # a告诉b哪些位置出错，b对其纠错
                for i in range(len(a_list_number)):
                    if a_list_number[i] != b_list_number[i]:
                        epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]

                        epiIndClosenessLsb = np.zeros(int(keyLen / segLen))

                        for j in range(int(keyLen / segLen)):
                            epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
                            epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))

                        min_b = np.argmin(epiIndClosenessLsb)
                        epiIndClosenessLsb[min_b] = keyLen * segLen
                        b_list_number[i] = np.argmin(epiIndClosenessLsb)

                        b_list = []

                        for i in range(len(b_list_number)):
                            number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
                            b_list += number

                        # print("keys of b:", len(b_list_number), b_list_number)

                        sum2 = 0
                        for i in range(0, min(len(a_list), len(b_list))):
                            sum2 += (a_list[i] == b_list[i])
            else:
                # 正式纠错
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
                # 正式纠错 end

        maxSum2 = max(sum2, maxSum2)
        print("sum2", maxSum2, sum1)

        if cnt >= 3 or maxSum2 == sum1:
            print("\033[0;32;40ma-b", maxSum2, maxSum2 / sum1, "\033[0m")
            print("a-e1", sum31, sum31 / sum1)
            print("a-e2", sum32, sum32 / sum1)
            print("a-n", sum4, sum4 / sum1)
            originSum += sum1
            correctSum += maxSum2
            randomSum1 += sum31
            randomSum2 += sum32
            noiseSum += sum4

            # decSum1 = min(len(a_list_number), len(b_list_number))
            # decSum2 = 0
            # decSum31 = 0
            # decSum32 = 0
            # decSum4 = 0
            # for i in range(0, decSum1):
            #     decSum2 += (a_list_number[i] == b_list_number[i])
            # for i in range(min(len(a_list_number), len(e1_list_number))):
            #     decSum31 += (a_list_number[i] == e1_list_number[i])
            # for i in range(min(len(a_list_number), len(e2_list_number))):
            #     decSum32 += (a_list_number[i] == e2_list_number[i])
            # for i in range(min(len(a_list_number), len(n_list_number))):
            #     decSum4 += (a_list_number[i] == n_list_number[i])
            # if decSum1 == 0:
            #     continue
            # if decSum2 == decSum1:
            #     print("\033[0;32;40ma-b dec", decSum2, decSum2 / decSum1, "\033[0m")
            # else:
            #     print("\033[0;31;40ma-b dec", "bad", decSum2, decSum2 / decSum1, "\033[0m")
            # print("a-e1", decSum31, decSum31 / decSum1)
            # print("a-e2", decSum32, decSum32 / decSum1)
            # print("a-n", decSum4, decSum4 / decSum1)
            # print("----------------------")
            # originDecSum += decSum1
            # correctDecSum += decSum2
            # randomDecSum1 += decSum31
            # randomDecSum2 += decSum32
            # noiseDecSum += decSum4

            originWholeSum += 1
            correctWholeSum = correctWholeSum + 1 if maxSum2 == sum1 else correctWholeSum
            randomWholeSum1 = randomWholeSum1 + 1 if sum31 == sum1 else randomWholeSum1
            randomWholeSum2 = randomWholeSum2 + 1 if sum32 == sum1 else randomWholeSum2
            noiseWholeSum = noiseWholeSum + 1 if sum4 == sum1 else noiseWholeSum
            break

    # coding = ""
    # for i in range(len(a_list)):
    #     coding += a_list[i]
    # codings += coding + "\n"
    #
    # with open('./key/' + fileName + ".txt", 'a', ) as f:
    #     f.write(codings)

print("\033[0;34;40ma-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10), "\033[0m")
print("a-e1 all", randomSum1, "/", originSum, "=", round(randomSum1 / originSum, 10))
print("a-e2 all", randomSum2, "/", originSum, "=", round(randomSum2 / originSum, 10))
print("a-n all", noiseSum, "/", originSum, "=", round(noiseSum / originSum, 10))
# print("a-b dec", correctDecSum, "/", originDecSum, "=", round(correctDecSum / originDecSum, 10))
# print("a-e1 dec", randomDecSum1, "/", originDecSum, "=", round(randomDecSum1 / originDecSum, 10))
# print("a-e2 dec", randomDecSum2, "/", originDecSum, "=", round(randomDecSum2 / originDecSum, 10))
# print("a-n dec", noiseDecSum, "/", originDecSum, "=", round(noiseDecSum / originDecSum, 10))
print("\033[0;34;40ma-b whole match", correctWholeSum, "/", originWholeSum, "=",
      round(correctWholeSum / originWholeSum, 10), "\033[0m")
print("a-e1 whole match", randomWholeSum1, "/", originWholeSum, "=", round(randomWholeSum1 / originWholeSum, 10))
print("a-e2 whole match", randomWholeSum2, "/", originWholeSum, "=", round(randomWholeSum2 / originWholeSum, 10))
print("a-n whole match", noiseWholeSum, "/", originWholeSum, "=", round(noiseWholeSum / originWholeSum, 10))
print("times", times)
print(overhead / originWholeSum)
