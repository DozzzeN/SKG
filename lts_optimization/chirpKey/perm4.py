import csv
import math
import time
from tkinter import messagebox

import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.fft import dct
from scipy.io import loadmat, savemat
from scipy.linalg import circulant, toeplitz
from scipy.stats import pearsonr, boxcox, ortho_group


def wthresh(data, threshold):
    for i in range(len(data)):
        if data[i] > threshold:
            data[i] = data[i] - threshold
        elif data[i] < -threshold:
            data[i] = data[i] + threshold
        else:
            data[i] = 0
    return data


def l2norm2(data):
    return np.sqrt(np.sum(np.square(data))) ** 2


def ass_pg_stls_f(A, b, N, K, lam, h, ni):
    # adaptive - step - size proximal - gradient
    AA = np.matmul(np.array(A).T, np.array(A))
    Ab = np.matmul(np.array(A).T, np.array(b))

    er2 = np.zeros(ni)  # error
    er0a = np.zeros(ni)  # missed detections
    er0b = np.zeros(ni)  # wrong detections
    xo = np.zeros(N)  # initialization of solution
    g = -2 * Ab
    mu0 = .2
    x = wthresh(-mu0 * g, mu0 * lam)
    y = 1 / (np.matmul(np.array(x).T, np.array(x)) + 1)
    c = y * l2norm2(np.matmul(A, x) - b)
    muo = mu0

    for nn in range(ni):
        # iterations loop
        # calculate gradient
        go = g  # g0
        co = c  # f1
        g = 2 * y * (np.matmul(AA, x) - Ab - co * x)  # gn

        #  calculate step - size
        if np.matmul(np.array(x - xo).T, (g - go)) == 0:
            mu = muo
        else:
            mus = np.matmul(np.array(x - xo).T, (x - xo)) / np.matmul(np.array(x - xo).T, (g - go))
            mum = np.matmul(np.array(x - xo).T, (g - go)) / np.matmul(np.array(g - go).T, (g - go))
            if mum / mus > .5:
                mu = mum
            else:
                mu = mus - mum / 2
            if mu <= 0:
                mu = muo

        # backtracking line-search
        while True:
            # proximal - gradient
            z = wthresh(x - mu * g, mu * lam)  # xn + 1
            y = 1 / (np.matmul(np.array(z).T, np.array(z)) + 1)
            c = y * l2norm2(np.matmul(A, z) - b)  # fn + 1
            if c <= co + np.matmul(np.array(z - x).T, g) + l2norm2(z - x) / (2 * mu):
                break
            mu = mu / 2
        muo = mu
        xo = x
        x = z

        # calculate errors
        er2[nn] = l2norm2(x - h)
        # ll = length(intersect(find(h), find(x)))
        # er0a(nn) = K - ll
        # er0b(nn) = length(find(x)) - ll
    return er2, er0a, er0b, x


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


fileName = ["../../data/data_mobile_indoor_1.mat",
            "../../data/data_mobile_outdoor_1.mat",
            "../../data/data_static_outdoor_1.mat",
            "../../data/data_static_indoor_1.mat"
            ]

# fileName = ["../../data/data_static_indoor_1.mat"]

# 是否排序
withoutSorts = [True, False]
# 是否添加噪声
addNoises = ["mul"]

bits = 2
for f in fileName:
    for addNoise in addNoises:
        for withoutSort in withoutSorts:
            print(f)
            rawData = loadmat(f)

            CSIa1Orig = rawData['A'][:, 0]
            CSIb1Orig = rawData['A'][:, 1]
            dataLen = len(CSIa1Orig)
            print("dataLen", dataLen)

            segLen = 1
            keyLen = 255 * segLen
            eta = 10
            iteration = 30

            print("segLen", segLen)
            print("keyLen", keyLen / segLen)

            originSum = 0
            correctSum = 0
            randomSum = 0

            originWholeSum = 0
            correctWholeSum = 0
            randomWholeSum = 0

            times = 0

            if withoutSort:
                if addNoise == "mul":
                    print("no sorting")
            if withoutSort:
                if addNoise == "":
                    print("no sorting and no perturbation")
            if withoutSort is False:
                if addNoise == "":
                    print("no perturbation")
                if addNoise == "mul":
                    print("normal")

            dataLenLoop = dataLen
            keyLenLoop = keyLen
            if f == "../../data/data_static_indoor_1.mat":
                dataLenLoop = int(dataLen / 5.5)
                keyLenLoop = int(keyLen / 5)
            for staInd in range(0, dataLenLoop, keyLenLoop):
                endInd = staInd + keyLen
                # print("range:", staInd, endInd)
                if endInd >= len(CSIa1Orig) or endInd >= len(CSIb1Orig):
                    break

                times += 1

                CSIa1Orig = rawData['A'][:, 0]
                CSIb1Orig = rawData['A'][:, 1]

                seed = np.random.randint(100000)
                np.random.seed(seed)

                # 固定随机置换的种子
                # if f == "../../data/data_static_indoor_1.mat":
                #     np.random.seed(0)
                #     combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig))
                #     np.random.shuffle(combineCSIx1Orig)
                #     CSIa1Orig, CSIb1Orig = zip(*combineCSIx1Orig)
                #     CSIa1Orig = np.array(CSIa1Orig)
                #     CSIb1Orig = np.array(CSIb1Orig)

                CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
                CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')

                tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
                tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
                tmpCSIe1 = np.random.normal(np.mean(tmpCSIa1), np.std(tmpCSIa1, ddof=1), len(tmpCSIa1))

                if addNoise == "mul":
                    # randomMatrix = np.random.randint(1, 4, size=(keyLen, keyLen))
                    np.random.seed(10000)
                    # randomMatrix = np.random.uniform(5, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
                    # 采用算法中相同均值和方差的随机矩阵，采用相同矩阵维度：维度越高，算法重构结果的误差越大
                    randomMatrix = np.random.uniform(0, 0.2, size=(keyLen, keyLen))
                    np.random.seed(10000)
                    randomVector = np.random.uniform(0, 0.2, keyLen)

                    # 随机正交矩阵
                    # randomMatrix = ortho_group.rvs(keyLen, keyLen)[0]
                    # 随机酉矩阵
                    # Q = ortho_group.rvs(keyLen, keyLen)[0]
                    # lambda1 = np.diag(np.random.normal(0, 0.1, keyLen))
                    # lambda2 = np.eye(keyLen) - lambda1 @ lambda1
                    # lambda2 = np.sqrt(lambda2)
                    # A1 = Q.T @ lambda1 @ Q
                    # B1 = Q.T @ lambda2 @ Q
                    # randomMatrix = A1 + np.ones((keyLen, keyLen)) * 1j * B1

                    tmpCSIa1 = (tmpCSIa1 - np.mean(tmpCSIa1)) / eta
                    tmpCSIb1 = (tmpCSIb1 - np.mean(tmpCSIb1)) / eta
                    tmpCSIe1 = (tmpCSIe1 - np.mean(tmpCSIe1)) / eta

                    # control the magnitude of the perturbation matrix
                    # tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1)) / eta
                    # if np.max(tmpCSIb1) - np.min(tmpCSIb1) == 0:
                    #     tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / np.max(tmpCSIb1) / eta
                    # else:
                    #     tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1)) / eta
                    # tmpCSIe1 = (tmpCSIe1 - np.min(tmpCSIe1)) / (np.max(tmpCSIe1) - np.min(tmpCSIe1)) / eta

                    # tmpCSIa1 = np.abs(np.fft.fft(tmpCSIa1))
                    # tmpCSIb1 = np.abs(np.fft.fft(tmpCSIb1))
                    # tmpCSIe1 = np.abs(np.fft.fft(tmpCSIe1))

                    # tmpCSIa1 = circulant(tmpCSIa1)
                    # tmpCSIb1 = circulant(tmpCSIb1)
                    # tmpCSIe1 = circulant(tmpCSIe1)
                    # tmpCSIa1 = np.matmul(randomMatrix, tmpCSIa1) + randomMatrix
                    # tmpCSIb1 = np.matmul(randomMatrix, tmpCSIb1) + randomMatrix
                    # tmpCSIe1 = np.matmul(randomMatrix, tmpCSIe1) + randomMatrix

                    # tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix) + randomVector
                    # tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix) + randomVector
                    # tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix) + randomVector
                    tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
                    tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)
                    tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix)
                    # tmpCSIe1 = np.matmul(np.ones(keyLen), randomMatrix)

                    # 相当于乘了一个置换矩阵 permutation matrix
                    # np.random.seed(0)
                    # combineCSIx1Orig = list(zip(tmpCSIa1, tmpCSIb1))
                    # np.random.shuffle(combineCSIx1Orig)
                    # tmpCSIa1, tmpCSIb1 = zip(*combineCSIx1Orig)
                    # tmpCSIa1 = np.array(tmpCSIa1)
                    # tmpCSIb1 = np.array(tmpCSIb1)
                else:
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)

                # 最后各自的密钥
                a_list = []
                b_list = []
                e_list = []

                # without sorting
                if withoutSort:
                    tmpCSIa1Ind = np.array(tmpCSIa1)
                    tmpCSIb1Ind = np.array(tmpCSIb1)
                    tmpCSIe1Ind = np.array(tmpCSIe1)
                else:
                    # control the magnitude of the perturbation matrix
                    tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
                    tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()
                    tmpCSIe1Ind = np.array(tmpCSIe1).argsort().argsort()

                    tmpCSIa1Ind = (tmpCSIa1Ind - np.min(tmpCSIa1Ind)) / (np.max(tmpCSIa1Ind) - np.min(tmpCSIa1Ind)) / eta / 10
                    tmpCSIb1Ind = (tmpCSIb1Ind - np.min(tmpCSIb1Ind)) / (np.max(tmpCSIb1Ind) - np.min(tmpCSIb1Ind)) / eta / 10
                    tmpCSIe1Ind = (tmpCSIe1Ind - np.min(tmpCSIe1Ind)) / (np.max(tmpCSIe1Ind) - np.min(tmpCSIe1Ind)) / eta / 10

                    # with shuffling
                    # np.random.seed(0)
                    # combineCSIx1Orig = list(zip(tmpCSIa1Ind, tmpCSIb1Ind, tmpCSIe1Ind))
                    # np.random.shuffle(combineCSIx1Orig)
                    # tmpCSIa1Ind, tmpCSIb1Ind, tmpCSIe1Ind = zip(*combineCSIx1Orig)
                    # tmpCSIa1Ind = list(tmpCSIa1Ind)
                    # tmpCSIb1Ind = list(tmpCSIb1Ind)
                    # tmpCSIe1Ind = list(tmpCSIe1Ind)

                np.random.seed(0)
                if bits == 1:
                    key = np.random.binomial(1, 0.5, int(keyLen / 5))
                else:
                    key = np.random.randint(1, 4, int(keyLen / 5))

                keyBin = []

                # sparse key
                for i in range(len(key)):
                    keyBin.append(key[i])
                    keyBin.extend([0, 0, 0, 0])

                # 效果差不多
                # shuffling = np.random.permutation(keyLen)
                # tmpCSIa1IndPerm = circulant(tmpCSIa1Ind[shuffling])
                # tmpCSIb1IndPerm = circulant(tmpCSIb1Ind[shuffling])
                # tmpCSIe1IndPerm = circulant(tmpCSIe1Ind[shuffling])

                tmpCSIa1IndPerm = circulant(tmpCSIa1Ind[::-1])
                tmpCSIb1IndPerm = circulant(tmpCSIb1Ind[::-1])
                tmpCSIe1IndPerm = circulant(tmpCSIe1Ind[::-1])

                # tmpCSIa1IndPerm = tmpCSIa1Ind
                # tmpCSIb1IndPerm = tmpCSIb1Ind
                # tmpCSIe1IndPerm = tmpCSIe1Ind

                # 效果差
                # tmpCSIa1IndPerm = toeplitz(tmpCSIa1Ind[::-1])
                # tmpCSIb1IndPerm = toeplitz(tmpCSIb1Ind[::-1])
                # tmpCSIe1IndPerm = toeplitz(tmpCSIe1Ind[::-1])

                tmpMulA1 = np.dot(tmpCSIa1IndPerm, keyBin)

                # tmpCSIb1IndPerm = np.vstack([tmpCSIb1IndPerm, np.ones(keyLen)]).T
                # tmpCSIe1IndPerm = np.vstack([tmpCSIe1IndPerm, np.ones(keyLen)]).T

                # a_list_number = keyBin
                [error, _, _, ae] = ass_pg_stls_f(tmpCSIa1IndPerm, tmpMulA1, keyLen, keyLen, 0.02, keyBin, iteration)
                [_, _, _, be] = ass_pg_stls_f(tmpCSIb1IndPerm, tmpMulA1, keyLen, keyLen, 0.02, keyBin, iteration)
                [_, _, _, ee] = ass_pg_stls_f(tmpCSIe1IndPerm, tmpMulA1, keyLen, keyLen, 0.02, keyBin, iteration)

                # print(np.mean(tmpCSIa1IndPerm))
                # print("error", error)
                # exit()

                a_list_number = []
                b_list_number = []
                e_list_number = []

                aee = []
                bee = []
                eee = []

                for i in range(0, len(ae), 5):
                    aee.append(ae[i])
                    bee.append(be[i])
                    eee.append(ee[i])

                if bits == 1:
                    for i in range(len(aee)):
                        if aee[i] > 0.5:
                            a_list_number.append(1)
                        elif aee[i] < -0.5:
                            a_list_number.append(1)
                        else:
                            a_list_number.append(0)

                        if bee[i] > 0.5:
                            b_list_number.append(1)
                        elif bee[i] < -0.5:
                            b_list_number.append(1)
                        else:
                            b_list_number.append(0)

                        if eee[i] > 0.5:
                            e_list_number.append(1)
                        elif eee[i] < -0.5:
                            e_list_number.append(1)
                        else:
                            e_list_number.append(0)

                        # 改用四分位数量化
                        # if aee[i] > np.percentile(aee, 75):
                        #     a_list_number.append(1)
                        # elif aee[i] < np.percentile(aee, 25):
                        #     a_list_number.append(1)
                        # else:
                        #     a_list_number.append(0)
                        #
                        # if bee[i] > np.percentile(bee, 75):
                        #     b_list_number.append(1)
                        # elif bee[i] < np.percentile(bee, 25):
                        #     b_list_number.append(1)
                        # else:
                        #     b_list_number.append(0)
                        #
                        # if eee[i] > np.percentile(eee, 75):
                        #     e_list_number.append(1)
                        # elif eee[i] < np.percentile(eee, 25):
                        #     e_list_number.append(1)
                        # else:
                        #     e_list_number.append(0)

                        # if aee[i] > np.mean(aee):
                        #     a_list_number.append(1)
                        # else:
                        #     a_list_number.append(0)
                        #
                        # if bee[i] > np.mean(aee):
                        #     b_list_number.append(1)
                        # else:
                        #     b_list_number.append(0)
                        #
                        # if eee[i] > np.mean(aee):
                        #     e_list_number.append(1)
                        # else:
                        #     e_list_number.append(0)
                else:
                    for i in range(len(aee)):
                        if aee[i] > np.percentile(aee, 75):
                            a_list_number.append(0)
                        elif aee[i] > np.percentile(aee, 50):
                            a_list_number.append(1)
                        elif aee[i] > np.percentile(aee, 25):
                            a_list_number.append(2)
                        else:
                            a_list_number.append(3)

                        if bee[i] > np.percentile(bee, 75):
                            b_list_number.append(0)
                        elif bee[i] > np.percentile(bee, 50):
                            b_list_number.append(1)
                        elif bee[i] > np.percentile(bee, 25):
                            b_list_number.append(2)
                        else:
                            b_list_number.append(3)

                        if eee[i] > np.percentile(eee, 75):
                            e_list_number.append(0)
                        elif eee[i] > np.percentile(eee, 50):
                            e_list_number.append(1)
                        elif eee[i] > np.percentile(eee, 25):
                            e_list_number.append(2)
                        else:
                            e_list_number.append(3)

                # 转成二进制，0填充成0000
                if bits == 2:
                    for i in range(len(a_list_number)):
                        number = bin(a_list_number[i])[2:].zfill(int(np.log2(4)))
                        a_list += number
                    for i in range(len(b_list_number)):
                        number = bin(b_list_number[i])[2:].zfill(int(np.log2(4)))
                        b_list += number
                    for i in range(len(e_list_number)):
                        number = bin(e_list_number[i])[2:].zfill(int(np.log2(4)))
                        e_list += number
                    sum1 = min(len(a_list), len(b_list))
                    sum2 = 0
                    sum3 = 0
                    for i in range(0, sum1):
                        sum2 += (a_list[i] == b_list[i])
                    for i in range(min(len(a_list), len(e_list))):
                        sum3 += (a_list[i] == e_list[i])
                else:
                    sum1 = min(len(a_list_number), len(b_list_number))
                    sum2 = 0
                    sum3 = 0
                    for i in range(0, sum1):
                        sum2 += (a_list_number[i] == b_list_number[i])
                    for i in range(min(len(a_list_number), len(e_list_number))):
                        sum3 += (a_list_number[i] == e_list_number[i])

                # print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
                originSum += sum1
                correctSum += sum2
                randomSum += sum3

                originWholeSum += 1
                correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
                randomWholeSum = randomWholeSum + 1 if sum3 == sum1 else randomWholeSum

            print("\033[0;34;40ma-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10), "\033[0m")
            print("\033[0;34;40ma-e all", randomSum, "/", originSum, "=", round(randomSum / originSum, 10),
                  "\033[0m")
            print("\033[0;34;40ma-b whole match", correctWholeSum, "/", originWholeSum, "=",
                  round(correctWholeSum / originWholeSum, 10), "\033[0m")
            print("\033[0;34;40ma-e whole match", randomWholeSum, "/", originWholeSum, "=",
                  round(randomWholeSum / originWholeSum, 10), "\033[0m")

            print(round(correctSum / originSum, 10), round(correctWholeSum / originWholeSum, 10),
                  round(originSum / times / keyLen, 10),
                  round(correctSum / times / keyLen, 10))
            if withoutSort:
                print("withoutSort")
            else:
                print("withSort")
            print("\n")
