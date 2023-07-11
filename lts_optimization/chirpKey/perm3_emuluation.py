import csv
import math
import time
from tkinter import messagebox

import pulp
import pywt
from scipy.fft import dct
from sklearn.decomposition import PCA
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.fft import dct
from scipy.io import loadmat, savemat
from scipy.linalg import circulant, toeplitz
from scipy.stats import pearsonr, boxcox, ortho_group
from scipy.optimize import least_squares, leastsq

import gurobipy as gp
from gurobipy import GRB
import logging

from RandomWayPoint import RandomWayPoint


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


def addNoiseFuc(origin, SNR):
    dataLen = len(origin)
    noise = np.random.normal(0, 1, size=dataLen)
    signal_power = np.sum(origin ** 2) / dataLen
    noise_power = np.sum(noise ** 2) / dataLen
    noise_variance = signal_power / (10 ** (SNR / 10))
    noise = noise * np.sqrt(noise_variance / noise_power)
    return origin + noise, noise


# 是否排序
withoutSorts = [True, False]
# 是否添加噪声
addNoises = ["mul"]

bits = 1

SNR = 10
model = RandomWayPoint(steps=10000, x_range=np.array([0, 11]), y_range=np.array([0, 11]))
trace_data = model.generate_trace(start_coor=[1, 1])
CSIa1OrigEmu = trace_data[:, 0]
CSIb1OrigEmu = addNoiseFuc(CSIa1OrigEmu, SNR)[0]

for addNoise in addNoises:
    for withoutSort in withoutSorts:
        CSIa1Orig = CSIa1OrigEmu.copy()
        CSIb1Orig = CSIb1OrigEmu.copy()

        dataLen = len(CSIa1Orig)
        print("dataLen", dataLen)

        segLen = 1
        keyLen = 256 * segLen

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
        for staInd in range(0, dataLenLoop, keyLenLoop):
            endInd = staInd + keyLen
            # print("range:", staInd, endInd)
            if endInd >= len(CSIa1Orig) or endInd >= len(CSIb1Orig):
                break

            times += 1

            CSIa1Orig = CSIa1OrigEmu.copy()
            CSIb1Orig = CSIb1OrigEmu.copy()

            seed = np.random.randint(100000)
            np.random.seed(seed)

            CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
            CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')

            tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
            tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
            tmpCSIe1 = np.random.normal(np.mean(tmpCSIa1), np.std(tmpCSIa1, ddof=1), len(tmpCSIa1))

            if addNoise == "mul":
                np.random.seed(10000)
                randomMatrix = np.random.uniform(5, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))

                # tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                # tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                # tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)

                tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1))
                if np.max(tmpCSIb1) - np.min(tmpCSIb1) == 0:
                    tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / np.max(tmpCSIb1)
                else:
                    tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1))
                tmpCSIe1 = (tmpCSIe1 - np.min(tmpCSIe1)) / (np.max(tmpCSIe1) - np.min(tmpCSIe1))

                # tmpCSIa1 = np.abs(np.fft.fft(tmpCSIa1))
                # tmpCSIb1 = np.abs(np.fft.fft(tmpCSIb1))
                # tmpCSIe1 = np.abs(np.fft.fft(tmpCSIe1))

                tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
                tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)
                tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix)
                # tmpCSIe1 = np.matmul(np.ones(keyLen), randomMatrix)
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
                tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
                tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()
                tmpCSIe1Ind = np.array(tmpCSIe1).argsort().argsort()

            np.random.seed(0)
            if bits == 1:
                keyBin = np.random.binomial(1, 0.5, keyLen)
            else:
                keyBin = np.random.randint(0, 4, keyLen)

            tmpCSIa1IndPerm = circulant(tmpCSIa1Ind[::-1])
            tmpCSIb1IndPerm = circulant(tmpCSIb1Ind[::-1])
            tmpCSIe1IndPerm = circulant(tmpCSIe1Ind[::-1])

            tmpMulA1 = np.dot(tmpCSIa1IndPerm, keyBin)

            def residuals(x, tmpMulA1, tmpCSIx1IndPerm):
                return tmpMulA1 - np.dot(tmpCSIx1IndPerm, x)
            a_list_number = leastsq(residuals, np.random.binomial(1, 0.5, keyLen), args=(tmpMulA1, tmpCSIa1IndPerm))[0]
            b_list_number = leastsq(residuals, np.random.binomial(1, 0.5, keyLen), args=(tmpMulA1, tmpCSIb1IndPerm))[0]
            e_list_number = leastsq(residuals, np.random.binomial(1, 0.5, keyLen), args=(tmpMulA1, tmpCSIe1IndPerm))[0]

            a_list_number = [round(abs(i)) for i in a_list_number]
            b_list_number = [round(abs(i)) for i in b_list_number]
            if bits == 1:
                e_list_number = [round(abs(i)) % 2 for i in e_list_number]
            else:
                e_list_number = [round(abs(i)) % 4 for i in e_list_number]

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
        print("\033[0;34;40ma-e all", randomSum, "/", originSum, "=", round(randomSum / originSum, 10), "\033[0m")
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
