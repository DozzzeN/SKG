import copy
import csv
import math
import sys
import time
from collections import Counter
from tkinter import messagebox

import bchlib
from scipy.signal import convolve, medfilt
import graycode
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


def normalize(data):
    return (np.array(data) - np.min(data)) / (np.max(data) - np.min(data))
    # if np.max(data) == np.min(data):
    #     return (data - np.min(data)) / np.max(data)
    # else:
    #     return (np.array(data) - np.min(data)) / (np.max(data) - np.min(data))


# fileName = ["../csi/csi_static_indoor_1_r"]

# "./data_alignment/csi_ear.mat"
# fileName = ["./data_alignment/csi_ear.mat"]

# fileName = ["../data/data_mobile_outdoor_1.mat"]

fileName = ["../data/data_mobile_indoor_1.mat",
            "../data/data_mobile_outdoor_1.mat",
            "../data/data_static_outdoor_1.mat",
            "../data/data_static_indoor_1.mat"
            ]

# fileName = ["../csi/csi_mobile_indoor_1_r",
#             "../csi/csi_mobile_outdoor_r",
#             "../csi/csi_static_indoor_1_r",
#             "../csi/csi_static_outdoor_r"]

# fileName = ["../csi/csi_static_indoor_1_r"]

# fileName = ["../data/data_NLOS.mat"]

# 是否排序
withoutSorts = [True]
# 是否添加噪声
addNoises = ["mul"]

bits = 2

process = ""

# 可选的方法
solutions = ["stls", "ls", "nnls", "ils", "gurobi_opt", "cvxpy_ip", "cvxpy_perturbed_ip",
             "cvxpy_perturbed_ls", "gd", "pulp_ip", "pulp_perturbed_ls", "scipy_regularized_perturbed_ls",
             "matrix_inv", "sils", "Tichonov_reg", "truncation_reg", "leastsq", "mmse"]

solution = "cvxpy_perturbed_ls"
print("Used solution:", solution)
error_correct = False

all_iterations = []
perturb_time = []
balance_time = []
encode_time = []
decode_time = []
solve_time = []
correct_time = []

isComplex = True
isEncoded = False
isGrouped = False
isBalanced = True

for f in fileName:
    Ba = []
    Bb = []
    Be = []
    y = []
    num = 1

    condition_numbers_before = []
    condition_numbers_before_without_mul = []
    condition_numbers_after = []
    correlation = []

    for addNoise in addNoises:
        for withoutSort in withoutSorts:
            print(f)
            rawData = loadmat(f)

            if f.find("data_alignment") != -1:
                CSIa1Orig = rawData['csi'][:, 0]
                CSIb1Orig = rawData['csi'][:, 1]
                CSIe1Orig = rawData['csi'][:, 2]
            elif f.find("csi") != -1:
                CSIa1Orig = rawData['testdata'][:, 0]
                CSIb1Orig = rawData['testdata'][:, 1]
            else:
                CSIa1Orig = rawData['A'][:, 0]
                CSIb1Orig = rawData['A'][:, 1]

            # plt.figure()
            # plt.title(f)
            # plt.plot(CSIa1Orig)
            # plt.plot(CSIb1Orig)
            # plt.show()

            print(f, pearsonr(CSIa1Orig, CSIb1Orig)[0])

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

            iterations_of_a = []
            iterations_of_b = []

            error_bits = []

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
            if f == "../data/data_static_indoor_1.mat":
                dataLenLoop = int(dataLen / 5.5)
                keyLenLoop = int(keyLen / 5)
            for staInd in range(0, dataLenLoop, keyLenLoop):
                if process == "pca":
                    keyLen = 256 * segLen
                endInd = staInd + keyLen
                # print("range:", staInd, endInd)
                if endInd >= len(CSIa1Orig) or endInd >= len(CSIb1Orig):
                    break

                times += 1

                if f.find("data_alignment") != -1:
                    CSIa1Orig = rawData['csi'][:, 0]
                    CSIb1Orig = rawData['csi'][:, 1]
                    CSIe1Orig = rawData['csi'][:, 2]
                elif f.find("csi") != -1:
                    CSIa1Orig = rawData['testdata'][:, 0]
                    CSIb1Orig = rawData['testdata'][:, 1]
                else:
                    CSIa1Orig = rawData['A'][:, 0]
                    CSIb1Orig = rawData['A'][:, 1]

                if f == "../data/data_NLOS.mat":
                    # 先整体shuffle一次
                    shuffleInd = np.random.permutation(dataLen)
                    CSIa1Orig = CSIa1Orig[shuffleInd]
                    CSIb1Orig = CSIb1Orig[shuffleInd]

                seed = np.random.randint(100000)
                np.random.seed(seed)

                CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
                CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')
                # CSIe1Orig = smooth(np.array(CSIe1Orig), window_len=30, window='flat')

                tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
                tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
                # tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
                tmpCSIe1 = np.random.normal(np.mean(tmpCSIa1), np.std(tmpCSIa1, ddof=1), len(tmpCSIa1))

                # 对于128 64的短密钥,加上以后会好一些,对于256 512的长密钥则无影响
                if np.isclose(np.max(tmpCSIa1), np.min(tmpCSIa1), 0.01):
                    print("\033[0;34;40mclose\033[0m")
                    continue

                if addNoise == "mul":
                    if process == "pca":
                        keyLen = 160 * segLen
                    # randomMatrix = np.random.randint(1, 4, size=(keyLen, keyLen))
                    np.random.seed(10000)
                    randomMatrix = np.random.uniform(5, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))

                    # 均值化
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)

                    tmpCSIa1IndPerm = circulant(tmpCSIa1[::-1])
                    condition_numbers_before_without_mul.append(np.linalg.cond(tmpCSIa1IndPerm))

                    correlation.append(pearsonr(tmpCSIa1, tmpCSIb1)[0])

                    # 标准化
                    # tmpCSIa1 = normalize(tmpCSIa1)
                    # if np.max(tmpCSIb1) == np.min(tmpCSIb1):
                    #     tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / np.max(tmpCSIb1)
                    # else:
                    #     tmpCSIb1 = normalize(tmpCSIb1)
                    # tmpCSIe1 = normalize(tmpCSIe1)

                    tmpCSIa1back = copy.deepcopy(tmpCSIa1)
                    tmpCSIb1back = copy.deepcopy(tmpCSIb1)
                    tmpCSIe1back = copy.deepcopy(tmpCSIe1)
                    perturb_start = time.time_ns() / 10 ** 6
                    tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
                    perturb_time.append(time.time_ns() / 10 ** 6 - perturb_start)
                    tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)
                    tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix)
                else:
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)

                # without sorting
                if withoutSort:
                    tmpCSIa1Ind = np.array(tmpCSIa1)
                    tmpCSIb1Ind = np.array(tmpCSIb1)
                    tmpCSIe1Ind = np.array(tmpCSIe1)
                else:
                    tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
                    tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()
                    tmpCSIe1Ind = np.array(tmpCSIe1).argsort().argsort()

                if solution == "stls":
                    # stls
                    tmpCSIa1IndPerm = tmpCSIa1Ind[::-1]
                    tmpCSIb1IndPerm = tmpCSIb1Ind[::-1]
                    tmpCSIe1IndPerm = tmpCSIe1Ind[::-1]
                else:
                    tmpCSIa1IndPerm = circulant(tmpCSIa1Ind[::-1])
                    tmpCSIb1IndPerm = circulant(tmpCSIb1Ind[::-1])
                    tmpCSIe1IndPerm = circulant(tmpCSIe1Ind[::-1])
                    # tmpCSIa1IndPerm = toeplitz(tmpCSIa1Ind[::-1])
                    # tmpCSIb1IndPerm = toeplitz(tmpCSIb1Ind[::-1])
                    # tmpCSIe1IndPerm = toeplitz(tmpCSIe1Ind[::-1])

                tmpCSIa1IndPerm = normalize(tmpCSIa1IndPerm)
                tmpCSIb1IndPerm = normalize(tmpCSIb1IndPerm)
                tmpCSIe1IndPerm = normalize(tmpCSIe1IndPerm)

                condition_numbers_before.append(np.linalg.cond(tmpCSIa1IndPerm))

                # private matrix equilibration via svd
                # 对普通的LS效果特别好
                if isComplex:
                    tmpCSIa1IndPerm = np.hstack((np.vstack((tmpCSIa1IndPerm, np.zeros((keyLen, keyLen)))),
                                                 np.vstack((np.zeros((keyLen, keyLen)), tmpCSIa1IndPerm))))
                    tmpCSIb1IndPerm = np.hstack((np.vstack((tmpCSIb1IndPerm, np.zeros((keyLen, keyLen)))),
                                                 np.vstack((np.zeros((keyLen, keyLen)), tmpCSIb1IndPerm))))
                    tmpCSIe1IndPerm = np.hstack((np.vstack((tmpCSIe1IndPerm, np.zeros((keyLen, keyLen)))),
                                                 np.vstack((np.zeros((keyLen, keyLen)), tmpCSIe1IndPerm))))
                    # print(np.allclose(A @ np.hstack((np.real(keyBin), np.imag(keyBin))),
                    #                   np.hstack((np.real(tmpCSIa1IndPerm @ keyBin), np.imag(tmpCSIa1IndPerm @ keyBin)))))

                    # 奇异值平滑
                    if isBalanced:
                        balance_start = time.time_ns() / 10 ** 6
                        U, S, Vt = np.linalg.svd(tmpCSIa1IndPerm)
                        S = medfilt(S, kernel_size=7)
                        D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
                        # D = np.sqrt(D)
                        tmpCSIa1IndPerm = U @ D @ Vt
                        balance_time.append(time.time_ns() / 10 ** 6 - balance_start)
                        # D = np.diag(1 / np.sqrt(S))
                        # tmpCSIa1IndPerm = U @ D @ np.diag(S) @ D @ Vt
                        U, S, Vt = np.linalg.svd(tmpCSIb1IndPerm)
                        S = medfilt(S, kernel_size=7)
                        D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
                        # D = np.sqrt(D)
                        tmpCSIb1IndPerm = U @ D @ Vt
                        # D = np.diag(1 / np.sqrt(S))
                        # tmpCSIb1IndPerm = U @ D @ np.diag(S) @ D @ Vt
                        U, S, Vt = np.linalg.svd(tmpCSIe1IndPerm)
                        S = medfilt(S, kernel_size=7)
                        D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
                        # D = np.sqrt(D)
                        tmpCSIe1IndPerm = U @ D @ Vt
                        # D = np.diag(1 / np.sqrt(S))
                        # tmpCSIe1IndPerm = U @ D @ np.diag(S) @ D @ Vt
                else:
                    U, S, Vt = np.linalg.svd(tmpCSIa1IndPerm)
                    D = np.diag(1 / np.sqrt(S))
                    tmpCSIa1IndPerm = U @ D @ np.diag(S) @ D @ Vt
                    U, S, Vt = np.linalg.svd(tmpCSIb1IndPerm)
                    D = np.diag(1 / np.sqrt(S))
                    tmpCSIb1IndPerm = U @ D @ np.diag(S) @ D @ Vt
                    U, S, Vt = np.linalg.svd(tmpCSIe1IndPerm)
                    D = np.diag(1 / np.sqrt(S))
                    tmpCSIe1IndPerm = U @ D @ np.diag(S) @ D @ Vt

                condition_numbers_after.append(np.linalg.cond(tmpCSIa1IndPerm))
    print("condition_numbers_before_without_mul_mean", "{:.2e}".format(np.mean(condition_numbers_before_without_mul)))
    print("condition_numbers_before_mean", np.mean(condition_numbers_before))
    print("condition_numbers_after_mean", np.mean(condition_numbers_after))
    print("correlation", np.mean(correlation))
    print()
