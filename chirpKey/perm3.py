import copy
import csv
import math
import sys
import time
from collections import Counter
from tkinter import messagebox

from scipy.signal import convolve
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
from scipy.optimize import least_squares, leastsq, nnls, minimize

import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
import xpress
import logging
# import matlab.engine

from pyldpc import make_ldpc, encode, decode, get_message

from tls import tls, stls, block_toeplitz, tls2, stls_qp, block_circulant


def precoding(a):
    bk_1 = [0]
    bk = []
    for i in range(len(a)):
        bk.append((a[i] - bk_1[i]) % 4)
        bk_1.append(bk[i])
    ck = []
    for i in range(len(a)):
        ck.append((bk[i] + bk_1[i]) % 4)
    return ck


# 给定数组b=[0,2,4,8]，给定另一个数组x，求出最靠近x的数，例如x=[1.2,2.1,3]时，返回[2,2,4]
def findClosest(basic, a):
    diff = np.abs(np.array(basic)[:, None] - np.array(a))
    min_idx = np.argmin(diff, axis=0)
    return np.array(basic)[min_idx]


def common_pca(ha, hb, he, k):
    ha = (ha - np.min(ha)) / (np.max(ha) - np.min(ha))
    hb = (hb - np.min(hb)) / (np.max(hb) - np.min(hb))
    he = (he - np.min(he)) / (np.max(he) - np.min(he))
    rha = np.dot(ha, ha.T) / len(ha)
    ua, sa, vha = np.linalg.svd(rha)
    # print("p", np.sum(sa) / np.sum(sa[:k]))
    vha = vha[:k, :]
    ya = vha @ ha
    yb = vha @ hb
    ye = vha @ he
    return np.array(ya), np.array(yb), np.array(ye)


def rec(a, bits, unmatched):
    res = copy.deepcopy(a)
    if bits == 1:
        for i in range(len(unmatched)):
            if round(res[unmatched[i] - 1]) == 0:
                res[unmatched[i]] = 1
            else:
                res[unmatched[i]] = 0
    elif bits == 2:
        for i in range(len(unmatched)):
            if round(res[unmatched[i] - 1]) == 0:
                res[unmatched[i]] = 1
            elif round(res[unmatched[i] - 1]) == 1:
                res[unmatched[i]] = 0
            elif round(res[unmatched[i] - 1]) == 2:
                res[unmatched[i]] = 3
            elif round(res[unmatched[i] - 1]) == 3:
                res[unmatched[i]] = 2

    return np.array([round(i) for i in res])


def autocorr(x):
    n = len(x)
    Rxx = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            Rxx[i, j] = np.mean(x[i] * np.conj(x[j]))

    return Rxx


def unitary_invariant_norm(A):
    return np.linalg.norm(A, ord=2)


def gd(A, y, lr, epochs):
    m = A.shape[0]
    n = A.shape[1]
    x = np.random.randn(n)
    for i in range(epochs):
        y_pred = np.dot(A, x)
        dx = (1 / m) * np.dot(A.T, (y_pred - y))
        x = x - lr * dx
    return x


def truncation(data, threshold):
    U, S, V, = np.linalg.svd(data)
    for i in range(len(S)):
        if S[i] < threshold:
            S[i] = 0
    Sr = np.zeros((data.shape[0], data.shape[1]))
    Sr[:data.shape[1], :data.shape[1]] = np.diag(S)
    return U @ Sr @ V


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


# fileName = ["../data/data_mobile_indoor_1.mat",
#             "../data/data_mobile_outdoor_1.mat",
#             "../data/data_static_outdoor_1.mat",
#             "../data/data_static_indoor_1.mat"
#             ]

fileName = ["../csi/csi_mobile_indoor_1_r",
            "../csi/csi_mobile_outdoor_r",
            "../csi/csi_static_indoor_1_r",
            "../csi/csi_static_outdoor_r"]

# fileName = ["../data/data_static_indoor_1.mat"]

# 是否排序
withoutSorts = [True, False]
withoutSorts = [True]
# 是否添加噪声
addNoises = ["mul"]

bits = 2

process = "pca"

# 可选的方法
solutions = ["stls", "ls", "nnls", "ils", "gurobi_opt", "cvxpy_ip", "cvxpy_perturbed_ip",
             "cvxpy_perturbed_ls", "gd", "pulp_ip", "pulp_perturbed_ls", "scipy_regularized_perturbed_ls",
             "matrix_inv", "sils", "Tichonov_reg", "truncation_reg", "leastsq", "mmse"]

solution = "Tichonov_reg"
# solution = "ls"
print("Used solution:", solution)
error_correct = False

all_iterations = []
used_time = []

for f in fileName:
    Ba = []
    Bb = []
    Be = []
    y = []
    num = 1

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
                    if process == "pca":
                        keyLen = 160 * segLen
                    # randomMatrix = np.random.randint(1, 4, size=(keyLen, keyLen))
                    np.random.seed(10000)
                    randomMatrix = np.random.uniform(5, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
                    # randomMatrix = np.random.uniform(5, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen)) + \
                    #                1j * np.random.uniform(5, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))

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

                    # 均值化
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)

                    # 标准化
                    # tmpCSIa1 = normalize(tmpCSIa1)
                    # if np.max(tmpCSIb1) == np.min(tmpCSIb1):
                    #     tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / np.max(tmpCSIb1)
                    # else:
                    #     tmpCSIb1 = normalize(tmpCSIb1)
                    # tmpCSIe1 = normalize(tmpCSIe1)

                    if process == "fft":
                        # 产生的结果是对称的，数据会降低一半
                        tmpCSIa1 = np.abs(np.fft.fft(tmpCSIa1))
                        tmpCSIb1 = np.abs(np.fft.fft(tmpCSIb1))
                        tmpCSIe1 = np.abs(np.fft.fft(tmpCSIe1))
                    elif process == "dct":
                        tmpCSIa1 = dct(tmpCSIa1)
                        tmpCSIb1 = dct(tmpCSIb1)
                        tmpCSIe1 = dct(tmpCSIe1)
                    elif process == "pca":
                        tmpCSIa1Reshape = np.array(tmpCSIa1).reshape(16, 16)
                        tmpCSIb1Reshape = np.array(tmpCSIb1).reshape(16, 16)
                        tmpCSIe1Reshape = np.array(tmpCSIe1).reshape(16, 16)
                        # pca = PCA(n_components=16)
                        # tmpCSIa1 = pca.fit_transform(tmpCSIa1Reshape).reshape(1, -1)[0]
                        # tmpCSIb1 = pca.fit_transform(tmpCSIb1Reshape).reshape(1, -1)[0]
                        # tmpCSIe1 = pca.fit_transform(tmpCSIe1Reshape).reshape(1, -1)[0]
                        tmpCSIa1, tmpCSIb1, tmpCSIe1 = common_pca(tmpCSIa1Reshape, tmpCSIb1Reshape,
                                                                  tmpCSIe1Reshape, 10)
                        tmpCSIa1 = tmpCSIa1.reshape(1, -1)[0]
                        tmpCSIb1 = tmpCSIb1.reshape(1, -1)[0]
                        tmpCSIe1 = tmpCSIe1.reshape(1, -1)[0]

                        # tmpCSIa1Reshape2 = circulant(tmpCSIa1)
                        # tmpCSIb1Reshape2 = circulant(tmpCSIb1)
                        # tmpCSIe1Reshape2 = circulant(tmpCSIe1)
                        # tmpCSIa1, tmpCSIb1, tmpCSIe1 = common_pca(tmpCSIa1Reshape2, tmpCSIb1Reshape2,
                        #                                           tmpCSIe1Reshape2, 100)
                        # tmpCSIa1 = tmpCSIa1.reshape(1, -1)[0]
                        # tmpCSIb1 = tmpCSIb1.reshape(1, -1)[0]
                        # tmpCSIe1 = tmpCSIe1.reshape(1, -1)[0]
                    elif process == "dwt":
                        wavelet = 'sym2'
                        wtCSIa1 = pywt.dwt(tmpCSIa1, wavelet)
                        tmpCSIa1 = list(wtCSIa1[0])
                        tmpCSIa1.extend(wtCSIa1[1])
                        tmpCSIa1 = tmpCSIa1[0: keyLen]
                        wtCSIb1 = pywt.dwt(tmpCSIb1, wavelet)
                        tmpCSIb1 = list(wtCSIb1[0])
                        tmpCSIb1.extend(wtCSIb1[1])
                        tmpCSIb1 = tmpCSIb1[0: keyLen]
                        wtCSIe1 = pywt.dwt(tmpCSIe1, wavelet)
                        tmpCSIe1 = list(wtCSIe1[0])
                        tmpCSIe1.extend(wtCSIe1[1])
                        tmpCSIe1 = tmpCSIe1[0: keyLen]

                    # tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix) + randomMatrix[0]
                    # tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix) + randomMatrix[0]
                    # tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix) + randomMatrix[0]
                    tmpCSIa1back = copy.deepcopy(tmpCSIa1)
                    tmpCSIb1back = copy.deepcopy(tmpCSIb1)
                    tmpCSIe1back = copy.deepcopy(tmpCSIe1)
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
                    tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
                    tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()
                    tmpCSIe1Ind = np.array(tmpCSIe1).argsort().argsort()

                    # 影响不大
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
                    keyBin = np.random.binomial(1, 0.5, keyLen)
                    basic = [0, 1]
                    # 生成浮点数的密钥效果差
                    # keyBin = np.random.random(keyLen)
                else:
                    # keyBin = np.random.randint(0, 4, keyLen)
                    # keyBin = np.random.random(keyLen) * 3
                    # basic = [0, 5, 10, 15]
                    # basic = [-2, -1, 1, 2]
                    # basic = [0, 1, 2, 3]
                    # basic = [0, 0.1, 0.2, 0.3]
                    basic = [0, 1, 2, 3]
                    keyBin = np.random.choice(basic, keyLen)

                # 影响不大
                # np.random.seed(0)
                # shuffling = np.random.permutation(keyLen)
                # tmpCSIa1IndPerm = circulant(tmpCSIa1Ind[shuffling])
                # tmpCSIb1IndPerm = circulant(tmpCSIb1Ind[shuffling])
                # tmpCSIe1IndPerm = circulant(tmpCSIe1Ind[shuffling])

                if solution == "stls":
                    # stls
                    tmpCSIa1IndPerm = tmpCSIa1Ind[::-1]
                    tmpCSIb1IndPerm = tmpCSIb1Ind[::-1]
                    tmpCSIe1IndPerm = tmpCSIe1Ind[::-1]
                else:
                    # 1 3 2
                    # 2 1 3
                    # 3 2 1
                    tmpCSIa1IndPerm = circulant(tmpCSIa1Ind[::-1])
                    tmpCSIb1IndPerm = circulant(tmpCSIb1Ind[::-1])
                    tmpCSIe1IndPerm = circulant(tmpCSIe1Ind[::-1])

                    # tmpCSIa1Mean = convolve(tmpCSIa1Ind, np.ones(keyLen), mode='same')
                    # tmpCSIb1Mean = convolve(tmpCSIb1Ind, np.ones(keyLen), mode='same')
                    # tmpCSIe1Mean = convolve(tmpCSIe1Ind, np.ones(keyLen), mode='same')
                    # # tmpCSIa1Mean = []
                    # # tmpCSIb1Mean = []
                    # # tmpCSIe1Mean = []
                    # # for i in range(0, len(tmpCSIa1Ind), 2):
                    # #     tmpCSIa1Mean.append(sum(tmpCSIa1Ind[i:i + 2]))
                    # #     tmpCSIb1Mean.append(sum(tmpCSIb1Ind[i:i + 2]))
                    # #     tmpCSIe1Mean.append(sum(tmpCSIe1Ind[i:i + 2]))
                    # tmpCSIa1IndPerm = block_circulant([block_circulant(tmpCSIa1Mean), block_circulant(tmpCSIa1Mean)])
                    # tmpCSIb1IndPerm = block_circulant([block_circulant(tmpCSIb1Mean), block_circulant(tmpCSIb1Mean)])
                    # tmpCSIe1IndPerm = block_circulant([block_circulant(tmpCSIe1Mean), block_circulant(tmpCSIe1Mean)])
                    # np.random.seed(0)
                    # if bits == 1:
                    #     keyBin = np.random.binomial(1, 0.5, int(keyLen * 2))
                    #     # 生成浮点数的密钥效果差
                    #     # keyBin = np.random.random(keyLen)
                    # else:
                    #     keyBin = np.random.randint(0, 4, int(keyLen * 2))
                    #     # keyBin = np.random.random(keyLen) * 3

                    # tmpCSIa1IndPerm = block_circulant([block_circulant(tmpCSIa1Ind[::-1]), block_circulant(tmpCSIa1Ind)])
                    # tmpCSIb1IndPerm = block_circulant([block_circulant(tmpCSIb1Ind[::-1]), block_circulant(tmpCSIb1Ind)])
                    # tmpCSIe1IndPerm = block_circulant([block_circulant(tmpCSIe1Ind[::-1]), block_circulant(tmpCSIe1Ind)])
                    # np.random.seed(0)
                    # if bits == 1:
                    #     keyBin = np.random.binomial(1, 0.5, keyLen * 2)
                    #     # 生成浮点数的密钥效果差
                    #     # keyBin = np.random.random(keyLen)
                    # else:
                    #     keyBin = np.random.randint(0, 4, keyLen * 2)
                    #     # keyBin = np.random.random(keyLen) * 3

                    # 效果差（每行/列不是都包含123，只是对角线元素一致）
                    # 1 2 3
                    # 2 1 2
                    # 3 2 1
                    # tmpCSIa1IndPerm = toeplitz(tmpCSIa1Ind[::-1])
                    # tmpCSIb1IndPerm = toeplitz(tmpCSIb1Ind[::-1])
                    # tmpCSIe1IndPerm = toeplitz(tmpCSIe1Ind[::-1])

                tmpCSIa1IndPerm = normalize(tmpCSIa1IndPerm)
                if np.max(tmpCSIb1IndPerm) == np.min(tmpCSIb1IndPerm):
                    tmpCSIb1IndPerm = (tmpCSIb1IndPerm - np.min(tmpCSIb1IndPerm)) / np.max(tmpCSIb1IndPerm)
                else:
                    tmpCSIb1IndPerm = normalize(tmpCSIb1IndPerm)
                tmpCSIe1IndPerm = normalize(tmpCSIe1IndPerm)

                # tmpCSIa1IndPerm = tmpCSIa1IndPerm - np.mean(tmpCSIa1IndPerm)
                # tmpCSIb1IndPerm = tmpCSIb1IndPerm - np.mean(tmpCSIb1IndPerm)
                # tmpCSIe1IndPerm = tmpCSIe1IndPerm - np.mean(tmpCSIe1IndPerm)

                # private matrix equilibration via svd
                # 对普通的LS效果特别好
                # U, S, Vt = np.linalg.svd(tmpCSIa1IndPerm)
                # D = np.diag(1 / np.sqrt(S))
                # tmpCSIa1IndPerm = U @ D @ np.diag(S) @ D @ Vt
                # U, S, Vt = np.linalg.svd(tmpCSIb1IndPerm)
                # D = np.diag(1 / np.sqrt(S))
                # tmpCSIb1IndPerm = U @ D @ np.diag(S) @ D @ Vt
                # U, S, Vt = np.linalg.svd(tmpCSIe1IndPerm)
                # D = np.diag(1 / np.sqrt(S))
                # tmpCSIe1IndPerm = U @ D @ np.diag(S) @ D @ Vt

                # private matrix equilibration via qr decomposition
                # Q, R = np.linalg.qr(tmpCSIa1IndPerm)
                # # 不能减少条件数
                # D = np.diag(1 / np.diag(R))
                # # tmpCSIa1IndPerm = Q @ R @ D
                # tmpCSIa1IndPerm = Q
                # Q, R = np.linalg.qr(tmpCSIb1IndPerm)
                # # D = np.diag(1 / np.diag(R))
                # # tmpCSIb1IndPerm = Q @ R @ D
                # tmpCSIb1IndPerm = Q
                # Q, R = np.linalg.qr(tmpCSIe1IndPerm)
                # # D = np.diag(1 / np.diag(R))
                # # tmpCSIe1IndPerm = Q @ R @ D
                # tmpCSIe1IndPerm = Q

                # common matrix equilibration
                # 无private好，因为B矩阵的条件数可能很大，不为1
                # U, S, Vt = np.linalg.svd(tmpCSIa1IndPerm)
                # E = U @ np.diag(1 / S) @ U.T
                # tmpCSIa1IndPerm = E @ tmpCSIa1IndPerm
                # tmpCSIb1IndPerm = E @ tmpCSIb1IndPerm
                # tmpCSIe1IndPerm = E @ tmpCSIe1IndPerm

                # np.random.seed(0)
                # tmpMulA1 = np.dot(tmpCSIa1IndPerm, keyBin + np.random.normal(0, 0.1, keyLen))
                tmpMulA1 = np.dot(tmpCSIa1IndPerm, keyBin)

                # # 特征向量作为密钥
                # k = 1
                # # 效果差
                # # tmpCSIa1back = circulant(tmpCSIa1back[::-1])
                # # ha = tmpCSIa1back - np.mean(tmpCSIa1back, axis=0)
                # # tmpCSIa1IndPerm = tmpCSIa1IndPerm + np.eye(keyLen) * 0.1
                # ha = tmpCSIa1IndPerm - np.mean(tmpCSIa1IndPerm, axis=0)
                # rha = np.cov(ha, rowvar=False)
                # U, S, Vt = np.linalg.svd(rha, full_matrices=False)
                # # 标准化到密钥空间
                # Vt = (Vt - np.min(Vt)) / (np.max(Vt) - np.min(Vt)) * (2 ** bits - 1)
                # # keyBin = np.round(Vt[keyLen - 2])
                # # 最大特征值对应的向量
                # # 如果先用均衡，则密钥取整以后不随机了
                # keyBin = np.round(Vt.T[:, : k]).reshape(-1)
                # # keyBin = np.mod(keyBin + np.random.binomial(1, 0.5, keyLen), 4)
                # tmpMulA1 = np.dot(tmpCSIa1IndPerm, keyBin)
                # # print(np.var(tmpMulA1), np.var(np.dot(tmpCSIa1IndPerm, Vt.T[:, : k].reshape(-1))),
                # #       np.var(np.dot(tmpCSIa1IndPerm, np.random.normal(0, 1, keyLen))))
                # rhb = np.cov(tmpCSIb1IndPerm - np.mean(tmpCSIb1IndPerm, axis=0), rowvar=False)
                # Vtb = np.linalg.svd(rhb, full_matrices=False)[2]
                # Vtb = (Vtb - np.min(Vtb)) / (np.max(Vtb) - np.min(Vtb)) * (2 ** bits - 1)
                # # print(pearsonr(Vt.T[:, :k].reshape(-1), Vtb.T[:, :k].reshape(-1))[0])

                # corr = pearsonr(keyBin, tmpMulA1)[0]
                # corr1 = pearsonr(keyBin, tmpCSIa1IndPerm[0])[0]
                # corr2 = pearsonr(np.array(Vt.T[:, : k]).reshape(-1), tmpMulA1)[0]
                # corr3 = pearsonr(np.array(Vt.T[:, : k]).reshape(-1), tmpCSIa1IndPerm[0])[0]

                if solution == "cvxpy_perturbed_ls":
                    # # 数据预处理：分块，然后生成循环矩阵
                    # block_shape = 1
                    # block_number = int(keyLen / block_shape / block_shape)
                    # tmpCSIa1Blocks = np.array(tmpCSIa1Ind).reshape(block_number, block_shape, block_shape)
                    # tmpCSIb1Blocks = np.array(tmpCSIb1Ind).reshape(block_number, block_shape, block_shape)
                    # tmpCSIe1Blocks = np.array(tmpCSIe1Ind).reshape(block_number, block_shape, block_shape)
                    # # tmpCSIa1Circulant = block_toeplitz(tmpCSIa1Blocks)
                    # # tmpCSIb1Circulant = block_toeplitz(tmpCSIb1Blocks)
                    # # tmpCSIe1Circulant = block_toeplitz(tmpCSIe1Blocks)
                    # tmpCSIa1Circulant = block_circulant(tmpCSIa1Blocks)
                    # tmpCSIb1Circulant = block_circulant(tmpCSIb1Blocks)
                    # tmpCSIe1Circulant = block_circulant(tmpCSIe1Blocks)
                    #
                    # tmpCSIa1Circulant = normalize(tmpCSIa1Circulant)
                    # if np.max(tmpCSIb1Circulant) == np.min(tmpCSIb1Circulant):
                    #     tmpCSIb1Circulant = (tmpCSIb1Circulant - np.min(tmpCSIb1Circulant)) / np.max(tmpCSIb1Circulant)
                    # else:
                    #     tmpCSIb1Circulant = normalize(tmpCSIb1Circulant)
                    # tmpCSIe1Circulant = normalize(tmpCSIe1Circulant)
                    #
                    # tmpCSIa1Blocks = normalize(tmpCSIa1Blocks)
                    # if np.max(tmpCSIb1Blocks) == np.min(tmpCSIb1Blocks):
                    #     tmpCSIb1Blocks = (tmpCSIb1Blocks - np.min(tmpCSIb1Blocks)) / np.max(tmpCSIb1Blocks)
                    # else:
                    #     tmpCSIb1Blocks = normalize(tmpCSIb1Blocks)
                    # tmpCSIb1Blocks = normalize(tmpCSIb1Blocks)
                    #
                    # np.random.seed(0)
                    # if bits == 1:
                    #     keyBin = np.random.binomial(1, 0.5, block_number * block_shape)
                    # else:
                    #     keyBin = np.random.randint(0, 4, block_number * block_shape)
                    #
                    # tmpMulA1 = np.dot(tmpCSIa1Circulant, keyBin)
                    # tmpCSIa1IndPerm = tmpCSIa1Circulant
                    # tmpCSIb1IndPerm = tmpCSIb1Circulant
                    # tmpCSIe1IndPerm = tmpCSIe1Circulant

                    # cvxpy perturbed least squares
                    # SCIP最慢，其他结果一致
                    lambda_ = 0
                    solvers = [cp.SCS, cp.GUROBI, cp.OSQP, cp.CVXOPT, cp.ECOS, cp.SCIP, cp.PROXQP, cp.MOSEK,
                               cp.CLARABEL, cp.NAG, cp.XPRESS]
                    solver = solvers[2]
                    x = cp.Variable(len(keyBin))
                    # 加正则项效果差
                    f_norm = np.linalg.norm(tmpCSIa1IndPerm, ord='fro')
                    obj = cp.Minimize(cp.sum_squares(tmpCSIa1IndPerm @ x - tmpMulA1) + lambda_ * cp.sum_squares(x))
                    prob = cp.Problem(obj, [x >= min(basic), x <= max(basic)])
                    solve_start = time.time()
                    if solver == cp.SCS:
                        prob.solve(solver=solver, max_iters=5000)
                    elif solver == cp.SCIP:
                        prob.solve(solver=solver, scip_params={"limits/time": 10})
                    elif solver == cp.OSQP:
                        # scaling=False表示禁用缩放以矩阵均衡
                        prob.solve(solver=solver, scaling=False)
                    else:
                        prob.solve(solver=solver)
                    used_time.append(time.time() - solve_start)
                    # print("num_iters of a: ", prob.solver_stats.num_iters)
                    iterations_of_a.append(prob.solver_stats.num_iters)
                    all_iterations.append(prob.solver_stats.num_iters)
                    a_list_number = [i.value for i in x]

                    x = cp.Variable(len(keyBin))
                    obj = cp.Minimize(cp.sum_squares(tmpCSIb1IndPerm @ x - tmpMulA1) + lambda_ * cp.sum_squares(x))
                    prob = cp.Problem(obj, [x >= min(basic), x <= max(basic)])
                    solve_start = time.time()
                    if solver == cp.SCS:
                        prob.solve(solver=solver, max_iters=5000)
                    elif solver == cp.SCIP:
                        prob.solve(solver=solver, scip_params={"limits/time": 10})
                    elif solver == cp.OSQP:
                        prob.solve(solver=solver, scaling=False)
                    else:
                        prob.solve(solver=solver)
                    used_time.append(time.time() - solve_start)
                    # print("num_iters of b: ", prob.solver_stats.num_iters)
                    iterations_of_b.append(prob.solver_stats.num_iters)
                    all_iterations.append(prob.solver_stats.num_iters)
                    b_list_number = [i.value for i in x]
                    # print(prob.solver_stats.solver_name,
                    #       np.sum(np.abs(np.array([x[i].value for i in range(keyLen)]) - np.array(a_list_number))))

                    x = cp.Variable(len(keyBin))
                    obj = cp.Minimize(cp.sum_squares(tmpCSIe1IndPerm @ x - tmpMulA1) + lambda_ * cp.sum_squares(x))
                    prob = cp.Problem(obj, [x >= min(basic), x <= max(basic)])
                    if solver == cp.SCS:
                        prob.solve(solver=solver, max_iters=5000)
                    elif solver == cp.SCIP:
                        prob.solve(solver=solver, scip_params={"limits/time": 10})
                    elif solver == cp.OSQP:
                        prob.solve(solver=solver, scaling=False)
                    else:
                        prob.solve(solver=solver)
                    e_list_number = [i.value for i in x]
                elif solution == "mmse":
                    # y = h * x + z
                    h_a = tmpCSIa1IndPerm
                    h_b = tmpCSIb1IndPerm
                    h_e = tmpCSIe1IndPerm
                    h_a_H = np.conj(np.transpose(h_a))
                    h_b_H = np.conj(np.transpose(h_b))
                    h_e_H = np.conj(np.transpose(h_e))

                    noise_signal_var = 1
                    # noise_signal_var = np.abs(np.var(h_b) - np.var(h_a)) / np.var(h_a)
                    a_list_number = h_a_H @ np.linalg.pinv(h_a @ h_a_H + noise_signal_var * np.eye(keyLen)) @ tmpMulA1
                    b_list_number = h_b_H @ np.linalg.pinv(h_b @ h_b_H + noise_signal_var * np.eye(keyLen)) @ tmpMulA1
                    e_list_number = h_e_H @ np.linalg.pinv(h_e @ h_e_H + noise_signal_var * np.eye(keyLen)) @ tmpMulA1
                elif solution == "dls":
                    # data least squares
                    solvers = [cp.SCS, cp.GUROBI, cp.OSQP, cp.CVXOPT, cp.ECOS, cp.SCIP, cp.PROXQP, cp.MOSEK,
                               cp.CLARABEL, cp.NAG, cp.XPRESS]
                    solver = solvers[2]
                    x = cp.Variable(len(keyBin))
                    # 加正则项效果差
                    # obj = cp.Minimize(cp.sum_squares(tmpCSIa1IndPerm @ x - tmpMulA1) + 0.1 * cp.sum_squares(x))
                    obj = cp.Minimize(cp.multiply(cp.sum_squares(tmpCSIa1IndPerm @ x - tmpMulA1),
                                                  cp.power(cp.sum_squares(x), -1)))
                    # prob = cp.Problem(obj)
                    # prob = cp.Problem(obj, [x >= 0, x <= 3])
                    prob = cp.Problem(obj, [x >= min(basic), x <= max(basic)])
                    if solver == cp.SCS:
                        prob.solve(solver=solver, max_iters=5000)
                    elif solver == cp.SCIP:
                        prob.solve(solver=solver, scip_params={"limits/time": 10})
                    else:
                        prob.solve(solver=solver)
                    a_list_number = [i.value for i in x]
                elif solution == "stls":
                    # stls
                    block_shape = 4
                    block_number = int(keyLen / block_shape / block_shape)
                    tmpCSIa1Blocks = np.array(tmpCSIa1IndPerm).reshape(block_number, block_shape, block_shape)
                    tmpCSIb1Blocks = np.array(tmpCSIb1IndPerm).reshape(block_number, block_shape, block_shape)
                    tmpCSIe1Blocks = np.array(tmpCSIe1IndPerm).reshape(block_number, block_shape, block_shape)
                    # tmpCSIa1Circulant = block_toeplitz(tmpCSIa1Blocks)
                    # tmpCSIb1Circulant = block_toeplitz(tmpCSIb1Blocks)
                    # tmpCSIe1Circulant = block_toeplitz(tmpCSIe1Blocks)
                    tmpCSIa1Circulant = block_circulant(tmpCSIa1Blocks)
                    tmpCSIb1Circulant = block_circulant(tmpCSIb1Blocks)
                    tmpCSIe1Circulant = block_circulant(tmpCSIe1Blocks)

                    tmpCSIa1Circulant = normalize(tmpCSIa1Circulant)
                    if np.max(tmpCSIb1Circulant) == np.min(tmpCSIb1Circulant):
                        tmpCSIb1Circulant = (tmpCSIb1Circulant - np.min(tmpCSIb1Circulant)) / np.max(tmpCSIb1Circulant)
                    else:
                        tmpCSIb1Circulant = normalize(tmpCSIb1Circulant)
                    tmpCSIe1Circulant = normalize(tmpCSIe1Circulant)

                    tmpCSIa1Blocks = normalize(tmpCSIa1Blocks)
                    if np.max(tmpCSIb1Blocks) == np.min(tmpCSIb1Blocks):
                        tmpCSIb1Blocks = (tmpCSIb1Blocks - np.min(tmpCSIb1Blocks)) / np.max(tmpCSIb1Blocks)
                    else:
                        tmpCSIb1Blocks = normalize(tmpCSIb1Blocks)
                    tmpCSIb1Blocks = normalize(tmpCSIb1Blocks)

                    np.random.seed(0)
                    if bits == 1:
                        keyBin = np.random.binomial(1, 0.5, block_number * block_shape)
                    else:
                        keyBin = np.random.randint(0, 4, block_number * block_shape)

                    tmpMulA1 = np.dot(tmpCSIa1Circulant, keyBin)
                    tmpMulA1 = tmpMulA1.reshape(block_number, block_shape)
                    a_list_number = stls(tmpCSIa1Blocks, tmpMulA1)
                    b_list_number = stls(tmpCSIb1Blocks, tmpMulA1)
                    e_list_number = stls(tmpCSIe1Blocks, tmpMulA1)
                    # a_list_number = stls_qp(tmpCSIa1Blocks, tmpMulA1)
                    # b_list_number = stls_qp(tmpCSIb1Blocks, tmpMulA1)
                    # e_list_number = stls_qp(tmpCSIe1Blocks, tmpMulA1)
                elif solution == "ls":
                    # least square
                    def residuals(x, tmpMulA1, tmpCSIx1IndPerm):
                        return tmpMulA1 - np.dot(tmpCSIx1IndPerm, x)


                    a_list_number = leastsq(residuals, np.random.binomial(1, 0.5, keyLen),
                                            args=(tmpMulA1, tmpCSIa1IndPerm))[0]
                    b_list_number = leastsq(residuals, np.random.binomial(1, 0.5, keyLen),
                                            args=(tmpMulA1, tmpCSIb1IndPerm))[0]
                    e_list_number = leastsq(residuals, np.random.binomial(1, 0.5, keyLen),
                                            args=(tmpMulA1, tmpCSIe1IndPerm))[0]

                    # theorem 5.3
                    # x = np.linalg.pinv(tmpCSIa1IndPerm) @ tmpMulA1
                    # x_hat = np.linalg.pinv(tmpCSIb1IndPerm) @ tmpMulA1
                    # r = tmpMulA1 - tmpCSIa1IndPerm @ x
                    # r_hat = tmpMulA1 - tmpCSIb1IndPerm @ x_hat
                    # r_hat_norm = unitary_invariant_norm(r_hat.reshape(-1, 1))
                    # r_norm = unitary_invariant_norm(r.reshape(-1, 1))
                    # epsilon = np.sqrt(r_hat_norm ** 2 - r_norm ** 2)
                    # E = (r_hat - r).reshape(-1, 1) @ x_hat.reshape(1, -1) / unitary_invariant_norm(x_hat.reshape(-1, 1)) ** 2
                    # E_ = tmpCSIb1IndPerm - tmpCSIa1IndPerm
                    # print(abs(E - E_).sum())
                    # print(unitary_invariant_norm(np.array(tmpMulA1 - (tmpCSIa1IndPerm + E) @ x_hat)))
                    # print(unitary_invariant_norm(np.array(tmpMulA1 - (tmpCSIa1IndPerm + E_) @ x_hat)))
                elif solution == "nnls":
                    # non negative least square
                    a_list_number = nnls(tmpCSIa1IndPerm, tmpMulA1)[0]
                    b_list_number = nnls(tmpCSIb1IndPerm, tmpMulA1)[0]
                    e_list_number = nnls(tmpCSIe1IndPerm, tmpMulA1)[0]
                elif solution == "ils":
                    # integer least square
                    # 直接调用matlab程序求解，运行地过慢，无法求出可行解，见后面的sils方法
                    eng = matlab.engine.start_matlab()
                    Ba = matlab.double(tmpCSIa1IndPerm)
                    Bb = matlab.double(tmpCSIb1IndPerm)
                    Be = matlab.double(tmpCSIe1IndPerm)
                    y = matlab.double(np.array(tmpMulA1).reshape(len(tmpMulA1), 1))
                    a_list_number = np.array(eng.sils(Ba, y, 1)).reshape(-1)
                    b_list_number = np.array(eng.sils(Bb, y, 1)).reshape(-1)
                    e_list_number = np.array(eng.sils(Be, y, 1)).reshape(-1)
                    eng.exit()
                elif solution == "gurobi_opt":
                    # gurobi optimization
                    vtype = GRB.CONTINUOUS
                    # 速度太慢，也无法求出可行解，退化成一般的最小二乘法
                    # vtype = GRB.INTEGER
                    model = gp.Model("Integer Quadratic Programming")
                    model.setParam('OutputFlag', 0)
                    model.setParam("LogToConsole", 0)
                    inputs = []
                    for i in range(len(keyBin)):
                        inputs.append(model.addVar(lb=0, ub=3, vtype=vtype, name=f'x{i}'))
                    obj = sum((np.dot(tmpCSIa1IndPerm, inputs) - tmpMulA1) ** 2)
                    model.setObjective(obj, GRB.MINIMIZE)
                    model.optimize()
                    if model.status == GRB.Status.OPTIMAL:
                        a_list_number = [round(i.x) for i in inputs]
                    else:
                        a_list_number = list(np.linalg.lstsq(tmpCSIa1IndPerm, tmpMulA1, rcond=None)[0])
                    model.close()

                    model = gp.Model("Integer Quadratic Programming")
                    model.setParam('OutputFlag', 0)
                    model.setParam("LogToConsole", 0)
                    inputs = []
                    for i in range(len(keyBin)):
                        inputs.append(model.addVar(lb=0, ub=3, vtype=vtype, name=f'x{i}'))
                    obj = sum((np.dot(tmpCSIb1IndPerm, inputs) - tmpMulA1) ** 2)
                    model.setObjective(obj, GRB.MINIMIZE)
                    model.optimize()
                    if model.status == GRB.Status.OPTIMAL:
                        b_list_number = [round(i.x) for i in inputs]
                    else:
                        b_list_number = list(np.linalg.lstsq(tmpCSIb1IndPerm, tmpMulA1, rcond=None)[0])
                    model.close()

                    model = gp.Model("Integer Quadratic Programming")
                    model.setParam('OutputFlag', 0)
                    model.setParam("LogToConsole", 0)
                    inputs = []
                    for i in range(len(keyBin)):
                        inputs.append(model.addVar(lb=0, ub=3, vtype=vtype, name=f'x{i}'))
                    obj = sum((np.dot(tmpCSIe1IndPerm, inputs) - tmpMulA1) ** 2)
                    model.setObjective(obj, GRB.MINIMIZE)
                    model.optimize()
                    if model.status == GRB.Status.OPTIMAL:
                        e_list_number = [round(i.x) for i in inputs]
                    else:
                        e_list_number = list(np.linalg.lstsq(tmpCSIe1IndPerm, tmpMulA1, rcond=None)[0])
                    model.close()
                elif solution == "cvxpy_ip":
                    # cvxpy integer programming
                    x = cp.Variable(len(keyBin), integer=True)
                    obj = cp.Minimize(cp.sum_squares(tmpCSIa1IndPerm @ x - tmpMulA1))
                    prob = cp.Problem(obj)
                    prob.solve()
                    a_list_number = [round(i.value) for i in x]

                    x = cp.Variable(len(keyBin), integer=True)
                    obj = cp.Minimize(cp.sum_squares(tmpCSIb1IndPerm @ x - tmpMulA1))
                    prob = cp.Problem(obj)
                    prob.solve()
                    b_list_number = [round(i.value) for i in x]

                    x = cp.Variable(len(keyBin), integer=True)
                    obj = cp.Minimize(cp.sum_squares(tmpCSIe1IndPerm @ x - tmpMulA1))
                    prob = cp.Problem(obj)
                    prob.solve()
                    e_list_number = [round(i.value) for i in x]
                elif solution == "gd":
                    # 梯度下降gd效果很差
                    lr = 0.001
                    epochs = 100000
                    a_list_number = gd(tmpCSIa1IndPerm, tmpMulA1, lr, epochs)
                    b_list_number = gd(tmpCSIb1IndPerm, tmpMulA1, lr, epochs)
                    e_list_number = gd(tmpCSIe1IndPerm, tmpMulA1, lr, epochs)
                elif solution == "pulp_ip":
                    # pulp integer programming
                    problem = pulp.LpProblem("Matrix_Constraint", pulp.LpMinimize)
                    # x = [pulp.LpVariable(f'x{i}', cat=pulp.LpInteger) for i in range(len(keyBin))]
                    x = [pulp.LpVariable(f'x{i}', lowBound=0, cat=pulp.LpInteger) for i in range(len(keyBin))]
                    # x = [pulp.LpVariable(f'x{i}', lowBound=0, upBound=3, cat=pulp.LpInteger) for i in range(len(keyBin))]
                    problem += pulp.lpSum(x)
                    for i in range(len(tmpMulA1)):
                        problem += pulp.lpSum([tmpCSIa1IndPerm[i][j] * x[j] for j in range(len(keyBin))]) == tmpMulA1[i]
                    problem.solve(pulp.PULP_CBC_CMD(msg=False))
                    a_list_number = [pulp.value(i) for i in x]

                    problem = pulp.LpProblem("Matrix_Constraint", pulp.LpMinimize)
                    # x = [pulp.LpVariable(f'x{i}', cat=pulp.LpInteger) for i in range(len(keyBin))]
                    x = [pulp.LpVariable(f'x{i}', lowBound=0, cat=pulp.LpInteger) for i in range(len(keyBin))]
                    # x = [pulp.LpVariable(f'x{i}', lowBound=0, upBound=3, cat=pulp.LpInteger) for i in range(len(keyBin))]
                    problem += pulp.lpSum(x)
                    for i in range(len(tmpMulA1)):
                        problem += pulp.lpSum([tmpCSIb1IndPerm[i][j] * x[j] for j in range(len(keyBin))]) == tmpMulA1[i]
                    problem.solve(pulp.PULP_CBC_CMD(msg=False))
                    b_list_number = [pulp.value(i) for i in x]

                    problem = pulp.LpProblem("Matrix_Constraint", pulp.LpMinimize)
                    # x = [pulp.LpVariable(f'x{i}', cat=pulp.LpInteger) for i in range(len(keyBin))]
                    x = [pulp.LpVariable(f'x{i}', lowBound=0, cat=pulp.LpInteger) for i in range(len(keyBin))]
                    # x = [pulp.LpVariable(f'x{i}', lowBound=0, upBound=3, cat=pulp.LpInteger) for i in range(len(keyBin))]
                    problem += pulp.lpSum(x)
                    for i in range(len(tmpMulA1)):
                        problem += pulp.lpSum([tmpCSIe1IndPerm[i][j] * x[j] for j in range(len(keyBin))]) == tmpMulA1[i]
                    problem.solve(pulp.PULP_CBC_CMD(msg=False))
                    e_list_number = [pulp.value(i) for i in x]
                elif solution == "pulp_perturbed_ls":
                    # pulp perturbed least squares
                    # 影响不大
                    r = 0.2
                    tmpCSIa1IndPerm = truncation(tmpCSIa1IndPerm, r)
                    tmpCSIb1IndPerm = truncation(tmpCSIb1IndPerm, r)
                    tmpCSIe1IndPerm = truncation(tmpCSIe1IndPerm, r)
                    problem = pulp.LpProblem("Regularized_Least_Squares", pulp.LpMinimize)
                    # x = [pulp.LpVariable(f'x{i}', lowBound=0, upBound=3, cat=pulp.LpInteger) for i in range(len(keyBin))]
                    x = [pulp.LpVariable(f'x{i}', lowBound=0, upBound=3) for i in range(len(keyBin))]
                    residuals = [pulp.LpVariable(f'r{i}', lowBound=0) for i in range(len(keyBin))]
                    problem += pulp.lpSum(residuals)
                    for i in range(len(tmpMulA1)):
                        problem += pulp.lpSum([tmpCSIa1IndPerm[i][j] * x[j] for j in range(len(keyBin))]) \
                                   - residuals[i] == tmpMulA1[i]
                    problem.solve(pulp.PULP_CBC_CMD(msg=False))
                    a_list_number = [pulp.value(i) for i in x]

                    problem = pulp.LpProblem("Regularized_Least_Squares", pulp.LpMinimize)
                    # x = [pulp.LpVariable(f'x{i}', lowBound=0, upBound=3, cat=pulp.LpInteger) for i in range(len(keyBin))]
                    x = [pulp.LpVariable(f'x{i}', lowBound=0, upBound=3) for i in range(len(keyBin))]
                    residuals = [pulp.LpVariable(f'r{i}', lowBound=0) for i in range(len(keyBin))]
                    problem += pulp.lpSum(residuals)  # 缺省的约束条件
                    for i in range(len(tmpMulA1)):
                        problem += pulp.lpSum([tmpCSIb1IndPerm[i][j] * x[j] for j in range(len(keyBin))]) \
                                   - residuals[i] == tmpMulA1[i]
                    problem.solve(pulp.PULP_CBC_CMD(msg=False))
                    b_list_number = [pulp.value(i) for i in x]

                    problem = pulp.LpProblem("Regularized_Least_Squares", pulp.LpMinimize)
                    # x = [pulp.LpVariable(f'x{i}', lowBound=0, upBound=3, cat=pulp.LpInteger) for i in range(len(keyBin))]
                    x = [pulp.LpVariable(f'x{i}', lowBound=0, upBound=3) for i in range(len(keyBin))]
                    residuals = [pulp.LpVariable(f'r{i}', lowBound=0) for i in range(len(keyBin))]
                    problem += pulp.lpSum(residuals)
                    for i in range(len(tmpMulA1)):
                        problem += pulp.lpSum([tmpCSIe1IndPerm[i][j] * x[j] for j in range(len(keyBin))]) \
                                   - residuals[i] == tmpMulA1[i]
                    problem.solve(pulp.PULP_CBC_CMD(msg=False))
                    e_list_number = [pulp.value(i) for i in x]
                elif solution == "scipy_regularized_perturbed_ls":
                    # scipy regularized perturbed least squares (scipy)
                    def loss(x, A, y, alpha=0.2):
                        return np.sum((A.dot(x) - y) ** 2) + alpha * np.sum(x ** 2)


                    a_list_number = minimize(loss, np.zeros(keyLen), args=(tmpCSIa1IndPerm, tmpMulA1)).x
                    b_list_number = minimize(loss, np.zeros(keyLen), args=(tmpCSIb1IndPerm, tmpMulA1)).x
                    e_list_number = minimize(loss, np.zeros(keyLen), args=(tmpCSIe1IndPerm, tmpMulA1)).x
                elif solution == "matrix_inv":
                    # 矩阵求逆
                    a_list_number = np.matmul(np.linalg.inv(tmpCSIa1IndPerm), tmpMulA1)
                    b_list_number = np.matmul(np.linalg.inv(tmpCSIb1IndPerm), tmpMulA1)
                    e_list_number = np.matmul(np.linalg.inv(tmpCSIe1IndPerm), tmpMulA1)
                elif solution == "sils":
                    # 将数据保存成mat，用matlab求解
                    a_inv = np.matmul(np.linalg.inv(tmpCSIa1IndPerm), tmpMulA1)
                    b_inv = np.matmul(np.linalg.inv(tmpCSIb1IndPerm), tmpMulA1)
                    # 下值过大必定无法通过sils求解
                    if np.sum(np.abs(tmpMulA1 - np.dot(tmpCSIb1IndPerm, np.round(b_inv, 0)))) < 65 and \
                            np.linalg.matrix_rank(tmpCSIa1IndPerm) == keyLen and np.linalg.matrix_rank(
                        tmpCSIb1IndPerm) == keyLen:
                        Ba.append(tmpCSIa1IndPerm)
                        Bb.append(tmpCSIb1IndPerm)
                        Be.append(tmpCSIe1IndPerm)
                        y.append(tmpMulA1)
                        print(num, np.sum(np.abs(tmpMulA1 - np.dot(tmpCSIb1IndPerm, keyBin))),
                              np.sum(np.abs(tmpMulA1 - np.dot(tmpCSIe1IndPerm, keyBin))))
                        # a_eig = np.linalg.svd(tmpCSIa1IndPerm)[1]
                        # b_eig = np.linalg.svd(tmpCSIb1IndPerm)[1]
                        # print(a_eig[np.argsort(a_eig)[::-1][:3]], b_eig[np.argsort(b_eig)[::-1][:3]])
                        print(max(b_inv), min(b_inv))
                        print("error", np.sum(np.abs(tmpMulA1 - np.dot(tmpCSIb1IndPerm, np.round(b_inv, 0)))))
                        num += 1
                        print(np.allclose(keyBin, np.round(b_inv, 0)))
                    else:
                        print("mismatch:", np.sum(np.abs(tmpMulA1 - np.dot(tmpCSIb1IndPerm, keyBin))))
                        # a_eig = np.linalg.svd(tmpCSIa1IndPerm)[1]
                        # b_eig = np.linalg.svd(tmpCSIb1IndPerm)[1]
                        # print(a_eig[np.argsort(a_eig)[::-1][:3]], b_eig[np.argsort(b_eig)[::-1][:3]])
                        # 最大值最小值不在[0,3]范围内必定无法通过sils求解，但有的在[0,3]内也会难以求解
                        print(max(b_inv), min(b_inv))
                        print("mis error", np.sum(np.abs(tmpMulA1 - np.dot(tmpCSIb1IndPerm, np.round(b_inv, 0)))))
                        print(np.allclose(keyBin, np.round(b_inv, 0)))
                elif solution == "Tichonov_reg":
                    # Tichonov regularization
                    alpha = 50
                    U, S, V, = np.linalg.svd(tmpCSIa1IndPerm)
                    V = V.T
                    S = np.diag(S)
                    beta = (U.T @ tmpMulA1)[:keyLen]
                    a_list_number = V @ np.linalg.inv(S.T @ S + alpha * np.eye(keyLen)) @ S.T @ beta

                    U, S, V, = np.linalg.svd(tmpCSIb1IndPerm)
                    V = V.T
                    S = np.diag(S)
                    beta = (U.T @ tmpMulA1)[:keyLen]
                    b_list_number = V @ np.linalg.inv(S.T @ S + alpha * np.eye(keyLen)) @ S.T @ beta

                    U, S, V, = np.linalg.svd(tmpCSIe1IndPerm)
                    V = V.T
                    S = np.diag(S)
                    beta = (U.T @ tmpMulA1)[:keyLen]
                    e_list_number = V @ np.linalg.inv(S.T @ S + alpha * np.eye(keyLen)) @ S.T @ beta
                elif solution == "truncation_reg":
                    # regularization by truncation
                    alpha = 50
                    r = 0.8
                    U, S, V, = np.linalg.svd(tmpCSIa1IndPerm)
                    V = V.T
                    for i in range(len(S)):
                        if S[i] < r:
                            S[i] = 0
                    # print(keyLen - len(np.nonzero(S)[0]))
                    # Sr = np.diag(S)
                    # Ar = U @ Sr @ V.T
                    # a_list_number = np.linalg.pinv(Ar) @ tmpMulA1
                    S = np.diag(S)
                    beta = (U.T @ tmpMulA1)[:keyLen]
                    a_list_number = V @ np.linalg.inv(S.T @ S + alpha * np.eye(keyLen)) @ S.T @ beta

                    U, S, V, = np.linalg.svd(tmpCSIb1IndPerm)
                    V = V.T
                    for i in range(len(S)):
                        if S[i] < r:
                            S[i] = 0
                    # Sr = np.diag(S)
                    # Ar = U @ Sr @ V.T
                    # b_list_number = np.linalg.pinv(Ar) @ tmpMulA1
                    # print(keyLen - len(np.nonzero(S)[0]))
                    S = np.diag(S)
                    beta = (U.T @ tmpMulA1)[:keyLen]
                    b_list_number = V @ np.linalg.inv(S.T @ S + alpha * np.eye(keyLen)) @ S.T @ beta

                    U, S, V, = np.linalg.svd(tmpCSIe1IndPerm)
                    V = V.T
                    for i in range(len(S)):
                        if S[i] < r:
                            S[i] = 0
                    # Sr = np.diag(S)
                    # Ar = U @ Sr @ V.T
                    # e_list_number = np.linalg.pinv(Ar) @ tmpMulA1
                    S = np.diag(S)
                    beta = (U.T @ tmpMulA1)[:keyLen]
                    e_list_number = V @ np.linalg.inv(S.T @ S + alpha * np.eye(keyLen)) @ S.T @ beta
                elif solution == "leastsq":
                    a_list_number = list(np.linalg.lstsq(tmpCSIa1IndPerm, tmpMulA1, rcond=None)[0])
                    b_list_number = list(np.linalg.lstsq(tmpCSIb1IndPerm, tmpMulA1, rcond=None)[0])
                    e_list_number = list(np.linalg.lstsq(tmpCSIe1IndPerm, tmpMulA1, rcond=None)[0])
                else:
                    raise Exception("incorrect solution!")

                # a_list_number = keyBin
                # b_list_number = keyBin
                # e_list_number = keyBin

                # 纠错
                # delta_b_list_number = np.array(a_list_number) - np.array(b_list_number)
                # delta_b_list_number = np.random.normal(np.mean(b_list_number), np.std(b_list_number, ddof=1),
                #                                        len(b_list_number)) - np.array(b_list_number)
                #
                # tmpMulB1 = np.dot(tmpCSIb1IndPerm, delta_b_list_number)
                #
                # a_list_number = list(np.linalg.lstsq(tmpCSIa1IndPerm, tmpMulB1, rcond=None)[0])
                # b_list_number = list(np.linalg.lstsq(tmpCSIb1IndPerm, tmpMulB1, rcond=None)[0])
                # e_list_number = list(np.linalg.lstsq(tmpCSIe1IndPerm, tmpMulB1, rcond=None)[0])

                # 量化
                # if bits == 1:
                #     alpha = 0.1
                #     q1A = np.mean(a_list_number) + alpha * np.std(a_list_number, ddof=1)
                #     q2A = np.mean(a_list_number) - alpha * np.std(a_list_number, ddof=1)
                #     q1B = np.mean(b_list_number) + alpha * np.std(b_list_number, ddof=1)
                #     q2B = np.mean(b_list_number) - alpha * np.std(b_list_number, ddof=1)
                #     q1E = np.mean(e_list_number) + alpha * np.std(e_list_number, ddof=1)
                #     q2E = np.mean(e_list_number) - alpha * np.std(e_list_number, ddof=1)
                #
                #     dropTmp = []
                #     for i in range(len(a_list_number)):
                #         if a_list_number[i] < q1A and a_list_number[i] > q2A:
                #             dropTmp.append(i)
                #         if b_list_number[i] < q1B and b_list_number[i] > q2B:
                #             dropTmp.append(i)
                #
                #     for i in range(len(a_list_number)):
                #         if i in dropTmp:
                #             continue
                #         if a_list_number[i] > q1A:
                #             a_list_number[i] = 1
                #         elif a_list_number[i] < q2A:
                #             a_list_number[i] = 0
                #         if b_list_number[i] > q1B:
                #             b_list_number[i] = 1
                #         elif b_list_number[i] < q2B:
                #             b_list_number[i] = 0
                #         if e_list_number[i] > q1E:
                #             e_list_number[i] = 1
                #         elif e_list_number[i] < q2E:
                #             e_list_number[i] = 0
                # elif bits == 2:
                #     alpha = 0.2
                #     q1A = np.mean(a_list_number) + alpha * np.std(a_list_number, ddof=1)
                #     q2A = np.mean(a_list_number)
                #     q3A = np.mean(a_list_number) - alpha * np.std(a_list_number, ddof=1)
                #     q1B = np.mean(b_list_number) + alpha * np.std(b_list_number, ddof=1)
                #     q2B = np.mean(b_list_number)
                #     q3B = np.mean(b_list_number) - alpha * np.std(b_list_number, ddof=1)
                #     q1E = np.mean(e_list_number) + alpha * np.std(e_list_number, ddof=1)
                #     q2E = np.mean(e_list_number)
                #     q3E = np.mean(e_list_number) - alpha * np.std(e_list_number, ddof=1)
                #
                #     for i in range(len(a_list_number)):
                #         if a_list_number[i] > q1A:
                #             a_list_number[i] = 3
                #         elif a_list_number[i] > q2A:
                #             a_list_number[i] = 2
                #         elif a_list_number[i] > q3A:
                #             a_list_number[i] = 1
                #         else:
                #             a_list_number[i] = 0
                #
                #         if b_list_number[i] > q1B:
                #             b_list_number[i] = 3
                #         elif b_list_number[i] > q2B:
                #             b_list_number[i] = 2
                #         elif b_list_number[i] > q3B:
                #             b_list_number[i] = 1
                #         else:
                #             b_list_number[i] = 0
                #
                #         if e_list_number[i] > q1E:
                #             e_list_number[i] = 3
                #         elif e_list_number[i] > q2E:
                #             e_list_number[i] = 2
                #         elif e_list_number[i] > q3E:
                #             e_list_number[i] = 1
                #         else:
                #             e_list_number[i] = 0
                #
                #     # 转成整数
                #     a_list_number = [round(abs(i)) for i in a_list_number]
                #     b_list_number = [round(abs(i)) for i in b_list_number]
                #     e_list_number = [round(abs(i)) for i in e_list_number]

                a_list_number1 = np.array([round(abs(i)) for i in a_list_number])
                # b_list_numbers = []
                # delta = 0.03
                # # 但有些出错的地方不一定是接近round发生错误的位置，如0.5附近的值
                # if abs(b_list_number[0] - 0.5) < delta:
                #     b_list_numbers.append([0])
                #     b_list_numbers.append([1])
                # elif abs(b_list_number[0] - 1.5) < delta:
                #     b_list_numbers.append([1])
                #     b_list_numbers.append([2])
                # elif abs(b_list_number[0] - 2.5) < delta:
                #     b_list_numbers.append([2])
                #     b_list_numbers.append([3])
                # else:
                #     b_list_numbers.append([round(b_list_number[0])])
                # for i in range(1, keyLen):
                #     if abs(b_list_number[i] - 0.5) < delta:
                #         # 每个当前密钥最新一位的可能值都要复制一份，添加0或1
                #         tmp_numbers = copy.deepcopy(b_list_numbers)
                #         final_numbers = []
                #         for j in range(len(tmp_numbers)):
                #             tmp_numbers[j].append(0)
                #             final_numbers.append(tmp_numbers[j])
                #         tmp_numbers = copy.deepcopy(b_list_numbers)
                #         for j in range(len(tmp_numbers)):
                #             tmp_numbers[j].append(1)
                #             final_numbers.append(tmp_numbers[j])
                #         b_list_numbers = final_numbers.copy()
                #     elif abs(b_list_number[i] - 1.5) < delta:
                #         tmp_numbers = copy.deepcopy(b_list_numbers)
                #         final_numbers = []
                #         for j in range(len(tmp_numbers)):
                #             tmp_numbers[j].append(1)
                #             final_numbers.append(tmp_numbers[j])
                #         tmp_numbers = copy.deepcopy(b_list_numbers)
                #         for j in range(len(tmp_numbers)):
                #             tmp_numbers[j].append(2)
                #             final_numbers.append(tmp_numbers[j])
                #         b_list_numbers = final_numbers.copy()
                #     elif abs(b_list_number[i] - 2.5) < delta:
                #         tmp_numbers = copy.deepcopy(b_list_numbers)
                #         final_numbers = []
                #         for j in range(len(tmp_numbers)):
                #             tmp_numbers[j].append(2)
                #             final_numbers.append(tmp_numbers[j])
                #         tmp_numbers = copy.deepcopy(b_list_numbers)
                #         for j in range(len(tmp_numbers)):
                #             tmp_numbers[j].append(3)
                #             final_numbers.append(tmp_numbers[j])
                #         b_list_numbers = final_numbers.copy()
                #     else:
                #         for j in range(len(b_list_numbers)):
                #             b_list_numbers[j].append(round(b_list_number[i]))
                # # 但是有些正确的密钥位对应的是大的MSE
                # min_squares = sum(abs(tmpCSIb1IndPerm @ b_list_numbers[0] - tmpMulA1))
                # b_list_number = b_list_numbers[0]
                # for i in range(1, len(b_list_numbers)):
                #     tmp_squares = sum(abs(tmpCSIb1IndPerm @ b_list_numbers[i] - tmpMulA1))
                #     if tmp_squares < min_squares:
                #         min_squares = tmp_squares
                #         b_list_number = b_list_numbers[i]
                b_list_number1 = np.array([round(abs(i)) for i in b_list_number])
                # if bits == 1:
                #     e_list_number = [round(abs(i)) % 2 for i in e_list_number]
                # else:
                #     e_list_number = [round(abs(i)) % 4 for i in e_list_number]
                e_list_number1 = np.array([round(abs(i)) for i in e_list_number])

                # a_list_number1 = np.array(findClosest(basic, a_list_number))
                # b_list_number1 = np.array(findClosest(basic, b_list_number))
                # e_list_number1 = np.array(findClosest(basic, e_list_number))

                # err = np.nonzero(a_list_number1 - b_list_number1)[0]
                # if err.size != 0:
                #     a_list_number1 = np.array(findClosest([0, 2, 4, 8], a_list_number))
                #     b_list_number1 = np.array(findClosest([0, 2, 4, 8], b_list_number))
                #     e_list_number1 = np.array(findClosest([0, 2, 4, 8], e_list_number))

                # err = np.nonzero(a_list_number1 - b_list_number1)[0]
                # x = a_list_number1
                # x_prime = b_list_number1
                # delta_x = x_prime - x
                # A = tmpCSIa1IndPerm
                # A_prime = tmpCSIb1IndPerm
                # delta_A = A_prime - A
                # condition1 = np.linalg.cond(A)
                # condition2 = np.linalg.cond(A_prime)
                # left = np.linalg.norm(delta_x) / np.linalg.norm(x_prime)
                # right = np.linalg.svd(delta_A)[1][0] / np.linalg.svd(A_prime)[1][0]
                # # print(err.size, left <= condition1 * right, left, condition1, condition2, right)

                if error_correct == True:
                    # 纠错：纠错前A发送了RSSa*x=b，纠错时B发送RSSb*x'=b'，效果变好
                    # np.random.seed(1)
                    # if bits == 1:
                    #     keyBin_b = np.random.binomial(1, 0.5, keyLen)
                    # else:
                    #     keyBin_b = np.random.randint(0, 4, keyLen)
                    #
                    # tmpMulB1 = np.dot(tmpCSIb1IndPerm, keyBin_b)
                    tmpMulB1 = np.dot(tmpCSIb1IndPerm, b_list_number1)

                    lambda_ = 0
                    solvers = [cp.SCS, cp.GUROBI, cp.OSQP, cp.CVXOPT, cp.ECOS, cp.SCIP, cp.PROXQP, cp.MOSEK,
                               cp.CLARABEL, cp.NAG, cp.XPRESS]
                    solver = solvers[2]
                    x = cp.Variable(len(keyBin))
                    # 加正则项效果差
                    f_norm = np.linalg.norm(tmpCSIa1IndPerm, ord='fro')
                    obj = cp.Minimize(cp.sum_squares(tmpCSIa1IndPerm @ x - tmpMulB1) + lambda_ * cp.sum_squares(x))
                    # prob = cp.Problem(obj)
                    prob = cp.Problem(obj, [x >= min(basic), x <= max(basic)])
                    if solver == cp.SCS:
                        prob.solve(solver=solver, max_iters=5000)
                    elif solver == cp.SCIP:
                        prob.solve(solver=solver, scip_params={"limits/time": 10})
                    elif solver == cp.OSQP:
                        # scaling=False表示禁用缩放以矩阵均衡
                        prob.solve(solver=solver, scaling=False)
                    else:
                        prob.solve(solver=solver)
                    a_float_list_number2 = [i.value for i in x]

                    x = cp.Variable(len(keyBin))
                    # 加正则项效果差
                    obj = cp.Minimize(cp.sum_squares(tmpCSIb1IndPerm @ x - tmpMulB1) + lambda_ * cp.sum_squares(x))
                    # prob = cp.Problem(obj)
                    prob = cp.Problem(obj, [x >= min(basic), x <= max(basic)])
                    if solver == cp.SCS:
                        prob.solve(solver=solver, max_iters=5000)
                    elif solver == cp.SCIP:
                        prob.solve(solver=solver, scip_params={"limits/time": 10})
                    elif solver == cp.OSQP:
                        prob.solve(solver=solver, scaling=False)
                    else:
                        prob.solve(solver=solver)
                    b_float_list_number2 = [i.value for i in x]

                    x = cp.Variable(len(keyBin))
                    # 加正则项效果差
                    obj = cp.Minimize(cp.sum_squares(tmpCSIe1IndPerm @ x - tmpMulB1) + lambda_ * cp.sum_squares(x))
                    # prob = cp.Problem(obj)
                    prob = cp.Problem(obj, [x >= min(basic), x <= max(basic)])
                    if solver == cp.SCS:
                        prob.solve(solver=solver, max_iters=5000)
                    elif solver == cp.SCIP:
                        prob.solve(solver=solver, scip_params={"limits/time": 10})
                    elif solver == cp.OSQP:
                        prob.solve(solver=solver, scaling=False)
                    else:
                        prob.solve(solver=solver)
                    e_float_list_number2 = [i.value for i in x]

                    a_list_number2 = np.array([round(abs(i)) for i in a_float_list_number2])
                    b_list_number2 = np.array([round(abs(i)) for i in b_float_list_number2])
                    e_list_number2 = np.array([round(abs(i)) for i in e_float_list_number2])

                    a_list_number_sum = np.array([round(abs(i / 2)) for i in a_list_number1 + a_list_number2])
                    b_list_number_sum = np.array([round(abs(i / 2)) for i in b_list_number1 + b_list_number2])
                    e_list_number_sum = np.array([round(abs(i / 2)) for i in e_list_number1 + e_list_number2])
                    errors1 = np.nonzero(a_list_number1 - b_list_number1)[0]
                    # print("Before rec errors1", len(errors1), errors1)
                    errors2 = np.nonzero(a_list_number2 - b_list_number2)[0]
                    # print("Before rec errors2", len(errors2), errors2)
                    errors_sum = np.nonzero(a_list_number_sum - b_list_number_sum)[0]
                    # print("Before rec errors_sum", len(errors_sum), errors_sum)

                    interval = 0.2
                    prone1 = []
                    for i in range(len(b_list_number)):
                        if (1 / 2 + interval > b_list_number[i] > 1 / 2 - interval) \
                                or (3 / 2 + interval > b_list_number[i] > 3 / 2 - interval) \
                                or (5 / 2 + interval > b_list_number[i] > 5 / 2 - interval):
                            prone1.append(i)
                    # print("prone1", len(prone1), np.array(prone1))
                    prone2 = []
                    for i in range(len(a_float_list_number2)):
                        if (1 / 2 + interval > a_float_list_number2[i] > 1 / 2 - interval) \
                                or (3 / 2 + interval > a_float_list_number2[i] > 3 / 2 - interval) \
                                or (5 / 2 + interval > a_float_list_number2[i] > 5 / 2 - interval):
                            prone2.append(i)
                    # print("prone2", len(prone2), np.array(prone2))
                    prone_sum = []
                    for i in range(len(a_list_number)):
                        if (1 / 2 + interval > a_list_number[i] + a_float_list_number2[i] > 1 / 2 - interval) \
                                or (3 / 2 + interval > a_list_number[i] + a_float_list_number2[i] > 3 / 2 - interval) \
                                or (5 / 2 + interval > a_list_number[i] + a_float_list_number2[i] > 5 / 2 - interval):
                            prone_sum.append(i)
                    # print("prone_sum", len(prone_sum), np.array(prone_sum))

                    mismatched = np.nonzero(a_list_number1 - a_list_number2)[0]
                    # print("Mismatch of a1 and a2", len(mismatched), mismatched)
                    if len(mismatched) != 0:
                        for i in mismatched:
                            isChanged = False
                            for j in range(i - 1, -1, -1):
                                if a_list_number1[j % keyLen] == a_list_number2[j % keyLen]:
                                    a_list_number1[i] = a_list_number1[j % keyLen]
                                    a_list_number2[i] = a_list_number2[j % keyLen]
                                    b_list_number1[i] = b_list_number1[j % keyLen]
                                    b_list_number2[i] = b_list_number2[j % keyLen]
                                    isChanged = True
                                    break
                            if isChanged == False:
                                for j in range(i + 1, keyLen):
                                    if a_list_number1[j % keyLen] == a_list_number2[j % keyLen]:
                                        a_list_number1[i] = a_list_number1[j % keyLen]
                                        a_list_number2[i] = a_list_number2[j % keyLen]
                                        b_list_number1[i] = b_list_number1[j % keyLen]
                                        b_list_number2[i] = b_list_number2[j % keyLen]
                                        isChanged = True
                                        break

                    a_list_number_sum = np.array([round(abs(i / 2)) for i in a_list_number1 + a_list_number2])
                    b_list_number_sum = np.array([round(abs(i / 2)) for i in b_list_number1 + b_list_number2])
                    e_list_number_sum = np.array([round(abs(i / 2)) for i in e_list_number1 + e_list_number2])

                    # a_list_number_sum = np.array([round(i) for i in a_list_number1 + a_list_number2])
                    # b_list_number_sum = np.array([round(i) for i in b_list_number1 + b_list_number2])
                    # e_list_number_sum = np.array([round(i) for i in e_list_number1 + e_list_number2])

                    # print(abs(np.array(a_list_number1) - np.array(b_list_number1)).sum())
                    # print(abs(np.array(a_list_number2) - np.array(b_list_number2)).sum())
                    # print(abs(np.array(a_list_number_sum) - np.array(b_list_number_sum)).sum())
                    # print()

                    errors = np.nonzero(a_list_number_sum - b_list_number_sum)[0]
                    # print("After rec", len(errors), errors)
                    # print()

                    if bits == 2:
                        a_list_number2 = np.array([round(abs(i)) % 4 for i in a_list_number_sum])
                        b_list_number2 = np.array([round(abs(i)) % 4 for i in b_list_number_sum])
                        e_list_number2 = np.array([round(abs(i)) % 4 for i in e_list_number_sum])
                    else:
                        a_list_number2 = np.array([round(abs(i)) % 2 for i in a_list_number_sum])
                        b_list_number2 = np.array([round(abs(i)) % 2 for i in b_list_number_sum])
                        e_list_number2 = np.array([round(abs(i)) % 2 for i in e_list_number_sum])

                    a_list_number1 = a_list_number2
                    b_list_number1 = b_list_number2
                    e_list_number1 = e_list_number2

                    # # 纠错：纠错前a发送了RSSa*key=b，纠错时b随机产生e，再利用自己得到的key'计算B*(key'+e)=c
                    # # a用c和key恢复key+e作为最终的密钥，效果更差
                    # np.random.seed(1)
                    # if bits == 1:
                    #     perturb = np.random.binomial(1, 0.5, keyLen)
                    # else:
                    #     perturb = np.random.randint(0, 4, keyLen)
                    #
                    # tmpCSIa1IndPerm = circulant(tmpMulA1[::-1])
                    #
                    # tmpCSIa1IndPerm = normalize(tmpCSIa1IndPerm)
                    #
                    # tmpMulB1 = np.dot(tmpCSIa1IndPerm, b_list_number1 + perturb)
                    # # tmpMulB1 = np.dot(tmpCSIa1IndPerm, perturb - b_list_number1)
                    #
                    # solvers = [cp.SCS, cp.GUROBI, cp.OSQP, cp.CVXOPT, cp.ECOS, cp.SCIP, cp.PROXQP, cp.MOSEK,
                    #            cp.CLARABEL, cp.NAG, cp.XPRESS]
                    # solver = solvers[2]
                    # x = cp.Variable(len(keyBin))
                    # # 加正则项效果差
                    # # obj = cp.Minimize(cp.sum_squares(tmpCSIa1IndPerm @ x - tmpMulA1) + 0.1 * cp.sum_squares(x))
                    # obj = cp.Minimize(cp.sum_squares(tmpCSIa1IndPerm @ x - tmpMulB1))
                    # # prob = cp.Problem(obj)
                    # prob = cp.Problem(obj, [x >= 0, x <= 6])
                    # # prob = cp.Problem(obj, [x >= -3, x <= 3])
                    # if solver == cp.SCS:
                    #     prob.solve(solver=solver, max_iters=5000)
                    # elif solver == cp.SCIP:
                    #     prob.solve(solver=solver, scip_params={"limits/time": 10})
                    # else:
                    #     prob.solve(solver=solver)
                    # perturbed_list_number = [i.value for i in x]
                    #
                    # a_list_number2 = np.array(perturbed_list_number) - a_list_number1
                    # b_list_number2 = np.array(perturbed_list_number) - b_list_number1
                    # e_list_number2 = np.array(perturbed_list_number) - e_list_number1
                    # # a_list_number2 = np.array(perturbed_list_number) + a_list_number1
                    # # b_list_number2 = np.array(perturbed_list_number) + b_list_number1
                    # # e_list_number2 = np.array(perturbed_list_number) + e_list_number1
                    #
                    # a_list_number2 = np.array([round(abs(i)) for i in a_list_number2])
                    # b_list_number2 = np.array([round(abs(i)) for i in b_list_number2])
                    # e_list_number2 = np.array([round(abs(i)) for i in e_list_number2])
                    #
                    # a_list_number1 = a_list_number2
                    # b_list_number1 = b_list_number2
                    # e_list_number1 = e_list_number2

                    # 纠错：纠错前a发送了RSSa*key=b，纠错时b随机产生y，再利用自己得到的key'计算key'*y=c
                    # a用c和key恢复y，y当作最终的密钥
                    # np.random.seed(0)
                    # if bits == 1:
                    #     keyBin1 = np.random.binomial(1, 0.5, keyLen)
                    # else:
                    #     keyBin1 = np.random.randint(0, 4, keyLen)
                    #
                    # tmpCSIa1IndPerm = circulant(a_list_number1[::-1])
                    # tmpCSIb1IndPerm = circulant(b_list_number1[::-1])
                    # tmpCSIe1IndPerm = circulant(e_list_number1[::-1])
                    #
                    # tmpCSIa1IndPerm = normalize(tmpCSIa1IndPerm)
                    # if np.max(tmpCSIb1IndPerm) == np.min(tmpCSIb1IndPerm):
                    #     tmpCSIb1IndPerm = (tmpCSIb1IndPerm - np.min(tmpCSIb1IndPerm)) / np.max(tmpCSIb1IndPerm)
                    # else:
                    #     tmpCSIb1IndPerm = normalize(tmpCSIb1IndPerm)
                    # tmpCSIe1IndPerm = normalize(tmpCSIe1IndPerm)
                    #
                    # tmpMulB1 = np.dot(tmpCSIa1IndPerm, keyBin1)
                    #
                    # solvers = [cp.SCS, cp.GUROBI, cp.OSQP, cp.CVXOPT, cp.ECOS, cp.SCIP, cp.PROXQP, cp.MOSEK,
                    #            cp.CLARABEL, cp.NAG, cp.XPRESS]
                    # solver = solvers[2]
                    # x = cp.Variable(len(keyBin))
                    # # 加正则项效果差
                    # # obj = cp.Minimize(cp.sum_squares(tmpCSIa1IndPerm @ x - tmpMulA1) + 0.1 * cp.sum_squares(x))
                    # obj = cp.Minimize(cp.sum_squares(tmpCSIa1IndPerm @ x - tmpMulB1))
                    # # prob = cp.Problem(obj)
                    # prob = cp.Problem(obj, [x >= 0, x <= 3])
                    # if solver == cp.SCS:
                    #     prob.solve(solver=solver, max_iters=5000)
                    # elif solver == cp.SCIP:
                    #     prob.solve(solver=solver, scip_params={"limits/time": 10})
                    # else:
                    #     prob.solve(solver=solver)
                    # a_list_number2 = [i.value for i in x]
                    #
                    # x = cp.Variable(len(keyBin))
                    # obj = cp.Minimize(cp.sum_squares(tmpCSIb1IndPerm @ x - tmpMulB1))
                    # # prob = cp.Problem(obj)
                    # prob = cp.Problem(obj, [x >= 0, x <= 3])
                    # if solver == cp.SCS:
                    #     prob.solve(solver=solver, max_iters=5000)
                    # elif solver == cp.SCIP:
                    #     prob.solve(solver=solver, scip_params={"limits/time": 10})
                    # else:
                    #     prob.solve(solver=solver)
                    # b_list_number2 = [i.value for i in x]
                    #
                    # x = cp.Variable(len(keyBin))
                    # obj = cp.Minimize(cp.sum_squares(tmpCSIe1IndPerm @ x - tmpMulB1))
                    # # prob = cp.Problem(obj)
                    # prob = cp.Problem(obj, [x >= 0, x <= 3])
                    # if solver == cp.SCS:
                    #     prob.solve(solver=solver, max_iters=5000)
                    # elif solver == cp.SCIP:
                    #     prob.solve(solver=solver, scip_params={"limits/time": 10})
                    # else:
                    #     prob.solve(solver=solver)
                    # e_list_number2 = [i.value for i in x]
                    #
                    # a_list_number2 = np.array([round(abs(i)) for i in a_list_number2])
                    # b_list_number2 = np.array([round(abs(i)) for i in b_list_number2])
                    # e_list_number2 = np.array([round(abs(i)) for i in e_list_number2])
                    #
                    # a_list_number1 = a_list_number2
                    # b_list_number1 = b_list_number2
                    # e_list_number1 = e_list_number2

                    # print(abs(a_list_number1 - b_list_number1).sum())
                    # print(abs(a_list_number2 - b_list_number2).sum())

                    # 纠错，同chirpkey中的，B计算b-RSSb*x'，然后用该结果作为b再求一次x''，效果更差
                    # delta_b = np.matmul(tmpCSIb1IndPerm, b_list_number1) - tmpMulA1
                    # delta_e = np.matmul(tmpCSIe1IndPerm, e_list_number1) - tmpMulA1
                    #
                    # # cvxpy perturbed least squares
                    # # SCIP最慢，其他结果一致
                    # solvers = [cp.SCS, cp.GUROBI, cp.OSQP, cp.CVXOPT, cp.ECOS, cp.SCIP, cp.PROXQP, cp.MOSEK,
                    #            cp.CLARABEL, cp.NAG, cp.XPRESS]
                    # solver = solvers[2]
                    #
                    # x = cp.Variable(len(keyBin))
                    # obj = cp.Minimize(cp.sum_squares(tmpCSIb1IndPerm @ x - delta_b))
                    # prob = cp.Problem(obj)
                    # # prob = cp.Problem(obj, [x >= 0, x <= 3])
                    # if solver == cp.SCS:
                    #     prob.solve(solver=solver, max_iters=5000)
                    # elif solver == cp.SCIP:
                    #     prob.solve(solver=solver, scip_params={"limits/time": 10})
                    # else:
                    #     prob.solve(solver=solver)
                    # delta_b_number = [i.value for i in x]
                    #
                    # x = cp.Variable(len(keyBin))
                    # obj = cp.Minimize(cp.sum_squares(tmpCSIe1IndPerm @ x - delta_e))
                    # prob = cp.Problem(obj)
                    # # prob = cp.Problem(obj, [x >= 0, x <= 3])
                    # if solver == cp.SCS:
                    #     prob.solve(solver=solver, max_iters=5000)
                    # elif solver == cp.SCIP:
                    #     prob.solve(solver=solver, scip_params={"limits/time": 10})
                    # else:
                    #     prob.solve(solver=solver)
                    # delta_e_number = [i.value for i in x]
                    #
                    # b_list_number2 = np.array(b_list_number) - np.array(delta_b_number)
                    # e_list_number2 = np.array(e_list_number) - np.array(delta_e_number)
                    #
                    # b_list_number2 = np.array([round(abs(i)) for i in b_list_number2])
                    # e_list_number2 = np.array([round(abs(i)) for i in e_list_number2])
                    # # print(abs(a_list_number1 - b_list_number1).sum())
                    # # print(abs(a_list_number1 - b_list_number2).sum())

                    # 纠错：将[0.5-0.2,0.5+0.2]的索引找出来，看作是易错的位置，此处的密钥值以前一位数据代替（熵减）
                    # 而且前一位的密钥暴露后此处的密钥泄露
                    # if np.allclose(a_list_number1, b_list_number1) is False:
                    #     print("before rec")
                    #     # 第一个相同的位置，且b的不能落入interval中
                    #     interval = 0.2
                    #     mask = a_list_number1 == b_list_number1
                    #     common_index = -1
                    #     for i in range(len(b_list_number)):
                    #         if bits == 1:
                    #             if mask[i] and (b_list_number[i] < 1 / 2 - interval or b_list_number[i] > 1 / 2 + interval):
                    #                 common_index = i
                    #                 break
                    #         elif bits == 2:
                    #             if mask[i] and ((b_list_number[i] < 1 / 2 - interval or b_list_number[i] > 1 / 2 + interval)
                    #                             or (b_list_number[i] < 3 / 2 - interval or b_list_number[
                    #                         i] > 3 / 2 + interval)
                    #                             or (b_list_number[i] < 5 / 2 - interval or b_list_number[
                    #                         i] > 5 / 2 + interval)):
                    #                 common_index = i
                    #                 break
                    #     # 以common_index为开头
                    #     a_list_number = np.roll(a_list_number, -common_index)
                    #     b_list_number = np.roll(b_list_number, -common_index)
                    #     e_list_number = np.roll(e_list_number, -common_index)
                    #
                    #     a_list_number1 = np.roll(a_list_number1, -common_index)
                    #     b_list_number1 = np.roll(b_list_number1, -common_index)
                    #     diff_index = []
                    #     for i in range(len(b_list_number)):
                    #         if bits == 1:
                    #             if 1 / 2 - interval <= b_list_number[i] <= 1 / 2 + interval:
                    #                 diff_index.append(i)
                    #         elif bits == 2:
                    #             if 1 / 2 - interval <= b_list_number[i] <= 1 / 2 + interval or \
                    #                     3 / 2 - interval <= b_list_number[i] <= 3 / 2 + interval or \
                    #                     5 / 2 - interval <= b_list_number[i] <= 5 / 2 + interval:
                    #                 diff_index.append(i)
                    #     diffs0 = []
                    #     for i in range(len(diff_index)):
                    #         diffs0.append([diff_index[i], b_list_number[diff_index[i]], a_list_number[diff_index[i]]])
                    #     print("易错的", len(diffs0), diffs0)
                    #     error = []
                    #     for i in range(len(a_list_number1)):
                    #         if a_list_number1[i] != b_list_number1[i]:
                    #             error.append([i, b_list_number[i], a_list_number[i]])
                    #     print("实际错误的", len(error), error)
                    #     a_list_number1 = rec(a_list_number, bits, diff_index)
                    #     b_list_number1 = rec(b_list_number, bits, diff_index)
                    #     e_list_number1 = rec(e_list_number, bits, diff_index)
                    #     # print(np.allclose(a_list_number1, b_list_number1))
                    #
                    # if np.allclose(a_list_number1, b_list_number1) is False:
                    #     print("after rec")
                    #     diff = np.nonzero(a_list_number1 - b_list_number1)[0]
                    #     diffs1 = []
                    #     for i in range(len(diff)):
                    #         diffs1.append([diff[i], b_list_number[diff[i]], a_list_number[diff[i]]])
                    #     print("仍然错误的", len(diffs1), diffs1)

                    # 另一种：利用frodo rec进行，容错范围必须在合适的范围内，可以找出易错的位置，将此处的密钥放在二进制的低位
                    # 然后不易出错的位置放在二进制的高位，使用rec进行纠错（更加随机）

                    # 用LDPC纠错，效果最好
                    # isCorrected = False
                    # if a_list_number1 != b_list_number1:
                    #     isCorrected = True
                    #     a_list = []
                    #     b_list = []
                    #     e_list = []
                    #     seed = np.random.RandomState(42)
                    #     n_code, d_v, d_c = 0, 0, 0
                    #     if len(a_list_number1) == 256:
                    #         n_code = 261
                    #         d_v = 2
                    #         d_c = 87
                    #     elif len(a_list_number1) == 512:
                    #         n_code = 540
                    #         d_v = 3
                    #         d_c = 54
                    #     H, G = make_ldpc(n_code, d_v, d_c, seed=seed, systematic=True, sparse=True)
                    #     k = G.shape[1]
                    #     snr = 20
                    #     ra = np.random.randint(2, size=n_code)
                    #     rb = np.random.randint(2, size=n_code)
                    #     re = np.random.randint(2, size=n_code)
                    #     for i in range(len(a_list_number1)):
                    #         a_list += '{:02b}'.format(graycode.tc_to_gray_code(a_list_number1[i]))
                    #     for i in range(len(b_list_number1)):
                    #         b_list += '{:02b}'.format(graycode.tc_to_gray_code(b_list_number1[i]))
                    #     for i in range(len(e_list_number1)):
                    #         e_list += '{:02b}'.format(graycode.tc_to_gray_code(e_list_number1[i]))
                    #     a_list = np.array(a_list).astype(int)
                    #     b_list = np.array(b_list).astype(int)
                    #     e_list = np.array(e_list).astype(int)
                    #     print("before", abs(np.array(a_list) - np.array(b_list)).sum())
                    #     ya = (encode(G, a_list, seed=seed) + ra) % 2
                    #     yb = (encode(G, b_list, seed=seed) + rb) % 2
                    #     ye = (encode(G, e_list, seed=seed) + re) % 2
                    #     xa = (decode(H, yb, snr) + ra) % 2
                    #     xb = (decode(H, ya, snr) + rb) % 2
                    #     xe = (decode(H, ye, snr) + re) % 2
                    #     a_list_number1 = get_message(G, xa)
                    #     b_list_number1 = get_message(G, xb)
                    #     e_list_number1 = get_message(G, xe)
                    #     print("after", abs(np.array(a_list_number1) - np.array(b_list_number1)).sum())

                    # 纠错：把密钥看作是矩阵A，再重新生成密钥x进行密钥生成，效果提升有限
                    # if a_list_number1 != b_list_number1:
                    #     errors = np.nonzero(np.array(a_list_number1) - np.array(b_list_number1))[0]
                    #     # print("before ecc", len(errors))
                    #     # for i in range(len(errors)):
                    #     #     print(errors[i], a_list_number[int(errors[i])], b_list_number[int(errors[i])])
                    #
                    #     np.random.seed(0)
                    #     randomMatrix = np.random.uniform(5, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
                    #     # 均值化
                    #     # a_list_number1 = a_list_number1 - np.mean(a_list_number1)
                    #     # b_list_number1 = b_list_number1 - np.mean(b_list_number1)
                    #     # e_list_number1 = e_list_number1 - np.mean(e_list_number1)
                    #     # 标准化
                    #     a_list_number1 = normalize(a_list_number1)
                    #     if np.max(b_list_number1) == np.min(b_list_number1):
                    #         b_list_number1 = (b_list_number1 - np.min(b_list_number1)) / np.max(b_list_number1)
                    #     else:
                    #         b_list_number1 = normalize(b_list_number1)
                    #     e_list_number1 = normalize(e_list_number1)
                    #     a_list_number1 = np.matmul(a_list_number1, randomMatrix)
                    #     b_list_number1 = np.matmul(b_list_number1, randomMatrix)
                    #     e_list_number1 = np.matmul(e_list_number1, randomMatrix)
                    #
                    #     np.random.seed(0)
                    #     if bits == 1:
                    #         keyBin = np.random.binomial(1, 0.5, keyLen)
                    #     else:
                    #         keyBin = np.random.randint(0, 4, keyLen)
                    #     tmpCSIa1IndPerm = circulant(a_list_number1[::-1])
                    #     tmpCSIb1IndPerm = circulant(b_list_number1[::-1])
                    #     tmpCSIe1IndPerm = circulant(e_list_number1[::-1])
                    #     tmpCSIa1IndPerm = normalize(tmpCSIa1IndPerm)
                    #     if np.max(tmpCSIb1IndPerm) == np.min(tmpCSIb1IndPerm):
                    #         tmpCSIb1IndPerm = (tmpCSIb1IndPerm - np.min(tmpCSIb1IndPerm)) / np.max(tmpCSIb1IndPerm)
                    #     else:
                    #         tmpCSIb1IndPerm = normalize(tmpCSIb1IndPerm)
                    #     tmpCSIe1IndPerm = normalize(tmpCSIe1IndPerm)
                    #     tmpMulA1 = np.dot(tmpCSIa1IndPerm, keyBin)
                    #
                    #     # cvxpy perturbed least squares
                    #     solvers = [cp.SCS, cp.GUROBI, cp.OSQP, cp.CVXOPT, cp.ECOS, cp.SCIP, cp.PROXQP, cp.MOSEK, cp.CLARABEL,
                    #                cp.NAG, cp.XPRESS]
                    #     solver = solvers[2]
                    #     x = cp.Variable(len(keyBin))
                    #     obj = cp.Minimize(cp.sum_squares(tmpCSIa1IndPerm @ x - tmpMulA1))
                    #     prob = cp.Problem(obj, [x >= 0, x <= 3])
                    #     if solver == cp.SCS:
                    #         prob.solve(solver=solver, max_iters=5000)
                    #     elif solver == cp.SCIP:
                    #         prob.solve(solver=solver, scip_params={"limits/time": 10})
                    #     else:
                    #         prob.solve(solver=solver)
                    #     a_list_number = [i.value for i in x]
                    #
                    #     x = cp.Variable(len(keyBin))
                    #     obj = cp.Minimize(cp.sum_squares(tmpCSIb1IndPerm @ x - tmpMulA1))
                    #     prob = cp.Problem(obj, [x >= 0, x <= 3])
                    #     if solver == cp.SCS:
                    #         prob.solve(solver=solver, max_iters=5000)
                    #     elif solver == cp.SCIP:
                    #         prob.solve(solver=solver, scip_params={"limits/time": 10})
                    #     else:
                    #         prob.solve(solver=solver)
                    #     b_list_number = [i.value for i in x]
                    #
                    #     x = cp.Variable(len(keyBin))
                    #     obj = cp.Minimize(cp.sum_squares(tmpCSIe1IndPerm @ x - tmpMulA1))
                    #     prob = cp.Problem(obj, [x >= 0, x <= 3])
                    #     if solver == cp.SCS:
                    #         prob.solve(solver=solver, max_iters=5000)
                    #     elif solver == cp.SCIP:
                    #         prob.solve(solver=solver, scip_params={"limits/time": 10})
                    #     else:
                    #         prob.solve(solver=solver)
                    #     e_list_number = [i.value for i in x]
                    #
                    #     a_list_number1 = [round(abs(i)) for i in a_list_number]
                    #     b_list_number1 = [round(abs(i)) for i in b_list_number]
                    #     e_list_number1 = [round(abs(i)) for i in e_list_number]
                    #
                    #     errors = np.nonzero(np.array(a_list_number1) - np.array(b_list_number1))[0]
                    #     # print("after ecc", len(errors))
                    #     # for i in range(len(errors)):
                    #     #     print(errors[i], a_list_number[int(errors[i])], b_list_number[int(errors[i])])
                    #     # print()

                # if np.allclose(a_list_number1, b_list_number1) is False:
                #     print()

                a_list_number = a_list_number1
                b_list_number = b_list_number1
                e_list_number = e_list_number1

                # 统计错误的bit对
                err = np.nonzero(a_list_number - b_list_number)[0]
                if err.size > 0:
                    for i in range(len(err)):
                        error_bits.append(str(a_list_number[err[i]]) + str(b_list_number[err[i]]))

                # print(len(np.nonzero(a_list_number1 - b_list_number1)[0]))

                # 转成二进制，0填充成0000
                # if bits == 2 and isCorrected is False:
                if bits == 2:
                    # for i in range(len(a_list_number)):
                    #     number = bin(a_list_number[i])[2:].zfill(int(np.log2(4)))
                    #     a_list += number
                    # for i in range(len(b_list_number)):
                    #     number = bin(b_list_number[i])[2:].zfill(int(np.log2(4)))
                    #     b_list += number
                    # for i in range(len(e_list_number)):
                    #     number = bin(e_list_number[i])[2:].zfill(int(np.log2(4)))
                    #     e_list += number
                    # gray码
                    for i in range(len(a_list_number)):
                        a_list += '{:02b}'.format(graycode.tc_to_gray_code(a_list_number[i]))
                    for i in range(len(b_list_number)):
                        b_list += '{:02b}'.format(graycode.tc_to_gray_code(b_list_number[i]))
                    for i in range(len(e_list_number)):
                        e_list += '{:02b}'.format(graycode.tc_to_gray_code(e_list_number[i]))

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
                # sum1 = min(len(a_list_number), len(b_list_number))
                # sum2 = 0
                # sum3 = 0
                # for i in range(0, sum1):
                #     sum2 += (a_list_number[i] == b_list_number[i])
                # for i in range(min(len(a_list_number), len(e_list_number))):
                #     sum3 += (a_list_number[i] == e_list_number[i])

                # if sum1 != sum2:
                # a_list_number = nnls(tmpCSIa1IndPerm, tmpMulA1)[0]
                # b_list_number = nnls(tmpCSIb1IndPerm, tmpMulA1)[0]
                # e_list_number = nnls(tmpCSIe1IndPerm, tmpMulA1)[0]
                # print()
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
            # print(round(correctSum / originSum, 10), round(correctWholeSum / originWholeSum, 10),
            #       round(originSum / times / (block_number * block_shape), 10),
            #       round(correctSum / times / (block_number * block_shape), 10))
            if withoutSort:
                print("withoutSort")
            else:
                print("withSort")
            print("iterations_of_a", max(iterations_of_a), min(iterations_of_a), np.mean(iterations_of_a))
            print("iterations_of_b", max(iterations_of_b), min(iterations_of_b), np.mean(iterations_of_b))
            print("error_bits", Counter(error_bits))
            print("\n")
    # savemat('integer_lsq/tmpCSIa1IndPerm.mat', {'Ba': Ba})
    # savemat('integer_lsq/tmpCSIb1IndPerm.mat', {'Bb': Bb})
    # savemat('integer_lsq/tmpCSIe1IndPerm.mat', {'Be': Be})
    # savemat('integer_lsq/tmpMulA1.mat', {'y': y})
messagebox.showinfo("提示", "测试结束")
print("all_iterations", max(all_iterations), min(all_iterations), np.mean(all_iterations))
print("used_time", max(used_time), min(used_time), np.mean(used_time))
