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
from scipy.optimize import least_squares, leastsq, nnls, minimize

import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
import xpress
import logging
import matlab.engine

from pyldpc import make_ldpc, encode, decode, get_message

from tls import tls, stls, block_toeplitz, tls2, stls_qp, block_circulant


def frequency(samples):
    samples = np.array(samples)
    total_samples = len(samples)

    # 使用字典来记录每个数值出现的次数
    frequency_count = {}
    for sample in samples:
        if sample in frequency_count:
            frequency_count[sample] += 1
        else:
            frequency_count[sample] = 1

    # 计算每个数值的频率，即概率分布
    frequency = []
    for sample in frequency_count:
        frequency.append(frequency_count[sample] / total_samples)

    return frequency


def minEntropy(probabilities):
    return -np.log2(np.max(probabilities) + 1e-12)


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
    if np.max(data) == np.min(data):
        return (data - np.min(data)) / np.max(data)
    else:
        return (np.array(data) - np.min(data)) / (np.max(data) - np.min(data))


# fileName = ["../csi/csi_static_indoor_1_r"]

# "./data_alignment/csi_ear.mat"
# fileName = ["./data_alignment/csi_ear.mat"]

# fileName = ["../csi/csi_static_indoor_1_r"]

fileName = ["../data/data_mobile_indoor_1.mat",
            "../data/data_mobile_outdoor_1.mat",
            "../data/data_static_outdoor_1.mat",
            "../data/data_static_indoor_1.mat"
            ]

# fileName = ["../csi/csi_mobile_indoor_1_r",
#             "../csi/csi_mobile_outdoor_r",
#             "../csi/csi_static_indoor_1_r",
#             "../csi/csi_static_outdoor_r"]

# fileName = ["../data/data_NLOS.mat"]

# RSS security strength random_keys
# mi1 0.8749999999769169    0.9485717192494804
# si1 0.8749999999769169    0.9554820236857732
# mo1 0.8421206992515399    0.954078616758146
# so1 0.8662013339915913    0.9513041484944179

# CSI security strength random_keys
# mi1 0.8578120595102735    0.9568964378389685
# si1 0.8747031490784669    0.9530242050914822
# mo1 0.8643604232734955    0.9466510840032142
# so1 0.8842500726561741    0.9499327587670594

# 是否添加噪声
addNoises = ["mul"]

bits = 2

isBalanced = True

for f in fileName:
    for addNoise in addNoises:
        print(f)
        rawData = loadmat(f)

        if f.find("data_alignment") != -1:
            CSIa1Orig = rawData['csi'][:, 0]
        elif f.find("csi") != -1:
            CSIa1Orig = rawData['testdata'][:, 0]
        else:
            CSIa1Orig = rawData['A'][:, 0]

        # 扩展数据
        CSIa1Orig = np.tile(CSIa1Orig, 5)

        dataLen = len(CSIa1Orig)
        print("dataLen", dataLen)

        segLen = 1
        keyLen = 8 * segLen

        bit_len = int(keyLen / segLen)

        print("segLen", segLen)
        print("keyLen", keyLen / segLen)

        originSum = 0
        correctSum = 0

        originWholeSum = 0
        correctWholeSum = 0

        times = 0

        keys = []
        keys_random = []

        # 至少测试10*bit_len次，保证每次结果均出现
        for staInd in range(0, dataLen):
            endInd = staInd + keyLen
            # print("range:", staInd, endInd)
            if endInd >= len(CSIa1Orig):
                print("too long")
                break

            if times == 100 * 2 ** bit_len:
                print("enough test")
                break

            times += 1

            # CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
            #
            # tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
            #
            # randomMatrix = np.random.uniform(5, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
            #
            # # 均值化
            # tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
            #
            # tmpCSIa1back = copy.deepcopy(tmpCSIa1)
            # tmpCSIa1 = tmpCSIa1 + np.random.uniform(0, 1, keyLen)
            # tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)

            # 最后各自的密钥
            a_list = []

            # tmpCSIa1Ind = np.array(tmpCSIa1)

            keyBin = np.random.randint(0, 2, keyLen)

            # tmpCSIa1IndPerm = circulant(tmpCSIa1Ind[::-1])
            #
            # tmpCSIa1IndPerm = normalize(tmpCSIa1IndPerm)
            #
            # # private matrix equilibration via svd
            # # 对普通的LS效果特别好
            # # 奇异值平滑
            # if isBalanced:
            #     try:
            #         U, S, Vt = np.linalg.svd(tmpCSIa1IndPerm)
            #     except np.linalg.LinAlgError:
            #         print("SVD Error")
            #         times -= 1
            #         continue
            #     S = medfilt(S, kernel_size=7)
            #     D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
            #     tmpCSIa1IndPerm = U @ D @ Vt
            #
            # tmpMulA1 = np.dot(tmpCSIa1IndPerm, keyBin)
            #
            # lambda_ = 0
            # solvers = [cp.SCS, cp.GUROBI, cp.OSQP, cp.CVXOPT, cp.ECOS, cp.SCIP, cp.PROXQP, cp.MOSEK,
            #            cp.CLARABEL, cp.NAG, cp.XPRESS]
            # solver = solvers[2]
            # x = cp.Variable(len(keyBin))
            # # 加正则项效果差
            # f_norm = np.linalg.norm(tmpCSIa1IndPerm, ord='fro')
            # obj = cp.Minimize(cp.sum_squares(tmpCSIa1IndPerm @ x - tmpMulA1) + lambda_ * cp.sum_squares(x))
            # prob = cp.Problem(obj, [x >= 0, x <= 1])
            # if solver == cp.SCS:
            #     prob.solve(solver=solver, max_iters=5000)
            # elif solver == cp.SCIP:
            #     prob.solve(solver=solver, scip_params={"limits/time": 10})
            # elif solver == cp.OSQP:
            #     prob.solve(solver=solver, scaling=False)
            # else:
            #     prob.solve(solver=solver)
            # a_list_number = [i.value for i in x]
            #
            # a_list_number = np.array([round(abs(i)) for i in a_list_number])
            #
            # keys.append("".join(map(str, a_list_number)))
            keys_random.append("".join(map(str, keyBin)))

        print(100 * 2 ** bit_len, times)

        # distribution = frequency(keys)
        distribution_random = frequency(keys_random)
        # print("minEntropy", minEntropy(distribution) / bit_len, "bit_len", bit_len, "keyLen", len(keys))
        print("minEntropy", minEntropy(distribution_random) / bit_len, "bit_len", bit_len, "keyLen", len(keys_random))
        print("\n")
