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
# import matlab.engine

from pyldpc import make_ldpc, encode, decode, get_message

from tls import tls, stls, block_toeplitz, tls2, stls_qp, block_circulant


def quartet_to_bytes(quartet_array):
    byte_array = bytearray()
    current_byte = 0
    bits_written = 0

    for quartet in quartet_array:
        current_byte = (current_byte << 2) | quartet
        bits_written += 2

        if bits_written == 8:
            byte_array.append(current_byte)
            current_byte = 0
            bits_written = 0

    if bits_written > 0:
        # 在最后一个字节上添加零位，以确保字节数组的长度是整数倍
        current_byte <<= 8 - bits_written
        byte_array.append(current_byte)

    return byte_array


# 将字节数组转换为四进制的数组
def bytes_to_quartet(byte_array):
    quartet_array = []
    current_byte = 0
    bits_read = 0

    for byte in byte_array:
        current_byte = (current_byte << 8) | byte
        bits_read += 8

        while bits_read >= 2:
            bits_read -= 2
            quartet = (current_byte >> bits_read) & 0b11
            quartet_array.append(quartet)

    return quartet_array


def binary_to_bytes(binary_array):
    byte_array = bytearray()
    for i in range(0, len(binary_array), 8):
        byte = binary_array[i:i + 8]
        byte_value = int(''.join(map(str, byte)), 2)
        byte_array.append(byte_value)
    return byte_array


def bytes_to_binary(byte_array):
    binary_array = []
    for byte_value in byte_array:
        byte_binary = bin(byte_value)[2:].zfill(8)  # 将字节值转换为二进制字符串并补零
        binary_array.extend(map(int, byte_binary))
    return binary_array


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

# fileName = ["../data/data_static_outdoor_1.mat"]

# fileName = ["../data/data_mobile_indoor_1.mat",
#             "../data/data_mobile_outdoor_1.mat",
#             "../data/data_static_outdoor_1.mat",
#             "../data/data_static_indoor_1.mat"
#             ]

fileName = ["../csi/csi_mobile_indoor_1_r",
            "../csi/csi_mobile_outdoor_r",
            "../csi/csi_static_indoor_1_r",
            "../csi/csi_static_outdoor_r"]

# fileName = ["../data/data_NLOS.mat"]

# 是否排序
withoutSorts = [True, False]
withoutSorts = [True]
# 是否添加噪声
addNoises = ["mul"]

bits = 2

solve_time = []
solve_time_com = []

process = ""

# 可选的方法
solutions = ["stls", "stls2", "tls", "ls", "nnls", "ils", "gurobi_opt", "cvxpy_ip", "cvxpy_perturbed_ip",
             "cvxpy_perturbed_ls", "gd", "pulp_ip", "pulp_perturbed_ls", "scipy_regularized_perturbed_ls",
             "matrix_inv", "sils", "Tichonov_reg", "truncation_reg", "leastsq", "mmse"]

solution = "cvxpy_perturbed_ls"
print("Used solution:", solution)

isComplex = True
isEncoded = True
isGrouped = True
isBalanced = False

for f in fileName:
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
                    tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
                    tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)
                    tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix)
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
                if isEncoded and bits == 2 and keyLen == 256 and solution != "stls" and solution != "stls2":
                    if isGrouped == False:
                        bch = bchlib.BCH(19, m=9)
                        max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        # 一部分是随机的密钥，一部分是ECC
                        real_key_dec = np.random.randint(0, 2, max_data_len * 8, dtype=np.int8)
                        real_key = binary_to_bytes(real_key_dec)
                        ecc = bch.encode(real_key)

                        real_key_quartet = bytes_to_binary(real_key)
                        # print("before", real_key_quartet)
                        packet = real_key + ecc
                        keyBin = bytes_to_binary(packet)
                    else:
                        # in groups
                        # two groups
                        # block_size = 128
                        # bch = bchlib.BCH(19, m=8)
                        # max_data_len = bch.n // 8 - (bch.ecc_bits + 14) // 8
                        # bch = bchlib.BCH(17, m=8)
                        # four groups
                        # block_size = 64
                        # bch = bchlib.BCH(9, m=7)
                        # eight groups
                        block_size = 32
                        bch = bchlib.BCH(6, m=6)
                        max_data_len = bch.n // 8 - bch.ecc_bits // 8
                        # sixteen groups
                        # block_size = 16
                        # bch = bchlib.BCH(5, m=5)
                        # max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        keyBin = []
                        real_key_quartet = []
                        this_encode_time = 0
                        for i in range(int(keyLen / block_size)):
                            real_key_dec = np.random.randint(0, 2, max_data_len * 8, dtype=np.int8)
                            real_key = binary_to_bytes(real_key_dec)
                            ecc = bch.encode(real_key)

                            real_key_quartet.extend(bytes_to_binary(real_key))
                            packet = real_key + ecc
                            keyBin.extend(bytes_to_binary(packet))

                    # print("before", len(keyBin), keyBin)
                    basic = [0, 1]
                elif isEncoded and bits == 2 and keyLen == 256 and (solution == "stls" or solution == "stls2"):
                    if isGrouped == False:
                        bch = bchlib.BCH(9, m=7)
                        max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        # 一部分是随机的密钥，一部分是ECC
                        real_key_dec = np.random.randint(0, 2, max_data_len * 8, dtype=np.int8)
                        real_key = binary_to_bytes(real_key_dec)
                        ecc = bch.encode(real_key)

                        real_key_quartet = bytes_to_binary(real_key)
                        # print("before", real_key_quartet)
                        packet = real_key + ecc
                        keyBin = bytes_to_binary(packet)
                    else:
                        # in groups
                        # two groups
                        # block_size = 32
                        # bch = bchlib.BCH(9, m=6)
                        # four groups
                        # block_size = 16
                        # bch = bchlib.BCH(5, m=5)
                        # max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        # eight groups
                        block_size = 32
                        bch = bchlib.BCH(6, m=6)
                        max_data_len = bch.n // 8 - bch.ecc_bits // 8
                        keyBin = []
                        real_key_quartet = []
                        this_encode_time = 0
                        for i in range(int(keyLen / 4 / block_size)):
                            real_key_dec = np.random.randint(0, 2, max_data_len * 8, dtype=np.int8)
                            real_key = binary_to_bytes(real_key_dec)
                            ecc = bch.encode(real_key)

                            real_key_quartet.extend(bytes_to_binary(real_key))
                            packet = real_key + ecc
                            keyBin.extend(bytes_to_binary(packet))

                    # print("before", len(keyBin), keyBin)
                    basic = [0, 1]
                elif isEncoded and bits == 2 and keyLen == 128 and solution != "stls" and solution != "stls2":
                    if isGrouped == False:
                        bch = bchlib.BCH(17, m=8)
                        max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        # 一部分是随机的密钥，一部分是ECC
                        real_key_dec = np.random.randint(0, 2, max_data_len * 8, dtype=np.int8)
                        real_key = binary_to_bytes(real_key_dec)
                        ecc = bch.encode(real_key)

                        real_key_quartet = bytes_to_binary(real_key)
                        # print("before", real_key_quartet)
                        packet = real_key + ecc
                        keyBin = bytes_to_binary(packet)
                    else:
                        # in groups
                        # two groups
                        block_size = 64
                        bch = bchlib.BCH(9, m=7)
                        # four groups
                        # block_size = 32
                        # bch = bchlib.BCH(7, m=6)
                        max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        keyBin = []
                        real_key_quartet = []
                        for i in range(int(keyLen / block_size)):
                            real_key_dec = np.random.randint(0, 2, max_data_len * 8, dtype=np.int8)
                            real_key = binary_to_bytes(real_key_dec)
                            ecc = bch.encode(real_key)

                            real_key_quartet.extend(bytes_to_binary(real_key))
                            packet = real_key + ecc
                            keyBin.extend(bytes_to_binary(packet))

                    # print("before", len(keyBin), keyBin)
                    basic = [0, 1]
                elif isEncoded and bits == 2 and keyLen == 64 and solution != "stls" and solution != "stls2":
                    if isGrouped == False:
                        bch = bchlib.BCH(9, m=7)
                        max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        # 一部分是随机的密钥，一部分是ECC
                        real_key_dec = np.random.randint(0, 2, max_data_len * 8, dtype=np.int8)
                        real_key = binary_to_bytes(real_key_dec)
                        ecc = bch.encode(real_key)

                        real_key_quartet = bytes_to_binary(real_key)
                        # print("before", real_key_quartet)
                        packet = real_key + ecc
                        keyBin = bytes_to_binary(packet)
                    else:
                        # in groups
                        # two groups
                        block_size = 32
                        bch = bchlib.BCH(9, m=6)
                        # four groups
                        # block_size = 16
                        # bch = bchlib.BCH(5, m=5)
                        max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        keyBin = []
                        real_key_quartet = []
                        for i in range(int(keyLen / block_size)):
                            real_key_dec = np.random.randint(0, 2, max_data_len * 8, dtype=np.int8)
                            real_key = binary_to_bytes(real_key_dec)
                            ecc = bch.encode(real_key)

                            real_key_quartet.extend(bytes_to_binary(real_key))
                            packet = real_key + ecc
                            keyBin.extend(bytes_to_binary(packet))

                    # print("before", len(keyBin), keyBin)
                    basic = [0, 1]
                elif isComplex:
                    if bits == 2:
                        basic = [0, 1]
                        keyBin = np.random.randint(0, 2, keyLen) + 1j * np.random.randint(0, 2, keyLen)
                    elif bits == 4:
                        basic = [0, 1, 2, 3]
                        keyBin = np.random.randint(0, 4, keyLen) + 1j * np.random.randint(0, 4, keyLen)
                elif bits == 1:
                    keyBin = np.random.binomial(1, 0.5, keyLen)
                    basic = [0, 1]
                elif bits == 2:
                    basic = [0, 1, 2, 3]
                    keyBin = np.random.choice(basic, keyLen)
                else:
                    raise Exception("bits error")

                if solution == "stls" or solution == "stls2":
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

                # tmpCSIa1IndPerm = tmpCSIa1IndPerm - np.mean(tmpCSIa1IndPerm)
                # tmpCSIb1IndPerm = tmpCSIb1IndPerm - np.mean(tmpCSIb1IndPerm)
                # tmpCSIe1IndPerm = tmpCSIe1IndPerm - np.mean(tmpCSIe1IndPerm)

                # private matrix equilibration via svd
                # 对普通的LS效果特别好
                if isComplex and solution != "stls" and solution != "stls2":
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
                        U, S, Vt = np.linalg.svd(tmpCSIa1IndPerm)
                        S = medfilt(S, kernel_size=7)
                        D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
                        tmpCSIa1IndPerm = U @ D @ Vt
                        U, S, Vt = np.linalg.svd(tmpCSIb1IndPerm)
                        S = medfilt(S, kernel_size=7)
                        D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
                        tmpCSIb1IndPerm = U @ D @ Vt
                        U, S, Vt = np.linalg.svd(tmpCSIe1IndPerm)
                        S = medfilt(S, kernel_size=7)
                        D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
                        tmpCSIe1IndPerm = U @ D @ Vt
                elif solution != "stls" and solution != "stls2":
                    # 奇异值平滑
                    if isBalanced:
                        U, S, Vt = np.linalg.svd(tmpCSIa1IndPerm)
                        D = np.diag(1 / np.sqrt(S))
                        tmpCSIa1IndPerm = U @ D @ np.diag(S) @ D @ Vt
                        U, S, Vt = np.linalg.svd(tmpCSIb1IndPerm)
                        D = np.diag(1 / np.sqrt(S))
                        tmpCSIb1IndPerm = U @ D @ np.diag(S) @ D @ Vt
                        U, S, Vt = np.linalg.svd(tmpCSIe1IndPerm)
                        D = np.diag(1 / np.sqrt(S))
                        tmpCSIe1IndPerm = U @ D @ np.diag(S) @ D @ Vt

                if np.iscomplex(keyBin).any() and isEncoded == False and solution != "stls" and solution != "stls2":
                    # 转化成实数的凸优化
                    y = np.real(keyBin)
                    z = np.imag(keyBin)
                    BigX = np.array([y, z]).reshape(2 * keyLen)
                    tmpMulA1 = np.dot(tmpCSIa1IndPerm, BigX)
                elif solution != "stls" and solution != "stls2":
                    np.random.seed(0)
                    # tmpMulA1 = np.dot(tmpCSIa1IndPerm, keyBin + np.random.normal(0, 0.1, keyLen))
                    tmpMulA1 = np.dot(tmpCSIa1IndPerm, keyBin)

                if solution == "leastsq":
                    a_list_number = list(np.linalg.lstsq(tmpCSIa1IndPerm, tmpMulA1, rcond=None)[0])
                    b_list_number = list(np.linalg.lstsq(tmpCSIb1IndPerm, tmpMulA1, rcond=None)[0])
                    e_list_number = list(np.linalg.lstsq(tmpCSIe1IndPerm, tmpMulA1, rcond=None)[0])
                elif solution == "matrix_inv":
                    # 矩阵求逆
                    a_list_number = np.matmul(np.linalg.inv(tmpCSIa1IndPerm), tmpMulA1)
                    b_list_number = np.matmul(np.linalg.inv(tmpCSIb1IndPerm), tmpMulA1)
                    e_list_number = np.matmul(np.linalg.inv(tmpCSIe1IndPerm), tmpMulA1)
                elif solution == "nnls":
                    # non negative least square
                    a_list_number = nnls(tmpCSIa1IndPerm, tmpMulA1)[0]
                    b_list_number = nnls(tmpCSIb1IndPerm, tmpMulA1)[0]
                    e_list_number = nnls(tmpCSIe1IndPerm, tmpMulA1)[0]
                elif solution == "ls":
                    # least square
                    def residuals(x, tmpMulA1, tmpCSIx1IndPerm):
                        return tmpMulA1 - np.dot(tmpCSIx1IndPerm, x)


                    a_list_number = leastsq(residuals, np.random.binomial(1, 0.5, len(tmpMulA1)),
                                            args=(tmpMulA1, tmpCSIa1IndPerm))[0]
                    b_list_number = leastsq(residuals, np.random.binomial(1, 0.5, len(tmpMulA1)),
                                            args=(tmpMulA1, tmpCSIb1IndPerm))[0]
                    e_list_number = leastsq(residuals, np.random.binomial(1, 0.5, len(tmpMulA1)),
                                            args=(tmpMulA1, tmpCSIe1IndPerm))[0]
                elif solution == "mmse":
                    # y = h * x + z
                    h_a = tmpCSIa1IndPerm
                    h_b = tmpCSIb1IndPerm
                    h_e = tmpCSIe1IndPerm
                    h_a_H = np.conj(np.transpose(h_a))
                    h_b_H = np.conj(np.transpose(h_b))
                    h_e_H = np.conj(np.transpose(h_e))

                    noise_signal_var = 0.1  # 越大越好，但也不安全
                    # noise_signal_var = np.abs(np.var(h_b) - np.var(h_a)) / np.var(h_a)
                    a_list_number = h_a_H @ np.linalg.pinv(h_a @ h_a_H + noise_signal_var * np.eye(len(tmpMulA1))) @ tmpMulA1
                    b_list_number = h_b_H @ np.linalg.pinv(h_b @ h_b_H + noise_signal_var * np.eye(len(tmpMulA1))) @ tmpMulA1
                    e_list_number = h_e_H @ np.linalg.pinv(h_e @ h_e_H + noise_signal_var * np.eye(len(tmpMulA1))) @ tmpMulA1
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

                    tmpCSIa1Blocks = (np.hstack((tmpCSIa1IndPerm, np.zeros(keyLen)))
                                      .reshape(block_number * 2, block_shape, block_shape))
                    tmpCSIb1Blocks = (np.hstack((tmpCSIb1IndPerm, np.zeros(keyLen)))
                                      .reshape(block_number * 2, block_shape, block_shape))
                    tmpCSIe1Blocks = (np.hstack((tmpCSIe1IndPerm, np.zeros(keyLen)))
                                      .reshape(block_number * 2, block_shape, block_shape))

                    tmpCSIa1Blocks = normalize(tmpCSIa1Blocks)
                    tmpCSIb1Blocks = normalize(tmpCSIb1Blocks)
                    tmpCSIb1Blocks = normalize(tmpCSIb1Blocks)

                    tmpCSIa1IndPerm = normalize(tmpCSIa1Circulant)
                    tmpCSIb1IndPerm = normalize(tmpCSIb1Circulant)
                    tmpCSIe1IndPerm = normalize(tmpCSIe1Circulant)

                    if isComplex:
                        blockLen = block_number * block_shape
                        tmpCSIa1IndPerm = np.hstack((np.vstack((tmpCSIa1IndPerm, np.zeros((blockLen, blockLen)))),
                                                     np.vstack((np.zeros((blockLen, blockLen)), tmpCSIa1IndPerm))))
                        tmpCSIb1IndPerm = np.hstack((np.vstack((tmpCSIb1IndPerm, np.zeros((blockLen, blockLen)))),
                                                     np.vstack((np.zeros((blockLen, blockLen)), tmpCSIb1IndPerm))))
                        tmpCSIe1IndPerm = np.hstack((np.vstack((tmpCSIe1IndPerm, np.zeros((blockLen, blockLen)))),
                                                     np.vstack((np.zeros((blockLen, blockLen)), tmpCSIe1IndPerm))))

                        # 奇异值平滑
                        if isBalanced:
                            U, S, Vt = np.linalg.svd(tmpCSIa1IndPerm)
                            S = medfilt(S, kernel_size=7)
                            D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
                            tmpCSIa1IndPerm = U @ D @ Vt
                            U, S, Vt = np.linalg.svd(tmpCSIb1IndPerm)
                            S = medfilt(S, kernel_size=7)
                            D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
                            tmpCSIb1IndPerm = U @ D @ Vt
                            U, S, Vt = np.linalg.svd(tmpCSIe1IndPerm)
                            S = medfilt(S, kernel_size=7)
                            D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
                            tmpCSIe1IndPerm = U @ D @ Vt
                    else:
                        # 奇异值平滑
                        if isBalanced:
                            U, S, Vt = np.linalg.svd(tmpCSIa1IndPerm)
                            D = np.diag(1 / np.sqrt(S))
                            tmpCSIa1IndPerm = U @ D @ np.diag(S) @ D @ Vt
                            U, S, Vt = np.linalg.svd(tmpCSIb1IndPerm)
                            D = np.diag(1 / np.sqrt(S))
                            tmpCSIb1IndPerm = U @ D @ np.diag(S) @ D @ Vt
                            U, S, Vt = np.linalg.svd(tmpCSIe1IndPerm)
                            D = np.diag(1 / np.sqrt(S))
                            tmpCSIe1IndPerm = U @ D @ np.diag(S) @ D @ Vt

                    if np.iscomplex(keyBin).any() and isEncoded == False:
                        # 转化成实数的凸优化
                        y = np.real(keyBin)
                        z = np.imag(keyBin)
                        BigX = np.array([y, z]).reshape(2 * keyLen)
                        tmpMulA1 = np.dot(tmpCSIa1IndPerm, BigX)
                    else:
                        np.random.seed(0)
                        # tmpMulA1 = np.dot(tmpCSIa1IndPerm, keyBin + np.random.normal(0, 0.1, keyLen))
                        tmpMulA1 = np.dot(tmpCSIa1IndPerm, keyBin)

                    # np.random.seed(0)
                    # if bits == 1:
                    #     keyBin = np.random.binomial(1, 0.5, block_number * block_shape)
                    # else:
                    #     keyBin = np.random.randint(0, 4, block_number * block_shape)

                    # tmpMulA1 = np.dot(tmpCSIa1Circulant, keyBin)
                    # 考虑complex情况用0填充，则长度扩大一倍
                    tmpMulA1 = tmpMulA1.reshape(block_number * 2, block_shape)
                    a_list_number = stls(tmpCSIa1Blocks, tmpMulA1)
                    b_list_number = stls(tmpCSIb1Blocks, tmpMulA1)
                    e_list_number = stls(tmpCSIe1Blocks, tmpMulA1)
                    # a_list_number = stls_qp(tmpCSIa1Blocks, tmpMulA1)
                    # b_list_number = stls_qp(tmpCSIb1Blocks, tmpMulA1)
                    # e_list_number = stls_qp(tmpCSIe1Blocks, tmpMulA1)

                    if isEncoded and bits == 2:
                        a_list_number = np.array([round(abs(i)) % 2 for i in a_list_number])
                        if isGrouped == False:
                            # 取出data部分
                            a_data_quartet = a_list_number[:max_data_len * 8]
                            a_data = binary_to_bytes(a_data_quartet)

                            a_ecc_quartet = a_list_number[max_data_len * 8:]
                            a_ecc = binary_to_bytes(a_ecc_quartet)
                            bch.decode(a_data, a_ecc)
                            bch.correct(a_data, a_ecc)
                            a_list_number = bytes_to_binary(a_data)
                            a_list_number.extend(bytes_to_binary(a_ecc))
                        else:
                            # in groups
                            a_decoded = []
                            for i in range(int(keyLen / 4 / block_size)):
                                a_list_number_tmp = a_list_number[i * block_size * 2:(i + 1) * block_size * 2]
                                a_data_quartet = a_list_number_tmp[:max_data_len * 8]
                                a_data = binary_to_bytes(a_data_quartet)

                                a_ecc_quartet = a_list_number_tmp[max_data_len * 8:]
                                a_ecc = binary_to_bytes(a_ecc_quartet)
                                bch.decode(a_data, a_ecc)
                                bch.correct(a_data, a_ecc)
                                a_decoded.extend(bytes_to_binary(a_data))
                                a_decoded.extend(bytes_to_binary(a_ecc))
                            a_list_number = a_decoded

                    if isEncoded and bits == 2:
                        b_list_number = np.array([round(abs(i)) % 2 for i in b_list_number])
                        if isGrouped == False:
                            # 取出data部分
                            b_data_quartet = b_list_number[:max_data_len * 8]
                            b_data = binary_to_bytes(b_data_quartet)

                            b_ecc_quartet = b_list_number[max_data_len * 8:]
                            b_ecc = binary_to_bytes(b_ecc_quartet)
                            bch.decode(b_data, b_ecc)
                            bch.correct(b_data, b_ecc)
                            b_list_number = bytes_to_binary(b_data)
                            b_list_number.extend(bytes_to_binary(b_ecc))
                        else:
                            # in groups
                            b_decoded = []
                            for i in range(int(keyLen / 4 / block_size)):
                                b_list_number_tmp = b_list_number[i * block_size * 2:(i + 1) * block_size * 2]
                                b_data_quartet = b_list_number_tmp[:max_data_len * 8]
                                b_data = binary_to_bytes(b_data_quartet)

                                b_ecc_quartet = b_list_number_tmp[max_data_len * 8:]
                                b_ecc = binary_to_bytes(b_ecc_quartet)
                                bch.decode(b_data, b_ecc)
                                bch.correct(b_data, b_ecc)
                                b_decoded.extend(bytes_to_binary(b_data))
                                b_decoded.extend(bytes_to_binary(b_ecc))
                            b_list_number = b_decoded

                    if isEncoded and bits == 2:
                        e_list_number = np.array([round(abs(i)) % 2 for i in e_list_number])
                        if isGrouped == False:
                            # 取出data部分
                            e_data_quartet = e_list_number[:max_data_len * 8]
                            e_data = binary_to_bytes(e_data_quartet)

                            e_ecc_quartet = e_list_number[max_data_len * 8:]
                            e_ecc = binary_to_bytes(e_ecc_quartet)
                            bch.decode(e_data, e_ecc)
                            bch.correct(e_data, e_ecc)
                            e_list_number = bytes_to_binary(e_data)
                            e_list_number.extend(bytes_to_binary(e_ecc))
                        else:
                            # in groups
                            e_decoded = []
                            for i in range(int(keyLen / 4 / block_size)):
                                e_list_number_tmp = e_list_number[i * block_size * 2:(i + 1) * block_size * 2]
                                e_data_quartet = e_list_number_tmp[:max_data_len * 8]
                                e_data = binary_to_bytes(e_data_quartet)

                                e_ecc_quartet = e_list_number_tmp[max_data_len * 8:]
                                e_ecc = binary_to_bytes(e_ecc_quartet)
                                bch.decode(e_data, e_ecc)
                                bch.correct(e_data, e_ecc)
                                e_decoded.extend(bytes_to_binary(e_data))
                                e_decoded.extend(bytes_to_binary(e_ecc))
                            e_list_number = e_decoded
                elif solution == "stls2":
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
                    tmpCSIb1Circulant = normalize(tmpCSIb1Circulant)
                    tmpCSIe1Circulant = normalize(tmpCSIe1Circulant)

                    tmpCSIa1Blocks = normalize(tmpCSIa1Blocks)
                    tmpCSIb1Blocks = normalize(tmpCSIb1Blocks)
                    tmpCSIb1Blocks = normalize(tmpCSIb1Blocks)

                    np.random.seed(0)
                    if bits == 1:
                        keyBin = np.random.binomial(1, 0.5, block_number * block_shape)
                    else:
                        keyBin = np.random.randint(0, 2, block_number * block_shape)

                    tmpMulA1 = np.dot(tmpCSIa1Circulant, keyBin)
                    tmpMulA1 = tmpMulA1.reshape(block_number, block_shape)
                    a_list_number = stls(tmpCSIa1Blocks, tmpMulA1)
                    b_list_number = stls(tmpCSIb1Blocks, tmpMulA1)
                    e_list_number = stls(tmpCSIe1Blocks, tmpMulA1)
                    # a_list_number = stls_qp(tmpCSIa1Blocks, tmpMulA1)
                    # b_list_number = stls_qp(tmpCSIb1Blocks, tmpMulA1)
                    # e_list_number = stls_qp(tmpCSIe1Blocks, tmpMulA1)
                elif solution == "tls":
                    # total least square
                    np.random.seed(0)
                    perturbation = np.random.normal(0, np.var(tmpMulA1) / 100, keyLen * 2)
                    a_list_number = tls(tmpCSIa1IndPerm, tmpMulA1 + perturbation)
                    b_list_number = tls(tmpCSIb1IndPerm, tmpMulA1 + perturbation)
                    e_list_number = tls(tmpCSIe1IndPerm, tmpMulA1 + perturbation)

                    # 与最小二乘结果一致？可能是因为当b无噪音扰动时，总体最小二乘退化成最小二乘
                    # a_list_number1 = list(np.linalg.lstsq(tmpCSIa1IndPerm, tmpMulA1, rcond=None)[0])
                    # b_list_number1 = list(np.linalg.lstsq(tmpCSIb1IndPerm, tmpMulA1, rcond=None)[0])
                    # e_list_number1 = list(np.linalg.lstsq(tmpCSIe1IndPerm, tmpMulA1, rcond=None)[0])
                    # a_list_number1 = np.matmul(np.linalg.inv(tmpCSIa1IndPerm), tmpMulA1)
                    # b_list_number1 = np.matmul(np.linalg.inv(tmpCSIb1IndPerm), tmpMulA1)
                    # e_list_number1 = np.matmul(np.linalg.inv(tmpCSIe1IndPerm), tmpMulA1)
                    # print(np.all(np.round(b_list_number) == np.round(b_list_number1)))

                    # a_list_number = tls2(tmpCSIa1IndPerm, tmpMulA1)
                    # b_list_number = tls2(tmpCSIb1IndPerm, tmpMulA1)
                    # e_list_number = tls2(tmpCSIe1IndPerm, tmpMulA1)
                elif solution == "Tichonov_reg":
                    # Tichonov regularization
                    alpha = 50
                    U, S, V, = np.linalg.svd(tmpCSIa1IndPerm)
                    V = V.T
                    S = np.diag(S)
                    beta = (U.T @ tmpMulA1)
                    a_list_number = V @ np.linalg.inv(S.T @ S + alpha * np.eye(keyLen * 2)) @ S.T @ beta

                    U, S, V, = np.linalg.svd(tmpCSIb1IndPerm)
                    V = V.T
                    S = np.diag(S)
                    beta = (U.T @ tmpMulA1)
                    b_list_number = V @ np.linalg.inv(S.T @ S + alpha * np.eye(keyLen * 2)) @ S.T @ beta

                    U, S, V, = np.linalg.svd(tmpCSIe1IndPerm)
                    V = V.T
                    S = np.diag(S)
                    beta = (U.T @ tmpMulA1)
                    e_list_number = V @ np.linalg.inv(S.T @ S + alpha * np.eye(keyLen * 2)) @ S.T @ beta

                    a_list_number = normalize(a_list_number)
                    b_list_number = normalize(b_list_number)
                    e_list_number = normalize(e_list_number)

                elif solution == "cvxpy_perturbed_ls":
                    # cvxpy perturbed least squares
                    # SCIP最慢，其他结果一致
                    # SCIP和PROXQP过慢
                    lambda_ = 0
                    solvers = [cp.SCS, cp.GUROBI, cp.OSQP, cp.CVXOPT, cp.ECOS, cp.SCIP, cp.PROXQP, cp.MOSEK,
                               cp.CLARABEL, cp.NAG, cp.XPRESS]
                    # 使用不同的求解器
                    solver = solvers[3]
                    if isComplex and isEncoded == False:
                        # x = cp.Variable(len(keyBin), complex=True)
                        x = cp.Variable(len(keyBin) * 2)
                    else:
                        x = cp.Variable(len(keyBin))
                    # 加正则项效果差
                    f_norm = np.linalg.norm(tmpCSIa1IndPerm, ord='fro')
                    obj = cp.Minimize(cp.sum_squares(tmpCSIa1IndPerm @ x - tmpMulA1) + lambda_ * cp.sum_squares(x))
                    prob = cp.Problem(obj, [x >= min(basic), x <= max(basic)])
                    solve_start = time.time_ns() / 10 ** 6
                    if solver == cp.SCS:
                        prob.solve(solver=solver, max_iters=5000)
                    elif solver == cp.SCIP:
                        prob.solve(solver=solver, scip_params={"limits/time": 10})
                    elif solver == cp.OSQP:
                        # scaling=False表示禁用缩放以矩阵均衡
                        prob.solve(solver=solver, scaling=False)
                        # prob.solve(solver=solver, scaling=False, warm_start=True)
                    else:
                        prob.solve(solver=solver)
                    solve_time_com.append(time.time_ns() / 10 ** 6 - solve_start)
                    # print("num_iters of a: ", prob.solver_stats.num_iters)
                    a_list_number = [i.value for i in x]
                    # 用time.time_ns()计算出来的时间偏高
                    solve_time.append(prob.solver_stats.solve_time)

                    if isComplex and isEncoded == False:
                        # x = cp.Variable(len(keyBin), complex=True)
                        x = cp.Variable(len(keyBin) * 2)
                    else:
                        x = cp.Variable(len(keyBin))
                    obj = cp.Minimize(cp.sum_squares(tmpCSIb1IndPerm @ x - tmpMulA1) + lambda_ * cp.sum_squares(x))
                    prob = cp.Problem(obj, [x >= min(basic), x <= max(basic)])
                    solve_start = time.time_ns() / 10 ** 6
                    if solver == cp.SCS:
                        prob.solve(solver=solver, max_iters=5000)
                    elif solver == cp.SCIP:
                        prob.solve(solver=solver, scip_params={"limits/time": 10})
                    elif solver == cp.OSQP:
                        prob.solve(solver=solver, scaling=False)
                        # prob.solve(solver=solver, scaling=False, warm_start=True)
                    else:
                        prob.solve(solver=solver)
                    solve_time_com.append(time.time_ns() / 10 ** 6 - solve_start)
                    b_list_number = [i.value for i in x]
                    solve_time.append(prob.solver_stats.solve_time)

                    if isComplex and isEncoded == False:
                        # x = cp.Variable(len(keyBin), complex=True)
                        x = cp.Variable(len(keyBin) * 2)
                    else:
                        x = cp.Variable(len(keyBin))
                    obj = cp.Minimize(cp.sum_squares(tmpCSIe1IndPerm @ x - tmpMulA1) + lambda_ * cp.sum_squares(x))
                    prob = cp.Problem(obj, [x >= min(basic), x <= max(basic)])
                    if solver == cp.SCS:
                        prob.solve(solver=solver, max_iters=5000)
                    elif solver == cp.SCIP:
                        prob.solve(solver=solver, scip_params={"limits/time": 10})
                    elif solver == cp.OSQP:
                        prob.solve(solver=solver, scaling=False)
                        # prob.solve(solver=solver, scaling=False, warm_start=True)
                    else:
                        prob.solve(solver=solver)
                    e_list_number = [i.value for i in x]
                else:
                    raise Exception("incorrect solution!")

                if isEncoded and bits == 2 and solution != "stls" and solution != "stls2":
                    a_list_number = np.array([round(abs(i)) % 2 for i in a_list_number])
                    if isGrouped == False:
                        # 取出data部分
                        a_data_quartet = a_list_number[:max_data_len * 8]
                        a_data = binary_to_bytes(a_data_quartet)

                        a_ecc_quartet = a_list_number[max_data_len * 8:]
                        a_ecc = binary_to_bytes(a_ecc_quartet)
                        bch.decode(a_data, a_ecc)
                        bch.correct(a_data, a_ecc)
                        a_list_number = bytes_to_binary(a_data)
                        a_list_number.extend(bytes_to_binary(a_ecc))
                    else:
                        # in groups
                        a_decoded = []
                        for i in range(int(keyLen / block_size)):
                            a_list_number_tmp = a_list_number[i * block_size * 2:(i + 1) * block_size * 2]
                            a_data_quartet = a_list_number_tmp[:max_data_len * 8]
                            a_data = binary_to_bytes(a_data_quartet)

                            a_ecc_quartet = a_list_number_tmp[max_data_len * 8:]
                            a_ecc = binary_to_bytes(a_ecc_quartet)
                            bch.decode(a_data, a_ecc)
                            bch.correct(a_data, a_ecc)
                            a_decoded.extend(bytes_to_binary(a_data))
                            a_decoded.extend(bytes_to_binary(a_ecc))
                        a_list_number = a_decoded

                if isEncoded and bits == 2 and solution != "stls" and solution != "stls2":
                    b_list_number = np.array([round(abs(i)) % 2 for i in b_list_number])
                    if isGrouped == False:
                        # 取出data部分
                        b_data_quartet = b_list_number[:max_data_len * 8]
                        b_data = binary_to_bytes(b_data_quartet)

                        b_ecc_quartet = b_list_number[max_data_len * 8:]
                        b_ecc = binary_to_bytes(b_ecc_quartet)
                        bch.decode(b_data, b_ecc)
                        bch.correct(b_data, b_ecc)
                        b_list_number = bytes_to_binary(b_data)
                        b_list_number.extend(bytes_to_binary(b_ecc))
                    else:
                        # in groups
                        b_decoded = []
                        for i in range(int(keyLen / block_size)):
                            b_list_number_tmp = b_list_number[i * block_size * 2:(i + 1) * block_size * 2]
                            b_data_quartet = b_list_number_tmp[:max_data_len * 8]
                            b_data = binary_to_bytes(b_data_quartet)

                            b_ecc_quartet = b_list_number_tmp[max_data_len * 8:]
                            b_ecc = binary_to_bytes(b_ecc_quartet)
                            bch.decode(b_data, b_ecc)
                            bch.correct(b_data, b_ecc)
                            b_decoded.extend(bytes_to_binary(b_data))
                            b_decoded.extend(bytes_to_binary(b_ecc))
                        b_list_number = b_decoded

                if isEncoded and bits == 2 and solution != "stls" and solution != "stls2":
                    e_list_number = np.array([round(abs(i)) % 2 for i in e_list_number])
                    if isGrouped == False:
                        # 取出data部分
                        e_data_quartet = e_list_number[:max_data_len * 8]
                        e_data = binary_to_bytes(e_data_quartet)

                        e_ecc_quartet = e_list_number[max_data_len * 8:]
                        e_ecc = binary_to_bytes(e_ecc_quartet)
                        bch.decode(e_data, e_ecc)
                        bch.correct(e_data, e_ecc)
                        e_list_number = bytes_to_binary(e_data)
                        e_list_number.extend(bytes_to_binary(e_ecc))
                    else:
                        # in groups
                        e_decoded = []
                        for i in range(int(keyLen / block_size)):
                            e_list_number_tmp = e_list_number[i * block_size * 2:(i + 1) * block_size * 2]
                            e_data_quartet = e_list_number_tmp[:max_data_len * 8]
                            e_data = binary_to_bytes(e_data_quartet)

                            e_ecc_quartet = e_list_number_tmp[max_data_len * 8:]
                            e_ecc = binary_to_bytes(e_ecc_quartet)
                            bch.decode(e_data, e_ecc)
                            bch.correct(e_data, e_ecc)
                            e_decoded.extend(bytes_to_binary(e_data))
                            e_decoded.extend(bytes_to_binary(e_ecc))
                        e_list_number = e_decoded

                a_list_number = np.array([round(abs(i)) % 2 for i in a_list_number])
                b_list_number = np.array([round(abs(i)) % 2 for i in b_list_number])
                e_list_number = np.array([round(abs(i)) % 2 for i in e_list_number])

                # 转成二进制，0填充成0000
                sum1 = min(len(a_list_number), len(b_list_number))
                sum2 = 0
                sum3 = 0
                for i in range(0, sum1):
                    sum2 += (a_list_number[i] == b_list_number[i])
                for i in range(min(len(a_list_number), len(e_list_number))):
                    sum3 += (a_list_number[i] == e_list_number[i])

                originSum += sum1
                correctSum += sum2
                randomSum += sum3

                originWholeSum += 1
                correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
                randomWholeSum = randomWholeSum + 1 if sum3 == sum1 else randomWholeSum

            print("\033[0;34;40ma-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 9), "\033[0m")
            print("\033[0;34;40ma-e all", randomSum, "/", originSum, "=", round(randomSum / originSum, 9),
                  "\033[0m")
            print("\033[0;34;40ma-b whole match", correctWholeSum, "/", originWholeSum, "=",
                  round(correctWholeSum / originWholeSum, 9), "\033[0m")
            print("\033[0;34;40ma-e whole match", randomWholeSum, "/", originWholeSum, "=",
                  round(randomWholeSum / originWholeSum, 9), "\033[0m")

            print(round(correctSum / originSum, 9), round(correctWholeSum / originWholeSum, 9),
                  round(originSum / times / keyLen, 9),
                  round(correctSum / times / keyLen, 9))
            if withoutSort:
                print("withoutSort")
            else:
                print("withSort")
            if isGrouped:
                print("Grouped", block_size)
            else:
                print("Not Grouped")
            if isBalanced:
                print("withBalance")
            else:
                print("withoutBalance")
            # print(solver)
            print("\n")
if len(solve_time) != 0:
    print("solve_time", np.round(max(solve_time), 5), np.round(min(solve_time), 5), np.round(np.mean(solve_time), 5))
if len(solve_time_com) != 0:
    print("solve_time", np.round(max(solve_time_com), 5), np.round(min(solve_time_com), 5), np.round(np.mean(solve_time_com), 5))

if fileName[0].find("csi") != -1:
    # savemat(str(solver) + '_CSI.mat', {"time": np.array(solve_time_com).T})
    savemat(str(solver) + '_CSI.mat', {"time": np.array(solve_time).T})
else:
    # savemat(str(solver) + '_RSS.mat', {"time": np.array(solve_time_com).T})
    savemat(str(solver) + '_RSS.mat', {"time": np.array(solve_time).T})

messagebox.showinfo("提示", "测试结束")

