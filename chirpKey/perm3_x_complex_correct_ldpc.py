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

# fileName = ["../csi/csi_static_indoor_1_r"]

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

process = ""

# 可选的方法
solutions = ["stls", "ls", "nnls", "ils", "gurobi_opt", "cvxpy_ip", "cvxpy_perturbed_ip",
             "cvxpy_perturbed_ls", "gd", "pulp_ip", "pulp_perturbed_ls", "scipy_regularized_perturbed_ls",
             "matrix_inv", "sils", "Tichonov_reg", "truncation_reg", "leastsq", "mmse"]

solution = "cvxpy_perturbed_ls"
print("Used solution:", solution)
error_correct = True

all_iterations = []
perturb_time = []
balance_time = []
encode_time = []
decode_time = []
solve_time = []
correct_time_of_A = []
correct_time_of_B = []

ldpc_time_of_A = []
ldpc_time_of_B = []

isComplex = True
isEncoded = True
isGrouped = True
isBalanced = True

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
            keyLen = 128 * segLen

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
                if isEncoded and bits == 4 and keyLen == 256:
                    bch = bchlib.BCH(22, m=10)
                    max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                    # 一部分是随机的密钥，一部分是ECC
                    real_key_dec = np.random.randint(0, 4, max_data_len * 4, dtype=np.int8)
                    real_key = quartet_to_bytes(real_key_dec)
                    encode_start = time.time_ns() / 10 ** 6
                    ecc = bch.encode(real_key)
                    encode_time.append(time.time_ns() / 10 ** 6 - encode_start)

                    real_key_quartet = bytes_to_quartet(real_key)
                    # print("before", real_key_quartet)
                    packet = real_key + ecc
                    keyBin = bytes_to_quartet(packet)
                    # 依然编码成二进制串
                    basic = [0, 1, 2, 3]
                elif isEncoded and bits == 2 and keyLen == 512:
                    if isGrouped == False:
                        bch = bchlib.BCH(22, m=10)
                        max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        # 一部分是随机的密钥，一部分是ECC
                        real_key_dec = np.random.randint(0, 2, max_data_len * 8, dtype=np.int8)
                        real_key = binary_to_bytes(real_key_dec)
                        encode_start = time.time_ns() / 10 ** 6
                        ecc = bch.encode(real_key)
                        encode_time.append(time.time_ns() / 10 ** 6 - encode_start)

                        real_key_quartet = bytes_to_binary(real_key)
                        # print("before", real_key_quartet)
                        packet = real_key + ecc
                        keyBin = bytes_to_binary(packet)
                    else:
                        # in groups
                        # two groups
                        block_size = 256
                        bch = bchlib.BCH(19, m=9)
                        # four groups
                        # block_size = 128
                        # bch = bchlib.BCH(17, m=8)
                        # eight groups
                        # block_size = 32
                        # bch = bchlib.BCH(7, m=6)
                        max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        keyBin = []
                        real_key_quartet = []
                        for i in range(int(keyLen / block_size)):
                            real_key_dec = np.random.randint(0, 2, max_data_len * 8, dtype=np.int8)
                            real_key = binary_to_bytes(real_key_dec)
                            encode_start = time.time_ns() / 10 ** 6
                            ecc = bch.encode(real_key)
                            encode_time.append(time.time_ns() / 10 ** 6 - encode_start)

                            real_key_quartet.extend(bytes_to_binary(real_key))
                            packet = real_key + ecc
                            keyBin.extend(bytes_to_binary(packet))

                    # print("before", len(keyBin), keyBin)
                    basic = [0, 1]
                elif isEncoded and bits == 2 and keyLen == 256:
                    if isGrouped == False:
                        bch = bchlib.BCH(19, m=9)
                        max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        # 一部分是随机的密钥，一部分是ECC
                        real_key_dec = np.random.randint(0, 2, max_data_len * 8, dtype=np.int8)
                        real_key = binary_to_bytes(real_key_dec)
                        encode_start = time.time_ns() / 10 ** 6
                        ecc = bch.encode(real_key)
                        encode_time.append(time.time_ns() / 10 ** 6 - encode_start)

                        real_key_quartet = bytes_to_binary(real_key)
                        # print("before", real_key_quartet)
                        packet = real_key + ecc
                        keyBin = bytes_to_binary(packet)
                    else:
                        # in groups
                        # two groups
                        block_size = 128
                        bch = bchlib.BCH(17, m=8)
                        # four groups
                        # block_size = 64
                        # bch = bchlib.BCH(9, m=7)
                        # eight groups
                        # block_size = 32
                        # t代表纠错个数
                        # bch = bchlib.BCH(9, m=6)
                        # bch = bchlib.BCH(7, m=6)
                        # bch = bchlib.BCH(5, m=6)
                        # max_data_len = bch.n // 8 - bch.ecc_bits // 8  # only for bchlib.BCH(5, m=6)
                        # sixteen groups
                        # block_size = 16
                        # bch = bchlib.BCH(5, m=5)
                        max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        keyBin = []
                        real_key_quartet = []
                        this_encode_time = 0
                        for i in range(int(keyLen / block_size)):
                            real_key_dec = np.random.randint(0, 2, max_data_len * 8, dtype=np.int8)
                            real_key = binary_to_bytes(real_key_dec)
                            encode_start = time.time_ns() / 10 ** 6
                            ecc = bch.encode(real_key)
                            this_encode_time += time.time_ns() / 10 ** 6 - encode_start

                            real_key_quartet.extend(bytes_to_binary(real_key))
                            packet = real_key + ecc
                            keyBin.extend(bytes_to_binary(packet))
                        encode_time.append(this_encode_time)

                    # print("before", len(keyBin), keyBin)
                    basic = [0, 1]
                elif isEncoded and bits == 2 and keyLen == 128:
                    if isGrouped == False:
                        bch = bchlib.BCH(17, m=8)
                        max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        # 一部分是随机的密钥，一部分是ECC
                        real_key_dec = np.random.randint(0, 2, max_data_len * 8, dtype=np.int8)
                        real_key = binary_to_bytes(real_key_dec)
                        encode_start = time.time_ns() / 10 ** 6
                        ecc = bch.encode(real_key)
                        encode_time.append(time.time_ns() / 10 ** 6 - encode_start)

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
                        this_encode_time = 0
                        for i in range(int(keyLen / block_size)):
                            real_key_dec = np.random.randint(0, 2, max_data_len * 8, dtype=np.int8)
                            real_key = binary_to_bytes(real_key_dec)
                            encode_start = time.time_ns() / 10 ** 6
                            ecc = bch.encode(real_key)
                            this_encode_time += time.time_ns() / 10 ** 6 - encode_start

                            real_key_quartet.extend(bytes_to_binary(real_key))
                            packet = real_key + ecc
                            keyBin.extend(bytes_to_binary(packet))
                        encode_time.append(this_encode_time)

                    # print("before", len(keyBin), keyBin)
                    basic = [0, 1]
                elif isEncoded and bits == 2 and keyLen == 64:
                    if isGrouped == False:
                        bch = bchlib.BCH(9, m=7)
                        max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        # 一部分是随机的密钥，一部分是ECC
                        real_key_dec = np.random.randint(0, 2, max_data_len * 8, dtype=np.int8)
                        real_key = binary_to_bytes(real_key_dec)
                        encode_start = time.time_ns() / 10 ** 6
                        ecc = bch.encode(real_key)
                        encode_time.append(time.time_ns() / 10 ** 6 - encode_start)

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
                        this_encode_time = 0
                        for i in range(int(keyLen / block_size)):
                            real_key_dec = np.random.randint(0, 2, max_data_len * 8, dtype=np.int8)
                            real_key = binary_to_bytes(real_key_dec)
                            encode_start = time.time_ns() / 10 ** 6
                            ecc = bch.encode(real_key)
                            this_encode_time += time.time_ns() / 10 ** 6 - encode_start

                            real_key_quartet.extend(bytes_to_binary(real_key))
                            packet = real_key + ecc
                            keyBin.extend(bytes_to_binary(packet))
                        encode_time.append(this_encode_time)

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

                # tmpCSIa1IndPerm = tmpCSIa1IndPerm - np.mean(tmpCSIa1IndPerm)
                # tmpCSIb1IndPerm = tmpCSIb1IndPerm - np.mean(tmpCSIb1IndPerm)
                # tmpCSIe1IndPerm = tmpCSIe1IndPerm - np.mean(tmpCSIe1IndPerm)

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

                if solution == "cvxpy_perturbed_ls":
                    # cvxpy perturbed least squares
                    # SCIP最慢，其他结果一致
                    lambda_ = 0
                    solvers = [cp.SCS, cp.GUROBI, cp.OSQP, cp.CVXOPT, cp.ECOS, cp.SCIP, cp.PROXQP, cp.MOSEK,
                               cp.CLARABEL, cp.NAG, cp.XPRESS]
                    solver = solvers[2]
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
                    solve_time.append(time.time_ns() / 10 ** 6 - solve_start)
                    # print("num_iters of a: ", prob.solver_stats.num_iters)
                    iterations_of_a.append(prob.solver_stats.num_iters)
                    all_iterations.append(prob.solver_stats.num_iters)
                    a_list_number = [i.value for i in x]

                    if isEncoded and bits == 4:
                        a_list_number = np.array([round(abs(i)) for i in a_list_number])
                        # 取出data部分
                        a_data_quartet = a_list_number[:max_data_len * 4]
                        a_data = quartet_to_bytes(a_data_quartet)

                        a_ecc_quartet = a_list_number[max_data_len * 4:]
                        a_ecc = quartet_to_bytes(a_ecc_quartet)
                        decode_start = time.time_ns() / 10 ** 6
                        bch.decode(a_data, a_ecc)
                        bch.correct(a_data, a_ecc)
                        decode_time.append(time.time_ns() / 10 ** 6 - decode_start)
                        a_list_number = bytes_to_quartet(a_data)
                        a_list_number.extend(bytes_to_quartet(a_ecc))
                        # print("after", len(a_list_number), a_list_number)
                        # exit()
                    elif isEncoded and bits == 2:
                        a_list_number = np.array([round(abs(i)) for i in a_list_number])
                        if isGrouped == False:
                            # 取出data部分
                            a_data_quartet = a_list_number[:max_data_len * 8]
                            a_data = binary_to_bytes(a_data_quartet)

                            a_ecc_quartet = a_list_number[max_data_len * 8:]
                            a_ecc = binary_to_bytes(a_ecc_quartet)
                            decode_start = time.time_ns() / 10 ** 6
                            bch.decode(a_data, a_ecc)
                            bch.correct(a_data, a_ecc)
                            decode_time.append(time.time_ns() / 10 ** 6 - decode_start)
                            a_list_number = bytes_to_binary(a_data)
                            a_list_number.extend(bytes_to_binary(a_ecc))
                        else:
                            # in groups
                            a_decoded = []
                            this_decode_time = 0
                            for i in range(int(keyLen / block_size)):
                                a_list_number_tmp = a_list_number[i * block_size * 2:(i + 1) * block_size * 2]
                                a_data_quartet = a_list_number_tmp[:max_data_len * 8]
                                a_data = binary_to_bytes(a_data_quartet)

                                a_ecc_quartet = a_list_number_tmp[max_data_len * 8:]
                                a_ecc = binary_to_bytes(a_ecc_quartet)
                                decode_start = time.time_ns() / 10 ** 6
                                bch.decode(a_data, a_ecc)
                                bch.correct(a_data, a_ecc)
                                this_decode_time += time.time_ns() / 10 ** 6 - decode_start
                                a_decoded.extend(bytes_to_binary(a_data))
                                a_decoded.extend(bytes_to_binary(a_ecc))
                            decode_time.append(this_decode_time)
                            a_list_number = a_decoded

                        # print("after", len(a_list_number), a_list_number)
                        # exit()
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
                    solve_time.append(time.time_ns() / 10 ** 6 - solve_start)
                    iterations_of_b.append(prob.solver_stats.num_iters)
                    all_iterations.append(prob.solver_stats.num_iters)
                    b_list_number = [i.value for i in x]

                    b_list_number_before_decode = np.array([abs(i) for i in b_list_number])

                    if isEncoded and bits == 4:
                        b_list_number = np.array([round(abs(i)) for i in b_list_number])
                        # 取出data部分
                        b_data_quartet = b_list_number[:max_data_len * 4]
                        b_data = quartet_to_bytes(b_data_quartet)

                        b_ecc_quartet = b_list_number[max_data_len * 4:]
                        b_ecc = quartet_to_bytes(b_ecc_quartet)
                        decode_start = time.time_ns() / 10 ** 6
                        bch.decode(b_data, b_ecc)
                        bch.correct(b_data, b_ecc)
                        decode_time.append(time.time_ns() / 10 ** 6 - decode_start)
                        b_list_number = bytes_to_quartet(b_data)
                        b_list_number.extend(bytes_to_quartet(b_ecc))
                    elif isEncoded and bits == 2:
                        b_list_number = np.array([round(abs(i)) for i in b_list_number])
                        if isGrouped == False:
                            # 取出data部分
                            b_data_quartet = b_list_number[:max_data_len * 8]
                            b_data = binary_to_bytes(b_data_quartet)

                            b_ecc_quartet = b_list_number[max_data_len * 8:]
                            b_ecc = binary_to_bytes(b_ecc_quartet)
                            decode_start = time.time_ns() / 10 ** 6
                            bch.decode(b_data, b_ecc)
                            bch.correct(b_data, b_ecc)
                            decode_time.append(time.time_ns() / 10 ** 6 - decode_start)
                            b_list_number = bytes_to_binary(b_data)
                            b_list_number.extend(bytes_to_binary(b_ecc))
                        else:
                            # in groups
                            b_decoded = []
                            this_decode_time = 0
                            for i in range(int(keyLen / block_size)):
                                b_list_number_tmp = b_list_number[i * block_size * 2:(i + 1) * block_size * 2]
                                b_data_quartet = b_list_number_tmp[:max_data_len * 8]
                                b_data = binary_to_bytes(b_data_quartet)

                                b_ecc_quartet = b_list_number_tmp[max_data_len * 8:]
                                b_ecc = binary_to_bytes(b_ecc_quartet)
                                decode_start = time.time_ns() / 10 ** 6
                                bch.decode(b_data, b_ecc)
                                bch.correct(b_data, b_ecc)
                                this_decode_time += time.time_ns() / 10 ** 6 - decode_start
                                b_decoded.extend(bytes_to_binary(b_data))
                                b_decoded.extend(bytes_to_binary(b_ecc))
                            decode_time.append(this_decode_time)
                            b_list_number = b_decoded

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

                    if isEncoded and bits == 4:
                        e_list_number = np.array([round(abs(i)) for i in e_list_number])
                        # 取出data部分
                        e_data_quartet = e_list_number[:max_data_len * 4]
                        e_data = quartet_to_bytes(e_data_quartet)

                        e_ecc_quartet = e_list_number[max_data_len * 4:]
                        e_ecc = quartet_to_bytes(e_ecc_quartet)
                        bch.decode(e_data, e_ecc)
                        bch.correct(e_data, e_ecc)
                        e_list_number = bytes_to_quartet(e_data)
                        e_list_number.extend(bytes_to_quartet(e_ecc))
                    elif isEncoded and bits == 2:
                        e_list_number = np.array([round(abs(i)) for i in e_list_number])
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

                a_list_number1 = np.array([round(abs(i)) for i in a_list_number])
                b_list_number1 = np.array([round(abs(i)) for i in b_list_number])
                e_list_number1 = np.array([round(abs(i)) for i in e_list_number])

                if error_correct == True and len(np.nonzero(a_list_number1 - b_list_number1)[0]) != 0:
                    errors = []
                    # for i in range(len(np.nonzero(a_list_number1 - b_list_number1)[0])):
                    #     errors.append(b_list_number_before_decode[i])
                    for i in range(len(a_list_number1)):
                        if a_list_number1[i] != b_list_number1[i]:
                            errors.append(i)
                    print("errors", errors)
                    # 纠错：纠错前B发送易错位置，A发送对应的RSSa*x=b
                    if isEncoded == False or isComplex == False:
                        raise Exception("error correct only for encoded complex")

                    close_index = []
                    epsilon = 0.1
                    for i in range(len(b_list_number_before_decode)):
                        if (b_list_number_before_decode[i] >= 0.5 - epsilon and
                                b_list_number_before_decode[i] <= 0.5 + epsilon):
                            close_index.append(i)
                    print("close_index", close_index)
                    close_index = errors

                    if close_index != []:
                        # 扩充close的个数，使得可以正常编码
                        while len(close_index) < 8:
                            correct_index = np.random.randint(0, len(a_list_number1) - 1)
                            if correct_index not in close_index:
                                close_index.append(correct_index)

                        a_list_number_short = []
                        for i in range(len(a_list_number1)):
                            if i in close_index:
                                a_list_number_short.append(a_list_number1[i])


                        def construct_sub_matrix(matrix, close_index):
                            sub_matrix = []
                            for i in range(len(matrix)):
                                if i in close_index:
                                    tmp = []
                                    isIn = False
                                    for j in range(len(matrix)):
                                        if j in close_index:
                                            tmp.append(matrix[i][j])
                                            isIn = True
                                    if isIn:
                                        sub_matrix.append(tmp)
                            return sub_matrix


                        # sub_tmpCSIa1IndPerm = construct_sub_matrix(tmpCSIa1IndPerm, close_index)
                        # sub_tmpCSIb1IndPerm = construct_sub_matrix(tmpCSIb1IndPerm, close_index)
                        # sub_tmpCSIe1IndPerm = construct_sub_matrix(tmpCSIe1IndPerm, close_index)

                        # 取前一半非零值,进行滤波,噪音扰动,均衡
                        shortKeyLen = len(close_index)
                        changed_index = [x for x in list(range(len(tmpCSIa1IndPerm))) if x not in close_index][
                                        :len(close_index)]
                        sub_tmpCSIa1IndPerm = construct_sub_matrix(tmpCSIa1IndPerm, changed_index)[0]
                        sub_tmpCSIb1IndPerm = construct_sub_matrix(tmpCSIb1IndPerm, changed_index)[0]
                        sub_tmpCSIe1IndPerm = construct_sub_matrix(tmpCSIe1IndPerm, changed_index)[0]

                        sub_tmpCSIa1IndPerm = smooth(np.array(sub_tmpCSIa1IndPerm), window_len=4, window='flat')[
                                              :shortKeyLen]
                        sub_tmpCSIb1IndPerm = smooth(np.array(sub_tmpCSIb1IndPerm), window_len=4, window='flat')[
                                              :shortKeyLen]
                        sub_tmpCSIe1IndPerm = smooth(np.array(sub_tmpCSIe1IndPerm), window_len=4, window='flat')[
                                              :shortKeyLen]

                        sub_tmpCSIa1IndPerm = sub_tmpCSIa1IndPerm - np.mean(sub_tmpCSIa1IndPerm)
                        sub_tmpCSIb1IndPerm = sub_tmpCSIb1IndPerm - np.mean(sub_tmpCSIb1IndPerm)
                        sub_tmpCSIe1IndPerm = sub_tmpCSIe1IndPerm - np.mean(sub_tmpCSIe1IndPerm)

                        np.random.seed(10000)
                        shortRandomMatrix = np.random.uniform(5, np.std(sub_tmpCSIa1IndPerm) * 4,
                                                              size=(shortKeyLen, shortKeyLen))
                        sub_tmpCSIa1IndPerm = np.matmul(sub_tmpCSIa1IndPerm, shortRandomMatrix)
                        sub_tmpCSIb1IndPerm = np.matmul(sub_tmpCSIb1IndPerm, shortRandomMatrix)
                        sub_tmpCSIe1IndPerm = np.matmul(sub_tmpCSIe1IndPerm, shortRandomMatrix)

                        sub_tmpCSIa1IndPerm = normalize(sub_tmpCSIa1IndPerm)
                        sub_tmpCSIb1IndPerm = normalize(sub_tmpCSIb1IndPerm)
                        sub_tmpCSIe1IndPerm = normalize(sub_tmpCSIe1IndPerm)

                        sub_tmpCSIa1IndPerm = circulant(sub_tmpCSIa1IndPerm[::-1])
                        sub_tmpCSIb1IndPerm = circulant(sub_tmpCSIb1IndPerm[::-1])
                        sub_tmpCSIe1IndPerm = circulant(sub_tmpCSIe1IndPerm[::-1])

                        # 奇异值平滑
                        if isBalanced:
                            U, S, Vt = np.linalg.svd(sub_tmpCSIa1IndPerm)
                            S = medfilt(S, kernel_size=3)
                            D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
                            sub_tmpCSIa1IndPerm = U @ D @ Vt
                            U, S, Vt = np.linalg.svd(sub_tmpCSIb1IndPerm)
                            S = medfilt(S, kernel_size=3)
                            D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
                            sub_tmpCSIb1IndPerm = U @ D @ Vt
                            U, S, Vt = np.linalg.svd(sub_tmpCSIe1IndPerm)
                            S = medfilt(S, kernel_size=3)
                            D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
                            sub_tmpCSIe1IndPerm = U @ D @ Vt

                        correct_solve_start = time.time_ns() / 10 ** 6
                        tmpMulA2 = np.dot(sub_tmpCSIa1IndPerm, a_list_number_short)
                        correct_solve_end = time.time_ns() / 10 ** 6
                        correct_time_of_A.append(correct_solve_end - correct_solve_start)

                        lambda_ = 0
                        solvers = [cp.SCS, cp.GUROBI, cp.OSQP, cp.CVXOPT, cp.ECOS, cp.SCIP, cp.PROXQP, cp.MOSEK,
                                   cp.CLARABEL, cp.NAG, cp.XPRESS]
                        solver = solvers[2]

                        x = cp.Variable(len(close_index))
                        # 加正则项效果差
                        f_norm = np.linalg.norm(sub_tmpCSIa1IndPerm, ord='fro')
                        obj = cp.Minimize(
                            cp.sum_squares(sub_tmpCSIa1IndPerm @ x - tmpMulA2) + lambda_ * cp.sum_squares(x))
                        # prob = cp.Problem(obj)
                        prob = cp.Problem(obj, [x >= min(basic), x <= max(basic)])
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
                        a_float_list_number2 = [i.value for i in x]

                        x = cp.Variable(len(close_index))
                        # 加正则项效果差
                        f_norm = np.linalg.norm(sub_tmpCSIb1IndPerm, ord='fro')
                        obj = cp.Minimize(
                            cp.sum_squares(sub_tmpCSIb1IndPerm @ x - tmpMulA2) + lambda_ * cp.sum_squares(x))
                        # prob = cp.Problem(obj)
                        prob = cp.Problem(obj, [x >= min(basic), x <= max(basic)])
                        correct_solve_start = time.time_ns() / 10 ** 6
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
                        correct_solve_end = time.time_ns() / 10 ** 6
                        correct_time_of_B.append(correct_solve_end - correct_solve_start)
                        b_float_list_number2 = [i.value for i in x]

                        x = cp.Variable(len(close_index))
                        # 加正则项效果差
                        obj = cp.Minimize(
                            cp.sum_squares(sub_tmpCSIe1IndPerm @ x - tmpMulA2) + lambda_ * cp.sum_squares(x))
                        # prob = cp.Problem(obj)
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
                        e_float_list_number2 = [i.value for i in x]

                        a_list_number2 = np.array([round(abs(i)) for i in a_float_list_number2])
                        b_list_number2 = np.array([round(abs(i)) for i in b_float_list_number2])
                        e_list_number2 = np.array([round(abs(i)) for i in e_float_list_number2])

                        a_list_number_sum = a_list_number1
                        b_list_number_sum = b_list_number1
                        e_list_number_sum = e_list_number1

                        for i in range(len(close_index)):
                            a_list_number_sum[close_index[i]] = a_list_number2[i]
                            b_list_number_sum[close_index[i]] = b_list_number2[i]
                            e_list_number_sum[close_index[i]] = e_list_number2[i]

                        if bits == 4:
                            a_list_number2 = np.array([round(abs(i)) % 4 for i in a_list_number_sum])
                            b_list_number2 = np.array([round(abs(i)) % 4 for i in b_list_number_sum])
                            e_list_number2 = np.array([round(abs(i)) % 4 for i in e_list_number_sum])
                        elif bits == 2:
                            a_list_number2 = np.array([round(abs(i)) % 2 for i in a_list_number_sum])
                            b_list_number2 = np.array([round(abs(i)) % 2 for i in b_list_number_sum])
                            e_list_number2 = np.array([round(abs(i)) % 2 for i in e_list_number_sum])

                        a_list_number1 = a_list_number2
                        b_list_number1 = b_list_number2
                        e_list_number1 = e_list_number2

                # 用LDPC纠错，效果最好
                errors = []
                for i in range(len(a_list_number1)):
                    if a_list_number1[i] != b_list_number1[i]:
                        errors.append(i)

                if error_correct == True and len(np.nonzero(a_list_number1 - b_list_number1)[0]) != 0:
                    a_list = []
                    b_list = []
                    e_list = []
                    seed = np.random.RandomState(42)
                    n_code, d_v, d_c = 0, 0, 0
                    if len(a_list_number1) == 128:
                        n_code = 135
                        d_v = 3
                        d_c = 45
                    elif len(a_list_number1) == 256:
                        n_code = 261
                        d_v = 2
                        d_c = 87
                    elif len(a_list_number1) == 512:
                        n_code = 540
                        d_v = 3
                        d_c = 54
                    elif len(a_list_number1) == 1024:
                        n_code = 1340
                        d_v = 35
                        d_c = 134
                    ldpc_initial_start = time.time_ns() / 10 ** 6
                    H, G = make_ldpc(n_code, d_v, d_c, seed=seed, systematic=True, sparse=True)
                    ldpc_initial_end = time.time_ns() / 10 ** 6

                    k = G.shape[1]
                    snr = 20
                    ra = np.random.randint(2, size=n_code)
                    rb = np.random.randint(2, size=n_code)
                    re = np.random.randint(2, size=n_code)
                    print("lpdc before", abs(np.array(a_list_number1) - np.array(b_list_number1)).sum(),
                          len(np.nonzero(a_list_number1 - b_list_number1)[0]))
                    ldpc_encode_a_start = time.time_ns() / 10 ** 6
                    ya = (encode(G, a_list_number1, seed=seed) + ra) % 2
                    ldpc_encode_a_end = time.time_ns() / 10 ** 6

                    ldpc_encode_b_start = time.time_ns() / 10 ** 6
                    yb = (encode(G, b_list_number1, seed=seed) + rb) % 2
                    ldpc_encode_b_end = time.time_ns() / 10 ** 6

                    ldpc_decode_a_start = time.time_ns() / 10 ** 6
                    xa = (decode(H, yb, snr) + ra) % 2
                    a_list_number1 = get_message(G, xa)
                    ldpc_decode_a_end = time.time_ns() / 10 ** 6

                    ldpc_decode_b_start = time.time_ns() / 10 ** 6
                    xb = (decode(H, ya, snr) + rb) % 2
                    b_list_number1 = get_message(G, xb)
                    ldpc_decode_b_end = time.time_ns() / 10 ** 6

                    xe = decode(H, ya, snr)
                    e_list_number1 = get_message(G, xe)
                    print("lpdc after", abs(np.array(a_list_number1) - np.array(b_list_number1)).sum(),
                          len(np.nonzero(a_list_number1 - b_list_number1)[0]))

                    ldpc_time_of_A.append(ldpc_initial_end - ldpc_initial_start + ldpc_encode_a_end -
                                          ldpc_encode_a_start + ldpc_decode_a_end - ldpc_decode_a_start)
                    ldpc_time_of_B.append(ldpc_encode_b_end - ldpc_encode_b_start +
                                          ldpc_decode_b_end - ldpc_decode_b_start)

                a_list_number = a_list_number1
                b_list_number = b_list_number1
                e_list_number = e_list_number1

                # 统计错误的bit对
                # err = np.nonzero(a_list_number - b_list_number)[0]
                # if err.size > 0:
                #     for i in range(len(err)):
                #         error_bits.append(str(a_list_number[err[i]]) + str(b_list_number[err[i]]))

                # 转成二进制，0填充成0000
                if bits == 4:
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
            print("iterations_of_a", max(iterations_of_a), min(iterations_of_a), np.mean(iterations_of_a))
            print("iterations_of_b", max(iterations_of_b), min(iterations_of_b), np.mean(iterations_of_b))
            iterations_of_a_b = iterations_of_a + iterations_of_b
            print("average iterations", np.round(max(iterations_of_a_b), 5), np.round(min(iterations_of_a_b), 5),
                  np.round(np.mean(iterations_of_a_b), 5))
            # print("error_bits", Counter(error_bits))
            print("\n")
messagebox.showinfo("提示", "测试结束")
print("all_iterations", max(all_iterations), min(all_iterations), np.mean(all_iterations))
if len(perturb_time) != 0:
    print("perturb_time", np.round(max(perturb_time), 5), np.round(min(perturb_time), 5),
          np.round(np.mean(perturb_time), 5))
if len(balance_time) != 0:
    print("balance_time", np.round(max(balance_time), 5), np.round(min(balance_time), 5),
          np.round(np.mean(balance_time), 5))
if len(encode_time) != 0:
    print("encode_time", np.round(max(encode_time), 5), np.round(min(encode_time), 5),
          np.round(np.mean(encode_time), 5))
if len(decode_time) != 0:
    print("decode_time", np.round(max(decode_time), 5), np.round(min(decode_time), 5),
          np.round(np.mean(decode_time), 5))
if len(solve_time) != 0:
    print("solve_time", np.round(max(solve_time), 5), np.round(min(solve_time), 5), np.round(np.mean(solve_time), 5))
if len(correct_time_of_A) != 0:
    print("correct_time_of_A", np.round(max(correct_time_of_A), 5), np.round(min(correct_time_of_A), 5),
          np.round(np.mean(correct_time_of_A), 5))
if len(correct_time_of_B) != 0:
    print("correct_time_of_B", np.round(max(correct_time_of_B), 5), np.round(min(correct_time_of_B), 5),
          np.round(np.mean(correct_time_of_B), 5))
if len(ldpc_time_of_A) != 0:
    print("ldpc_time_of_A", np.round(max(ldpc_time_of_A), 5), np.round(min(ldpc_time_of_A), 5),
          np.round(np.mean(ldpc_time_of_A), 5))
if len(ldpc_time_of_B) != 0:
    print("ldpc_time_of_B", np.round(max(ldpc_time_of_B), 5), np.round(min(ldpc_time_of_B), 5),
          np.round(np.mean(ldpc_time_of_B), 5))
