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

from segmentation.methods import dtw_metric
from tls import tls, stls, block_toeplitz, tls2, stls_qp, block_circulant

import random

def find_sub_opt_segment_method_sliding(data1, data2, min_len, max_len, threshold=np.inf):
    cal_dtw_times = 0

    segments = []
    n = len(data1)
    i = 0

    # 随机选择第一个分段的长度
    random.seed(100000)
    first_segment_len = random.randint(min_len, max_len)
    segments.append(data1[:first_segment_len])
    i += first_segment_len

    while i < n:
        segment_found = False
        max_min_distance = 0
        best_segment = None

        # 尝试在规定的区间内找到一个合适的分段长度
        for length in range(min_len, max_len + 1):
            if i + length > n:
                break  # 超出数组范围

            new_segment = data2[i:i + length]
            min_distance = float('inf')

            # 计算新分段与之前所有分段的最小DTW距离
            for segment in segments:
                cal_dtw_times += 1
                distance = dtw_metric(segment, new_segment)
                if distance < threshold:
                    min_distance = distance

            # 选择最小DTW距离最大的分段
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_segment = new_segment

        if best_segment is not None:
            segments.append(best_segment)
            i += len(best_segment)
            segment_found = True

        # 如果在规定区间内没有找到合适的分段长度，则将剩余部分作为最后一个分段
        if not segment_found:
            segments.append(data1[i:])
            break

    # 处理最后一个分段的长度
    if len(segments[-1]) < min_len:
        segments.pop()

    segment_method = []
    for segment in segments:
        segment_method.append(len(segment))

    # print("dtw times:", cal_dtw_times)
    return segment_method


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

# fileName = ["./data_alignment/csi_ear.mat",
#             "./data_alignment/csi_par.mat",
#             "./data_alignment/csi_sar.mat"]

fileName = ["../data/data_mobile_indoor_1.mat"]

# fileName = ["../data/data_mobile_indoor_1.mat",
#             "../data/data_mobile_outdoor_1.mat",
#             "../data/data_static_outdoor_1.mat",
#             "../data/data_static_indoor_1.mat"
#             ]

# fileName = ["../csi/csi_mobile_indoor_1_r",
#             "../csi/csi_mobile_outdoor_r",
#             "../csi/csi_static_indoor_1_r",
#             "../csi/csi_static_outdoor_r"]

# fileName = ["../data/data_NLOS.mat",
#             "../data/data_static_indoor_1.mat"]

# fileName = ["../data/data_static_indoor_1.mat"]

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
correct_time = []

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

                # stalking attack
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
                if f.find("data_alignment") != -1:
                    tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
                else:
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
                        # bch = bchlib.BCH(22, m=10)
                        # max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        # bch = bchlib.BCH(60, m=10)
                        # max_data_len = bch.n // 8 - (bch.ecc_bits + 56) // 8
                        bch = bchlib.BCH(58, m=10)
                        max_data_len = bch.n // 8 - (bch.ecc_bits + 56) // 8
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
                        bch = bchlib.BCH(31, m=9)
                        max_data_len = bch.n // 8 - (bch.ecc_bits + 14) // 8
                        # four groups
                        # block_size = 128
                        # bch = bchlib.BCH(17, m=8)
                        # eight groups
                        # block_size = 64
                        # bch = bchlib.BCH(9, m=7)
                        # max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
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
                        # bch = bchlib.BCH(19, m=9)
                        # max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        # bch = bchlib.BCH(36, m=9)
                        # max_data_len = bch.n // 8 - (bch.ecc_bits + 56) // 8
                        bch = bchlib.BCH(31, m=9)
                        max_data_len = bch.n // 8 - (bch.ecc_bits + 14) // 8

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
                        bch = bchlib.BCH(19, m=8)
                        max_data_len = bch.n // 8 - (bch.ecc_bits + 14) // 8
                        # four groups
                        # block_size = 64
                        # bch = bchlib.BCH(10, m=7)
                        # max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        # bch = bchlib.BCH(11, m=7)
                        # max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        # eight groups
                        # block_size = 32
                        # bch = bchlib.BCH(9, m=6)
                        # max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        # bch = bchlib.BCH(6, m=6)
                        # bch = bchlib.BCH(1, m=6)
                        # max_data_len = bch.n // 8 - bch.ecc_bits // 8
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
                        # bch = bchlib.BCH(17, m=8)
                        # max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        # bch = bchlib.BCH(21, m=8)
                        # max_data_len = bch.n // 8 - (bch.ecc_bits + 21) // 8
                        bch = bchlib.BCH(19, m=8)
                        max_data_len = bch.n // 8 - (bch.ecc_bits + 14) // 8
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
                        bch = bchlib.BCH(10, m=7)
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
                        # bch = bchlib.BCH(9, m=7)
                        # bch = bchlib.BCH(10, m=7)
                        # max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                        bch = bchlib.BCH(11, m=7)
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
                        bch = bchlib.BCH(6, m=6)
                        max_data_len = bch.n // 8 - (bch.ecc_bits) // 8
                        # four groups
                        # block_size = 16
                        # bch = bchlib.BCH(5, m=5)
                        # max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
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
                    # segment_method_ori = find_sub_opt_segment_method_sliding(tmpCSIa1Ind, tmpCSIa1Ind, 3, 5)
                    # if len(segment_method_ori) != 16:
                    #     continue
                    # tmpCSIa1IndPerm = []
                    # tmpCSIb1IndPerm = []
                    # tmpCSIe1IndPerm = []
                    # for i in range(len(segment_method_ori)):
                    #     full_zero_a = tmpCSIa1Ind[i * segment_method_ori[i]: (i + 1) * segment_method_ori[i]]
                    #     full_zero_a = full_zero_a.tolist() + np.zeros(len(tmpCSIa1Ind) - len(full_zero_a)).tolist()
                    #     tmpCSIa1IndPerm.append(full_zero_a)
                    #     full_zero_b = tmpCSIb1Ind[i * segment_method_ori[i]: (i + 1) * segment_method_ori[i]]
                    #     full_zero_b = full_zero_b.tolist() + np.zeros(len(tmpCSIb1Ind) - len(full_zero_b)).tolist()
                    #     tmpCSIb1IndPerm.append(full_zero_b)
                    #     full_zero_e = tmpCSIe1Ind[i * segment_method_ori[i]: (i + 1) * segment_method_ori[i]]
                    #     full_zero_e = full_zero_e.tolist() + np.zeros(len(tmpCSIe1Ind) - len(full_zero_e)).tolist()
                    #     tmpCSIe1IndPerm.append(full_zero_e)
                    # # 填充至64位
                    # tmpCSIa1IndPerm = np.tile(np.array(tmpCSIa1IndPerm).T, int(keyLen / len(tmpCSIa1IndPerm)))
                    # tmpCSIb1IndPerm = np.tile(np.array(tmpCSIb1IndPerm).T, int(keyLen / len(tmpCSIb1IndPerm)))
                    # tmpCSIe1IndPerm = np.tile(np.array(tmpCSIe1IndPerm).T, int(keyLen / len(tmpCSIe1IndPerm)))
                    # tmpCSIa1IndPerm = tmpCSIa1IndPerm.T
                    # tmpCSIb1IndPerm = tmpCSIb1IndPerm.T
                    # tmpCSIe1IndPerm = tmpCSIe1IndPerm.T

                    tmpCSIa1IndPerm = circulant(tmpCSIa1Ind[::-1])
                    tmpCSIb1IndPerm = circulant(tmpCSIb1Ind[::-1])
                    tmpCSIe1IndPerm = circulant(tmpCSIe1Ind[::-1])
                    # tmpCSIa1IndPerm = toeplitz(tmpCSIa1Ind[::-1])
                    # tmpCSIb1IndPerm = toeplitz(tmpCSIb1Ind[::-1])
                    # tmpCSIe1IndPerm = toeplitz(tmpCSIe1Ind[::-1])

                # tmpCSIa1IndPerm = normalize(tmpCSIa1IndPerm)
                # tmpCSIb1IndPerm = normalize(tmpCSIb1IndPerm)
                # tmpCSIe1IndPerm = normalize(tmpCSIe1IndPerm)

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
                        # if isBalanced:
                        #     prob.solve(solver=solver, scaling=False)
                        # else:
                        #     prob.solve(solver=solver)
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
                        # if isBalanced:
                        #     prob.solve(solver=solver, scaling=False)
                        # else:
                        #     prob.solve(solver=solver)
                        # prob.solve(solver=solver, scaling=False, warm_start=True)
                    else:
                        prob.solve(solver=solver)
                    solve_time.append(time.time_ns() / 10 ** 6 - solve_start)
                    iterations_of_b.append(prob.solver_stats.num_iters)
                    all_iterations.append(prob.solver_stats.num_iters)
                    b_list_number = [i.value for i in x]

                    b_list_number_before_decode = np.array([round(abs(i)) for i in b_list_number])

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
                        # if isBalanced:
                        #     prob.solve(solver=solver, scaling=False)
                        # else:
                        #     prob.solve(solver=solver)
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

                if error_correct == True:
                    # 纠错：纠错前A发送了RSSa*x=b，纠错时B发送RSSb*x'=b'，效果变好
                    if isEncoded == False or isComplex == False:
                        raise Exception("error correct only for encoded complex")

                    # tmpMulB1 = np.dot(tmpCSIb1IndPerm, b_list_number_before_decode)
                    # 效果好
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
                    correct_solve_start = time.time_ns() / 10 ** 6
                    if solver == cp.SCS:
                        prob.solve(solver=solver, max_iters=5000)
                    elif solver == cp.SCIP:
                        prob.solve(solver=solver, scip_params={"limits/time": 10})
                    elif solver == cp.OSQP:
                        # scaling=False表示禁用缩放以矩阵均衡
                        prob.solve(solver=solver, scaling=False)
                        # if isBalanced:
                        #     prob.solve(solver=solver, scaling=False)
                        # else:
                        #     prob.solve(solver=solver)
                        # prob.solve(solver=solver, scaling=False, warm_start=True)
                    else:
                        prob.solve(solver=solver)
                    correct_solve_end = time.time_ns() / 10 ** 6
                    a_float_list_number2 = [i.value for i in x]

                    if isEncoded and bits == 4:
                        a_float_list_number2 = np.array([round(abs(i)) for i in a_float_list_number2])
                        # 取出data部分
                        a_data_quartet = a_float_list_number2[:max_data_len * 4]
                        a_data = quartet_to_bytes(a_data_quartet)

                        a_ecc_quartet = a_float_list_number2[max_data_len * 4:]
                        a_ecc = quartet_to_bytes(a_ecc_quartet)
                        correct_correct_start = time.time_ns() / 10 ** 6
                        bch.decode(a_data, a_ecc)
                        bch.correct(a_data, a_ecc)
                        correct_correct_end = time.time_ns() / 10 ** 6
                        correct_time.append(correct_correct_end - correct_correct_start +
                                            correct_solve_end - correct_solve_start)
                        a_float_list_number2 = bytes_to_quartet(a_data)
                        a_float_list_number2.extend(bytes_to_quartet(a_ecc))
                        # print("after", len(a_list_number), a_list_number)
                        # exit()
                    elif isEncoded and bits == 2:
                        a_float_list_number2 = np.array([round(abs(i)) for i in a_float_list_number2])
                        if isGrouped == False:
                            # 取出data部分
                            a_data_quartet = a_float_list_number2[:max_data_len * 8]
                            a_data = binary_to_bytes(a_data_quartet)

                            a_ecc_quartet = a_float_list_number2[max_data_len * 8:]
                            a_ecc = binary_to_bytes(a_ecc_quartet)
                            correct_correct_start = time.time_ns() / 10 ** 6
                            bch.decode(a_data, a_ecc)
                            bch.correct(a_data, a_ecc)
                            correct_correct_end = time.time_ns() / 10 ** 6
                            correct_time.append(correct_correct_end - correct_correct_start +
                                                correct_solve_end - correct_solve_start)
                            a_float_list_number2 = bytes_to_binary(a_data)
                            a_float_list_number2.extend(bytes_to_binary(a_ecc))
                        else:
                            # in groups
                            a_decoded = []
                            for i in range(int(keyLen / block_size)):
                                a_list_number_tmp = a_float_list_number2[i * block_size * 2:(i + 1) * block_size * 2]
                                a_data_quartet = a_list_number_tmp[:max_data_len * 8]
                                a_data = binary_to_bytes(a_data_quartet)

                                a_ecc_quartet = a_list_number_tmp[max_data_len * 8:]
                                a_ecc = binary_to_bytes(a_ecc_quartet)
                                correct_correct_start = time.time_ns() / 10 ** 6
                                bch.decode(a_data, a_ecc)
                                bch.correct(a_data, a_ecc)
                                correct_correct_end = time.time_ns() / 10 ** 6
                                correct_time.append(correct_correct_end - correct_correct_start +
                                                    correct_solve_end - correct_solve_start)
                                a_decoded.extend(bytes_to_binary(a_data))
                                a_decoded.extend(bytes_to_binary(a_ecc))
                            a_float_list_number2 = a_decoded

                        # print("after", a_list_number)
                        # exit()

                    x = cp.Variable(len(keyBin))
                    # 加正则项效果差
                    obj = cp.Minimize(cp.sum_squares(tmpCSIb1IndPerm @ x - tmpMulB1) + lambda_ * cp.sum_squares(x))
                    # prob = cp.Problem(obj)
                    prob = cp.Problem(obj, [x >= min(basic), x <= max(basic)])
                    correct_solve_start = time.time_ns() / 10 ** 6
                    if solver == cp.SCS:
                        prob.solve(solver=solver, max_iters=5000)
                    elif solver == cp.SCIP:
                        prob.solve(solver=solver, scip_params={"limits/time": 10})
                    elif solver == cp.OSQP:
                        prob.solve(solver=solver, scaling=False)
                        # if isBalanced:
                        #     prob.solve(solver=solver, scaling=False)
                        # else:
                        #     prob.solve(solver=solver)
                        # prob.solve(solver=solver, scaling=False, warm_start=True)
                    else:
                        prob.solve(solver=solver)
                    correct_solve_end = time.time_ns() / 10 ** 6
                    b_float_list_number2 = [i.value for i in x]

                    if isEncoded and bits == 4:
                        b_float_list_number2 = np.array([round(abs(i)) for i in b_float_list_number2])
                        # 取出data部分
                        b_data_quartet = b_float_list_number2[:max_data_len * 4]
                        b_data = quartet_to_bytes(b_data_quartet)

                        b_ecc_quartet = b_float_list_number2[max_data_len * 4:]
                        b_ecc = quartet_to_bytes(b_ecc_quartet)
                        correct_correct_start = time.time_ns() / 10 ** 6
                        bch.decode(b_data, b_ecc)
                        bch.correct(b_data, b_ecc)
                        correct_correct_end = time.time_ns() / 10 ** 6
                        correct_time.append(correct_correct_end - correct_correct_start +
                                            correct_solve_end - correct_solve_start)
                        b_float_list_number2 = bytes_to_quartet(b_data)
                        b_float_list_number2.extend(bytes_to_quartet(b_ecc))
                    elif isEncoded and bits == 2:
                        b_float_list_number2 = np.array([round(abs(i)) for i in b_float_list_number2])
                        if isGrouped == False:
                            # 取出data部分
                            b_data_quartet = b_float_list_number2[:max_data_len * 8]
                            b_data = binary_to_bytes(b_data_quartet)

                            b_ecc_quartet = b_float_list_number2[max_data_len * 8:]
                            b_ecc = binary_to_bytes(b_ecc_quartet)
                            correct_correct_start = time.time_ns() / 10 ** 6
                            bch.decode(b_data, b_ecc)
                            bch.correct(b_data, b_ecc)
                            correct_correct_end = time.time_ns() / 10 ** 6
                            correct_time.append(correct_correct_end - correct_correct_start +
                                                correct_solve_end - correct_solve_start)
                            b_float_list_number2 = bytes_to_binary(b_data)
                            b_float_list_number2.extend(bytes_to_binary(b_ecc))
                        else:
                            # in groups
                            b_decoded = []
                            for i in range(int(keyLen / block_size)):
                                b_list_number_tmp = b_float_list_number2[i * block_size * 2:(i + 1) * block_size * 2]
                                b_data_quartet = b_list_number_tmp[:max_data_len * 8]
                                b_data = binary_to_bytes(b_data_quartet)

                                b_ecc_quartet = b_list_number_tmp[max_data_len * 8:]
                                b_ecc = binary_to_bytes(b_ecc_quartet)
                                correct_correct_start = time.time_ns() / 10 ** 6
                                bch.decode(b_data, b_ecc)
                                bch.correct(b_data, b_ecc)
                                correct_correct_end = time.time_ns() / 10 ** 6
                                correct_time.append(correct_correct_end - correct_correct_start +
                                                    correct_solve_end - correct_solve_start)
                                b_decoded.extend(bytes_to_binary(b_data))
                                b_decoded.extend(bytes_to_binary(b_ecc))
                            b_float_list_number2 = b_decoded

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
                        # if isBalanced:
                        #     prob.solve(solver=solver, scaling=False)
                        # else:
                        #     prob.solve(solver=solver)
                        # prob.solve(solver=solver, scaling=False, warm_start=True)
                    else:
                        prob.solve(solver=solver)
                    e_float_list_number2 = [i.value for i in x]

                    if isEncoded and bits == 4:
                        e_float_list_number2 = np.array([round(abs(i)) for i in e_float_list_number2])
                        # 取出data部分
                        e_data_quartet = e_float_list_number2[:max_data_len * 4]
                        e_data = quartet_to_bytes(e_data_quartet)

                        e_ecc_quartet = e_float_list_number2[max_data_len * 4:]
                        e_ecc = quartet_to_bytes(e_ecc_quartet)
                        bch.decode(e_data, e_ecc)
                        bch.correct(e_data, e_ecc)
                        e_float_list_number2 = bytes_to_quartet(e_data)
                        e_float_list_number2.extend(bytes_to_quartet(e_ecc))
                    elif isEncoded and bits == 2:
                        e_float_list_number2 = np.array([round(abs(i)) for i in e_float_list_number2])
                        if isGrouped == False:
                            # 取出data部分
                            e_data_quartet = e_float_list_number2[:max_data_len * 8]
                            e_data = binary_to_bytes(e_data_quartet)

                            e_ecc_quartet = e_float_list_number2[max_data_len * 8:]
                            e_ecc = binary_to_bytes(e_ecc_quartet)
                            bch.decode(e_data, e_ecc)
                            bch.correct(e_data, e_ecc)
                            e_float_list_number2 = bytes_to_binary(e_data)
                            e_float_list_number2.extend(bytes_to_binary(e_ecc))
                        else:
                            # in groups
                            e_decoded = []
                            for i in range(int(keyLen / block_size)):
                                e_list_number_tmp = e_float_list_number2[i * block_size * 2:(i + 1) * block_size * 2]
                                e_data_quartet = e_list_number_tmp[:max_data_len * 8]
                                e_data = binary_to_bytes(e_data_quartet)

                                e_ecc_quartet = e_list_number_tmp[max_data_len * 8:]
                                e_ecc = binary_to_bytes(e_ecc_quartet)
                                bch.decode(e_data, e_ecc)
                                bch.correct(e_data, e_ecc)
                                e_decoded.extend(bytes_to_binary(e_data))
                                e_decoded.extend(bytes_to_binary(e_ecc))
                            e_float_list_number2 = e_decoded

                    a_list_number2 = np.array([round(abs(i)) for i in a_float_list_number2])
                    b_list_number2 = np.array([round(abs(i)) for i in b_float_list_number2])
                    e_list_number2 = np.array([round(abs(i)) for i in e_float_list_number2])

                    a_list_number_sum = np.array([round(abs(i / 2)) for i in a_list_number1 + a_list_number2])
                    b_list_number_sum = np.array([round(abs(i / 2)) for i in b_list_number1 + b_list_number2])
                    e_list_number_sum = np.array([round(abs(i / 2)) for i in e_list_number1 + e_list_number2])

                    mismatched = np.nonzero(a_list_number1 - a_list_number2)[0]
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

                    # errors = np.nonzero(a_list_number_sum - b_list_number_sum)[0]
                    # print("After rec", len(errors), errors)
                    # print()

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

                a_list_number = a_list_number1
                b_list_number = b_list_number1
                e_list_number = e_list_number1

                # # 统计错误的bit对
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
if len(correct_time) != 0:
    print("correct_time", np.round(max(correct_time), 5), np.round(min(correct_time), 5),
          np.round(np.mean(correct_time), 5))
