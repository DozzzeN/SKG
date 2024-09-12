import copy
import csv
import math
import sys
import time
from collections import Counter
from tkinter import messagebox

import bchlib
from reedsolo import RSCodec
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
import matlab.engine

from pyldpc import make_ldpc, encode, decode, get_message

from tls import tls, stls, block_toeplitz, tls2, stls_qp, block_circulant


def transform_neg_one_to_zero(x):
    for i in range(len(x)):
        if x[i] == -1:
            x[i] = 0
    return x

def transform_zero_to_neg_one(x):
    for i in range(len(x)):
        if x[i] == 0:
            x[i] = -1
    return x


# 两个字节数组的差异，即不同的位数
def bytearray_diff(a, b):
    if len(a) != len(b):
        raise ValueError("array length not equal")
    diff = 0
    for i in range(len(a)):
        diff += 1 if a[i] != b[i] else 0
    return diff

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


fileName = ["../data/data_mobile_indoor_1.mat",
            "../data/data_mobile_outdoor_1.mat",
            "../data/data_static_outdoor_1.mat",
            "../data/data_static_indoor_1.mat"
            ]

# fileName = ["../data/data_static_indoor_1.mat"]

# 是否排序
withoutSorts = [True, False]
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

all_iterations = []
used_time = []
isComplex = True
isEncoded = True

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

                CSIa1Orig = rawData['A'][:, 0]
                CSIb1Orig = rawData['A'][:, 1]

                seed = np.random.randint(100000)
                np.random.seed(seed)

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
                if isEncoded and bits == 4:
                    # bch = bchlib.BCH(22, m=10)
                    # max_data_len = bch.n // 8 - (bch.ecc_bits + 7) // 8
                    # # 一部分是随机的密钥，一部分是ECC
                    # real_key_dec = np.random.randint(0, 4, max_data_len * 4, dtype=np.int8)
                    # real_key = quartet_to_bytes(real_key_dec)
                    # ecc = bch.encode(real_key)
                    #
                    # real_key_quartet = bytes_to_quartet(real_key)
                    # # print("before", real_key_quartet)
                    # packet = real_key + ecc
                    # keyBin = bytes_to_quartet(packet)
                    # # 依然编码成二进制串
                    # basic = [0, 1, 2, 3]
                    pass
                elif isEncoded and bits == 2:
                    n = 512
                    d_v = 16
                    d_c = 32
                    snr = 20
                    seed = np.random.RandomState(42)
                    H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)
                    k = G.shape[1]
                    real_key = np.random.randint(2, size=k)
                    keyBin = transform_neg_one_to_zero(encode(G, real_key, seed=seed))
                    # print("before", len(real_key), real_key)
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
                if isComplex:
                    tmpCSIa1IndPerm = np.hstack((np.vstack((tmpCSIa1IndPerm, np.zeros((keyLen, keyLen)))),
                                                 np.vstack((np.zeros((keyLen, keyLen)), tmpCSIa1IndPerm))))
                    tmpCSIb1IndPerm = np.hstack((np.vstack((tmpCSIb1IndPerm, np.zeros((keyLen, keyLen)))),
                                                 np.vstack((np.zeros((keyLen, keyLen)), tmpCSIb1IndPerm))))
                    tmpCSIe1IndPerm = np.hstack((np.vstack((tmpCSIe1IndPerm, np.zeros((keyLen, keyLen)))),
                                                 np.vstack((np.zeros((keyLen, keyLen)), tmpCSIe1IndPerm))))
                    # print(np.allclose(A @ np.hstack((np.real(keyBin), np.imag(keyBin))),
                    #                   np.hstack((np.real(tmpCSIa1IndPerm @ keyBin), np.imag(tmpCSIa1IndPerm @ keyBin)))))

                    # 原始，奇异值，均衡
                    U, S, Vt = np.linalg.svd(tmpCSIa1IndPerm)
                    # 奇异值平滑
                    # D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
                    # tmpCSIa1IndPerm = U @ D @ Vt
                    D = np.diag(1 / np.sqrt(S))
                    tmpCSIa1IndPerm = U @ D @ np.diag(S) @ D @ Vt
                    U, S, Vt = np.linalg.svd(tmpCSIb1IndPerm)
                    # D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
                    # tmpCSIb1IndPerm = U @ D @ Vt
                    D = np.diag(1 / np.sqrt(S))
                    tmpCSIb1IndPerm = U @ D @ np.diag(S) @ D @ Vt
                    U, S, Vt = np.linalg.svd(tmpCSIe1IndPerm)
                    # D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
                    # tmpCSIe1IndPerm = U @ D @ Vt
                    D = np.diag(1 / np.sqrt(S))
                    tmpCSIe1IndPerm = U @ D @ np.diag(S) @ D @ Vt
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

                    if isEncoded and bits == 4:
                        # a_list_number = np.array([round(abs(i)) for i in a_list_number])
                        # # 取出data部分
                        # a_data_quartet = a_list_number[:max_data_len * 4]
                        # a_data = quartet_to_bytes(a_data_quartet)
                        #
                        # a_ecc_quartet = a_list_number[max_data_len * 4:]
                        # a_ecc = quartet_to_bytes(a_ecc_quartet)
                        # bch.decode(a_data, a_ecc)
                        # bch.correct(a_data, a_ecc)
                        # a_list_number = bytes_to_quartet(a_data)
                        # a_list_number.extend(bytes_to_quartet(a_ecc))
                        # # print("after", len(a_list_number), a_list_number)
                        # # exit()
                        pass
                    elif isEncoded and bits == 2:
                        a_list_number = np.array([round(abs(i)) for i in a_list_number])
                        a_list_number = decode(H, transform_zero_to_neg_one(a_list_number), snr)
                        # print("after", len(get_message(G, a_list_number)), get_message(G, a_list_number))
                        # exit()
                    if isComplex and isEncoded == False:
                        # x = cp.Variable(len(keyBin), complex=True)
                        x = cp.Variable(len(keyBin) * 2)
                    else:
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
                    iterations_of_b.append(prob.solver_stats.num_iters)
                    all_iterations.append(prob.solver_stats.num_iters)
                    b_list_number = [i.value for i in x]

                    if isEncoded and bits == 4:
                        # b_list_number = np.array([round(abs(i)) for i in b_list_number])
                        # # 取出data部分
                        # b_data_quartet = b_list_number[:max_data_len * 4]
                        # b_data = quartet_to_bytes(b_data_quartet)
                        #
                        # b_ecc_quartet = b_list_number[max_data_len * 4:]
                        # b_ecc = quartet_to_bytes(b_ecc_quartet)
                        # bch.decode(b_data, b_ecc)
                        # bch.correct(b_data, b_ecc)
                        # b_list_number = bytes_to_quartet(b_data)
                        # b_list_number.extend(bytes_to_quartet(b_ecc))
                        pass
                    elif isEncoded and bits == 2:
                        b_list_number = np.array([round(abs(i)) for i in b_list_number])
                        b_list_number = decode(H, transform_zero_to_neg_one(b_list_number), snr)

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
                    else:
                        prob.solve(solver=solver)
                    e_list_number = [i.value for i in x]

                    if isEncoded and bits == 4:
                        # e_list_number = np.array([round(abs(i)) for i in e_list_number])
                        # # 取出data部分
                        # e_data_quartet = e_list_number[:max_data_len * 4]
                        # e_data = quartet_to_bytes(e_data_quartet)
                        #
                        # e_ecc_quartet = e_list_number[max_data_len * 4:]
                        # e_ecc = quartet_to_bytes(e_ecc_quartet)
                        # bch.decode(e_data, e_ecc)
                        # bch.correct(e_data, e_ecc)
                        # e_list_number = bytes_to_quartet(e_data)
                        # e_list_number.extend(bytes_to_quartet(e_ecc))
                        pass
                    elif isEncoded and bits == 2:
                        e_list_number = np.array([round(abs(i)) for i in e_list_number])
                        e_list_number = decode(H, transform_zero_to_neg_one(e_list_number), snr)

                a_list_number1 = np.array([round(abs(i)) for i in a_list_number])
                b_list_number1 = np.array([round(abs(i)) for i in b_list_number])
                e_list_number1 = np.array([round(abs(i)) for i in e_list_number])

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
            print("iterations_of_a", max(iterations_of_a), min(iterations_of_a), np.mean(iterations_of_a))
            print("iterations_of_b", max(iterations_of_b), min(iterations_of_b), np.mean(iterations_of_b))
            # print("error_bits", Counter(error_bits))
            print("\n")
messagebox.showinfo("提示", "测试结束")
print("all_iterations", max(all_iterations), min(all_iterations), np.mean(all_iterations))
print("used_time", max(used_time), min(used_time), np.mean(used_time))
