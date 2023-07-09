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
            keyLen = 32 * segLen

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
                    randomMatrix = np.random.uniform(5, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))

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

                    # tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    # tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    # tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)

                    tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1))
                    if np.max(tmpCSIb1) - np.min(tmpCSIb1) == 0:
                        tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / np.max(tmpCSIb1)
                    else:
                        tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1))
                    tmpCSIe1 = (tmpCSIe1 - np.min(tmpCSIe1)) / (np.max(tmpCSIe1) - np.min(tmpCSIe1))

                    tmpCSIa1 = np.abs(np.fft.fft(tmpCSIa1))
                    tmpCSIb1 = np.abs(np.fft.fft(tmpCSIb1))
                    tmpCSIe1 = np.abs(np.fft.fft(tmpCSIe1))

                    # tmpCSIa1 = dct(tmpCSIa1)
                    # tmpCSIb1 = dct(tmpCSIb1)
                    # tmpCSIe1 = dct(tmpCSIe1)

                    # tmpCSIa1Reshape = np.array(tmpCSIa1).reshape(16, 16)
                    # tmpCSIb1Reshape = np.array(tmpCSIb1).reshape(16, 16)
                    # tmpCSIe1Reshape = np.array(tmpCSIe1).reshape(16, 16)
                    # pca = PCA(n_components=16)
                    # tmpCSIa1 = pca.fit_transform(tmpCSIa1Reshape).reshape(1, -1)[0]
                    # tmpCSIb1 = pca.fit_transform(tmpCSIb1Reshape).reshape(1, -1)[0]
                    # tmpCSIe1 = pca.fit_transform(tmpCSIe1Reshape).reshape(1, -1)[0]

                    # wavelet = 'sym2'
                    # wtCSIa1 = pywt.dwt(tmpCSIa1, wavelet)
                    # tmpCSIa1 = list(wtCSIa1[0])
                    # tmpCSIa1.extend(wtCSIa1[1])
                    # tmpCSIa1 = tmpCSIa1[0: keyLen]
                    # wtCSIb1 = pywt.dwt(tmpCSIb1, wavelet)
                    # tmpCSIb1 = list(wtCSIb1[0])
                    # tmpCSIb1.extend(wtCSIb1[1])
                    # tmpCSIb1 = tmpCSIb1[0: keyLen]
                    # wtCSIe1 = pywt.dwt(tmpCSIe1, wavelet)
                    # tmpCSIe1 = list(wtCSIe1[0])
                    # tmpCSIe1.extend(wtCSIe1[1])
                    # tmpCSIe1 = tmpCSIe1[0: keyLen]

                    # tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix) + randomMatrix[0]
                    # tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix) + randomMatrix[0]
                    # tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix) + randomMatrix[0]
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

                    # with shuffling
                    np.random.seed(0)
                    combineCSIx1Orig = list(zip(tmpCSIa1Ind, tmpCSIb1Ind, tmpCSIe1Ind))
                    np.random.shuffle(combineCSIx1Orig)
                    tmpCSIa1Ind, tmpCSIb1Ind, tmpCSIe1Ind = zip(*combineCSIx1Orig)
                    tmpCSIa1Ind = list(tmpCSIa1Ind)
                    tmpCSIb1Ind = list(tmpCSIb1Ind)
                    tmpCSIe1Ind = list(tmpCSIe1Ind)

                np.random.seed(0)
                if bits == 1:
                    keyBin = np.random.binomial(1, 0.5, keyLen)
                else:
                    keyBin = np.random.randint(0, 4, keyLen)

                # shuffling = np.random.permutation(keyLen)
                # tmpCSIa1IndPerm = circulant(tmpCSIa1Ind[shuffling])
                # tmpCSIb1IndPerm = circulant(tmpCSIb1Ind[shuffling])
                # tmpCSIe1IndPerm = circulant(tmpCSIe1Ind[shuffling])

                tmpCSIa1IndPerm = circulant(tmpCSIa1Ind[::-1])
                tmpCSIb1IndPerm = circulant(tmpCSIb1Ind[::-1])
                tmpCSIe1IndPerm = circulant(tmpCSIe1Ind[::-1])

                # 效果差
                # tmpCSIa1IndPerm = toeplitz(tmpCSIa1Ind[::-1])
                # tmpCSIb1IndPerm = toeplitz(tmpCSIb1Ind[::-1])
                # tmpCSIe1IndPerm = toeplitz(tmpCSIe1Ind[::-1])

                tmpMulA1 = np.dot(tmpCSIa1IndPerm, keyBin)

                # tmpCSIb1IndPerm = np.vstack([tmpCSIb1IndPerm, np.ones(keyLen)]).T
                # tmpCSIe1IndPerm = np.vstack([tmpCSIe1IndPerm, np.ones(keyLen)]).T

                # def residuals(x, tmpMulA1, tmpCSIx1IndPerm):
                #     return tmpMulA1 - np.dot(tmpCSIx1IndPerm, x)
                # a_list_number = leastsq(residuals, np.random.binomial(1, 0.5, keyLen), args=(tmpMulA1, tmpCSIa1IndPerm))[0]
                # b_list_number = leastsq(residuals, np.random.binomial(1, 0.5, keyLen), args=(tmpMulA1, tmpCSIb1IndPerm))[0]
                # e_list_number = leastsq(residuals, np.random.binomial(1, 0.5, keyLen), args=(tmpMulA1, tmpCSIe1IndPerm))[0]

                # # gurobi optimization
                # vtype = GRB.CONTINUOUS
                # # 速度太慢，也无法求出可行解，退化成一般的最小二乘法
                # # vtype = GRB.INTEGER
                # model = gp.Model("Integer Quadratic Programming")
                # model.setParam('OutputFlag', 0)
                # model.setParam("LogToConsole", 0)
                # inputs = []
                # for i in range(len(keyBin)):
                #     inputs.append(model.addVar(lb=0, ub=3, vtype=vtype, name=f'x{i}'))
                # obj = sum((np.dot(tmpCSIa1IndPerm, inputs) - tmpMulA1) ** 2)
                # model.setObjective(obj, GRB.MINIMIZE)
                # model.optimize()
                # if model.status == GRB.Status.OPTIMAL:
                #     a_list_number = [round(i.x) for i in inputs]
                # else:
                #     a_list_number = list(np.linalg.lstsq(tmpCSIa1IndPerm, tmpMulA1, rcond=None)[0])
                # model.close()
                #
                # model = gp.Model("Integer Quadratic Programming")
                # model.setParam('OutputFlag', 0)
                # model.setParam("LogToConsole", 0)
                # inputs = []
                # for i in range(len(keyBin)):
                #     inputs.append(model.addVar(lb=0, ub=3, vtype=vtype, name=f'x{i}'))
                # obj = sum((np.dot(tmpCSIb1IndPerm, inputs) - tmpMulA1) ** 2)
                # model.setObjective(obj, GRB.MINIMIZE)
                # model.optimize()
                # if model.status == GRB.Status.OPTIMAL:
                #     b_list_number = [round(i.x) for i in inputs]
                # else:
                #     b_list_number = list(np.linalg.lstsq(tmpCSIb1IndPerm, tmpMulA1, rcond=None)[0])
                # model.close()
                #
                # model = gp.Model("Integer Quadratic Programming")
                # model.setParam('OutputFlag', 0)
                # model.setParam("LogToConsole", 0)
                # inputs = []
                # for i in range(len(keyBin)):
                #     inputs.append(model.addVar(lb=0, ub=3, vtype=vtype, name=f'x{i}'))
                # obj = sum((np.dot(tmpCSIe1IndPerm, inputs) - tmpMulA1) ** 2)
                # model.setObjective(obj, GRB.MINIMIZE)
                # model.optimize()
                # if model.status == GRB.Status.OPTIMAL:
                #     e_list_number = [round(i.x) for i in inputs]
                # else:
                #     e_list_number = list(np.linalg.lstsq(tmpCSIe1IndPerm, tmpMulA1, rcond=None)[0])
                # model.close()

                # print(f'keyBin: {keyBin}')
                # print("mismatch:", np.sum(np.array(a_list_number) != np.array(b_list_number)))
                # print(f'a_list_number: {a_list_number}')
                # print(f'b_list_number: {b_list_number}')
                # print(f'e_list_number: {e_list_number}')
                # exit()

                # pulp integer programming
                problem = pulp.LpProblem("Matrix_Constraint", pulp.LpMinimize)
                x = [pulp.LpVariable(f'x{i}', lowBound=0, cat=pulp.LpInteger) for i in range(len(keyBin))]
                problem += pulp.lpSum(x)
                for i in range(len(tmpMulA1)):
                    problem += pulp.lpSum([tmpCSIa1IndPerm[i][j] * x[j] for j in range(len(keyBin))]) == tmpMulA1[i]
                problem.solve(pulp.PULP_CBC_CMD(msg=False))
                a_list_number = [pulp.value(i) for i in x]

                problem = pulp.LpProblem("Matrix_Constraint", pulp.LpMinimize)
                x = [pulp.LpVariable(f'x{i}', lowBound=0, cat=pulp.LpInteger) for i in range(len(keyBin))]
                problem += pulp.lpSum(x)
                for i in range(len(tmpMulA1)):
                    problem += pulp.lpSum([tmpCSIb1IndPerm[i][j] * x[j] for j in range(len(keyBin))]) == tmpMulA1[i]
                problem.solve(pulp.PULP_CBC_CMD(msg=False))
                b_list_number = [pulp.value(i) for i in x]

                problem = pulp.LpProblem("Matrix_Constraint", pulp.LpMinimize)
                x = [pulp.LpVariable(f'x{i}', lowBound=0, cat=pulp.LpInteger) for i in range(len(keyBin))]
                problem += pulp.lpSum(x)
                for i in range(len(tmpMulA1)):
                    problem += pulp.lpSum([tmpCSIe1IndPerm[i][j] * x[j] for j in range(len(keyBin))]) == tmpMulA1[i]
                problem.solve(pulp.PULP_CBC_CMD(msg=False))
                e_list_number = [pulp.value(i) for i in x]

                # pulp regularized perturbed least squares
                # alpha = 0.2
                # problem = pulp.LpProblem("Regularized_Least_Squares", pulp.LpMinimize)
                # # x = [pulp.LpVariable(f'x{i}', lowBound=0, upBound=3, cat=pulp.LpInteger) for i in range(len(keyBin))]
                # x = [pulp.LpVariable(f'x{i}', lowBound=0, upBound=3) for i in range(len(keyBin))]
                # residuals = [pulp.LpVariable(f'r{i}', lowBound=0) for i in range(len(keyBin))]
                # problem += pulp.lpSum(residuals) + alpha * pulp.lpSum(x)
                # for i in range(len(tmpMulA1)):
                #     problem += pulp.lpSum([tmpCSIa1IndPerm[i][j] * x[j] for j in range(len(keyBin))]) \
                #                - residuals[i] == tmpMulA1[i]
                # problem.solve(pulp.PULP_CBC_CMD(msg=False))
                # a_list_number = [pulp.value(i) for i in x]
                #
                # problem = pulp.LpProblem("Regularized_Least_Squares", pulp.LpMinimize)
                # # x = [pulp.LpVariable(f'x{i}', lowBound=0, upBound=3, cat=pulp.LpInteger) for i in range(len(keyBin))]
                # x = [pulp.LpVariable(f'x{i}', lowBound=0, upBound=3) for i in range(len(keyBin))]
                # residuals = [pulp.LpVariable(f'r{i}', lowBound=0) for i in range(len(keyBin))]
                # problem += pulp.lpSum(residuals) + alpha * pulp.lpSum(x)
                # for i in range(len(tmpMulA1)):
                #     problem += pulp.lpSum([tmpCSIb1IndPerm[i][j] * x[j] for j in range(len(keyBin))]) \
                #                - residuals[i] == tmpMulA1[i]
                # problem.solve(pulp.PULP_CBC_CMD(msg=False))
                # b_list_number = [pulp.value(i) for i in x]
                #
                # problem = pulp.LpProblem("Regularized_Least_Squares", pulp.LpMinimize)
                # # x = [pulp.LpVariable(f'x{i}', lowBound=0, upBound=3, cat=pulp.LpInteger) for i in range(len(keyBin))]
                # x = [pulp.LpVariable(f'x{i}', lowBound=0, upBound=3) for i in range(len(keyBin))]
                # residuals = [pulp.LpVariable(f'r{i}', lowBound=0) for i in range(len(keyBin))]
                # problem += pulp.lpSum(residuals) + alpha * pulp.lpSum(x)
                # for i in range(len(tmpMulA1)):
                #     problem += pulp.lpSum([tmpCSIe1IndPerm[i][j] * x[j] for j in range(len(keyBin))]) \
                #                - residuals[i] == tmpMulA1[i]
                # problem.solve(pulp.PULP_CBC_CMD(msg=False))
                # e_list_number = [pulp.value(i) for i in x]

                # a_list_number = list(np.linalg.lstsq(tmpCSIa1IndPerm, tmpMulA1, rcond=None)[0])
                # b_list_number = list(np.linalg.lstsq(tmpCSIb1IndPerm, tmpMulA1, rcond=None)[0])
                # e_list_number = list(np.linalg.lstsq(tmpCSIe1IndPerm, tmpMulA1, rcond=None)[0])

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