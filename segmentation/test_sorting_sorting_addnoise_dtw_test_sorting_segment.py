import copy
import hashlib
import itertools
import math
import random
import sys
import time
from collections import Counter
from datetime import datetime

import graycode
import scipy.stats
from dtaidistance import dtw_ndim
from scipy.optimize import minimize

from segmentation.methods import find_opt_segment_method, segment_sequence, compute_min_dtw, \
    find_sub_opt_segment_method_down, find_sub_opt_segment_method_up, find_opt_segment_method_from_candidate, \
    search_segment_method_with_matching, find_segments, find_all_cover_intervals_iter_up, get_segment_lengths, \
    find_all_cover_intervals_iter, search_segment_method_with_min_match_dtw, find_special_array, step_discrete_corr, \
    find_peaks_in_segments, search_segment_method, search_segment_method_with_offset, find_min_threshold, find_offset, \
    dtw_metric, find_plateau_length, generate_segment_lengths, normal2uniform, smooth, compute_threshold, \
    diff_sq_integral_rough, find_opt_segment_method_cond, search_segment_method_with_min_cross_dtw, \
    find_special_array_min_mean_var, search_segment_method_with_min_dtw, find_special_array_min_min_mean_var, \
    search_index_with_segment, find_special_array_mean_var, find_sub_opt_segment_method_down_cond, \
    find_special_array_min_mean, find_sub_opt_segment_method_merge, find_sub_opt_segment_method_merge_heap_fast, \
    find_gaps, merge_gaps, find_covering_intervals, find_sub_opt_segment_method_sliding_threshold, \
    find_sub_opt_segment_method_sliding, find_out_of_range_intervals, find_before_intervals, compute_all_dtw, \
    common_pca, find_sub_opt_shuffle_method_sliding_threshold, find_best_matching_pair, \
    find_best_matching_pair_filter, find_best_matching_pair_shuffled, find_sub_opt_shuffle_method_sliding, \
    find_best_matching_pair_threshold, find_best_matching_pair_threshold_filter, \
    find_all_sub_opt_shuffle_method_sliding, find_sub_opt_shuffle_method_sliding_threshold_random, \
    find_sub_opt_shuffle_method_sliding_threshold_even, find_all_matching_pairs_threshold, compute_max_min_euclidean, \
    combine_indices_iteratively, calculate_distances, inverse_permutation

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr, boxcox, spearmanr, kurtosis, skew

from segmentation.test_partition import partition

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def writeToFile():
    # 获取当前时间戳，并生成文件名
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"./output/output_{timestamp}.txt"

    # 保存当前脚本内容到文件
    script_filename = __file__
    with open(script_filename, 'r', encoding='utf-8') as script_file:
        script_content = script_file.read()

    with open(filename, 'w', encoding='utf-8') as script_backup_file:
        script_backup_file.write(script_content)

    # 使用Logger类将标准输出重定向到文件和控制台
    sys.stdout = Logger(filename)

# 将当前代码和结果写入文件
# writeToFile()

# fileName = ["../data/data_mobile_indoor_1.mat",
#             "../data/data_mobile_outdoor_1.mat",
#             "../data/data_static_outdoor_1.mat",
#             "../data/data_static_indoor_1.mat"
#             ]

# 基于自适应分段索引匹配的密钥生成
fileName = ["../data/data_mobile_indoor_1.mat"]

isPrint = False
isUp = False
isCoxBox = True
isCorrect = True
isDegrade = False
withIndexValue = True
withProb = False
withNewMatrix = True
withNewMatrix2 = False
withNewMatrix3 = False
withNewMatrix4 = False

isShuffle = False
isSegSort = False
isValueShuffle = False

# 是否纠错
rec = False

segLen = 4
segNum = 4
keyLen = segNum * segLen
tell = True

# 是否排序
# withoutSorts = [True, False]
withoutSorts = [False]
# 是否添加噪声
# addNoises = ["pca", "mul", "add", ""]
addNoises = ["mul"]
segment_option = "find_search"
print(segment_option)

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

            if f == "../data/data_NLOS.mat":
                # 先整体shuffle一次
                seed = np.random.randint(100000)
                np.random.seed(seed)
                shuffleInd = np.random.permutation(len(CSIa1Orig))
                CSIa1Orig = CSIa1Orig[shuffleInd]
                CSIb1Orig = CSIb1Orig[shuffleInd]

            # 为了使用cox-box处理，将滤波放在总循环前面，结果与old版本的有略微的不一致
            CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
            CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')
            CSIe1Orig = np.random.normal(np.mean(CSIb1Orig), np.std(CSIb1Orig), len(CSIa1Orig))

            if isCoxBox:
                CSIa1Orig = (CSIa1Orig - np.min(CSIa1Orig)) / (np.max(CSIa1Orig) - np.min(CSIa1Orig))
                CSIb1Orig = (CSIb1Orig - np.min(CSIb1Orig)) / (np.max(CSIb1Orig) - np.min(CSIb1Orig))
                CSIe1Orig = (CSIe1Orig - np.min(CSIe1Orig)) / (np.max(CSIe1Orig) - np.min(CSIe1Orig))
                CSIa1Orig = scipy.stats.boxcox(np.abs(CSIa1Orig) + 1e-4)[0]
                CSIb1Orig = scipy.stats.boxcox(np.abs(CSIb1Orig) + 1e-4)[0]
                CSIa1Orig = normal2uniform(CSIa1Orig) * 2
                CSIb1Orig = normal2uniform(CSIb1Orig) * 2
                CSIa1Orig = np.array(CSIa1Orig)
                CSIb1Orig = np.array(CSIb1Orig)
                CSIe1Orig = np.random.normal(np.mean(CSIb1Orig), np.std(CSIb1Orig), len(CSIa1Orig))

            dataLen = len(CSIa1Orig)
            print("dataLen", dataLen)

            print("segLen", segLen)
            print("keyLen", keyLen / segLen)

            indices = list(itertools.permutations(range(4)))

            originSum = 0
            correctSum = 0
            randomSum = 0

            originDecSum = 0
            correctDecSum = 0
            randomDecSum = 0

            originWholeSum = 0
            correctWholeSum = 0
            randomWholeSum = 0

            originSegSum = 0
            correctSegSum = 0
            randomSegSum = 0

            segmentMaxDist = []
            evenMaxDist = []
            badSegments = 0

            times = 0
            overhead = 0

            optimize_time = []
            shuffling_time = []
            search_time = []

            if withoutSort:
                if addNoise != "":
                    print("no sorting")
                elif addNoise == "":
                    print("no sorting and no perturbation")
            else:
                if addNoise == "":
                    print("no perturbation")
                elif addNoise == "mul":
                    print("normal mul")
                elif addNoise == "add":
                    print("normal add")
            if isShuffle:
                print("with shuffle")
            if isSegSort:
                print("with segSort")

            dataLenLoop = dataLen
            keyLenLoop = keyLen

            isRunBack = False
            runBackCounts = 0

            shuffle_patterns = []
            if f == "../data/data_static_indoor_1.mat":
                dataLenLoop = int(dataLen / 5.5)
                keyLenLoop = int(keyLen / 5)
            staInd = -keyLenLoop

            indices_counters = {}
            indices_counters_B = {}
            while staInd < dataLenLoop:
                staInd += keyLenLoop
                start = time.time_ns()
                keyLen = segNum * segLen
                endInd = staInd + keyLen
                if endInd >= len(CSIa1Orig):
                    break

                # 如果有错误，则退化成原始分段
                findError = False

                tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
                tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
                tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]

                if np.isclose(np.max(tmpCSIa1), np.min(tmpCSIa1)) or np.isclose(np.max(tmpCSIb1), np.min(tmpCSIb1)):
                    continue

                if isRunBack:
                    # 随机置换
                    # np.random.seed(staInd + runBackCounts)
                    # combineCSIx1Orig = list(zip(tmpCSIa1, tmpCSIb1, tmpCSIe1))
                    # np.random.shuffle(combineCSIx1Orig)
                    # tmpCSIa1, tmpCSIb1, tmpCSIe1 = zip(*combineCSIx1Orig)
                    # tmpCSIa1 = np.array(tmpCSIa1)
                    # tmpCSIb1 = np.array(tmpCSIb1)
                    # tmpCSIe1 = np.array(tmpCSIe1)

                    # 乘随机矩阵
                    np.random.seed(staInd + runBackCounts)
                    randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
                    tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
                    tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)
                    tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix)

                    runBackCounts += 1
                else:
                    runBackCounts = 0

                # 目的是把加噪音+无排序的结果降下来
                if addNoise == "mul" and isRunBack is False:
                    np.random.seed(10000)
                    randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
                    # randomMatrix = np.random.uniform(0, 1, size=(keyLen, keyLen))
                    # randomMatrix = np.random.normal(0, 1, size=(keyLen, keyLen))
                    # 均值化
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
                    tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
                    tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)
                    tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix)

                else:
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    tmpCSIe1 = np.random.normal(np.mean(tmpCSIa1), np.std(tmpCSIa1), keyLen)

                optimize_start = time.time_ns()
                # 寻找最优噪声矩阵
                if withNewMatrix:
                    resCSIa1 = [0] * int(keyLen / 2) + [1] * int(keyLen / 2)
                    np.random.seed(staInd)
                    resCSIa1 = np.random.permutation(resCSIa1)

                    # 无影响
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)

                    randomMatrix = np.linalg.lstsq(np.array(tmpCSIa1).reshape(1, -1), np.array(resCSIa1).reshape(1, -1), rcond=None)[0]
                    randomMatrix = randomMatrix - np.mean(randomMatrix)
                    randomMatrix = (randomMatrix - np.min(randomMatrix)) / (np.max(randomMatrix) - np.min(randomMatrix))
                    np.random.seed(staInd)
                    randomMatrix += np.random.uniform(0, 1, (keyLen, keyLen))
                    # randomMatrix += np.random.uniform(0, 0.5, (keyLen, keyLen))
                    # randomMatrix += np.random.uniform(np.mean(CSIa1Orig) - np.std(CSIa1Orig), np.mean(CSIa1Orig) + np.std(CSIa1Orig), (keyLen, keyLen))
                    # randomMatrix += np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
                    # randomMatrix += np.random.normal(0, 0.5, size=(keyLen, keyLen))
                    # randomMatrix += np.random.normal(0, 5, size=(keyLen, keyLen))
                    tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
                    tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)
                    tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix)

                    # 无影响
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)

                # 寻找最优噪声矩阵
                if withNewMatrix2:
                    # 准确率高，但攻击者也会猜到
                    resCSIa1 = [0] * int(keyLen / 2) + [1] * int(keyLen / 2)
                    np.random.seed(staInd)
                    resCSIa1 = np.random.permutation(resCSIa1)

                    # 无影响
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)

                    randomMatrixA = np.linalg.lstsq(np.array(tmpCSIa1).reshape(1, -1), np.array(resCSIa1).reshape(1, -1), rcond=None)[0]
                    randomMatrixA = randomMatrixA - np.mean(randomMatrixA)
                    randomMatrixA = (randomMatrixA - np.min(randomMatrixA)) / (np.max(randomMatrixA) - np.min(randomMatrixA))
                    randomMatrixB = np.linalg.lstsq(np.array(tmpCSIb1).reshape(1, -1), np.array(resCSIa1).reshape(1, -1), rcond=None)[0]
                    randomMatrixB = randomMatrixB - np.mean(randomMatrixB)
                    randomMatrixB = (randomMatrixB - np.min(randomMatrixB)) / (np.max(randomMatrixB) - np.min(randomMatrixB))
                    randomMatrixE = np.linalg.lstsq(np.array(tmpCSIe1).reshape(1, -1), np.array(resCSIa1).reshape(1, -1), rcond=None)[0]
                    randomMatrixE = randomMatrixE - np.mean(randomMatrixE)
                    randomMatrixE = (randomMatrixE - np.min(randomMatrixE)) / (np.max(randomMatrixE) - np.min(randomMatrixE))
                    tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrixA)
                    tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrixB)
                    tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrixE)

                    # 无影响
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)

                # 寻找最优噪声矩阵
                if withNewMatrix3:
                    # optimize time segment time total search time
                    # 2.303344322 0.004338428 0.002073774
                    # (9718 / 9720 = 0.999794239) 0.999794239 0.999176955 0.5 0.499897119  /  2.303344322 0.004338428 0.002073774
                    # 使用向量化操作替换循环来计算欧式距离
                    def sum_distance(solution1, solution2):
                        solution1 = np.array(solution1)
                        solution2 = np.array(solution2)
                        diff = solution1[:, np.newaxis, :] - solution2[np.newaxis, :, :]
                        distances = np.sum(diff ** 2, axis=2)
                        total = np.sum(distances)
                        return total

                    # 定义目标函数
                    def objective(K, A, n):
                        K = K.reshape(n, n)  # 将 K 重塑为 n x n 矩阵
                        B = A @ K  # B = A * K, 这里 B 是一个 1 x n 的向量

                        # 对 B 进行均值化和归一化
                        B = B - np.mean(B)
                        B = (B - np.min(B)) / (np.max(B) - np.min(B))

                        sum_bi_squared = np.sum(B ** 2)
                        sum_bi = np.sum(B)
                        return -(n * sum_bi_squared - sum_bi ** 2)  # 因为我们使用最小化，所以取负数


                    # 初始猜测的矩阵 K
                    initial_K = np.random.normal(0, 1, size=(keyLen, keyLen)).flatten()  # 生成一个 n x n 的初始 K，并展平成一维数组
                    optimal_K = minimize(objective, initial_K, args=(tmpCSIa1, keyLen), method='l-bfgs-b').x.reshape(keyLen, keyLen)
                    tmpCSIa1 = np.matmul(tmpCSIa1, optimal_K)
                    optimal_K = minimize(objective, initial_K, args=(tmpCSIb1, keyLen), method='l-bfgs-b').x.reshape(keyLen, keyLen)
                    tmpCSIb1 = np.matmul(tmpCSIb1, optimal_K)
                    optimal_K = minimize(objective, initial_K, args=(tmpCSIe1, keyLen), method='l-bfgs-b').x.reshape(keyLen, keyLen)
                    tmpCSIe1 = np.matmul(tmpCSIe1, optimal_K)

                    # 无影响
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)

                # 寻找最优噪声矩阵
                if withNewMatrix4:
                    # 过慢
                    def objective(K_flat, A):
                        # 将展平的 K 恢复为 N x N 矩阵
                        N = len(A)
                        K = K_flat.reshape(N, N)

                        # 计算 B = A @ K
                        B = np.dot(A, K)

                        # 对 B 进行均值化和归一化
                        B = B - np.mean(B)
                        B = (B - np.min(B)) / (np.max(B) - np.min(B))

                        # 计算最小欧式距离之和
                        min_distance_sum = 0
                        for i in range(len(B)):
                            distances = np.linalg.norm(B[i] - np.delete(B, i))
                            min_distance_sum += np.min(distances)

                        # 我们需要最大化目标函数，因此返回其负值
                        return -min_distance_sum


                    def constraint(K_flat, A):
                        # 将展平的 K 恢复为 N x N 矩阵
                        N = len(A)
                        K = K_flat.reshape(N, N)

                        # 计算 B = A @ K
                        B = np.dot(A, K)

                        # 约束：B 的所有元素应在 [0, 1] 范围内
                        return np.concatenate((B - 0, 1 - B))


                    def optimize_matrix(A, method):
                        N = len(A)
                        # 初始猜测的 K
                        initial_K = np.random.rand(N, N)
                        initial_K_flat = initial_K.flatten()

                        # 定义约束条件
                        cons = ({'type': 'ineq', 'fun': constraint, 'args': (A,)})

                        # 优化目标函数，包含边界和约束条件
                        if method == 'slsqp' or method == 'trust-constr':
                            result = minimize(objective, initial_K_flat, args=(A,), method=method, constraints=cons)
                        else:
                            result = minimize(objective, initial_K_flat, args=(A,), method=method)

                        # 将优化后的 K 恢复为 N x N 矩阵
                        K_optimal = result.x.reshape(N, N)

                        return K_optimal

                    K_optimal = optimize_matrix(tmpCSIa1, 'l-bfgs-b')
                    tmpCSIa1 = np.dot(tmpCSIa1, K_optimal)
                    K_optimal = optimize_matrix(tmpCSIb1, 'l-bfgs-b')
                    tmpCSIb1 = np.dot(tmpCSIb1, K_optimal)
                    K_optimal = optimize_matrix(tmpCSIe1, 'l-bfgs-b')
                    tmpCSIe1 = np.dot(tmpCSIe1, K_optimal)

                    # 无影响
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)

                optimize_end = time.time_ns()
                optimize_time.append((optimize_end - optimize_start) / 1e9)

                # 最后各自的密钥
                a_list = []
                b_list = []
                e_list = []

                # without sorting
                if withIndexValue:
                    # 将测量值和其索引结合成二维数组
                    tmpCSIa1Index = np.array(tmpCSIa1).argsort().argsort()
                    tmpCSIb1Index = np.array(tmpCSIb1).argsort().argsort()
                    tmpCSIe1Index = np.array(tmpCSIe1).argsort().argsort()

                    # 将index和value放缩到同样的区间内
                    tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1))
                    tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1))
                    tmpCSIe1 = (tmpCSIe1 - np.min(tmpCSIe1)) / (np.max(tmpCSIe1) - np.min(tmpCSIe1))
                    tmpCSIa1Index = (tmpCSIa1Index - np.min(tmpCSIa1Index)) / (np.max(tmpCSIa1Index) - np.min(tmpCSIa1Index))
                    tmpCSIb1Index = (tmpCSIb1Index - np.min(tmpCSIb1Index)) / (np.max(tmpCSIb1Index) - np.min(tmpCSIb1Index))
                    tmpCSIe1Index = (tmpCSIe1Index - np.min(tmpCSIe1Index)) / (np.max(tmpCSIe1Index) - np.min(tmpCSIe1Index))

                    tmpCSIa1Ind = np.array(list(zip(tmpCSIa1, tmpCSIa1Index)))
                    tmpCSIb1Ind = np.array(list(zip(tmpCSIb1, tmpCSIb1Index)))
                    tmpCSIe1Ind = np.array(list(zip(tmpCSIe1, tmpCSIe1Index)))
                    tmpCSIa1Prod = tmpCSIa1 * tmpCSIa1Index
                    tmpCSIb1Prod = tmpCSIb1 * tmpCSIb1Index
                else:
                    if withoutSort:
                        tmpCSIa1Ind = np.array(tmpCSIa1)
                        tmpCSIb1Ind = np.array(tmpCSIb1)
                        tmpCSIe1Ind = np.array(tmpCSIe1)
                    else:
                        tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
                        tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()
                        tmpCSIe1Ind = np.array(tmpCSIe1).argsort().argsort()

                minEpiIndClosenessLsb = np.zeros(int(keyLen / segLen), dtype=int)
                minEpiIndClosenessLse = np.zeros(int(keyLen / segLen), dtype=int)
                # with segSort
                if isSegSort:
                    if withoutSort is False:
                        for i in range(int(keyLen / segLen)):
                            epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]
                            epiIndb1 = tmpCSIb1Ind[i * segLen:(i + 1) * segLen]

                            np.random.seed(i)
                            combineEpiIndx1 = list(zip(epiInda1, epiIndb1))
                            np.random.shuffle(combineEpiIndx1)
                            epiInda1, epiIndb1 = zip(*combineEpiIndx1)

                            tmpCSIa1Ind[i * segLen:(i + 1) * segLen] = epiInda1
                            tmpCSIb1Ind[i * segLen:(i + 1) * segLen] = epiIndb1
                        # print(pearsonr(tmpCSIa1Ind, tmpCSIb1Ind)[0])

                if segment_option == "find_search":
                    start_time = time.time_ns()
                    shuffling_start = time.time_ns()

                    # 滑动窗口分段
                    if withIndexValue:
                        threshold = 1
                    else:
                        if withoutSort:
                            threshold = 0.2
                        else:
                            threshold = 15
                    # 设置阈值
                    # shuffle_method_ori = find_sub_opt_shuffle_method_sliding_threshold(
                    #     tmpCSIa1Ind, tmpCSIa1Ind, 4, indices, threshold)
                    # shuffle_method_ori = find_sub_opt_shuffle_method_sliding_threshold_random(
                    #     tmpCSIa1Ind, tmpCSIa1Ind, 4, indices, threshold)
                    shuffle_method_ori, indices_counters = find_sub_opt_shuffle_method_sliding_threshold_even(
                        tmpCSIa1Ind, tmpCSIa1Ind, 4, indices, indices_counters, threshold)
                    print(staInd, shuffle_method_ori)
                    # indices_counters = dict(sorted(indices_counters.items(), key=lambda x: x[1], reverse=True))
                    # print(staInd, indices_counters)

                    # 不设置阈值
                    # shuffle_method_ori = find_sub_opt_shuffle_method_sliding(tmpCSIa1Ind, tmpCSIa1Ind, 4, indices)
                    # shuffle_method_ori = find_all_sub_opt_shuffle_method_sliding(tmpCSIa1Ind, tmpCSIa1Ind, 4, indices)
                    shuffling_end = time.time_ns()
                    shuffling_time.append((shuffling_end - shuffling_start) / 1e9)
                    for index in shuffle_method_ori:
                        shuffle_patterns.append("".join(np.array(index).astype(str)))

                    # if isRunBack is False:
                    #     print("staInd", staInd, "shuffle_method", shuffle_method_ori)

                    if withIndexValue:
                        tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIa1Ind))
                        tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIb1Ind))
                        tmpCSIe1IndReshape = np.array(tmpCSIe1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIe1Ind))
                    else:
                        tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)
                        tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(int(keyLen / segLen), segLen)
                        tmpCSIe1IndReshape = np.array(tmpCSIe1Ind).reshape(int(keyLen / segLen), segLen)
                    tmpCSIa1IndReshape = [list(tmpCSIa1IndReshape[i][list(shuffle_method_ori[i])]) for i in
                                          range(len(tmpCSIa1IndReshape))]

                    # tmpCSIa1EvenSegment = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIa1Ind))
                    # dist_opt_seg = compute_min_dtw(tmpCSIa1IndReshape, tmpCSIa1IndReshape)
                    # dist_even_seg = compute_min_dtw(tmpCSIa1EvenSegment, tmpCSIa1EvenSegment)
                    # if (dist_opt_seg < dist_even_seg):
                    #     badSegments += 1
                    #     tmpCSIa1IndReshape = tmpCSIa1EvenSegment
                    #     segment_method_ori = [segLen] * int(keyLen / segLen)
                    #     min_length = min(segment_method_ori)
                    #     max_length = max(segment_method_ori)
                    #     num_segments = len(segment_method_ori)
                    #     measurements_len = sum(segment_method_ori)
                    # segmentMaxDist.append(dist_opt_seg)
                    # evenMaxDist.append(dist_even_seg)
                elif segment_option == "sub_window":
                    start_time = time.time_ns()
                    if withIndexValue:
                        threshold = 1
                    else:
                        if withoutSort:
                            threshold = 0.2
                        else:
                            threshold = 15
                    # shuffle_method = find_sub_opt_shuffle_method_sliding_threshold(
                    #     tmpCSIa1Ind, tmpCSIa1Ind, 4, indices, threshold)
                    shuffle_method, indices_counters = find_sub_opt_shuffle_method_sliding_threshold_even(
                        tmpCSIa1Ind, tmpCSIa1Ind, 4, indices, indices_counters, threshold)
                    # 不设定阈值的准确率更高
                    # shuffle_method = find_sub_opt_shuffle_method_sliding(
                    #     tmpCSIa1Ind, tmpCSIa1Ind, 4, indices)
                    # 穷举第一个分段置换顺序的准确率最高，但速度也最慢
                    # shuffle_method = find_all_sub_opt_shuffle_method_sliding(
                    #     tmpCSIa1Ind, tmpCSIa1Ind, 4, indices)
                    print("staInd", staInd, "shuffle_method", shuffle_method)

                    for index in shuffle_method:
                        shuffle_patterns.append("".join(np.array(index).astype(str)))

                    if withIndexValue:
                        tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIa1Ind))
                        tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIb1Ind))
                        tmpCSIe1IndReshape = np.array(tmpCSIe1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIe1Ind))
                    else:
                        tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)
                        tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(int(keyLen / segLen), segLen)
                        tmpCSIe1IndReshape = np.array(tmpCSIe1Ind).reshape(int(keyLen / segLen), segLen)

                    tmpCSIa1IndReshape = [list(tmpCSIa1IndReshape[i][list(shuffle_method[i])]) for i in
                                          range(len(tmpCSIa1IndReshape))]
                    tmpCSIb1IndReshape = [list(tmpCSIb1IndReshape[i][list(shuffle_method[i])]) for i in
                                          range(len(tmpCSIb1IndReshape))]
                    tmpCSIe1IndReshape = [list(tmpCSIe1IndReshape[i][list(shuffle_method[i])]) for i in
                                          range(len(tmpCSIe1IndReshape))]
                elif segment_option == "sub_window_ind":
                    start_time = time.time_ns()
                    if withIndexValue:
                        threshold = 1
                    else:
                        if withoutSort:
                            threshold = 0.2
                        else:
                            threshold = 15
                    shuffle_method_A = find_sub_opt_shuffle_method_sliding_threshold(
                        tmpCSIa1Ind, tmpCSIa1Ind, 4, indices, threshold)
                    shuffle_method_B = find_sub_opt_shuffle_method_sliding_threshold(
                        tmpCSIb1Ind, tmpCSIb1Ind, 4, indices, threshold)
                    # shuffle_method_A = find_sub_opt_shuffle_method_sliding(
                    #     tmpCSIa1Ind, tmpCSIa1Ind, 4, indices)
                    # shuffle_method_B = find_sub_opt_shuffle_method_sliding(
                    #     tmpCSIb1Ind, tmpCSIb1Ind, 4, indices)
                    print("staInd", staInd, "shuffle_method", shuffle_method_A, shuffle_method_B)

                    if withIndexValue:
                        tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIa1Ind))
                        tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIb1Ind))
                        tmpCSIe1IndReshape = np.array(tmpCSIe1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIe1Ind))
                    else:
                        tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)
                        tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(int(keyLen / segLen), segLen)
                        tmpCSIe1IndReshape = np.array(tmpCSIe1Ind).reshape(int(keyLen / segLen), segLen)

                    tmpCSIa1IndReshape = [list(tmpCSIa1IndReshape[i][list(shuffle_method_A[i])]) for i in
                                          range(len(tmpCSIa1IndReshape))]
                    tmpCSIb1IndReshape = [list(tmpCSIb1IndReshape[i][list(shuffle_method_B[i])]) for i in
                                          range(len(tmpCSIb1IndReshape))]
                    tmpCSIe1IndReshape = [list(tmpCSIe1IndReshape[i][list(shuffle_method_A[i])]) for i in
                                          range(len(tmpCSIe1IndReshape))]
                else:
                    start_time = time.time_ns()
                    # 原方法：固定分段
                    if withIndexValue:
                        tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIa1Ind))
                        tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIb1Ind))
                        tmpCSIe1IndReshape = np.array(tmpCSIe1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIe1Ind))
                    else:
                        tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)
                        tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(int(keyLen / segLen), segLen)
                        tmpCSIe1IndReshape = np.array(tmpCSIe1Ind).reshape(int(keyLen / segLen), segLen)

                tmpCSIa1Bck = tmpCSIa1Ind.copy()
                permutation = list(range(int(keyLen / segLen)))
                combineMetric = list(zip(tmpCSIa1IndReshape, permutation))
                np.random.seed(staInd)
                np.random.shuffle(combineMetric)
                tmpCSIa1IndReshape, permutation = zip(*combineMetric)
                if withIndexValue:
                    tmpCSIa1IndReshape = np.array(tmpCSIa1IndReshape).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIa1Ind))
                else:
                    tmpCSIa1IndReshape = np.array(tmpCSIa1IndReshape).reshape(int(keyLen / segLen), segLen)

                if segment_option == "find_search":
                    search_start = time.time_ns()
                    tmpCSIa1BckReshape = np.array(tmpCSIa1Bck).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIa1Ind))
                    # threshold = compute_max_min_euclidean(tmpCSIa1BckReshape, tmpCSIb1IndReshape)
                    threshold = 0.1
                    print("threshold", threshold)

                    def product(subarray):
                        # 计算子分段内元素的乘积
                        from functools import reduce
                        import operator
                        return reduce(operator.mul, subarray, 1)

                    # 按照某种顺序先进行匹配，提升运行速度
                    # for i in range(len(tmpCSIa1IndReshape)):
                    #     tmpCSIa1IndReshape[i] = np.sort(tmpCSIa1IndReshape[i], axis=0)
                    #     tmpCSIb1IndReshape[i] = np.sort(tmpCSIb1IndReshape[i], axis=0)
                    #     tmpCSIe1IndReshape[i] = np.sort(tmpCSIe1IndReshape[i], axis=0)
                    #     tmpCSIa1BckReshape[i] = np.sort(tmpCSIa1BckReshape[i], axis=0)
                    #
                    #     # 准确率更高
                    #     # tmpCSIa1IndReshape[i] = np.sort(tmpCSIa1IndReshape[i], axis=1)
                    #     # tmpCSIb1IndReshape[i] = np.sort(tmpCSIb1IndReshape[i], axis=1)
                    #     # tmpCSIe1IndReshape[i] = np.sort(tmpCSIe1IndReshape[i], axis=1)
                    #     # tmpCSIa1BckReshape[i] = np.sort(tmpCSIa1BckReshape[i], axis=1)
                    #
                    #     # tmpCSIa1IndReshape[i] = sorted(tmpCSIa1IndReshape[i], key=lambda subarray: np.sum(subarray))
                    #     # tmpCSIb1IndReshape[i] = sorted(tmpCSIb1IndReshape[i], key=lambda subarray: np.sum(subarray))
                    #     # tmpCSIe1IndReshape[i] = sorted(tmpCSIe1IndReshape[i], key=lambda subarray: np.sum(subarray))
                    #     # tmpCSIa1BckReshape[i] = sorted(tmpCSIa1BckReshape[i], key=lambda subarray: np.sum(subarray))
                    #
                    #     # tmpCSIa1IndReshape[i] = sorted(tmpCSIa1IndReshape[i], key=lambda subarray: product(subarray))
                    #     # tmpCSIb1IndReshape[i] = sorted(tmpCSIb1IndReshape[i], key=lambda subarray: product(subarray))
                    #     # tmpCSIe1IndReshape[i] = sorted(tmpCSIe1IndReshape[i], key=lambda subarray: product(subarray))
                    #     # tmpCSIa1BckReshape[i] = sorted(tmpCSIa1BckReshape[i], key=lambda subarray: product(subarray))
                    # indices_a = find_best_matching_pair_shuffled(tmpCSIa1IndReshape, tmpCSIa1BckReshape)
                    # indices_b = find_best_matching_pair_shuffled(tmpCSIa1IndReshape, tmpCSIb1IndReshape)
                    # indices_e = find_best_matching_pair_shuffled(tmpCSIa1IndReshape, tmpCSIe1IndReshape)

                    # indices_a, shuffle_methods_a = find_best_matching_pair(tmpCSIa1IndReshape, tmpCSIa1BckReshape)
                    # indices_b, shuffle_methods_b = find_best_matching_pair(tmpCSIa1IndReshape, tmpCSIb1IndReshape)
                    # indices_e, shuffle_methods_e = find_best_matching_pair(tmpCSIa1IndReshape, tmpCSIe1IndReshape)
                    # 滤重
                    # indices_a, shuffle_methods_a = find_best_matching_pair_filter(tmpCSIa1IndReshape, tmpCSIa1BckReshape)
                    # indices_b, shuffle_methods_b = find_best_matching_pair_filter(tmpCSIa1IndReshape, tmpCSIb1IndReshape)
                    # indices_e, shuffle_methods_e = find_best_matching_pair_filter(tmpCSIa1IndReshape, tmpCSIe1IndReshape)
                    # 阈值
                    # indices_a, shuffle_methods_a = find_best_matching_pair_threshold(tmpCSIa1IndReshape, tmpCSIa1BckReshape, threshold)
                    # indices_b, shuffle_methods_b = find_best_matching_pair_threshold(tmpCSIa1IndReshape, tmpCSIb1IndReshape, threshold)
                    # indices_e, shuffle_methods_e = find_best_matching_pair_threshold(tmpCSIa1IndReshape, tmpCSIe1IndReshape, threshold)
                    # 阈值滤重
                    # indices_a, shuffle_methods_a = find_best_matching_pair_threshold_filter(tmpCSIa1IndReshape, tmpCSIa1BckReshape, threshold)
                    # indices_b, shuffle_methods_b = find_best_matching_pair_threshold_filter(tmpCSIa1IndReshape, tmpCSIb1IndReshape, threshold)
                    # indices_e, shuffle_methods_e = find_best_matching_pair_threshold_filter(tmpCSIa1IndReshape, tmpCSIe1IndReshape, threshold)

                    # 阈值 找出所有配对
                    indices_a, shuffle_methods_a = find_all_matching_pairs_threshold(tmpCSIa1IndReshape, tmpCSIa1BckReshape, threshold)
                    indices_b, shuffle_methods_b = find_all_matching_pairs_threshold(tmpCSIa1IndReshape, tmpCSIb1IndReshape, threshold)
                    indices_e, shuffle_methods_e = find_all_matching_pairs_threshold(tmpCSIa1IndReshape, tmpCSIe1IndReshape, threshold)

                    all_indices_a = combine_indices_iteratively(indices_a)
                    all_shuffle_methods_a = combine_indices_iteratively(shuffle_methods_a)
                    all_indices_b = combine_indices_iteratively(indices_b)
                    all_shuffle_methods_b = combine_indices_iteratively(shuffle_methods_b)
                    all_indices_e = combine_indices_iteratively(indices_e)
                    all_shuffle_methods_e = combine_indices_iteratively(shuffle_methods_e)

                    a_dists = [calculate_distances(tmpCSIa1IndReshape, tmpCSIa1BckReshape, index) for index in all_indices_a]
                    b_dists = [calculate_distances(tmpCSIa1IndReshape, tmpCSIb1IndReshape, index) for index in all_indices_b]
                    e_dists = [calculate_distances(tmpCSIa1IndReshape, tmpCSIe1IndReshape, index) for index in all_indices_e]

                    a_list_numbers = [[index[1] for index in indices] for indices in all_indices_a]
                    b_list_numbers = [[index[1] for index in indices] for indices in all_indices_b]
                    e_list_numbers = [[index[1] for index in indices] for indices in all_indices_e]

                    # a_list_number = a_list_numbers[np.argmin(a_dists)]
                    # key_hash = hash(tuple(a_list_number))
                    # for b_list_number in b_list_numbers:
                    #     if key_hash != hash(tuple(b_list_number)):
                    #         key_counter = dict(Counter(b_list_number))
                    #         most_common_keys = []
                    #         for key, value in key_counter.items():
                    #             if value > 1:
                    #                 most_common_keys.append(key)
                    #         most_common_keys_indices = []
                    #         for key in most_common_keys:
                    #             for i in range(len(b_list_number)):
                    #                 if b_list_number[i] == key:
                    #                     most_common_keys_indices.append(i)
                    #         a_list_number_short = []
                    #         most_common_keys_indices = np.sort(most_common_keys_indices)
                    #         for i in range(len(most_common_keys_indices)):
                    #             a_list_number_short.append(a_list_number[most_common_keys_indices[i]])
                    #         key_hash_short = hash(tuple(a_list_number_short))
                    #         b_list_number_short = a_list_number_short
                    #         b_list_number_short.reverse()
                    #         for i in range(len(b_list_number)):
                    #             if i in most_common_keys_indices:
                    #                 b_list_number[i] = b_list_number_short.pop()

                    # 滤重
                    # for i in range(len(a_list_numbers) - 1, -1, -1):
                    #     if np.array_equal(np.sort(a_list_numbers[i]), list(range(len(a_list_numbers[i])))) is False:
                    #         del a_list_numbers[i]
                    #         del a_dists[i]
                    # for i in range(len(b_list_numbers) - 1, -1, -1):
                    #     if np.array_equal(np.sort(b_list_numbers[i]), list(range(len(b_list_numbers[i])))) is False:
                    #         del b_list_numbers[i]
                    #         del b_dists[i]
                    # for i in range(len(e_list_numbers) - 1, -1, -1):
                    #     if np.array_equal(np.sort(e_list_numbers[i]), list(range(len(e_list_numbers[i])))) is False:
                    #         del e_list_numbers[i]
                    #         del e_dists[i]

                    a_list_number = []
                    b_list_number = []
                    e_list_number = []

                    if len(a_dists) != 0:
                        a_list_number = a_list_numbers[np.argmin(a_dists)]
                    if len(b_dists) != 0:
                        b_list_number = b_list_numbers[np.argmin(b_dists)]
                    if len(e_dists) != 0:
                        e_list_number = e_list_numbers[np.argmin(e_dists)]

                    search_end = time.time_ns()
                    search_time.append((search_end - search_start) / 1e9)

                    if a_list_number != b_list_number:
                        # 大概率与原来的不同
                        # print(shuffle_method_ori, np.array(all_shuffle_methods_a)[[np.argmin(a_dists)]])
                        # print("不匹配原来置换")

                        key_hash = hash(tuple(a_list_number))
                        for i in range(len(b_list_numbers)):
                            if key_hash == hash(tuple(b_list_numbers[i])):
                                b_list_number = b_list_numbers[i]
                                break

                    # indices_a = [index[1] for index in indices_a]
                    # indices_b = [index[1] for index in indices_b]
                    # indices_e = [index[1] for index in indices_e]
                    #
                    # a_list_number = list(indices_a)
                    # b_list_number = list(indices_b)
                    # e_list_number = list(indices_e)

                    key_hash = hash(tuple(a_list_number))
                    if key_hash != hash(tuple(b_list_number)):
                        if isPrint:
                            print("self correction")
                        key_counter = dict(Counter(b_list_number))
                        most_common_keys = []
                        for key, value in key_counter.items():
                            if value > 1:
                                most_common_keys.append(key)
                        most_common_keys_indices = []
                        for key in most_common_keys:
                            for i in range(len(b_list_number)):
                                if b_list_number[i] == key:
                                    most_common_keys_indices.append(i)
                        a_list_number_short = []
                        most_common_keys_indices = np.sort(most_common_keys_indices)
                        for i in range(len(most_common_keys_indices)):
                            a_list_number_short.append(a_list_number[most_common_keys_indices[i]])
                        key_hash_short = hash(tuple(a_list_number_short))
                        b_list_number_short = a_list_number_short
                        b_list_number_short.reverse()
                        for i in range(len(b_list_number)):
                            if i in most_common_keys_indices:
                                b_list_number[i] = b_list_number_short.pop()

                    if isPrint:
                        if a_list_number != b_list_number:
                            print("staInd", staInd, "error")
                        else:
                            print("staInd", staInd)
                        # print("a", "segment_method", shuffle_method_ori, a_list_number)
                        # print("a", "segment_method", shuffle_methods_a, a_list_number)
                        # print("b", "segment_method", shuffle_methods_b, b_list_number)
                        print()

                    # self correction
                    if isCorrect:
                        # 自行纠错重复的部分
                        key_hash = hash(tuple(a_list_number))
                        if key_hash != hash(tuple(b_list_number)):
                            if isPrint:
                                print("self correction")
                            key_counter = dict(Counter(b_list_number))
                            most_common_keys = []
                            for key, value in key_counter.items():
                                if value > 1:
                                    most_common_keys.append(key)
                            most_common_keys_indices = []
                            for key in most_common_keys:
                                for i in range(len(b_list_number)):
                                    if b_list_number[i] == key:
                                        most_common_keys_indices.append(i)
                            a_list_number_short = []
                            most_common_keys_indices = np.sort(most_common_keys_indices)
                            for i in range(len(most_common_keys_indices)):
                                a_list_number_short.append(a_list_number[most_common_keys_indices[i]])
                            key_hash_short = hash(tuple(a_list_number_short))
                            # used_keys = []
                            # for i in range(len(b_list_number)):
                            #     if i not in most_common_keys_indices:
                            #         used_keys.append(b_list_number[i])
                            # unused_keys = list(set(list(range(int(keyLen / segLen)))) - set(used_keys))
                            # short_indices = list(itertools.permutations(range(len(unused_keys))))
                            # b_list_number_short = unused_keys
                            # for i in range(len(short_indices)):
                            #     b_list_number_short = list(np.array(b_list_number_short)[list(short_indices[i])])
                            #     if hash(tuple(b_list_number_short)) == key_hash_short:
                            #         if isPrint:
                            #             print("self correction finished")
                            #         break
                            b_list_number_short = a_list_number_short
                            b_list_number_short.reverse()
                            for i in range(len(b_list_number)):
                                if i in most_common_keys_indices:
                                    b_list_number[i] = b_list_number_short.pop()

                        # correction
                        key_hash = hash(tuple(a_list_number))
                        if key_hash != hash(tuple(b_list_number)):
                            # filter
                            indices_b_filter, shuffle_methods_b = find_best_matching_pair_filter(tmpCSIa1IndReshape, tmpCSIb1IndReshape)
                            indices_b_filter = [index[1] for index in indices_b_filter]
                            b_list_number = list(indices_b_filter)
                            # ind
                            if key_hash != hash(tuple(b_list_number)):
                                if withIndexValue:
                                    threshold = 1
                                else:
                                    if withoutSort:
                                        threshold = 0.2
                                    else:
                                        threshold = 15
                                shuffle_methods_b = find_sub_opt_shuffle_method_sliding_threshold(
                                    tmpCSIb1Ind, tmpCSIb1Ind, 4, indices, threshold)
                                # shuffle_methods_b, indices_counters_B = find_sub_opt_shuffle_method_sliding_threshold_even(
                                #     tmpCSIb1Ind, tmpCSIb1Ind, 4, indices, indices_counters_B, threshold)
                                tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIb1Ind))
                                tmpCSIb1IndReshape = [list(tmpCSIb1IndReshape[i][list(shuffle_methods_b[i])]) for i in
                                                      range(len(tmpCSIb1IndReshape))]
                                indices_b_ind = find_best_matching_pair_shuffled(tmpCSIa1IndReshape, tmpCSIb1IndReshape)
                                b_list_number = list(indices_b_ind)
                            if isPrint:
                                print("correction")
                                if a_list_number != b_list_number:
                                    print("staInd", staInd, "error again")
                                else:
                                    print("staInd", staInd)
                                print("a", a_list_number, "segment_method", shuffle_method_ori)
                                print("b", b_list_number, "segment_method", shuffle_methods_b)
                                print()

                        # 回退多次
                        if a_list_number != b_list_number and runBackCounts < 5:
                            if isPrint:
                                print("回退" + str(runBackCounts) + "次")
                            isRunBack = True
                            staInd -= keyLenLoop
                            continue
                        elif a_list_number != b_list_number and isRunBack is True:
                            runBackCounts = 0
                            isRunBack = False
                        else:
                            runBackCounts = 0
                            isRunBack = False

                        if a_list_number == b_list_number:
                            runBackCounts = 0
                else:
                    for i in range(int(keyLen / segLen)):
                        epiInda1 = tmpCSIa1IndReshape[i]

                        epiIndClosenessLsb = np.zeros(int(keyLen / segLen))
                        epiIndClosenessLse = np.zeros(int(keyLen / segLen))

                        for j in range(int(keyLen / segLen)):
                            epiIndb1 = tmpCSIb1IndReshape[j]
                            epiInde1 = tmpCSIe1IndReshape[j]

                            # 欧式距离度量更好
                            epiIndClosenessLsb[j] = np.sum(np.square(epiIndb1 - np.array(epiInda1)))
                            # epiIndClosenessLsb[j] = dtw_metric(epiIndb1, np.array(epiInda1))
                            # epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))
                            # epiIndClosenessLsb[j] = distance.cosine(epiIndb1, np.array(epiInda1))
                            # epiIndClosenessLsb[j] = abs(sum(epiIndb1) - sum(np.array(epiInda1)))

                            epiIndClosenessLse[j] = np.sum(np.square(epiInde1 - np.array(epiInda1)))
                            # epiIndClosenessLse[j] = dtw_metric(epiInde1, np.array(epiInda1))
                            # epiIndClosenessLse[j] = sum(abs(epiInde1 - np.array(epiInda1)))

                        minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)
                        minEpiIndClosenessLse[i] = np.argmin(epiIndClosenessLse)

                    a_list_number = list(permutation)
                    b_list_number = list(minEpiIndClosenessLsb)
                    e_list_number = list(minEpiIndClosenessLse)

                # 转成二进制
                max_digit = str(math.ceil(np.log2(len(a_list_number))))
                for i in range(len(a_list_number)):
                    a_list += str('{:0' + max_digit + 'b}').format(graycode.tc_to_gray_code(a_list_number[i]))
                for i in range(len(b_list_number)):
                    b_list += str('{:0' + max_digit + 'b}').format(graycode.tc_to_gray_code(b_list_number[i]))
                for i in range(len(e_list_number)):
                    e_list += str('{:0' + max_digit + 'b}').format(graycode.tc_to_gray_code(e_list_number[i]))

                # for i in range(len(a_list_number)):
                #     number = bin(a_list_number[i])[2:].zfill(int(np.log2(len(a_list_number))))
                #     a_list += number
                # for i in range(len(b_list_number)):
                #     number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
                #     b_list += number

                print("a", a_list)
                print("b", b_list)
                print("e", e_list)
                sum1 = min(len(a_list), len(b_list))
                sum2 = 0
                sum3 = 0

                for i in range(0, sum1):
                    sum2 += (a_list[i] == b_list[i])

                for i in range(min(len(a_list), len(e_list))):
                    sum3 += (a_list[i] == e_list[i])

                end = time.time_ns()
                overhead += end - start
                # 从分段开始记时

                # 自适应纠错
                if sum1 != sum2 and rec:
                    if tell:
                        # print("correction")
                        # a告诉b哪些位置出错，b对其纠错
                        for i in range(len(a_list_number)):
                            if a_list_number[i] != b_list_number[i]:
                                epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]

                                epiIndClosenessLsb = np.zeros(int(keyLen / segLen))

                                for j in range(int(keyLen / segLen)):
                                    epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
                                    epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))

                                # 第一个找到的错误的，将其距离置为最大，下次找到的就是第二个，作为正确结果
                                min_b = np.argmin(epiIndClosenessLsb)
                                epiIndClosenessLsb[min_b] = keyLen * segLen
                                b_list_number[i] = np.argmin(epiIndClosenessLsb)

                                b_list = []

                                max_digit = str(math.ceil(np.log2(len(b_list_number))))
                                for i in range(len(b_list_number)):
                                    b_list += str('{:0' + max_digit + 'b}').format(
                                        graycode.tc_to_gray_code(b_list_number[i]))

                                # for i in range(len(b_list_number)):
                                #     number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
                                #     b_list += number

                                sum2 = 0
                                for i in range(0, min(len(a_list), len(b_list))):
                                    sum2 += (a_list[i] == b_list[i])

                # print("key", len(a_list), a_list)

                # print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
                originSum += sum1
                correctSum += sum2
                randomSum += sum3

                originWholeSum += 1
                correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
                randomWholeSum = randomWholeSum + 1 if sum3 == sum1 else randomWholeSum

                times += 1

            print("\033[0;34;40ma-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 9), "\033[0m")
            print("\033[0;34;40ma-b whole match", correctWholeSum, "/", originWholeSum, "=",
                  round(correctWholeSum / originWholeSum, 9), "\033[0m")
            print("\033[0;34;40ma-e all", randomSum, "/", originSum, "=", round(randomSum / originSum, 9), "\033[0m")
            print("\033[0;34;40ma-e whole match", randomWholeSum, "/", originWholeSum, "=",
                  round(randomWholeSum / originWholeSum, 9), "\033[0m")

            print(round(correctSum / originSum, 9), round(correctWholeSum / originWholeSum, 9),
                  round(originSum / times / keyLen, 9),
                  round(correctSum / times / keyLen, 9))
            # 分段匹配的情况
            if originDecSum != 0:
                print("\033[0;32;40ma-b seg", correctSegSum, "/", originSegSum, "=",
                      round(correctSegSum / originSegSum, 9), "\033[0m")
            if segmentMaxDist != []:
                print("segmented max distance", np.mean(segmentMaxDist), np.std(segmentMaxDist))
                print("even max distance", np.mean(evenMaxDist), np.std(evenMaxDist))
                print("badSegments", badSegments / times)
            if withoutSort:
                print("withoutSort")
            else:
                print("withSort")
            if withIndexValue:
                print("withIndexValue")
            else:
                print("withoutIndexValue")
            if len(shuffling_time) != 0:
                print("optimize time", "segment time", "total search time")
                # 求AB进行搜索的平均时间
                print(np.round(np.mean(optimize_time), 9),
                      np.round(np.mean(shuffling_time), 9),
                      np.round(np.mean(search_time) / 3, 9))

                print("(" + str(correctSum), "/", originSum, "=", str(round(correctSum / originSum, 9)) + ")",
                      round(correctSum / originSum, 9), round(correctWholeSum / originWholeSum, 9),
                      round(originSum / times / keyLen, 9),
                      round(correctSum / times / keyLen, 9), " / ",
                      np.round(np.mean(optimize_time), 9),
                      np.round(np.mean(shuffling_time), 9),
                      np.round(np.mean(search_time) / 3, 9))
            print("shuffle_patterns", Counter(shuffle_patterns))
            print("\n")

            # 画出置换类型占比图
            # all_experimental_segments = dict(Counter(shuffle_patterns))
            # # 计算总分段个数
            # total_experimental = 0
            # for counter in all_experimental_segments:
            #     total_experimental += all_experimental_segments[counter]
            #
            # experimental_ratios = {key: value / total_experimental for key, value in
            #                        all_experimental_segments.items()}
            # experimental_ratios = dict(sorted(experimental_ratios.items(), key=lambda x: x[0]))
            #
            # experimental_labels = list(experimental_ratios.keys())
            # experimental_sizes = list(experimental_ratios.values())
            #
            # plt.figure(figsize=(8, 6))
            # plt.bar(experimental_labels, experimental_sizes, color='orange')
            # plt.xlabel('Shuffle Pattern')
            # plt.ylabel('Proportion')
            # plt.xticks(rotation=45)
            # plt.title('Experimental Shuffle Pattern Proportions 4 * ' + str(segNum))
            # plt.tight_layout()
            # plt.show()
