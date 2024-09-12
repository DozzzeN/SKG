import hashlib
import math
import random
import sys
import time
from datetime import datetime

import graycode
import scipy.stats
from dtaidistance import dtw_ndim

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
    common_pca, optimize_random_matrix_max, optimize_random_matrix_max_min, search_random_matrix_max_min, \
    search_random_matrix_uniform

from optimization.test_gen_matrix import generate_matrix_and_solve_diag
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
isCoxBox = False
isCorrect = False
isDegrade = False
isGray = True
withIndexValue = False
withProb = False

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
# 最优分段opt/opt_pair，次优分段sub_down/sub_down_pair，各自分段ind，合并分段sub_up/sub_up_pair
# 随机分段rdm，随机分段且限制最小分段数rdm_cond，发送打乱后的数据然后推测分段
# 通过自相关确定分段类型，再根据公布的数据进行分段匹配snd
# 不根据相关系数只根据数据来进行搜索find
# 固定分段类型fix，双方从固定分段类型里面挑选
# 根据公布的索引搜索分段find_search/find_search_pair
segment_option = ""
print(segment_option)
# match, cross, both
search_method = "match"
print(search_method)

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

            segment_time = []
            search_time = []
            final_search_time = []
            total_time = []

            search_space = []

            value_mean_pearson_ab = []
            value_mean_spearman_ab = []
            value_mean_pearson_ae = []
            value_mean_spearman_ae = []

            index_mean_pearson_ab = []
            index_mean_spearman_ab = []
            index_mean_pearson_ae = []
            index_mean_spearman_ae = []

            both_mean_pearson_ab = []
            prob_mean_pearson_ab = []

            both_mean_with_matrix_pearson_ae = []

            # 所有子区间个数
            all_intervals_number = []
            # 所有间隙个数
            all_gaps_number = []
            # 所有子区间长度
            all_intervals_length = []
            # 所有间隙长度
            all_gaps_length = []
            # 存在间隙的个数
            have_gaps_number = 0
            # 总密钥数
            all_key_number = 0

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
            if f == "../data/data_static_indoor_1.mat":
                dataLenLoop = int(dataLen / 5.5)
                keyLenLoop = int(keyLen / 5)
            staInd = -keyLenLoop
            # for staInd in range(0, dataLenLoop, keyLenLoop):
            while staInd < dataLenLoop:
                staInd += keyLenLoop
                start = time.time_ns()
                keyLen = segNum * segLen
                endInd = staInd + keyLen
                # print("range:", staInd, endInd)
                if endInd >= len(CSIa1Orig):
                    break

                # 有些情况被跳过，故放在密钥生成后进行计数
                # times += 1

                # 如果有错误，则退化成原始分段
                findError = False

                tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
                tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
                tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]


                # 根据值过滤
                # if np.isclose(np.max(tmpCSIa1), np.min(tmpCSIa1)) or np.isclose(np.max(tmpCSIb1), np.min(tmpCSIb1)):
                #     continue

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

                # tmpCSIa1 = np.array(diff_sq_integral_rough(tmpCSIa1))
                # tmpCSIb1 = np.array(diff_sq_integral_rough(tmpCSIb1))
                # tmpCSIe1 = np.array(diff_sq_integral_rough(tmpCSIe1))

                # tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1))
                # tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1))
                # tmpCSIe1 = (tmpCSIe1 - np.min(tmpCSIe1)) / (np.max(tmpCSIe1) - np.min(tmpCSIe1))
                # tmpCSIa1 = scipy.stats.boxcox(np.abs(tmpCSIa1) + 1e-4)[0]
                # tmpCSIb1 = scipy.stats.boxcox(np.abs(tmpCSIb1) + 1e-4)[0]
                # tmpCSIe1 = scipy.stats.boxcox(np.abs(tmpCSIe1) + 1e-4)[0]
                # tmpCSIa1 = normal2uniform(tmpCSIa1) * 2
                # tmpCSIb1 = normal2uniform(tmpCSIb1) * 2
                # tmpCSIe1 = normal2uniform(tmpCSIe1) * 2
                # tmpCSIa1 = np.array(tmpCSIa1)
                # tmpCSIb1 = np.array(tmpCSIb1)
                # tmpCSIe1 = np.array(tmpCSIe1)

                # 目的是把加噪音+无排序的结果降下来
                if addNoise == "add" and isRunBack is False:
                    np.random.seed(10000)
                    randomMatrix = np.random.uniform(0, 1, keyLen)
                    # 均值化
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    # tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1))
                    # tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1))

                    tmpCSIe1 = np.random.normal(np.mean(tmpCSIa1), np.std(tmpCSIa1), keyLen)
                    tmpCSIa1 = tmpCSIa1 + randomMatrix
                    tmpCSIb1 = tmpCSIb1 + randomMatrix
                    tmpCSIe1 = tmpCSIe1 + randomMatrix
                elif addNoise == "mul" and isRunBack is False:
                    # randomMatrix = np.random.randint(0, 2, size=(keyLen, keyLen))
                    # randomMatrix = np.random.uniform(0, 1, size=(keyLen, keyLen))
                    np.random.seed(10000)
                    randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
                    # randomMatrix = np.random.uniform(0, 1, size=(keyLen, keyLen))
                    # randomMatrix = np.random.normal(0, 1, size=(keyLen, keyLen))
                    # 均值化
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)

                    # 再次标准化+均值化的结果不变
                    # tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1))
                    # tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1))
                    # tmpCSIe1 = (tmpCSIe1 - np.min(tmpCSIe1)) / (np.max(tmpCSIe1) - np.min(tmpCSIe1))
                    # # 第二次均值化保证数据的均值为0，否则攻击者的数据和用户的强相关
                    # tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    # tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    # tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)

                    tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
                    tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)
                    tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix)

                    # 乘性扰动后均值化或标准化则无影响
                    # tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    # tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    # tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
                    # tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1))
                    # tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1))
                    # tmpCSIe1 = (tmpCSIe1 - np.min(tmpCSIe1)) / (np.max(tmpCSIe1) - np.min(tmpCSIe1))

                    # print("corr of a and e", pearsonr(tmpCSIa1, tmpCSIe1)[0])
                    # 散点图
                    # plt.figure()
                    # combinetmpCSIa1 = list(zip(tmpCSIa1, list(range(keyLen))))
                    # combinetmpCSIa1 = sorted(combinetmpCSIa1, key=lambda x: x[0])
                    # tmpCSIa1, indexa1 = zip(*combinetmpCSIa1)
                    # combinetmpCSIb1 = list(zip(tmpCSIb1, list(range(keyLen))))
                    # combinetmpCSIb1 = sorted(combinetmpCSIb1, key=lambda x: x[0])
                    # tmpCSIb1, indexb1 = zip(*combinetmpCSIb1)
                    # print("indexa1", indexa1)
                    # print("indexb1", indexb1)
                    # plt.scatter(list(range(keyLen)), np.array(tmpCSIa1)[list(indexa1)])
                    # plt.scatter(list(range(keyLen)), np.array(tmpCSIb1)[list(indexb1)])
                    # plt.show()
                    # exit()

                    # np.random.seed(10000)
                    # randomMatrix = np.random.uniform(0, 1, keyLen)
                    # tmpCSIe1 = np.random.normal(np.mean(tmpCSIa1), np.std(tmpCSIa1), keyLen)
                    # tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1))
                    # tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1))
                    # tmpCSIe1 = (tmpCSIe1 - np.min(tmpCSIe1)) / (np.max(tmpCSIe1) - np.min(tmpCSIe1))
                    # tmpCSIa1 = tmpCSIa1 * randomMatrix
                    # tmpCSIb1 = tmpCSIb1 * randomMatrix
                    # tmpCSIe1 = tmpCSIe1 * randomMatrix
                    # tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    # tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    # tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)

                    # 相当于乘了一个置换矩阵 permutation matrix
                    # np.random.seed(0)
                    # combineCSIx1Orig = list(zip(tmpCSIa1, tmpCSIb1))
                    # np.random.shuffle(combineCSIx1Orig)
                    # tmpCSIa1, tmpCSIb1 = zip(*combineCSIx1Orig)
                    # tmpCSIa1 = np.array(tmpCSIa1)
                    # tmpCSIb1 = np.array(tmpCSIb1)
                elif addNoise == "pca" and isRunBack is False:
                    factor = 3 / 4
                    keyLen = int(segNum * segLen * factor)
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)

                    tmpCSIa1Reshape = np.array(tmpCSIa1).reshape(int(np.sqrt(segNum * segLen)), int(np.sqrt(segNum * segLen)))
                    tmpCSIb1Reshape = np.array(tmpCSIb1).reshape(int(np.sqrt(segNum * segLen)), int(np.sqrt(segNum * segLen)))
                    tmpCSIe1Reshape = np.array(tmpCSIe1).reshape(int(np.sqrt(segNum * segLen)), int(np.sqrt(segNum * segLen)))
                    # pca = PCA(n_components=16)
                    # tmpCSIa1 = pca.fit_transform(tmpCSIa1Reshape).reshape(1, -1)[0]
                    # tmpCSIb1 = pca.fit_transform(tmpCSIb1Reshape).reshape(1, -1)[0]
                    # tmpCSIe1 = pca.fit_transform(tmpCSIe1Reshape).reshape(1, -1)[0]
                    tmpCSIa1, tmpCSIb1, tmpCSIe1 = common_pca(tmpCSIa1Reshape, tmpCSIb1Reshape, tmpCSIe1Reshape, int(np.sqrt(segNum * segLen) * factor))
                    tmpCSIa1 = tmpCSIa1.reshape(1, -1)[0]
                    tmpCSIb1 = tmpCSIb1.reshape(1, -1)[0]
                    tmpCSIe1 = tmpCSIe1.reshape(1, -1)[0]
                else:
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    # tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1))
                    # tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1))
                    tmpCSIe1 = np.random.normal(np.mean(tmpCSIa1), np.std(tmpCSIa1), keyLen)

                # 乘完矩阵以后进行cox-box处理
                # plt.figure()
                # plt.plot(tmpCSIa1)
                # plt.plot(tmpCSIb1)
                # plt.show()
                # tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1))
                # tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1))
                # tmpCSIe1 = (tmpCSIe1 - np.min(tmpCSIe1)) / (np.max(tmpCSIe1) - np.min(tmpCSIe1))
                # tmpCSIa1 = scipy.stats.boxcox(np.abs(tmpCSIa1) + 1e-4)[0]
                # tmpCSIb1 = scipy.stats.boxcox(np.abs(tmpCSIb1) + 1e-4)[0]
                # tmpCSIe1 = scipy.stats.boxcox(np.abs(tmpCSIe1) + 1e-4)[0]
                # tmpCSIa1 = normal2uniform(tmpCSIa1) * 2
                # tmpCSIb1 = normal2uniform(tmpCSIb1) * 2
                # tmpCSIe1 = normal2uniform(tmpCSIe1) * 2
                # tmpCSIa1 = np.array(tmpCSIa1)
                # tmpCSIb1 = np.array(tmpCSIb1)
                # tmpCSIe1 = np.array(tmpCSIe1)
                # plt.figure()
                # plt.plot(tmpCSIa1)
                # plt.plot(tmpCSIb1)
                # plt.show()

                # tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1)) + 1e-4
                # tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1)) + 1e-4
                # tmpCSIe1 = (tmpCSIe1 - np.min(tmpCSIe1)) / (np.max(tmpCSIe1) - np.min(tmpCSIe1)) + 1e-4
                # tmpCSIa1 = scipy.stats.boxcox(np.abs(tmpCSIa1))[0]
                # tmpCSIb1 = scipy.stats.boxcox(np.abs(tmpCSIb1))[0]
                # tmpCSIe1 = scipy.stats.boxcox(np.abs(tmpCSIe1))[0]
                # tmpCSIa1 = normal2uniform(tmpCSIa1) * 2
                # tmpCSIb1 = normal2uniform(tmpCSIb1) * 2
                # tmpCSIe1 = normal2uniform(tmpCSIe1) * 2

                # 最后各自的密钥
                a_list = []
                b_list = []
                e_list = []

                # with value shuffling
                if isValueShuffle:
                    np.random.seed(10000)
                    combineCSIx1Orig = list(zip(tmpCSIa1, tmpCSIb1))
                    np.random.shuffle(combineCSIx1Orig)
                    tmpCSIa1, tmpCSIb1 = zip(*combineCSIx1Orig)
                    tmpCSIa1 = list(tmpCSIa1)
                    tmpCSIb1 = list(tmpCSIb1)

                # without sorting
                # print(pearsonr(tmpCSIa1, tmpCSIb1)[0])
                if withIndexValue:
                    # 将测量值和其索引结合成二维数组
                    tmpCSIa1Index = np.array(tmpCSIa1).argsort().argsort()
                    tmpCSIb1Index = np.array(tmpCSIb1).argsort().argsort()
                    tmpCSIe1Index = np.array(tmpCSIe1).argsort().argsort()

                    index_mean_pearson_ab.append(pearsonr(tmpCSIa1Index, tmpCSIb1Index)[0])
                    value_mean_pearson_ab.append(pearsonr(tmpCSIa1, tmpCSIb1)[0])

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
                    both_mean_pearson_ab.append(
                        pearsonr(np.array(tmpCSIa1Ind).flatten(), np.array(tmpCSIb1Ind).flatten())[0])

                    tmpCSIa1Prod = tmpCSIa1 * tmpCSIa1Index
                    tmpCSIb1Prod = tmpCSIb1 * tmpCSIb1Index
                    prob_mean_pearson_ab.append(pearsonr(tmpCSIa1Prod, tmpCSIb1Prod)[0])
                elif withProb:
                    tmpCSIa1Index = np.array(tmpCSIa1).argsort().argsort()
                    tmpCSIb1Index = np.array(tmpCSIb1).argsort().argsort()
                    tmpCSIe1Index = np.array(tmpCSIe1).argsort().argsort()

                    tmpCSIa1Ind = tmpCSIa1 * tmpCSIa1Index
                    tmpCSIb1Ind = tmpCSIb1 * tmpCSIb1Index
                    tmpCSIe1Ind = tmpCSIe1 * tmpCSIe1Index
                else:
                    if withoutSort:
                        tmpCSIa1Ind = np.array(tmpCSIa1)
                        tmpCSIb1Ind = np.array(tmpCSIb1)
                        tmpCSIe1Ind = np.array(tmpCSIe1)
                    else:
                        tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
                        tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()
                        tmpCSIe1Ind = np.array(tmpCSIe1).argsort().argsort()

                        # with shuffling
                        if isShuffle:
                            np.random.seed(0)
                            combineCSIx1Orig = list(zip(tmpCSIa1Ind, tmpCSIb1Ind))
                            np.random.shuffle(combineCSIx1Orig)
                            tmpCSIa1Ind, tmpCSIb1Ind = zip(*combineCSIx1Orig)
                            tmpCSIa1Ind = list(tmpCSIa1Ind)
                            tmpCSIb1Ind = list(tmpCSIb1Ind)
                            # print(pearsonr(tmpCSIa1Ind, tmpCSIb1Ind)[0])

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

                start_time = time.time_ns()
                # 原方法：固定分段
                if withIndexValue:
                    tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(
                        int(keyLen / segLen), segLen, np.ndim(tmpCSIa1Ind))
                    tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(
                        int(keyLen / segLen), segLen, np.ndim(tmpCSIb1Ind))
                    tmpCSIe1IndReshape = np.array(tmpCSIe1Ind).reshape(
                        int(keyLen / segLen), segLen, np.ndim(tmpCSIe1Ind))
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
                    tmpCSIa1Ind = np.vstack(tmpCSIa1IndReshape)
                else:
                    tmpCSIa1Ind = np.hstack((tmpCSIa1IndReshape))

                for i in range(int(keyLen / segLen)):
                    epiInda1 = tmpCSIa1IndReshape[i]

                    epiIndClosenessLsb = np.zeros(int(keyLen / segLen))
                    epiIndClosenessLse = np.zeros(int(keyLen / segLen))

                    for j in range(int(keyLen / segLen)):
                        epiIndb1 = tmpCSIb1IndReshape[j]
                        epiInde1 = tmpCSIe1IndReshape[j]

                        # 欧式距离度量更好
                        # epiIndClosenessLsb[j] = sum(np.square(epiIndb1 - np.array(epiInda1)))
                        epiIndClosenessLsb[j] = dtw_metric(epiIndb1, np.array(epiInda1))
                        # epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))
                        # epiIndClosenessLsb[j] = distance.cosine(epiIndb1, np.array(epiInda1))
                        # epiIndClosenessLsb[j] = abs(sum(epiIndb1) - sum(np.array(epiInda1)))

                        epiIndClosenessLse[j] = dtw_metric(epiInde1, np.array(epiInda1))
                        # epiIndClosenessLse[j] = sum(abs(epiInde1 - np.array(epiInda1)))

                    minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)
                    minEpiIndClosenessLse[i] = np.argmin(epiIndClosenessLse)

                a_list_number = list(permutation)
                b_list_number = list(minEpiIndClosenessLsb)
                e_list_number = list(minEpiIndClosenessLse)

                total_time.append((time.time_ns() - start_time) / 1e9)

                # print("a", a_list_number)
                # print("b", b_list_number)
                # print("e", e_list_number)

                # 转成二进制
                if isGray:
                    max_digit = str(math.ceil(np.log2(len(a_list_number))))
                    for i in range(len(a_list_number)):
                        a_list += str('{:0' + max_digit + 'b}').format(graycode.tc_to_gray_code(a_list_number[i]))
                    for i in range(len(b_list_number)):
                        b_list += str('{:0' + max_digit + 'b}').format(graycode.tc_to_gray_code(b_list_number[i]))
                    for i in range(len(e_list_number)):
                        e_list += str('{:0' + max_digit + 'b}').format(graycode.tc_to_gray_code(e_list_number[i]))
                else:
                    for i in range(len(a_list_number)):
                        number = bin(a_list_number[i])[2:].zfill(int(np.log2(len(a_list_number))))
                        a_list += number
                    for i in range(len(b_list_number)):
                        number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
                        b_list += number
                    for i in range(len(e_list_number)):
                        number = bin(e_list_number[i])[2:].zfill(int(np.log2(len(e_list_number))))
                        e_list += number

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
                total_time.append((end - start_time) / 1e9)
                # print("time:", end - start)

                # print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
                originSum += sum1
                correctSum += sum2
                randomSum += sum3

                originWholeSum += 1
                correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
                randomWholeSum = randomWholeSum + 1 if sum3 == sum1 else randomWholeSum

                times += 1

            print()
            # print("value_mean_pearson_ab", np.mean(value_mean_pearson_ab))
            # print("value_mean_pearson_ae", np.mean(value_mean_pearson_ae))
            # print("value_mean_spearman_ab", np.mean(value_mean_spearman_ab))
            # print("value_mean_spearman_ae", np.mean(value_mean_spearman_ae))
            # print("index_mean_pearson_ab", np.mean(index_mean_pearson_ab))
            # print("index_mean_pearson_ae", np.mean(index_mean_pearson_ae))
            # print("index_mean_spearman_ab", np.mean(index_mean_spearman_ab))
            # print("index_mean_spearman_ae", np.mean(index_mean_spearman_ae))
            print("both_mean_pearson_ab", np.mean(both_mean_pearson_ab))
            # print("prob_mean_pearson_ab", np.mean(prob_mean_pearson_ab))
            print("both_mean_with_matrix_pearson_ae", np.mean(both_mean_with_matrix_pearson_ae))
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
            if isGray:
                print("with gray code")
            else:
                print("without gray code")
            if len(segment_time) != 0:
                print("segment time", "search time", "final search time")
                print(np.round(np.mean(segment_time), 9), np.round(np.mean(search_time), 9),
                      np.round(np.mean(final_search_time), 9))
                print("search space", np.round(np.mean(search_space), 9))

                print("(" + str(correctSum), "/", originSum, "=", str(round(correctSum / originSum, 9)) + ")",
                      round(correctSum / originSum, 9), round(correctWholeSum / originWholeSum, 9),
                      round(originSum / times / keyLen, 9),
                      round(correctSum / times / keyLen, 9), " / ",
                      np.round(np.mean(segment_time), 9), np.round(np.mean(search_time), 9),
                      np.round(np.mean(final_search_time), 9), " / ",
                      np.round(np.mean(search_space), 9))
            else:
                print("total time", np.round(np.mean(total_time), 9))
            # print("gaps number ratio", sum(all_gaps_number) / sum(all_intervals_number))
            # print("gaps length ratio", sum(all_gaps_length) / sum(all_intervals_length))
            # print("mean gap number and length", np.mean(all_gaps_number), np.mean(all_gaps_length))
            # print("have gaps ratio", have_gaps_number / all_key_number)
            print("\n")
