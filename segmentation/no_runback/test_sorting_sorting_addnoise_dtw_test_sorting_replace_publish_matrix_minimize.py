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
    common_pca, search_random_matrix_max_min, search_random_matrix_uniform, insert_random_numbers, \
    modify_segments_position_compared_with_one_segments, modify_segments_with_even_addition, \
    replace_segments_position_compared_with_all_segments, compute_max_index_dtw, \
    replace_segments_position_compared_with_all_segments_no_limit, \
    replace_segments_position_compared_with_all_segments_no_limit_pair, optimize_random_block_vector_max_min, \
    entropyPerm

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr, boxcox, spearmanr, kurtosis, skew

from segmentation.test_partition import partition
from pyentrp import entropy as ent
import EntropyHub as eh
from tqdm import tqdm

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
    filename = f"../output/output_{timestamp}.txt"

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
fileName = ["../../data/data_mobile_indoor_1.mat"]

isExtraPrint = False
isPrint = False
isUp = False
isCoxBox = True
isCorrect = True
isDegrade = False
isGray = False
withIndexValue = True
withProb = False

isShuffle = False
isSegSort = False
isValueShuffle = False

isMean = True
isFilter = True
isPerm = False
isInterleave = False
isPerturb = False
isLocalPerm = False

# 是否纠错
rec = False

segLen = 4
segNums = [4, 8]
for segNum in segNums:
    print("segNum", segNum)
    keyLen = segNum * segLen
    tell = True

    replace_number = segLen * 2

    # 是否排序
    # withoutSorts = [True, False]
    withoutSorts = [False]
    # 是否添加噪声
    # addNoises = ["pca", "mul", "add", ""]
    addNoises = [""]
    # 最优分段opt/opt_pair，次优分段sub_down/sub_down_pair，各自分段ind，合并分段sub_up/sub_up_pair
    # 随机分段rdm，随机分段且限制最小分段数rdm_cond，发送打乱后的数据然后推测分段
    # 通过自相关确定分段类型，再根据公布的数据进行分段匹配snd
    # 不根据相关系数只根据数据来进行搜索find
    # 固定分段类型fix，双方从固定分段类型里面挑选
    # 根据公布的索引搜索分段find_search/find_search_pair
    segment_option = "find_search"
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

                # 首次置换
                if isPerm:
                    np.random.seed(100000)
                    shuffleInd = np.random.permutation(len(CSIa1Orig))

                    # print(ent.multiscale_entropy(CSIa1Orig / np.max(CSIa1Orig), 3, maxscale=1))
                    CSIa1Orig = CSIa1Orig[shuffleInd]
                    # print(ent.multiscale_entropy(CSIa1Orig / np.max(CSIa1Orig), 3, maxscale=1))
                    CSIb1Orig = CSIb1Orig[shuffleInd]
                    CSIe1Orig = CSIe1Orig[shuffleInd]

                # print(pearsonr(CSIa1Orig, CSIb1Orig)[0])

                # 交织
                if isInterleave:
                    CSIa1Orig, CSIb1Orig, CSIe1Orig = entropyPerm(CSIa1Orig, CSIb1Orig, CSIe1Orig, len(CSIa1Orig), 10000)

                if isPerturb:
                    np.random.seed(10000)
                    randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(len(CSIa1Orig), len(CSIa1Orig)))
                    # 均值化
                    CSIa1Orig = CSIa1Orig - np.mean(CSIa1Orig)
                    CSIb1Orig = CSIb1Orig - np.mean(CSIb1Orig)
                    CSIe1Orig = CSIe1Orig - np.mean(CSIe1Orig)

                    CSIa1Orig = np.matmul(CSIa1Orig, randomMatrix)
                    CSIb1Orig = np.matmul(CSIb1Orig, randomMatrix)
                    CSIe1Orig = np.matmul(CSIe1Orig, randomMatrix)

                if isCoxBox:
                    CSIa1Orig = (CSIa1Orig - np.min(CSIa1Orig)) / (np.max(CSIa1Orig) - np.min(CSIa1Orig))
                    CSIb1Orig = (CSIb1Orig - np.min(CSIb1Orig)) / (np.max(CSIb1Orig) - np.min(CSIb1Orig))
                    CSIe1Orig = (CSIe1Orig - np.min(CSIe1Orig)) / (np.max(CSIe1Orig) - np.min(CSIe1Orig))
                    CSIa1Orig = scipy.stats.boxcox(np.abs(CSIa1Orig) + 1e-4)[0]
                    CSIb1Orig = scipy.stats.boxcox(np.abs(CSIb1Orig) + 1e-4)[0]
                    CSIe1Orig = scipy.stats.boxcox(np.abs(CSIe1Orig) + 1e-4)[0]
                    CSIa1Orig = normal2uniform(CSIa1Orig) * 2
                    CSIb1Orig = normal2uniform(CSIb1Orig) * 2
                    CSIe1Orig = normal2uniform(CSIe1Orig) * 2
                    CSIa1Orig = np.array(CSIa1Orig)
                    CSIb1Orig = np.array(CSIb1Orig)
                    CSIe1Orig = np.array(CSIe1Orig)
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

                before_add_mean_pearson_ab = []
                after_add_mean_pearson_ab = []
                before_add_mean_pearson_ae = []
                after_add_mean_pearson_ae = []

                corr_ab = []
                max_min_dtw_ab = []

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

                entropy_a = []

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

                if f == "../data/data_static_indoor_1.mat":
                    dataLenLoop = int(dataLen / 5.5)
                    keyLenLoop = int(keyLen / 5)
                # staInd = -keyLenLoop
                for staInd in tqdm(range(0, dataLenLoop, keyLenLoop)):
                # while staInd < dataLenLoop:
                #     staInd += keyLenLoop
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

                    if np.isclose(np.max(tmpCSIa1), np.min(tmpCSIa1)) or np.isclose(np.max(tmpCSIb1), np.min(tmpCSIb1)):
                        continue

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
                    if addNoise == "add":
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
                    elif addNoise == "mul":
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

                        # print(pearsonr(tmpCSIa1, tmpCSIb1)[0])
                        tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
                        tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)
                        tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix)
                        # print(pearsonr(tmpCSIa1, tmpCSIb1)[0])

                        if isLocalPerm:
                            np.random.seed(staInd)
                            shuffleInd = np.random.permutation(keyLen)
                            tmpCSIa1 = tmpCSIa1[shuffleInd]
                            tmpCSIb1 = tmpCSIb1[shuffleInd]
                            tmpCSIe1 = tmpCSIe1[shuffleInd]

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
                    elif addNoise == "pca":
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

                        # 局部置换
                        if isLocalPerm:
                            np.random.seed(staInd)
                            shuffleInd = np.random.permutation(keyLen)
                            tmpCSIa1 = tmpCSIa1[shuffleInd]
                            tmpCSIb1 = tmpCSIb1[shuffleInd]
                            tmpCSIe1 = tmpCSIe1[shuffleInd]

                    entropy_a.append(eh.DispEn(tmpCSIa1)[0])
                    corr_ab.append(pearsonr(tmpCSIa1, tmpCSIb1)[0])

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

                    # 随机插入一个分段个数的随机数
                    # before_add_mean_pearson_ab.append(abs(pearsonr(tmpCSIa1, tmpCSIb1)[0]))
                    # tmpCSIa1 = insert_random_numbers(tmpCSIa1, segLen, staInd + 100000)
                    # tmpCSIb1 = insert_random_numbers(tmpCSIb1, segLen, staInd + 100000)
                    # tmpCSIe1 = insert_random_numbers(tmpCSIe1, segLen, staInd + 100000)
                    # keyLen += segLen
                    # after_add_mean_pearson_ab.append(abs(pearsonr(tmpCSIa1, tmpCSIb1)[0]))

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

                    # simulate noisy measurement
                    np.random.seed(staInd)
                    # simMatrix = np.random.normal(0, 1, size=(keyLen, keyLen))
                    # simCSIa1 = np.matmul(tmpCSIa1, simMatrix)
                    # simCSIb1 = np.matmul(tmpCSIb1, simMatrix)
                    # simCSIe1 = np.matmul(tmpCSIe1, simMatrix)
                    addMatrix = np.random.normal(0, 0.1, keyLen)
                    simCSIa1 = tmpCSIa1 + addMatrix
                    simCSIb1 = tmpCSIb1 + addMatrix
                    simCSIe1 = tmpCSIe1 + addMatrix

                    # without sorting
                    # print(pearsonr(tmpCSIa1, tmpCSIb1)[0])
                    if withIndexValue:
                        # 将测量值和其索引结合成二维数组
                        tmpCSIa1Index = np.array(tmpCSIa1).argsort().argsort()
                        tmpCSIb1Index = np.array(tmpCSIb1).argsort().argsort()
                        tmpCSIe1Index = np.array(tmpCSIe1).argsort().argsort()

                        simCSIa1Index = np.array(simCSIa1).argsort().argsort()
                        index_mean_pearson_ab.append(pearsonr(tmpCSIa1Index, tmpCSIb1Index)[0])
                        value_mean_pearson_ab.append(pearsonr(tmpCSIa1, tmpCSIb1)[0])

                        # 将index和value放缩到同样的区间内
                        tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1))
                        tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1))
                        tmpCSIe1 = (tmpCSIe1 - np.min(tmpCSIe1)) / (np.max(tmpCSIe1) - np.min(tmpCSIe1))

                        simCSIa1 = (simCSIa1 - np.min(simCSIa1)) / (np.max(simCSIa1) - np.min(simCSIa1))

                        tmpCSIa1Index = (tmpCSIa1Index - np.min(tmpCSIa1Index)) / (np.max(tmpCSIa1Index) - np.min(tmpCSIa1Index))
                        tmpCSIb1Index = (tmpCSIb1Index - np.min(tmpCSIb1Index)) / (np.max(tmpCSIb1Index) - np.min(tmpCSIb1Index))
                        tmpCSIe1Index = (tmpCSIe1Index - np.min(tmpCSIe1Index)) / (np.max(tmpCSIe1Index) - np.min(tmpCSIe1Index))

                        simCSIa1Index = (simCSIa1Index - np.min(simCSIa1Index)) / (np.max(simCSIa1Index) - np.min(simCSIa1Index))

                        if isMean:
                            tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                            tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                            tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
                            simCSIa1 = simCSIa1 - np.mean(simCSIa1)

                            tmpCSIa1Index = tmpCSIa1Index - np.mean(tmpCSIa1Index)
                            tmpCSIb1Index = tmpCSIb1Index - np.mean(tmpCSIb1Index)
                            tmpCSIe1Index = tmpCSIe1Index - np.mean(tmpCSIe1Index)
                            simCSIa1Index = simCSIa1Index - np.mean(simCSIa1Index)

                        tmpCSIa1Ind = np.array(list(zip(tmpCSIa1, tmpCSIa1Index)))
                        tmpCSIb1Ind = np.array(list(zip(tmpCSIb1, tmpCSIb1Index)))
                        tmpCSIe1Ind = np.array(list(zip(tmpCSIe1, tmpCSIe1Index)))

                        simCSIa1Ind = np.array(list(zip(simCSIa1, simCSIa1Index)))
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
                            simCSIa1Ind = np.array(simCSIa1)
                            simCSIb1Ind = np.array(simCSIb1)
                            simCSIe1Ind = np.array(simCSIe1)
                        else:
                            # print("corr", pearsonr(tmpCSIa1, tmpCSIb1)[0], pearsonr(tmpCSIa1, simCSIa1)[0],
                            #       pearsonr(simCSIa1, simCSIb1)[0])
                            # print("e corr", pearsonr(tmpCSIa1, tmpCSIe1)[0], pearsonr(tmpCSIa1, simCSIe1)[0],
                            #         pearsonr(simCSIa1, simCSIe1)[0])
                            # print("scorr", spearmanr(tmpCSIa1, tmpCSIb1)[0], spearmanr(tmpCSIa1, simCSIa1)[0],
                            #       spearmanr(simCSIa1, simCSIb1)[0])
                            # print("e scorr", spearmanr(tmpCSIa1, tmpCSIe1)[0], spearmanr(tmpCSIa1, simCSIe1)[0],
                            #         spearmanr(simCSIa1, simCSIe1)[0])
                            # value_mean_pearson_ab.append(pearsonr(tmpCSIa1, tmpCSIb1)[0])
                            # value_mean_spearman_ab.append(spearmanr(tmpCSIa1, tmpCSIb1)[0])
                            # value_mean_pearson_ae.append(pearsonr(tmpCSIa1, tmpCSIe1)[0])
                            # value_mean_spearman_ae.append(spearmanr(tmpCSIa1, tmpCSIe1)[0])

                            tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
                            tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()
                            tmpCSIe1Ind = np.array(tmpCSIe1).argsort().argsort()
                            simCSIa1Ind = np.array(simCSIa1).argsort().argsort()
                            simCSIb1Ind = np.array(simCSIb1).argsort().argsort()
                            simCSIe1Ind = np.array(simCSIe1).argsort().argsort()

                            # print("corr", pearsonr(tmpCSIa1Ind, tmpCSIb1Ind)[0], pearsonr(tmpCSIa1Ind, simCSIa1Ind)[0],
                            #       pearsonr(simCSIa1Ind, simCSIb1Ind)[0])
                            # print("e corr", pearsonr(tmpCSIa1Ind, tmpCSIe1Ind)[0], pearsonr(tmpCSIa1Ind, simCSIe1Ind)[0],
                            #         pearsonr(simCSIa1Ind, simCSIe1Ind)[0])
                            # index_mean_pearson_ab.append(pearsonr(tmpCSIa1Ind, tmpCSIb1Ind)[0])
                            # index_mean_spearman_ab.append(spearmanr(tmpCSIa1Ind, tmpCSIb1Ind)[0])
                            # index_mean_pearson_ae.append(pearsonr(tmpCSIa1Ind, tmpCSIe1Ind)[0])
                            # index_mean_spearman_ae.append(spearmanr(tmpCSIa1Ind, tmpCSIe1Ind)[0])

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

                    if segment_option == "find_search":
                        start_time = time.time_ns()
                        segment_start = time.time_ns()

                        # 滑动窗口分段
                        if withIndexValue:
                            segment_method_ori = find_sub_opt_segment_method_sliding(
                                tmpCSIa1Ind, tmpCSIa1Ind, 3, 5)
                        else:
                            if withoutSort is True:
                                segment_method_ori = find_sub_opt_segment_method_sliding(
                                    tmpCSIa1Ind, tmpCSIa1Ind, 3, 5)
                                # segment_method_ori = find_sub_opt_segment_method_sliding_threshold(
                                #     tmpCSIa1Ind, tmpCSIa1Ind, 3, 5, 0.005)
                            else:
                                segment_method_ori = find_sub_opt_segment_method_sliding_threshold(
                                    tmpCSIa1Ind, tmpCSIa1Ind, 3, 5, 60)

                        if isExtraPrint:
                            print()
                            print("staInd", staInd, "segment_method", len(segment_method_ori), segment_method_ori)

                        # 是否过滤
                        if isFilter:
                            if sum(segment_method_ori) != keyLen:
                                continue

                        tmpCSIa1IndReshape = segment_sequence(tmpCSIa1Ind, segment_method_ori)
                        tmpCSIb1IndReshape = segment_sequence(tmpCSIb1Ind, segment_method_ori)
                        tmpCSIe1IndReshape = segment_sequence(tmpCSIe1Ind, segment_method_ori)
                        simCSIa1IndReshape = segment_sequence(simCSIa1Ind, segment_method_ori)

                        # 不过滤时按照当前分段结果进行补全
                        if isFilter is False:
                            if withIndexValue:
                                tmpCSIa1Ind = np.vstack(tmpCSIa1IndReshape)
                                tmpCSIb1Ind = np.vstack(tmpCSIb1IndReshape)
                                tmpCSIe1Ind = np.vstack(tmpCSIe1IndReshape)
                            else:
                                tmpCSIa1Ind = np.hstack((tmpCSIa1IndReshape))
                                tmpCSIb1Ind = np.hstack((tmpCSIb1IndReshape))
                                tmpCSIe1Ind = np.hstack((tmpCSIe1IndReshape))

                        # plt.figure()
                        # plt.plot(tmpCSIa1Ind)
                        # plt.show()

                        K = optimize_random_block_vector_max_min(tmpCSIa1Ind, segment_method_ori)
                        tmpCSIa1Ind = K @ tmpCSIa1Ind
                        tmpCSIb1Ind = K @ tmpCSIb1Ind
                        tmpCSIe1Ind = K @ tmpCSIe1Ind
                        simCSIa1Ind = K @ simCSIa1Ind

                        # plt.figure()
                        # plt.plot(tmpCSIa1Ind)
                        # plt.show()
                        #
                        # plt.figure()
                        # plt.imshow(K, cmap='hot', interpolation='nearest')
                        # plt.colorbar()
                        # plt.show()

                        tmpCSIa1IndReshape = segment_sequence(tmpCSIa1Ind, segment_method_ori)
                        tmpCSIb1IndReshape = segment_sequence(tmpCSIb1Ind, segment_method_ori)
                        tmpCSIe1IndReshape = segment_sequence(tmpCSIe1Ind, segment_method_ori)
                        simCSIa1IndReshape = segment_sequence(simCSIa1Ind, segment_method_ori)

                        if isExtraPrint:
                            print("max dtw of ab", compute_max_index_dtw(tmpCSIa1IndReshape, tmpCSIb1IndReshape))

                            print("max min dtw of ab / aa / bb",
                                  compute_min_dtw(tmpCSIa1IndReshape, tmpCSIb1IndReshape),
                                  compute_min_dtw(tmpCSIa1IndReshape, tmpCSIa1IndReshape),
                                  compute_min_dtw(tmpCSIb1IndReshape, tmpCSIb1IndReshape))

                        # 替换原始数据
                        max_same_position_dist = compute_min_dtw(tmpCSIa1IndReshape, tmpCSIb1IndReshape)
                        # tmpCSIa1IndReshape, modifications = replace_segments_position_compared_with_all_segments(
                        #     tmpCSIa1IndReshape, 0.3, max_same_position_dist + 0.2, segLen * 2, staInd + 100000)
                        # tmpCSIa1IndReshape, modifications = replace_segments_position_compared_with_all_segments(
                        #     tmpCSIa1IndReshape, 0.1, segLen * 2, staInd + 100000)
                        tmpCSIa1IndReshape, modifications = replace_segments_position_compared_with_all_segments_no_limit(
                            tmpCSIa1IndReshape, replace_number, staInd + 100000)
                        # tmpCSIa1IndReshape, modifications = replace_segments_position_compared_with_all_segments_no_limit_pair(
                        #     tmpCSIa1IndReshape, simCSIa1IndReshape, segLen * 2, staInd + 100000)
                        # tmpCSIa1IndReshape, modifications = replace_segments_position_compared_with_all_segments_no_limit_pair(
                        #     tmpCSIa1IndReshape, tmpCSIb1IndReshape, segLen * 2, staInd + 100000)

                        before_add_mean_pearson_ab.append(abs(pearsonr(np.hstack(tmpCSIa1Ind), np.hstack(tmpCSIb1Ind))[0]))
                        before_add_mean_pearson_ae.append(abs(pearsonr(np.hstack(tmpCSIa1Ind), np.hstack(tmpCSIe1Ind))[0]))

                        if withIndexValue:
                            tmpCSIa1IndTmp = np.vstack(tmpCSIa1IndReshape)
                        else:
                            tmpCSIa1IndTmp = np.hstack((tmpCSIa1IndReshape))

                        transformMatrix = np.linalg.lstsq(tmpCSIa1Ind.T, tmpCSIa1IndTmp.T, rcond=None)[0]

                        # np.random.seed(10000)
                        # transformMatrix += np.random.uniform(0, np.std(CSIa1Orig) * 4, transformMatrix.shape)
                        # np.random.seed(10000 + staInd)
                        # transformMatrix += np.random.uniform(0, 1, transformMatrix.shape)
                        # 无需均值化
                        # transformMatrix = transformMatrix - np.mean(transformMatrix)
                        tmpCSIa1IndTmp = np.matmul(tmpCSIa1Ind.T, transformMatrix).T
                        tmpCSIb1IndTmp = np.matmul(tmpCSIb1Ind.T, transformMatrix).T
                        tmpCSIe1IndTmp = np.matmul(tmpCSIe1Ind.T, transformMatrix).T

                        # diag matrix (安全性差一些)
                        # transformMatrix = np.diag(np.array(np.hstack(tmpCSIa1IndTmp)) /
                        #                           (np.array(np.hstack(tmpCSIa1Ind)) + 0.001 * np.ones(keyLen * 2)))
                        # tmpCSIa1IndTmp = np.array(transformMatrix @ np.hstack(tmpCSIa1Ind)).reshape(keyLen, 2)
                        # tmpCSIb1IndTmp = np.array(transformMatrix @ np.hstack(tmpCSIb1Ind)).reshape(keyLen, 2)
                        # tmpCSIe1IndTmp = np.array(transformMatrix @ np.hstack(tmpCSIe1Ind)).reshape(keyLen, 2)

                        # plt.figure()
                        # plt.imshow(transformMatrix, cmap='hot', interpolation='nearest')
                        # plt.colorbar()
                        # plt.show()
                        # exit()

                        after_add_mean_pearson_ab.append(abs(pearsonr(np.hstack(tmpCSIa1IndTmp), np.hstack(tmpCSIb1IndTmp))[0]))
                        after_add_mean_pearson_ae.append(abs(pearsonr(np.hstack(tmpCSIa1IndTmp), np.hstack(tmpCSIe1IndTmp))[0]))

                        tmpCSIa1IndBck = tmpCSIa1Ind
                        tmpCSIb1IndBck = tmpCSIb1Ind
                        tmpCSIe1IndBck = tmpCSIe1Ind
                        tmpCSIa1Ind = tmpCSIa1IndTmp
                        tmpCSIb1Ind = tmpCSIb1IndTmp
                        tmpCSIe1Ind = tmpCSIe1IndTmp

                        tmpCSIa1IndReshape = segment_sequence(tmpCSIa1Ind, segment_method_ori)
                        tmpCSIb1IndReshape = segment_sequence(tmpCSIb1Ind, segment_method_ori)
                        tmpCSIe1IndReshape = segment_sequence(tmpCSIe1Ind, segment_method_ori)

                        if isExtraPrint:
                            print("max dtw of ab", compute_max_index_dtw(tmpCSIa1IndReshape, tmpCSIb1IndReshape))

                        if isExtraPrint:
                            print("max min dtw of ab / aa / bb",
                                compute_min_dtw(tmpCSIa1IndReshape, tmpCSIb1IndReshape),
                                compute_min_dtw(tmpCSIa1IndReshape, tmpCSIa1IndReshape),
                                compute_min_dtw(tmpCSIb1IndReshape, tmpCSIb1IndReshape))

                        # # 滑动窗口分段
                        # if withIndexValue:
                        #     segment_method_ori1 = find_sub_opt_segment_method_sliding(
                        #         tmpCSIa1Ind, tmpCSIa1Ind, 3, 5)
                        # else:
                        #     if withoutSort is True:
                        #         segment_method_ori1 = find_sub_opt_segment_method_sliding(
                        #             tmpCSIa1Ind, tmpCSIa1Ind, 3, 5)
                        #         # segment_method_ori = find_sub_opt_segment_method_sliding_threshold(
                        #         #     tmpCSIa1Ind, tmpCSIa1Ind, 3, 5, 0.005)
                        #     else:
                        #         segment_method_ori1 = find_sub_opt_segment_method_sliding_threshold(
                        #             tmpCSIa1Ind, tmpCSIa1Ind, 3, 5, 60)
                        #
                        # print("max min dtw of ab / aa / bb",
                        #     compute_min_dtw(segment_sequence(tmpCSIa1Ind, segment_method_ori1),
                        #                     segment_sequence(tmpCSIb1Ind, segment_method_ori1)),
                        #     compute_min_dtw(segment_sequence(tmpCSIa1Ind, segment_method_ori1),
                        #                     segment_sequence(tmpCSIa1Ind, segment_method_ori1)),
                        #     compute_min_dtw(segment_sequence(tmpCSIb1Ind, segment_method_ori1),
                        #                     segment_sequence(tmpCSIb1Ind, segment_method_ori1)))

                        # if withIndexValue:
                        #     print("corr flatten", pearsonr(tmpCSIa1Ind.reshape(1, -1)[0], tmpCSIb1Ind.reshape(1, -1)[0])[0])
                        #     print("corr value", pearsonr(tmpCSIa1Ind[:, 0], tmpCSIb1Ind[:, 0])[0])
                        #     print("corr index", pearsonr(tmpCSIa1Ind[:, 1], tmpCSIb1Ind[:, 1])[0])
                        #
                        #     print("var flatten", np.var(tmpCSIa1Ind.reshape(1, -1)), np.var(tmpCSIb1Ind.reshape(1, -1)))
                        #     print("var value", np.var(tmpCSIa1Ind[:, 0]), np.var(tmpCSIb1Ind[:, 0]))
                        #     print("var index", np.var(tmpCSIa1Ind[:, 1]), np.var(tmpCSIb1Ind[:, 1]))
                        #
                        #     print("kuro flatten", kurtosis(tmpCSIa1Ind.reshape(-1, 1)), kurtosis(tmpCSIb1Ind.reshape(-1, 1)))
                        #     print("kuro value", kurtosis(tmpCSIa1Ind[:, 0].reshape(-1, 1)), kurtosis(tmpCSIb1Ind[:, 0].reshape(-1, 1)))
                        #     print("kuro index", kurtosis(tmpCSIa1Ind[:, 1].reshape(-1, 1)), kurtosis(tmpCSIb1Ind[:, 1].reshape(-1, 1)))
                        #
                        #     print("skew flatten", skew(tmpCSIa1Ind.reshape(-1, 1)), skew(tmpCSIb1Ind.reshape(-1, 1)))
                        #     print("skew value", skew(tmpCSIa1Ind[:, 0]), skew(tmpCSIb1Ind[:, 0]))
                        #     print("skew index", skew(tmpCSIa1Ind[:, 1]), skew(tmpCSIb1Ind[:, 1]))
                        #
                        #     print("cv flatten", np.std(tmpCSIa1Ind.reshape(-1, 1)) / np.mean(tmpCSIa1Ind.reshape(-1, 1)),
                        #             np.std(tmpCSIb1Ind.reshape(-1, 1)) / np.mean(tmpCSIb1Ind.reshape(-1, 1)))
                        #     print("cv value", np.std(tmpCSIa1Ind[:, 0]) / np.mean(tmpCSIa1Ind[:, 0]),
                        #             np.std(tmpCSIb1Ind[:, 0]) / np.mean(tmpCSIb1Ind[:, 0]))
                        #     print("cv index", np.std(tmpCSIa1Ind[:, 1]) / np.mean(tmpCSIa1Ind[:, 1]),
                        #             np.std(tmpCSIb1Ind[:, 1]) / np.mean(tmpCSIb1Ind[:, 1]))
                        #
                        #     eigenvalues = np.linalg.eigvals(randomMatrix)
                        #     print("matrix eigenvalue", np.max(eigenvalues),  np.min(eigenvalues), np.mean(eigenvalues))
                        #     print("matrix cond", np.linalg.cond(randomMatrix))
                        #
                        #     print("max min dtw of ab / aa / bb",
                        #           compute_min_dtw(segment_sequence(tmpCSIa1Ind, segment_method_ori),
                        #                                          segment_sequence(tmpCSIb1Ind, segment_method_ori)),
                        #           compute_min_dtw(segment_sequence(tmpCSIa1Ind, segment_method_ori),
                        #                                          segment_sequence(tmpCSIa1Ind, segment_method_ori)),
                        #           compute_min_dtw(segment_sequence(tmpCSIb1Ind, segment_method_ori),
                        #                                          segment_sequence(tmpCSIb1Ind, segment_method_ori)))
                        #
                        #     print("sum dtw of ab / aa / bb",
                        #           np.sum(compute_all_dtw(segment_sequence(tmpCSIa1Ind, segment_method_ori),
                        #                           segment_sequence(tmpCSIb1Ind, segment_method_ori))),
                        #           np.sum(compute_all_dtw(segment_sequence(tmpCSIa1Ind, segment_method_ori),
                        #                           segment_sequence(tmpCSIa1Ind, segment_method_ori))),
                        #           np.sum(compute_all_dtw(segment_sequence(tmpCSIb1Ind, segment_method_ori),
                        #                           segment_sequence(tmpCSIb1Ind, segment_method_ori))))
                        # else:
                        #     print("corr", pearsonr(tmpCSIa1Ind, tmpCSIb1Ind)[0])
                        #     print("var", np.var(tmpCSIa1Ind), np.var(tmpCSIb1Ind))
                        #
                        #     print("kuro", kurtosis(tmpCSIa1Ind), kurtosis(tmpCSIb1Ind))
                        #     print("skew", skew(tmpCSIa1Ind), skew(tmpCSIb1Ind))
                        #     print("cv", np.std(tmpCSIa1Ind) / np.mean(tmpCSIa1Ind),
                        #           np.std(tmpCSIb1Ind) / np.mean(tmpCSIb1Ind))
                        #
                        #     eigenvalues = np.linalg.eigvals(randomMatrix)
                        #     print("matrix eigenvalue", np.max(eigenvalues), np.min(eigenvalues), np.mean(eigenvalues))
                        #     print("matrix cond", np.linalg.cond(randomMatrix))
                        #     print("max min dtw of ab / aa / bb",
                        #           compute_min_dtw(segment_sequence(tmpCSIa1Ind, segment_method_ori),
                        #                                          segment_sequence(tmpCSIb1Ind, segment_method_ori)),
                        #           compute_min_dtw(segment_sequence(tmpCSIa1Ind, segment_method_ori),
                        #                                          segment_sequence(tmpCSIa1Ind, segment_method_ori)),
                        #           compute_min_dtw(segment_sequence(tmpCSIb1Ind, segment_method_ori),
                        #                                          segment_sequence(tmpCSIb1Ind, segment_method_ori)))

                        if min(segment_method_ori) < 3:
                            if isExtraPrint:
                                print("小于最小长度")
                            continue

                        # 限定分段个数为4
                        # if len(segment_method_ori) != int(keyLen / segLen):
                        #     print("未找到次优解")
                        #     # segment_method_ori = [4, 4, 4, 4]
                        #     if isDegrade:
                        #         findError = True
                        #     else:
                        #         continue

                        min_length = min(segment_method_ori)
                        max_length = max(segment_method_ori)
                        num_segments = len(segment_method_ori)
                        measurements_len = sum(segment_method_ori)

                        tmpCSIa1IndReshape = segment_sequence(tmpCSIa1Ind, segment_method_ori)

                        # 补全至keyLen长度用于等长分段
                        if len(tmpCSIa1Ind) < keyLen:
                            if withIndexValue:
                                tmpCSIa1Ind = np.vstack((tmpCSIa1Ind, tmpCSIa1IndBck[-(keyLen - len(tmpCSIa1Ind)):]))
                                tmpCSIb1Ind = np.vstack((tmpCSIb1Ind, tmpCSIb1IndBck[-(keyLen - len(tmpCSIb1Ind)):]))
                                tmpCSIe1Ind = np.vstack((tmpCSIe1Ind, tmpCSIe1IndBck[-(keyLen - len(tmpCSIe1Ind)):]))
                            else:
                                tmpCSIa1Ind = np.hstack((tmpCSIa1Ind, tmpCSIa1IndBck[-(keyLen - len(tmpCSIa1Ind)):]))
                                tmpCSIb1Ind = np.hstack((tmpCSIb1Ind, tmpCSIb1IndBck[-(keyLen - len(tmpCSIb1Ind)):]))
                                tmpCSIe1Ind = np.hstack((tmpCSIe1Ind, tmpCSIe1IndBck[-(keyLen - len(tmpCSIe1Ind)):]))

                        tmpCSIa1EvenSegment = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIa1Ind))
                        dist_opt_seg = compute_min_dtw(tmpCSIa1IndReshape, tmpCSIa1IndReshape)
                        dist_even_seg = compute_min_dtw(tmpCSIa1EvenSegment, tmpCSIa1EvenSegment)
                        if (dist_opt_seg < dist_even_seg):
                            if isExtraPrint:
                                print("staInd", staInd, "最优分段最大差距", dist_opt_seg, "平均分段最大差距", dist_even_seg)
                            badSegments += 1
                            tmpCSIa1IndReshape = tmpCSIa1EvenSegment
                            segment_method_ori = [segLen] * int(keyLen / segLen)
                            min_length = min(segment_method_ori)
                            max_length = max(segment_method_ori)
                            num_segments = len(segment_method_ori)
                            measurements_len = sum(segment_method_ori)

                            max_min_dtw_ab.append(dist_even_seg)
                        else:
                            max_min_dtw_ab.append(dist_even_seg)

                        segmentMaxDist.append(dist_opt_seg)
                        evenMaxDist.append(dist_even_seg)

                        segment_end = time.time_ns()

                    else:
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
                    # if a_min_index == 0:
                    #     permutation = list(range(int(keyLen / segLen)))
                    # else:
                    #     permutation = list(range(int(keyLen / segLen) + 1))
                    # permutation = list(range(int(keyLen / segLen)))
                    if segment_option != "find_search":
                        # 如果还以此分段进行划分,则超过此分段的索引会被截掉
                        permutation = list(range(int(keyLen / segLen)))
                    else:
                        permutation = list(range(num_segments))
                    combineMetric = list(zip(tmpCSIa1IndReshape, permutation))
                    np.random.seed(staInd)
                    np.random.shuffle(combineMetric)
                    tmpCSIa1IndReshape, permutation = zip(*combineMetric)
                    if withIndexValue:
                        tmpCSIa1Ind = np.vstack(tmpCSIa1IndReshape)
                    else:
                        tmpCSIa1Ind = np.hstack((tmpCSIa1IndReshape))

                    if segment_option == "find_search" or segment_option == "find_search_pair":
                        if findError is False:
                            search_start = time.time_ns()
                            # 计算对应位置最大差距作为阈值
                            threshold = compute_threshold(tmpCSIa1Bck, tmpCSIb1Ind)
                            if isExtraPrint:
                                print("threshold", threshold)
                            if withIndexValue:
                                base_threshold = threshold
                                threshold /= 4
                            else:
                                if withoutSort is True:
                                    base_threshold = threshold
                                    threshold /= 4
                                    # threshold = 0.005
                                else:
                                    threshold = 3
                                # threshold = int(threshold / 2)
                                # if threshold >= 9:
                                #     continue
                            # 在阈值内匹配相近的索引
                            if withIndexValue:
                                all_segments_A = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIa1Bck, threshold)
                                all_segments_B = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIb1Ind, threshold)
                                all_segments_E = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIe1Ind, threshold)
                            else:
                                if withoutSort is True:
                                    all_segments_A = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIa1Bck, threshold)
                                    all_segments_B = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIb1Ind, threshold)
                                    all_segments_E = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIe1Ind, threshold)
                                else:
                                    all_segments_A = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIa1Bck, threshold + 1)
                                    all_segments_B = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIb1Ind, threshold + 1)
                                    all_segments_E = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIe1Ind, threshold + 1)
                            # 根据相近的索引组合成若干个子分段
                            if isUp:
                                segments_A = find_segments(all_segments_A, 2, keyLen)
                                segments_B = find_segments(all_segments_B, 2, keyLen)
                                segments_E = find_segments(all_segments_E, 2, keyLen)
                            else:
                                if withIndexValue:
                                    # 有时候找到的分段总个数不等于所使用的测量值个数，故需要重新规定总长度进行搜索
                                    segments_A = find_segments(all_segments_A, 3, measurements_len)
                                    segments_B = find_segments(all_segments_B, 3, measurements_len)
                                    segments_E = find_segments(all_segments_E, 3, measurements_len)

                                    while len(segments_B) < int(len(segments_A) / 2) and threshold < base_threshold:
                                        if isExtraPrint:
                                            print("threshold", threshold)
                                        threshold += base_threshold / 8
                                        all_segments_B = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIb1Ind,
                                                                                             threshold)
                                        segments_B = find_segments(all_segments_B, 3, measurements_len)
                                else:
                                    if withoutSort is True:
                                        # 有时候找到的分段总个数不等于所使用的测量值个数，故需要重新规定总长度进行搜索
                                        segments_A = find_segments(all_segments_A, 3, measurements_len)
                                        segments_B = find_segments(all_segments_B, 3, measurements_len)
                                        segments_E = find_segments(all_segments_E, 3, measurements_len)

                                        while len(segments_B) < int(len(segments_A) / 2) and threshold < base_threshold:
                                            if isExtraPrint:
                                                print("threshold", threshold)
                                            threshold += base_threshold / 8
                                            all_segments_B = search_segment_method_with_matching(tmpCSIa1Ind, tmpCSIb1Ind,
                                                                                                 threshold)
                                            segments_B = find_segments(all_segments_B, 3, measurements_len)
                                    else:
                                        segments_A = find_segments(all_segments_A, 3, keyLen)
                                        segments_B = find_segments(all_segments_B, 3, keyLen)
                                        segments_E = find_segments(all_segments_E, 3, keyLen)
                                        # segments_A = find_segments(all_segments_A, min_length, keyLen)
                                        # segments_B = find_segments(all_segments_B, min_length, keyLen)
                                        # segments_E = find_segments(all_segments_E, min_length, keyLen)

                            if segment_option == "find_search_pair":
                                # 配对寻找分段时有可能无法拼成一个整体分段
                                if segments_A[-1][1] != keyLen:
                                    segments_A.append([segments_A[-1][1], keyLen])

                            if len(segments_A) == 0:
                                if isExtraPrint:
                                    print("A的分段为空")
                                if isDegrade:
                                    findError = True
                                else:
                                    continue

                            if len(segments_B) == 0:
                                if isExtraPrint:
                                    print("B的分段为空")
                                segments_B = segments_A

                            if withIndexValue:
                                gaps = find_gaps(segments_B, [0, measurements_len])
                            else:
                                if withoutSort is True:
                                    gaps = find_gaps(segments_B, [0, measurements_len])
                                else:
                                    gaps = find_gaps(segments_B, [0, keyLen])
                            if len(gaps) != 0:
                                segments_B_bck = segments_B.copy()
                                # segments_B = merge_gaps(segments_B, gaps)
                                # gap_too_width = False
                                # for gap in gaps:
                                #     if gap[1] - gap[0] > 2 * max_length:
                                #         gap_too_width = True
                                #         break
                                # if gap_too_width:
                                #     print("第一个间隙过大")
                                #     continue

                                add_intervals = find_covering_intervals(segments_A, gaps)
                                if len(add_intervals) == 0:
                                    if isExtraPrint:
                                        print("A的分段无法覆盖间隙")
                                        print("segments_A", segments_A)
                                        print("segments_B", segments_B)
                                    if isDegrade:
                                        findError = True
                                    else:
                                        continue
                                segments_B.extend(add_intervals)
                                segments_B.sort(key=lambda x: x[0])

                                all_gaps_number.append(len(add_intervals))
                                all_gaps_length.append(sum([interval[1] - interval[0] for interval in add_intervals]))
                                have_gaps_number += 1

                            all_intervals_number.append(len(segments_B))
                            all_intervals_length.append(sum([interval[1] - interval[0] for interval in segments_B]))

                            # 根据子分段构成一个覆盖总分段长度的组合
                            if isUp:
                                segment_method_A = find_all_cover_intervals_iter_up(segments_A, (0, keyLen), 2)
                                segment_method_B = find_all_cover_intervals_iter_up(segments_B, (0, keyLen), 2)
                                segment_method_E = find_all_cover_intervals_iter_up(segments_E, (0, keyLen), 2)
                            else:
                                if withIndexValue:
                                    segment_method_A = find_all_cover_intervals_iter(
                                        segments_A, (0, measurements_len), min_length, max_length)[0]
                                    segment_method_B, all_segment_method_B = find_all_cover_intervals_iter(
                                        segments_B, (0, measurements_len), min_length, max_length)
                                    segment_method_E = find_all_cover_intervals_iter(
                                        segments_A, (0, measurements_len), min_length, max_length)[0]
                                else:
                                    if withoutSort is True:
                                        segment_method_A = find_all_cover_intervals_iter(
                                            segments_A, (0, measurements_len), min_length, max_length)[0]
                                        segment_method_B, all_segment_method_B = find_all_cover_intervals_iter(
                                            segments_B, (0, measurements_len), min_length, max_length)
                                        segment_method_E = find_all_cover_intervals_iter(
                                            segments_A, (0, measurements_len), min_length, max_length)[0]
                                    else:
                                        segment_method_A = find_all_cover_intervals_iter(
                                            segments_A, (0, keyLen), min_length, max_length)[0]
                                        segment_method_B, all_segment_method_B = find_all_cover_intervals_iter(
                                            segments_B, (0, keyLen), min_length, max_length)
                                        segment_method_E = find_all_cover_intervals_iter(
                                            segments_A, (0, keyLen), min_length, max_length)[0]
                                    # segment_method_A = find_all_cover_intervals_iter(segments_A, (0, keyLen), 3, 5)[0]
                                    # segment_method_B = find_all_cover_intervals_iter(segments_B, (0, keyLen), 3, 5)[0]
                                    # segment_method_E = find_all_cover_intervals_iter(segments_E, (0, keyLen), 3, 5)[0]
                                    # segment_method_A = find_all_cover_intervals_iter(segments_A, (0, keyLen), 3, keyLen)[0]
                                    # segment_method_B = find_all_cover_intervals_iter(segments_B, (0, keyLen), 3, keyLen)[0]
                                    # segment_method_E = find_all_cover_intervals_iter(segments_E, (0, keyLen), 3, keyLen)[0]

                            # if len(segment_method_B) == 0:
                            #     print("未找到合适分段1")
                            #     if isDegrade:
                            #         findError = True
                            #     else:
                            #         continue

                            if (withProb or keyLen >= 4 * 16) and (segment_method_A == -1 or segment_method_B == -1):
                                if isExtraPrint:
                                    print("分段数过多")
                                if isDegrade:
                                    findError = True
                                else:
                                    continue

                            if findError is False:
                                if len(segment_method_B) == 0:
                                    # 找出最长的分段方式
                                    segment_with_max_length = []
                                    max_segment_length = 0
                                    for k in all_segment_method_B.keys():
                                        for solution in all_segment_method_B[k]:
                                            if len(solution) > max_segment_length:
                                                segment_with_max_length = solution
                                                max_segment_length = len(solution)
                                    if isExtraPrint:
                                        print("segment_with_max_length", segment_with_max_length)
                                    if len(segment_with_max_length) != 0:
                                        short_segment = find_gaps(segment_with_max_length, [0, measurements_len])
                                        double_add_intervals = find_covering_intervals(segments_A, short_segment)
                                        if len(double_add_intervals) != 0:
                                            segments_B.extend(double_add_intervals)
                                            # 找出间隙之后的分段，有可能就是该分段不一致导致的无法推出连续分段
                                            double_add_before_intervals = find_before_intervals(segments_A, double_add_intervals)
                                            segments_B.extend(double_add_before_intervals)
                                            segments_B.sort(key=lambda x: x[0])
                                        else:
                                            segments_B = segments_A
                                        # 找出不在范围内的分段
                                        out_of_segment = find_out_of_range_intervals(segments_B, min_length, max_length)
                                        triple_add_intervals = find_covering_intervals(segments_A, out_of_segment)
                                        if len(triple_add_intervals) != 0:
                                            segments_B.extend(triple_add_intervals)
                                            segments_B.sort(key=lambda x: x[0])
                                        segment_method_B = find_all_cover_intervals_iter(
                                            segments_B, (0, measurements_len), min_length, max_length)[0]
                                    else:
                                        segments_B = segments_A
                                if (withProb or keyLen >= 4 * 16) and segment_method_B == -1:
                                    if isExtraPrint:
                                        print("分段数过多")
                                    if isDegrade:
                                        findError = True
                                    else:
                                        continue
                                if len(segment_method_B) == 0:
                                    segments_B = segments_A
                                segment_method_B = find_all_cover_intervals_iter(
                                    segments_B, (0, measurements_len), min_length, max_length)[0]
                                if (withProb or keyLen >= 4 * 16) and segment_method_B == -1:
                                    if isExtraPrint:
                                        print("分段数过多")
                                    if isDegrade:
                                        findError = True
                                    else:
                                        continue

                        if findError is False:
                            # 根据子分段索引得到子分段长度
                            segment_length_A = get_segment_lengths(segment_method_A)
                            segment_length_B = get_segment_lengths(segment_method_B)
                            # if len(segment_length_B) == 0:
                            #     print("未找到合适分段1")
                            #     segment_length_B = segment_length_A
                            segment_length_E = get_segment_lengths(segment_method_E)
                            search_end = time.time_ns()

                            # segment_length_A.append([4] * int(keyLen / segLen))
                            # segment_length_B.append([4] * int(keyLen / segLen))

                            if len(segment_length_A) > 2000 or len(segment_length_B) > 2000:
                                if isExtraPrint:
                                    print("分段数过多")
                                if isDegrade:
                                    findError = True
                                else:
                                    continue

                            if isPrint:
                                print("sorted index", np.array(tmpCSIa1Bck).tolist(), np.array(tmpCSIb1Ind).tolist())
                                print("published index", np.array(tmpCSIa1Ind).tolist())
                                print("threshold", threshold)
                                # print("all_segments_A", all_segments_A)
                                # print("all_segments_B", all_segments_B)
                                print("segments_A", segments_A)
                                print("segments_B", segments_B)
                                if len(gaps) != 0:
                                    print("gaps", gaps)
                                    print("segments_B_bck", segments_B_bck)
                                print("segment_length_A", len(segment_length_A), list(segment_length_A))
                                print("segment_length_B", len(segment_length_B), list(segment_length_B))

                            a_list_numbers = []
                            a_segments = []
                            a_dists = []
                            b_list_numbers = []
                            b_segments = []
                            b_dists = []
                            e_list_numbers = []
                            e_segments = []
                            e_dists = []
                            final_search_start = time.time_ns()

                            for i in range(len(segment_length_A)):
                                # if len(segment_length_A[i]) != num_segments:
                                #     continue
                                # if len(segment_length_A[i]) != int(keyLen / segLen):
                                #     continue
                                # if len(segment_length_A[i]) < int(keyLen / segLen) - 1 or \
                                #         len(segment_length_A[i]) > int(keyLen / segLen) + 1:
                                #     continue
                                a_list_number_tmp, dists = search_index_with_segment(
                                    tmpCSIa1Bck, tmpCSIa1Ind, np.array(segment_length_A[i]).astype(int))
                                if np.array_equal(np.sort(a_list_number_tmp),
                                                  list(range(0, len(a_list_number_tmp)))) is False:
                                    continue
                                a_segments.append(segment_length_A[i])
                                a_list_numbers.append(a_list_number_tmp)
                                a_dists.append(dists)
                                if isPrint:
                                    print("a_list_number", a_list_number_tmp, "segment", segment_length_A[i],
                                          "min", np.min(dists), "max", np.max(dists), "mean", np.mean(dists),
                                          "var", np.var(dists), "dists", dists)

                                # data_seg = segment_sequence(tmpCSIa1Bck, segment_length_A[i])
                                # published_seg = segment_sequence(tmpCSIa1Ind, np.array(segment_length_A[i])[a_list_number_tmp])
                                # all_dists = max(compute_all_dtw(data_seg, published_seg))
                                # a_dists.append(all_dists)
                                # if isPrint:
                                #     print("a_list_number", a_list_number_tmp, "segment", segment_length_A[i],
                                #           "max", np.max(all_dists), "mean", np.mean(all_dists), "var", np.var(all_dists),
                                #           "dists", all_dists)
                            for i in range(len(segment_length_B)):
                                # if len(segment_length_B[i]) != num_segments:
                                #     continue
                                # if len(segment_length_B[i]) != int(keyLen / segLen):
                                #     continue
                                # if len(segment_length_B[i]) < int(keyLen / segLen) - 1 or \
                                #         len(segment_length_B[i]) > int(keyLen / segLen) + 1:
                                #     continue
                                b_list_number_tmp, dists = search_index_with_segment(
                                    tmpCSIb1Ind, tmpCSIa1Ind, np.array(segment_length_B[i]).astype(int))
                                if np.array_equal(np.sort(b_list_number_tmp),
                                                  list(range(0, len(b_list_number_tmp)))) is False:
                                    continue
                                b_segments.append(segment_length_B[i])
                                b_list_numbers.append(b_list_number_tmp)
                                b_dists.append(dists)
                                if isPrint:
                                    print("b_list_number", b_list_number_tmp, "segment", segment_length_B[i],
                                          "min", np.min(dists), "max", np.max(dists), "mean", np.mean(dists),
                                          "var", np.var(dists), "dists", dists)

                                # data_seg = segment_sequence(tmpCSIb1Ind, segment_length_B[i])
                                # published_seg = segment_sequence(tmpCSIa1Ind, np.array(segment_length_B[i])[b_list_number_tmp])
                                # all_dists = max(compute_all_dtw(data_seg, published_seg))
                                # b_dists.append(all_dists)
                                # if isPrint:
                                #     print("b_list_number", b_list_number_tmp, "segment", segment_length_B[i],
                                #           "max", np.max(all_dists), "mean", np.mean(all_dists), "var", np.var(all_dists),
                                #           "dists", all_dists)

                            # print("a")
                            #
                            # for i in range(len(a_segments)):
                            #     data_seg = segment_sequence(tmpCSIa1Bck, a_segments[i])
                            #     indices = a_list_numbers[i]
                            #     published_seg = segment_sequence(tmpCSIa1Ind, np.array(a_segments[i])[indices])
                            #     print([list(x) for x in data_seg])
                            #     print([list(x) for x in published_seg])
                            #     all_dists = compute_all_dtw(data_seg, published_seg)
                            #     print(min(all_dists), np.mean(min(all_dists)), np.min(min(all_dists)))
                            #     print(max(all_dists), np.mean(max(all_dists)), np.min(max(all_dists)))
                            #     print(np.mean(all_dists, axis=1), np.mean(np.mean(all_dists, axis=1)),
                            #           np.min(np.mean(all_dists, axis=1)))
                            #     print(np.mean(all_dists), np.var(all_dists), np.median(all_dists))
                            #     print()
                            #
                            # print("b")
                            #
                            # for i in range(len(b_segments)):
                            #     data_seg = segment_sequence(tmpCSIb1Ind, b_segments[i])
                            #     indices = b_list_numbers[i]
                            #     published_seg = segment_sequence(tmpCSIa1Ind, np.array(b_segments[i])[indices])
                            #     print([list(x) for x in data_seg])
                            #     print([list(x) for x in published_seg])
                            #     all_dists = compute_all_dtw(data_seg, published_seg)
                            #     print(min(all_dists), np.mean(min(all_dists)), np.min(min(all_dists)))
                            #     print(max(all_dists), np.mean(max(all_dists)), np.min(max(all_dists)))
                            #     print(np.mean(all_dists, axis=1), np.mean(np.mean(all_dists, axis=1)),
                            #           np.min(np.mean(all_dists, axis=1)))
                            #     print(np.mean(all_dists), np.var(all_dists), np.median(all_dists))
                            #     print()

                            for i in range(len(segment_length_E)):
                                # if len(segment_length_E[i]) != num_segments:
                                #     continue
                                # if len(segment_length_E[i]) != int(keyLen / segLen):
                                #     continue
                                # if len(segment_length_E[i]) < int(keyLen / segLen) - 1 or \
                                #         len(segment_length_E[i]) > int(keyLen / segLen) + 1:
                                #     continue
                                e_list_number_tmp, dists = search_index_with_segment(
                                    tmpCSIe1Ind, tmpCSIa1Ind, np.array(segment_length_E[i]).astype(int))
                                if np.array_equal(np.sort(e_list_number_tmp),
                                                  list(range(0, len(e_list_number_tmp)))) is False:
                                    continue
                                e_segments.append(segment_length_E[i])
                                e_list_numbers.append(e_list_number_tmp)
                                e_dists.append(dists)
                                # if isPrint:
                                #     print("e_list_number", e_list_number_tmp, "segment", segment_length_E[i],
                                #           "max", np.max(dists), "mean", np.mean(dists), "var", np.var(dists), "dists", dists)

                            if len(a_segments) == 0 or len(b_segments) == 0:
                                if isExtraPrint:
                                    print("未找到合适分段2")
                                # a_list_numbers = []
                                # a_segments = []
                                # a_dists = []
                                # b_list_numbers = []
                                # b_segments = []
                                # b_dists = []
                                # e_list_numbers = []
                                # e_segments = []
                                # e_dists = []
                                # segment_length_B = segment_length_A
                                # segment_length_E = segment_length_A
                                # for i in range(len(segment_length_A)):
                                #     a_list_number_tmp, dists = search_index_with_segment(
                                #         tmpCSIa1Bck, tmpCSIa1Ind, np.array(segment_length_A[i]).astype(int))
                                #     if np.array_equal(np.sort(a_list_number_tmp),
                                #                       list(range(0, len(a_list_number_tmp)))) is False:
                                #         continue
                                #     a_segments.append(segment_length_A[i])
                                #     a_list_numbers.append(a_list_number_tmp)
                                #     a_dists.append(dists)
                                # for i in range(len(segment_length_B)):
                                #     b_list_number_tmp, dists = search_index_with_segment(
                                #         tmpCSIb1Ind, tmpCSIa1Ind, np.array(segment_length_B[i]).astype(int))
                                #     if np.array_equal(np.sort(b_list_number_tmp),
                                #                       list(range(0, len(b_list_number_tmp)))) is False:
                                #         continue
                                #     b_segments.append(segment_length_B[i])
                                #     b_list_numbers.append(b_list_number_tmp)
                                #     b_dists.append(dists)
                                # for i in range(len(segment_length_E)):
                                #     e_list_number_tmp, dists = search_index_with_segment(
                                #         tmpCSIe1Ind, tmpCSIa1Ind, np.array(segment_length_E[i]).astype(int))
                                #     if np.array_equal(np.sort(e_list_number_tmp),
                                #                       list(range(0, len(e_list_number_tmp)))) is False:
                                #         continue
                                #     e_segments.append(segment_length_E[i])
                                #     e_list_numbers.append(e_list_number_tmp)
                                #     e_dists.append(dists)

                                if isDegrade:
                                    findError = True
                                else:
                                    continue

                            if findError is False:
                                if isExtraPrint:
                                    print("search space", len(a_segments), len(b_segments))
                                search_space.append((len(a_segments) + len(b_segments)) / 2)

                                max_dist_A = np.inf
                                max_dist_B = np.inf
                                max_dist_E = np.inf
                                mean_dist_A = np.inf
                                mean_dist_B = np.inf
                                mean_dist_E = np.inf
                                var_dist_A = np.inf
                                var_dist_B = np.inf
                                var_dist_E = np.inf

                                # 按照mean最大->var最小的次序排序
                                # a_min_index = find_special_array_mean_var(a_dists)[0][0]
                                # a_list_number = a_list_numbers[a_min_index]
                                # a_segment = a_segments[a_min_index]
                                # b_min_index = find_special_array_mean_var(b_dists)[0][0]
                                # b_list_number = b_list_numbers[b_min_index]
                                # b_segment = b_segments[b_min_index]
                                # if len(e_segments) != 0:
                                #     e_min_index = find_special_array_mean_var(e_dists)[0][0]
                                #     e_list_number = e_list_numbers[e_min_index]
                                #     e_segment = e_segments[e_min_index]
                                # else:
                                #     e_list_number = np.random.permutation(len(tmpCSIe1Ind))

                                # 按照mean最大->var最小的次序排序
                                # a_min_index = find_special_array_mean_var(a_dists)[-1][0]
                                # a_list_number = a_list_numbers[a_min_index]
                                # a_segment = a_segments[a_min_index]
                                # b_min_index = find_special_array_mean_var(b_dists)[-1][0]
                                # b_list_number = b_list_numbers[b_min_index]
                                # b_segment = b_segments[b_min_index]
                                # if len(e_segments) != 0:
                                #     e_min_index = find_special_array_mean_var(e_dists)[-1][0]
                                #     e_list_number = e_list_numbers[e_min_index]
                                #     e_segment = e_segments[e_min_index]
                                # else:
                                #     e_list_number = np.random.permutation(len(tmpCSIe1Ind))

                                # 按照min最小->mean最小的次序排序
                                a_min_index = find_special_array_min_mean(a_dists)[0][0]
                                a_list_number = a_list_numbers[a_min_index]
                                a_segment = a_segments[a_min_index]
                                b_min_index = find_special_array_min_mean(b_dists)[0][0]
                                b_list_number = b_list_numbers[b_min_index]
                                b_segment = b_segments[b_min_index]
                                if len(e_segments) != 0:
                                    e_min_index = find_special_array_min_mean(e_dists)[0][0]
                                    e_list_number = e_list_numbers[e_min_index]
                                    e_segment = e_segments[e_min_index]
                                else:
                                    e_list_number = np.random.permutation(len(tmpCSIe1Ind))

                                final_search_end = time.time_ns()
                                if isExtraPrint:
                                    print(a_segment, b_segment)
                                    print(a_list_number, b_list_number)

                                # max_dist_A = np.inf
                                # max_dist_B = np.inf
                                # max_dist_E = np.inf
                                # mean_dist_A = np.inf
                                # mean_dist_B = np.inf
                                # mean_dist_E = np.inf
                                # var_dist_A = np.inf
                                # var_dist_B = np.inf
                                # var_dist_E = np.inf
                                # a_list_number = []
                                # b_list_number = []
                                # e_list_number = []
                                # a_segment = []
                                # b_segment = []
                                # e_segment = []
                                #
                                # max_dist_A = np.inf
                                # max_dist_B = np.inf
                                # max_dist_E = np.inf
                                # mean_dist_A = np.inf
                                # mean_dist_B = np.inf
                                # mean_dist_E = np.inf
                                # var_dist_A = np.inf
                                # var_dist_B = np.inf
                                # var_dist_E = np.inf
                                # a_list_number = []
                                # b_list_number = []
                                # e_list_number = []
                                # a_segment = []
                                # b_segment = []
                                # e_segment = []
                                #
                                # a_list_numbers = []
                                # a_dists = []
                                # a_segments = []
                                # b_list_numbers = []
                                # b_dists = []
                                # b_segments = []
                                # e_list_numbers = []
                                # e_dists = []
                                # e_segments = []
                                # final_search_start = time.time_ns()
                                # a_match_dists = []
                                # a_cross_dists = []
                                # b_match_dists = []
                                # b_cross_dists = []
                                # e_match_dists = []
                                # e_cross_dists = []
                                # for i in range(len(segment_length_A)):
                                #     if len(segment_length_A[i]) != int(keyLen / segLen):
                                #         continue
                                #     # if len(segment_length_A[i]) < int(keyLen / segLen) - 1 or \
                                #     #         len(segment_length_A[i]) > int(keyLen / segLen) + 1:
                                #     #     continue
                                #
                                #     dists = []
                                #     match_dists = []
                                #     cross_dists = []
                                #     if search_method == "match":
                                #         # 按min_min搜索：每个分段和其他所有分段最小的最小值
                                #         a_list_number_tmp, dists = search_segment_method_with_min_match_dtw(
                                #             tmpCSIa1Bck, tmpCSIa1Ind, np.array(segment_length_A[i]).astype(int))
                                #     elif search_method == "cross":
                                #         # 按max_min搜索：每个匹配分段之间排除最小值，找出和其他所有分段最大的最小值
                                #         a_list_number_tmp, dists = search_segment_method_with_min_cross_dtw(
                                #             tmpCSIa1Bck, tmpCSIa1Ind, np.array(segment_length_A[i]).astype(int))
                                #     else:
                                #         # 综合前两个方式搜索
                                #         a_list_number_tmp, match_dists, cross_dists = search_segment_method_with_min_dtw(
                                #             tmpCSIa1Bck, tmpCSIa1Ind, np.array(segment_length_A[i]).astype(int))
                                #
                                #     if np.array_equal(np.sort(a_list_number_tmp), list(range(0, len(a_list_number_tmp)))) is False:
                                #         continue
                                #     a_list_numbers.append(a_list_number_tmp)
                                #     if search_method == "both":
                                #         a_match_dists.append(match_dists)
                                #         a_cross_dists.append(cross_dists)
                                #     else:
                                #         a_dists.append(dists)
                                #     a_segments.append(segment_length_A[i])
                                #     if isPrint:
                                #         if search_method == "both":
                                #             print("a_list_number", a_list_number_tmp, "segment", segment_length_A[i],
                                #                   "match_min", np.min(match_dists), "cross_min", np.min(cross_dists),
                                #                   "mean", np.mean(cross_dists), "match_dists", match_dists,
                                #                   "cross_dists", cross_dists)
                                #         else:
                                #             print("a_list_number", a_list_number_tmp, "segment", segment_length_A[i],
                                #                   "max", np.max(dists), "mean", np.mean(dists), "var", np.var(dists),
                                #                   "dists", dists)
                                # for i in range(len(segment_length_B)):
                                #     if len(segment_length_B[i]) != int(keyLen / segLen):
                                #         continue
                                #     # if len(segment_length_B[i]) < int(keyLen / segLen) - 1 or \
                                #     #         len(segment_length_B[i]) > int(keyLen / segLen) + 1:
                                #     #     continue
                                #
                                #     dists = []
                                #     match_dists = []
                                #     cross_dists = []
                                #     if search_method == "match":
                                #         # 按min_min搜索：每个分段和其他所有分段最小的最小值
                                #         b_list_number_tmp, dists = search_segment_method_with_min_match_dtw(
                                #             tmpCSIb1Ind, tmpCSIa1Ind, np.array(segment_length_B[i]).astype(int))
                                #     elif search_method == "cross":
                                #         # 按max_min搜索：每个匹配分段之间排除最小值，找出和其他所有分段最大的最小值
                                #         b_list_number_tmp, dists = search_segment_method_with_min_cross_dtw(
                                #             tmpCSIb1Ind, tmpCSIa1Ind, np.array(segment_length_B[i]).astype(int))
                                #     else:
                                #         # 综合前两个方式搜索
                                #         b_list_number_tmp, match_dists, cross_dists = search_segment_method_with_min_dtw(
                                #             tmpCSIb1Ind, tmpCSIa1Ind, np.array(segment_length_B[i]).astype(int))
                                #
                                #     if np.array_equal(np.sort(b_list_number_tmp), list(range(0, len(b_list_number_tmp)))) is False:
                                #         continue
                                #     b_list_numbers.append(b_list_number_tmp)
                                #     if search_method == "both":
                                #         b_match_dists.append(match_dists)
                                #         b_cross_dists.append(cross_dists)
                                #     else:
                                #         b_dists.append(dists)
                                #     b_segments.append(segment_length_B[i])
                                #     if isPrint:
                                #         if search_method == "both":
                                #             print("b_list_number", b_list_number_tmp, "segment", segment_length_B[i],
                                #                   "match_min", np.min(match_dists), "cross_min", np.min(cross_dists),
                                #                   "mean", np.mean(cross_dists), "match_dists", match_dists,
                                #                   "cross_dists", cross_dists)
                                #         else:
                                #             print("b_list_number", b_list_number_tmp, "segment", segment_length_B[i],
                                #                   "max", np.max(dists), "mean", np.mean(dists), "var", np.var(dists),
                                #                   "dists", dists)
                                # for i in range(len(segment_length_E)):
                                #     if len(segment_length_E[i]) != int(keyLen / segLen):
                                #         continue
                                #     # if len(segment_length_E[i]) < int(keyLen / segLen) - 1 or \
                                #     #         len(segment_length_E[i]) > int(keyLen / segLen) + 1:
                                #     #     continue
                                #
                                #     dists = []
                                #     match_dists = []
                                #     cross_dists = []
                                #     if search_method == "match":
                                #         # 按min_min搜索：每个分段和其他所有分段最小的最小值
                                #         e_list_number_tmp, dists = search_segment_method_with_min_match_dtw(
                                #             tmpCSIe1Ind, tmpCSIa1Ind, np.array(segment_length_E[i]).astype(int))
                                #     elif search_method == "cross":
                                #         # 按max_min搜索：每个匹配分段之间排除最小值，找出和其他所有分段最大的最小值
                                #         e_list_number_tmp, dists = search_segment_method_with_min_cross_dtw(
                                #             tmpCSIe1Ind, tmpCSIa1Ind, np.array(segment_length_E[i]).astype(int))
                                #     else:
                                #         # 综合前两个方式搜索
                                #         e_list_number_tmp, match_dists, cross_dists = search_segment_method_with_min_dtw(
                                #             tmpCSIe1Ind, tmpCSIa1Ind, np.array(segment_length_E[i]).astype(int))
                                #
                                #     if np.array_equal(np.sort(e_list_number_tmp), list(range(0, len(e_list_number_tmp)))) is False:
                                #         continue
                                #     e_list_numbers.append(e_list_number_tmp)
                                #     if search_method == "both":
                                #         e_match_dists.append(match_dists)
                                #         e_cross_dists.append(cross_dists)
                                #     else:
                                #         e_dists.append(dists)
                                #     e_segments.append(segment_length_E[i])
                                #
                                # if len(a_segments) == 0 or len(b_segments) == 0:
                                #     print("未找到合适分段2")
                                #     print()
                                #     continue
                                # print("search space", len(a_segments), len(b_segments))
                                # search_space.append((len(a_segments) + len(b_segments)) / 2)
                                #
                                # if search_method == "match":
                                #     # 按照min最大->mean最大->var最小的次序排序
                                #     a_min_index = find_special_array(a_dists)[0][0]
                                #     a_list_number = a_list_numbers[a_min_index]
                                #     a_segment = a_segments[a_min_index]
                                #     b_min_index = find_special_array(b_dists)[0][0]
                                #     b_list_number = b_list_numbers[b_min_index]
                                #     b_segment = b_segments[b_min_index]
                                #     if len(e_segments) != 0:
                                #         e_min_index = find_special_array(e_dists)[0][0]
                                #         e_list_number = e_list_numbers[e_min_index]
                                #         e_segment = e_segments[e_min_index]
                                #     else:
                                #         e_list_number = np.random.permutation(len(tmpCSIe1Ind))
                                # elif search_method == "cross":
                                #     # 按max_min搜索：每个匹配分段之间排除最小值，找出和其他所有分段最大的最小值
                                #     # 然后按照min最大->mean最大->var最小的次序排序
                                #     a_min_index = find_special_array_min_mean_var(a_dists)[0][0]
                                #     a_list_number = a_list_numbers[a_min_index]
                                #     a_segment = a_segments[a_min_index]
                                #     b_min_index = find_special_array_min_mean_var(b_dists)[0][0]
                                #     b_list_number = b_list_numbers[b_min_index]
                                #     b_segment = b_segments[b_min_index]
                                #     if len(e_segments) != 0:
                                #         e_min_index = find_special_array_min_mean_var(e_dists)[0][0]
                                #         e_list_number = e_list_numbers[e_min_index]
                                #         e_segment = e_segments[e_min_index]
                                #     else:
                                #         e_list_number = np.random.permutation(len(tmpCSIe1Ind))
                                # else:
                                #     # 先按照min最小->cross_min最大->mean最大的次序排序
                                #     a_min_index = find_special_array_min_min_mean_var(a_match_dists, a_cross_dists)[0][0]
                                #     a_list_number = a_list_numbers[a_min_index]
                                #     a_segment = a_segments[a_min_index]
                                #     b_min_index = find_special_array_min_min_mean_var(b_match_dists, b_cross_dists)[0][0]
                                #     b_list_number = b_list_numbers[b_min_index]
                                #     b_segment = b_segments[b_min_index]
                                #     if len(e_segments) != 0:
                                #         e_min_index = find_special_array_min_min_mean_var(e_match_dists, e_cross_dists)[0][0]
                                #         e_list_number = e_list_numbers[e_min_index]
                                #         e_segment = e_segments[e_min_index]
                                #     else:
                                #         e_list_number = np.random.permutation(len(tmpCSIe1Ind))
                                #
                                # final_search_end = time.time_ns()

                                ######################################### correction part #########################################

                                if isCorrect and a_list_number != b_list_number:
                                    if isExtraPrint:
                                        print("correction")
                                    key_hash = hash(tuple(a_list_number))

                                    for i in range(len(segment_length_B)):
                                        b_list_number_tmp, dists = search_index_with_segment(
                                            tmpCSIb1Ind, tmpCSIa1Ind, np.array(segment_length_B[i]).astype(int))
                                        if hash(tuple(b_list_number_tmp)) == key_hash:
                                            b_list_number = b_list_number_tmp
                                            b_segment = segment_length_B[i]
                                            break

                                    # 仍然不正确，如果Alice的推测分段不是原始分段的话，使用原始分段
                                    if a_list_number != b_list_number and segment_method_ori != a_segment:
                                        a_list_number, dists = search_index_with_segment(
                                            tmpCSIa1Bck, tmpCSIa1Ind, np.array(segment_method_ori).astype(int))
                                        if isExtraPrint:
                                            print("double correction")
                                        key_hash = hash(tuple(a_list_number))

                                        for i in range(len(b_list_numbers)):
                                            b_list_number_tmp = b_list_numbers[i]
                                            if hash(tuple(b_list_number_tmp)) == key_hash:
                                                b_list_number = b_list_number_tmp
                                                b_segment = segment_length_B[i]
                                                break

                                    # if a_list_number != b_list_number:
                                    #     print("未找到合适分段3")
                                    #     findError = True

                                # if a_list_number != b_list_number:
                                #     key_hash = hash(tuple(a_list_number))
                                #
                                #     for i in range(len(segment_length_B)):
                                #         b_list_number_tmp, dists = search_segment_method_with_metric(
                                #             tmpCSIb1Ind, tmpCSIa1Ind, np.array(segment_length_B[i]).astype(int))
                                #         if hash(tuple(b_list_number_tmp)) == key_hash:
                                #             b_list_number = b_list_number_tmp
                                #             b_segment = segment_length_B[i]
                                #             break

                                # if a_list_number != b_list_number:
                                #     print("mismatch")
                                #     print("a_list_number", a_list_number, "a_segment", a_segment,
                                #           "b_list_number", b_list_number, "b_segment", b_segment)
                                #     # 找出第二小的分段
                                #     if len(segment_length_A) >= 2:
                                #         a_min_index = find_special_array(a_dists)[1][0]
                                #         a_list_number = a_list_numbers[a_min_index]
                                #         a_segment = segment_length_A[a_min_index]
                                #     if len(segment_length_B) >= 2:
                                #         b_min_index = find_special_array(b_dists)[1][0]
                                #         b_list_number = b_list_numbers[b_min_index]
                                #         b_segment = segment_length_B[b_min_index]
                                # if a_list_number != b_list_number:
                                #     # 密钥有重复的元素
                                #     if np.array_equal(np.sort(b_list_number), list(range(0, len(b_list_number)))) is False:
                                #         print("b_repeat")
                                #         max_dist_B = np.inf
                                #         mean_dist_B = np.inf
                                #         var_dist_B = np.inf
                                #
                                #         b_list_numbers = []
                                #         b_dists = []
                                #         b_segments = []
                                #         for i in range(len(segment_length_B)):
                                #             b_list_number_tmp, dists = search_segment_method_with_metric(
                                #                 tmpCSIb1Ind, tmpCSIa1Ind, np.array(segment_length_B[i]).astype(int))
                                #             if np.array_equal(np.sort(b_list_number_tmp),
                                #                               list(range(0, len(b_list_number_tmp)))) is False:
                                #                 continue
                                #             b_list_numbers.append(b_list_number_tmp)
                                #             b_dists.append(dists)
                                #             b_segments.append(segment_length_B[i])
                                #             print("b_list_number", b_list_number_tmp, "segment", segment_length_B[i],
                                #                   "max", np.max(dists), "mean", np.mean(dists), "dists", dists)
                                #             b_min_index = find_special_array(b_dists)[0][0]
                                #             b_list_number = b_list_numbers[b_min_index]
                                #             b_segment = b_segments[b_min_index]
                                #     if np.array_equal(np.sort(a_list_number), list(range(0, len(a_list_number)))) is False:
                                #         print("a_repeat")
                                #         max_dist_A = np.inf
                                #         mean_dist_A = np.inf
                                #         var_dist_A = np.inf
                                #
                                #         a_list_numbers = []
                                #         a_dists = []
                                #         a_segments = []
                                #         for i in range(len(segment_length_A)):
                                #             a_list_number_tmp, dists = search_segment_method_with_metric(
                                #                 tmpCSIa1Bck, tmpCSIa1Ind, np.array(segment_length_A[i]).astype(int))
                                #             if np.array_equal(np.sort(a_list_number_tmp),
                                #                               list(range(0, len(a_list_number_tmp)))) is False:
                                #                 continue
                                #
                                #             a_list_numbers.append(a_list_number_tmp)
                                #             a_dists.append(dists)
                                #             a_segments.append(segment_length_A[i])
                                #             print("a_list_number", a_list_number_tmp, "segment", segment_length_A[i],
                                #                   "max", np.max(dists), "mean", np.mean(dists), "dists", dists)
                                #             a_min_index = find_special_array(a_dists)[0][0]
                                #             a_list_number = a_list_numbers[a_min_index]
                                #             a_segment = a_segments[a_min_index]
                                #     # 仍然有重复的
                                #     print("a_list_number", a_list_number, "a_segment", a_segment,
                                #           "b_list_number", b_list_number, "b_segment", b_segment)
                                # if a_list_number != b_list_number:
                                #     if np.array_equal(np.sort(a_list_number), list(range(0, len(a_list_number)))) is False:
                                #         print("a also repeat")
                                #         a_segment = find_opt_segment_method_from_candidate(
                                #             tmpCSIa1Bck, segLen - 1, segLen + 1, int(keyLen / segLen))
                                #         a_list_number = search_segment_method(
                                #             tmpCSIa1Bck, tmpCSIa1Ind, np.array(a_segment).astype(int),
                                #             int(keyLen / segLen / 2) - 1, 0)
                                #     if np.array_equal(np.sort(b_list_number), list(range(0, len(b_list_number)))) is False:
                                #         print("b also repeat")
                                #         b_segment = find_opt_segment_method_from_candidate(
                                #             tmpCSIb1Ind, segLen - 1, segLen + 1, int(keyLen / segLen))
                                #         b_list_number = search_segment_method(
                                #             tmpCSIb1Ind, tmpCSIa1Ind, np.array(b_segment).astype(int),
                                #             int(keyLen / segLen / 2) - 1, 0)
                                # if a_list_number != b_list_number:
                                #     # A推导出来的和自己的次优分段个数不一致时
                                #     if len(segment_method_ori) != len(a_segment):
                                #         print("unequal length")
                                #         a_segment = segment_method_ori
                                #         a_list_number = search_segment_method(
                                #             tmpCSIa1Bck, tmpCSIa1Ind, np.array(a_segment).astype(int),
                                #             int(keyLen / segLen / 2) - 1, 0)
                                #
                                # if a_list_number != b_list_number:
                                #     key_hash = hash(tuple(a_list_number))
                                #
                                #     for i in range(len(segment_length_B)):
                                #         b_list_number_tmp, dists = search_segment_method_with_metric(
                                #             tmpCSIb1Ind, tmpCSIa1Ind, np.array(segment_length_B[i]).astype(int))
                                #         if hash(tuple(b_list_number_tmp)) == key_hash:
                                #             b_list_number = b_list_number_tmp
                                #             b_segment = segment_length_B[i]
                                #             break

                                if findError is False:
                                    if isExtraPrint:
                                        print("a_list_number", a_list_number, "a_segment", a_segment,
                                          "b_list_number", b_list_number, "b_segment", b_segment)
                                        if np.array_equal(a_list_number, b_list_number) is False:
                                            if np.array_equal(a_segment, b_segment) is False:
                                                print("key mismatch and segment mismatch")
                                            else:
                                                print("key mismatch but segment match")
                                            print("threshold", threshold)
                                        else:
                                            print("threshold", threshold)
                                        if np.array_equal(a_segment, segment_method_ori) is False:
                                            print("cannot find original segment")
                                        print("segment time (s)", (segment_end - segment_start) / 1e9,
                                              "search time (s)", (search_end - search_start) / 1e9,
                                              "final search time (s)", (final_search_end - final_search_start) / 1e9)

                                    segment_time.append((segment_end - segment_start) / 1e9)
                                    search_time.append((search_end - search_start) / 1e9)
                                    final_search_time.append((final_search_end - final_search_start) / 1e9)

                                    sum1 = min(len(a_segment), len(b_segment))
                                    sum2 = 0

                                    for i in range(0, sum1):
                                        sum2 += (a_segment[i] == b_segment[i])

                                    originSegSum += sum1
                                    correctSegSum += sum2
                    else:
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

                    if findError:
                        if isExtraPrint:
                            print("退化成等长分段")
                        start_time = time.time_ns()
                        # 原方法：等长分段
                        tmpCSIa1IndReshape = np.array(tmpCSIa1Bck).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIa1Bck))
                        tmpCSIb1IndReshape = np.array(tmpCSIb1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIb1Ind))
                        tmpCSIe1IndReshape = np.array(tmpCSIe1Ind).reshape(int(keyLen / segLen), segLen, np.ndim(tmpCSIe1Ind))

                        permutation = list(range(int(keyLen / segLen)))
                        combineMetric = list(zip(tmpCSIa1IndReshape, permutation))
                        np.random.seed(staInd)
                        np.random.shuffle(combineMetric)
                        tmpCSIa1IndReshape, permutation = zip(*combineMetric)
                        tmpCSIa1Ind = np.hstack((tmpCSIa1IndReshape))

                        for i in range(int(keyLen / segLen)):
                            epiInda1 = tmpCSIa1IndReshape[i]

                            epiIndClosenessLsb = np.zeros(int(keyLen / segLen))
                            epiIndClosenessLse = np.zeros(int(keyLen / segLen))

                            for j in range(int(keyLen / segLen)):
                                epiIndb1 = tmpCSIb1IndReshape[j]
                                epiInde1 = tmpCSIe1IndReshape[j]

                                if withIndexValue:
                                    # epiIndClosenessLsb[j] = dtw_ndim.distance(epiIndb1, np.array(epiInda1))
                                    # epiIndClosenessLse[j] = dtw_ndim.distance(epiInde1, np.array(epiInda1))
                                    epiIndClosenessLsb[j] = np.sum(np.abs(epiIndb1 - np.array(epiInda1)))
                                    epiIndClosenessLse[j] = np.sum(np.abs(epiInde1 - np.array(epiInda1)))
                                else:
                                    # 欧式距离度量更好
                                    # epiIndClosenessLsb[j] = sum(np.square(epiIndb1 - np.array(epiInda1)))
                                    # epiIndClosenessLsb[j] = dtw_metric(epiIndb1, np.array(epiInda1))
                                    epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))
                                    # epiIndClosenessLsb[j] = distance.cosine(epiIndb1, np.array(epiInda1))
                                    # epiIndClosenessLsb[j] = abs(sum(epiIndb1) - sum(np.array(epiInda1)))

                                    # epiIndClosenessLse[j] = dtw_metric(epiInde1, np.array(epiInda1))
                                    epiIndClosenessLse[j] = sum(abs(epiInde1 - np.array(epiInda1)))

                            minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)
                            minEpiIndClosenessLse[i] = np.argmin(epiIndClosenessLse)

                        a_list_number = list(permutation)
                        b_list_number = list(minEpiIndClosenessLsb)
                        e_list_number = list(minEpiIndClosenessLse)

                        threshold = compute_threshold(tmpCSIa1Bck, tmpCSIb1Ind)
                        if isPrint:
                            print("sorted index", tmpCSIa1Bck, tmpCSIb1Ind)
                            print("threshold", threshold)
                            print("a_list_number", a_list_number, "b_list_number", b_list_number)
                            if np.array_equal(a_list_number, b_list_number) is False:
                                print("key mismatch and find error")
                                print("threshold", threshold)
                            else:
                                print("threshold", threshold)

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

                    if sum1 != sum2:
                        if isExtraPrint:
                            print("mismatch")

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

                                    for i in range(len(b_list_number)):
                                        number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
                                        b_list += number

                                    sum2 = 0
                                    for i in range(0, min(len(a_list), len(b_list))):
                                        sum2 += (a_list[i] == b_list[i])

                    # print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
                    originSum += sum1
                    correctSum += sum2
                    randomSum += sum3

                    originWholeSum += 1
                    correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
                    randomWholeSum = randomWholeSum + 1 if sum3 == sum1 else randomWholeSum

                    times += 1

                    all_key_number += 1

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
                # print("both_mean_with_matrix_pearson_ae", np.mean(both_mean_with_matrix_pearson_ae))
                print("before_add_mean_pearson_ab", np.mean(before_add_mean_pearson_ab))
                print("after_add_mean_pearson_ab", np.mean(after_add_mean_pearson_ab))
                print("before_add_mean_pearson_ae", np.mean(before_add_mean_pearson_ae))
                print("after_add_mean_pearson_ae", np.mean(after_add_mean_pearson_ae))
                print("entropy after", np.mean(entropy_a), np.var(entropy_a))
                print("max min dtw ab", np.mean(max_min_dtw_ab), np.var(max_min_dtw_ab))
                print("corr ab", np.mean(corr_ab), np.var(corr_ab))

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
                if isMean:
                    print("with mean")
                else:
                    print("without mean")
                if isFilter:
                    print("with filter")
                else:
                    print("without filter")
                if isPerm:
                    print("with permutation")
                else:
                    print("without permutation")
                if isInterleave:
                    print("with interleave")
                else:
                    print("without interleave")
                if isPerturb:
                    print("with perturb")
                else:
                    print("without perturb")
                if isLocalPerm:
                    print("with local permutation")
                else:
                    print("without local permutation")
                print("replace_number", replace_number)
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
                print("gaps number ratio", sum(all_gaps_number) / sum(all_intervals_number))
                print("gaps length ratio", sum(all_gaps_length) / sum(all_intervals_length))
                print("mean gap number and length", np.mean(all_gaps_number), np.mean(all_gaps_length))
                print("have gaps ratio", have_gaps_number / all_key_number)
                print("\n")
