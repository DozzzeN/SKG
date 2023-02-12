import math
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def sortSegPermOfA(data, segLen):
    data_seg = []
    data_back = data.copy()
    data_back.sort()
    # 排序后按照interval的长度分段
    for i in range(0, len(data_back), segLen):
        tmp = []
        for j in range(segLen):
            tmp.append(data_back[i + j])
        tmp.append(int(i / segLen))  # 原始的索引
        data_seg.append(tmp)
    # 找出间距最小的分段
    # data_seg_sort = []
    # for i in range(len(data_seg)):
    #     data_seg_sort.append(sum(data_seg[i][0: interval]))
    # data_seg_sort.sort()
    # min_diff = sys.maxsize
    # min_index = -1
    # for i in range(len(data_seg_sort) - 1):
    #     min_diff = min(min_diff, abs(data_seg_sort[i + 1] - data_seg_sort[i]))
    # for i in range(len(data_seg_sort) - 1):
    #     if math.isclose(min_diff, abs(data_seg_sort[i + 1] - data_seg_sort[i]), rel_tol=1e-5):
    #         min_index = i
    # for i in range(len(data_seg) - 1, - 1, -1):
    #     if data_seg[i][len(data_seg[i]) - 1] == min_index:
    #         del data_seg[i]
    # 删除
    # for i in range(len(data_seg) - 2, -1, -1):
    #     if sum(data_seg[i + 1][0: interval]) - sum(data_seg[i][0: interval]) < 0.15:
    #         del data_seg[i + 1]
    # 置换
    perm = np.random.permutation(list(range(len(data_seg))))
    data_seg = np.array(data_seg)[perm]
    data_seg = data_seg.tolist()
    publish = []
    data_seg_back = []
    for i in range(len(data_seg)):
        # data_seg_back.append(np.mean(data_seg[i][0: len(data_seg[i]) - 1]))
        data_seg_back.append(sum(data_seg[i][0: len(data_seg[i]) - 1]))
    for j in range(len(data_seg)):
        for k in range(len(data_seg[j]) - 1):
            for i in range(len(data)):
                if math.isclose(data[i], data_seg[j][k], rel_tol=1e-5):
                    publish.append(i)
                    # 不重复选
                    data[i] = sys.maxsize
                    break
    return data_seg_back, publish, list(perm)


def sortSegPermOfB(publish, data, segLen):
    data_perm = []
    for i in range(len(publish)):
        data_perm.append(data[publish[i]])
    data_seg = []
    for i in range(0, len(data_perm), segLen):
        # tmp = 0
        # sample = data_perm[i:i + segLen].copy()
        # sample.sort()
        # data_seg.append(sample[3] + sample[2] + sample[0] + sample[1])
        # 欧式距离，平方和
        # tmp = 0
        # for j in range(segLen):
        #     tmp += (data_perm[i + j] * data_perm[i + j])
        # data_seg.append(tmp)
        # 求和
        data_seg.append(sum(data_perm[i:i + segLen]))
        # 求积
        # tmp = 1
        # for j in range(segLen):
        #     tmp *= data_perm[i + j]
        # data_seg.append(tmp)
        # 采样式求平均
        # data_seg.append(sampleSum(data_perm[i:i + segLen], segLen - 2))
        # 中值滤波
        # data_seg.append(medianFilter(data_perm[i:i + segLen]))
    perm = list(np.argsort(data_seg))
    perm = list(np.argsort(perm))
    return data_seg, perm


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


start_time = time.time()
fileName = "../data/data_mobile_outdoor_1.mat"
rawData = loadmat(fileName)


CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

dataLen = len(CSIa1Orig)

segLen = 7
keyLen = 128 * segLen

originSum = 0
correctSum = 0
randomSum = 0
noiseSum = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum = 0
noiseWholeSum = 0

times = 0
addNoise = "mul"

for staInd in range(0, dataLen, keyLen):
    endInd = staInd + keyLen
    print("range:", staInd, endInd)
    if endInd >= len(CSIa1Orig):
        break
    times += 1

    CSIa1Orig = rawData['A'][:, 0]
    CSIb1Orig = rawData['A'][:, 1]

    CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
    CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')

    CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=len(CSIa1Orig))
    seed = np.random.randint(100000)
    np.random.seed(seed)
    if addNoise == "add":
        noiseOrig = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=len(CSIa1Orig))
        CSIa1Orig = (CSIa1Orig - np.mean(CSIa1Orig)) + noiseOrig
        CSIb1Orig = (CSIb1Orig - np.mean(CSIb1Orig)) + noiseOrig
        CSIe1Orig = (CSIe1Orig - np.mean(CSIe1Orig)) + noiseOrig
        CSIn1Orig = noiseOrig

        tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
        tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
        tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
        tmpNoise = CSIn1Orig[range(staInd, endInd, 1)]
    else:
        # 固定随机置换的种子
        # np.random.seed(0)
        # combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig))
        # np.random.shuffle(combineCSIx1Orig)
        # CSIa1Orig, CSIb1Orig = zip(*combineCSIx1Orig)

        CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
        CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')

        tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
        tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
        tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]

        # randomMatrix = np.random.normal(np.mean(tmpCSIa1), np.std(tmpCSIa1), size=(keyLen, keyLen))
        randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
        tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
        tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
        tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
        tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
        tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)
        tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix)
        tmpNoise = np.matmul(np.ones(keyLen), randomMatrix)

    # 最后各自的密钥
    a_list = []
    b_list = []
    e_list = []
    n_list = []

    a_metric, publish, a_list_number = sortSegPermOfA(list(tmpCSIa1), segLen)
    b_metric, b_list_number = sortSegPermOfB(publish, list(tmpCSIb1), segLen)
    e_metric, e_list_number = sortSegPermOfB(publish, list(tmpCSIe1), segLen)
    n_metric, n_list_number = sortSegPermOfB(publish, list(tmpNoise), segLen)

    # 转成二进制，0填充成0000
    for i in range(len(a_list_number)):
        number = bin(a_list_number[i])[2:].zfill(int(np.log2(len(a_list_number))))
        a_list += number
    for i in range(len(b_list_number)):
        number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
        b_list += number
    for i in range(len(e_list_number)):
        number = bin(e_list_number[i])[2:].zfill(int(np.log2(len(e_list_number))))
        e_list += number
    for i in range(len(n_list_number)):
        number = bin(n_list_number[i])[2:].zfill(int(np.log2(len(n_list_number))))
        n_list += number

    # 对齐密钥，随机补全
    for i in range(len(a_list) - len(e_list)):
        e_list += str(np.random.randint(0, 2))
    for i in range(len(a_list) - len(n_list)):
        n_list += str(np.random.randint(0, 2))

    # print("keys of a:", len(a_list), a_list)
    print("keys of a:", len(a_list_number), a_list_number)
    # print("keys of b:", len(b_list), b_list)
    print("keys of b:", len(b_list_number), b_list_number)
    # print("keys of e:", len(e_list), e_list)
    print("keys of e:", len(e_list_number), e_list_number)
    # print("keys of n:", len(n_list), n_list)
    print("keys of n:", len(n_list_number), n_list_number)

    sum1 = min(len(a_list), len(b_list))
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for i in range(0, sum1):
        sum2 += (a_list[i] == b_list[i])
    for i in range(min(len(a_list), len(e_list))):
        sum3 += (a_list[i] == e_list[i])
    for i in range(min(len(a_list), len(n_list))):
        sum4 += (a_list[i] == n_list[i])

    if sum1 == 0:
        continue
    if sum2 == sum1:
        print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
    else:
        print("\033[0;31;40ma-b", "bad", sum1 - sum2, sum2, sum2 / sum1, "\033[0m")
    print("a-e", sum3, sum3 / sum1)
    print("a-n", sum4, sum4 / sum1)
    print("----------------------")
    originSum += sum1
    correctSum += sum2
    randomSum += sum3
    noiseSum += sum4

    originWholeSum += 1
    correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
    randomWholeSum = randomWholeSum + 1 if sum3 == sum1 else randomWholeSum
    noiseWholeSum = noiseWholeSum + 1 if sum4 == sum1 else noiseWholeSum

print("a-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10))
print("a-e all", randomSum, "/", originSum, "=", round(randomSum / originSum, 10))
print("a-n all", noiseSum, "/", originSum, "=", round(noiseSum / originSum, 10))
print("a-b whole match", correctWholeSum, "/", originWholeSum, "=", round(correctWholeSum / originWholeSum, 10))
print("a-e whole match", randomWholeSum, "/", originWholeSum, "=", round(randomWholeSum / originWholeSum, 10))
print("a-n whole match", noiseWholeSum, "/", originWholeSum, "=", round(noiseWholeSum / originWholeSum, 10))
print("times", times)
print("测试结束，耗时" + str(round(time.time() - start_time, 3)), "s")
