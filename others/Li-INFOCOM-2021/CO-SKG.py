import csv
import math
import time

import graycode
import numpy as np
from pyentrp import entropy as ent
import scipy.signal
from dtw import dtw
from dtw import accelerated_dtw
from matplotlib import pyplot as plt
from scipy.fft import dct
from scipy.io import loadmat
from scipy.spatial import distance
from scipy.stats import pearsonr, boxcox
from sklearn.decomposition import PCA


def common_pca(ha, k):
    ha = (ha - np.min(ha)) / (np.max(ha) - np.min(ha))
    rha = np.dot(ha, ha.T) / len(ha)
    ua, sa, vha = np.linalg.svd(rha)
    vha = vha[:k, :]
    ya = vha @ ha
    return np.array(ya)

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


rawData = loadmat("../../data/data_static_outdoor_1.mat")

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
# stalking attack
CSIe2Orig = loadmat("../../data/data_static_indoor_1.mat")['A'][:, 0]
dataLen = min(len(CSIe2Orig), len(CSIa1Orig))

segLen = 6
keyLen = 64 * segLen

lossySum = 0
originSum = 0
correctSum = 0
randomSum1 = 0
randomSum2 = 0
noiseSum1 = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum1 = 0
randomWholeSum2 = 0
noiseWholeSum1 = 0

times = 0

for staInd in range(0, int(dataLen), keyLen):
    CSIa1Orig = rawData['A'][:, 0][0: dataLen]
    CSIb1Orig = rawData['A'][:, 1][0: dataLen]

    endInd = staInd + keyLen
    print("range:", staInd, endInd)
    if endInd >= len(CSIa1Orig):
        break
    times += 1

    np.random.seed(0)

    # imitation attack
    CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=len(CSIa1Orig))

    # CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
    # CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')
    # CSIe1Orig = smooth(np.array(CSIe1Orig), window_len=30, window='flat')
    # CSIe2Orig = smooth(np.array(CSIe2Orig), window_len=30, window='flat')

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
    tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
    tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]

    # inference attack
    tmpCSIn1 = np.random.normal(loc=0, scale=1, size=keyLen)

    # randomize
    tmpCSIa1 = tmpCSIa1 * tmpCSIn1
    tmpCSIb1 = tmpCSIb1 * tmpCSIn1
    tmpCSIe1 = tmpCSIe1 * tmpCSIn1
    tmpCSIe2 = tmpCSIe2 * tmpCSIn1

    # rearrangement
    tmpCSIa1Reshape = np.array(tmpCSIa1).reshape(int(keyLen / segLen), segLen)
    tmpCSIb1Reshape = np.array(tmpCSIb1).reshape(int(keyLen / segLen), segLen)
    tmpCSIe1Reshape = np.array(tmpCSIe1).reshape(int(keyLen / segLen), segLen)
    tmpCSIe2Reshape = np.array(tmpCSIe2).reshape(int(keyLen / segLen), segLen)
    tmpCSIn1Reshape = np.array(tmpCSIn1).reshape(int(keyLen / segLen), segLen)

    # tmpCSIa1Reshape1 = common_pca(tmpCSIa1Reshape, int(0.999 * len(tmpCSIa1Reshape[0])))
    # tmpCSIb1Reshape1 = common_pca(tmpCSIb1Reshape, int(0.999 * len(tmpCSIb1Reshape[0])))
    # tmpCSIe1Reshape1 = common_pca(tmpCSIe1Reshape, int(0.999 * len(tmpCSIe1Reshape[0])))
    # tmpCSIe2Reshape1 = common_pca(tmpCSIe2Reshape, int(0.999 * len(tmpCSIe2Reshape[0])))
    # tmpCSIn1Reshape1 = common_pca(tmpCSIn1Reshape, int(0.999 * len(tmpCSIn1Reshape[0])))
    # tmpCSIa1Reshape1 = tmpCSIa1Reshape1.reshape(1, -1)[0]
    # tmpCSIb1Reshape1 = tmpCSIb1Reshape1.reshape(1, -1)[0]
    # tmpCSIe1Reshape1 = tmpCSIe1Reshape1.reshape(1, -1)[0]
    # tmpCSIe2Reshape1 = tmpCSIe2Reshape1.reshape(1, -1)[0]
    # tmpCSIn1Reshape1 = tmpCSIn1Reshape1.reshape(1, -1)[0]

    # transform
    eta = 0.999
    pca = PCA(n_components=int(eta * len(tmpCSIa1Reshape[0])))
    tmpCSIa1Reshape = pca.fit_transform(tmpCSIa1Reshape)
    tmpCSIb1Reshape = pca.fit_transform(tmpCSIb1Reshape)
    tmpCSIe1Reshape = pca.fit_transform(tmpCSIe1Reshape)
    tmpCSIe2Reshape = pca.fit_transform(tmpCSIe2Reshape)
    tmpCSIn1Reshape = pca.fit_transform(tmpCSIn1Reshape)

    # adaptive quantization
    lw = 64
    tmpCSIa1Reshape = tmpCSIa1Reshape.reshape(1, -1)[0]
    tmpCSIb1Reshape = tmpCSIb1Reshape.reshape(1, -1)[0]
    tmpCSIe1Reshape = tmpCSIe1Reshape.reshape(1, -1)[0]
    tmpCSIe2Reshape = tmpCSIe2Reshape.reshape(1, -1)[0]
    tmpCSIn1Reshape = tmpCSIn1Reshape.reshape(1, -1)[0]

    tmpCSIa1Reshape = tmpCSIa1Reshape.reshape(int(len(tmpCSIa1Reshape) / lw), lw)
    tmpCSIb1Reshape = tmpCSIb1Reshape.reshape(int(len(tmpCSIb1Reshape) / lw), lw)
    tmpCSIe1Reshape = tmpCSIe1Reshape.reshape(int(len(tmpCSIe1Reshape) / lw), lw)
    tmpCSIe2Reshape = tmpCSIe2Reshape.reshape(int(len(tmpCSIe2Reshape) / lw), lw)
    tmpCSIn1Reshape = tmpCSIn1Reshape.reshape(int(len(tmpCSIn1Reshape) / lw), lw)

    # 最后各自的密钥
    a_list = []
    b_list = []
    e1_list = []
    e2_list = []
    n1_list = []

    a_list_number = []
    b_list_number = []
    e1_list_number = []
    e2_list_number = []
    n1_list_number = []

    alpha = 0.8
    for i in range(len(tmpCSIa1Reshape)):
        q1A1 = np.mean(tmpCSIa1Reshape[i]) + alpha * np.std(tmpCSIa1Reshape[i])
        q2A1 = np.mean(tmpCSIa1Reshape[i])
        q3A1 = np.mean(tmpCSIa1Reshape[i]) - alpha * np.std(tmpCSIa1Reshape[i])
        q1B1 = np.mean(tmpCSIb1Reshape[i]) + alpha * np.std(tmpCSIb1Reshape[i])
        q2B1 = np.mean(tmpCSIb1Reshape[i])
        q3B1 = np.mean(tmpCSIb1Reshape[i]) - alpha * np.std(tmpCSIb1Reshape[i])
        q1N1 = np.mean(tmpCSIn1Reshape[i]) + alpha * np.std(tmpCSIn1Reshape[i])
        q2N1 = np.mean(tmpCSIn1Reshape[i])
        q3N1 = np.mean(tmpCSIn1Reshape[i]) - alpha * np.std(tmpCSIn1Reshape[i])
        q1E1 = np.mean(tmpCSIe1Reshape[i]) + alpha * np.std(tmpCSIe1Reshape[i])
        q2E1 = np.mean(tmpCSIe1Reshape[i])
        q3E1 = np.mean(tmpCSIe1Reshape[i]) - alpha * np.std(tmpCSIe1Reshape[i])
        q1E2 = np.mean(tmpCSIe2Reshape[i]) + alpha * np.std(tmpCSIe2Reshape[i])
        q2E2 = np.mean(tmpCSIe2Reshape[i])
        q3E2 = np.mean(tmpCSIe2Reshape[i]) - alpha * np.std(tmpCSIe2Reshape[i])

        for j in range(len(tmpCSIa1Reshape[0])):
            if tmpCSIa1Reshape[i][j] > q1A1:
                a_list_number.append(0)
            elif tmpCSIa1Reshape[i][j] > q2A1:
                a_list_number.append(1)
            elif tmpCSIa1Reshape[i][j] > q3A1:
                a_list_number.append(2)
            elif tmpCSIa1Reshape[i][j] < q3A1:
                a_list_number.append(3)

            if tmpCSIb1Reshape[i][j] > q1B1:
                b_list_number.append(0)
            elif tmpCSIb1Reshape[i][j] > q2B1:
                b_list_number.append(1)
            elif tmpCSIb1Reshape[i][j] > q3B1:
                b_list_number.append(2)
            elif tmpCSIb1Reshape[i][j] < q3B1:
                b_list_number.append(3)

            if tmpCSIe1Reshape[i][j] > q1E1:
                e1_list_number.append(0)
            elif tmpCSIe1Reshape[i][j] > q2E1:
                e1_list_number.append(1)
            elif tmpCSIe1Reshape[i][j] > q3E1:
                e1_list_number.append(2)
            elif tmpCSIe1Reshape[i][j] < q3E1:
                e1_list_number.append(3)

            if tmpCSIe2Reshape[i][j] > q1E2:
                e2_list_number.append(0)
            elif tmpCSIe2Reshape[i][j] > q2E2:
                e2_list_number.append(1)
            elif tmpCSIe2Reshape[i][j] > q3E2:
                e2_list_number.append(2)
            elif tmpCSIe2Reshape[i][j] < q3E2:
                e2_list_number.append(3)

            if tmpCSIn1Reshape[i][j] > q1N1:
                n1_list_number.append(0)
            elif tmpCSIn1Reshape[i][j] > q2N1:
                n1_list_number.append(1)
            elif tmpCSIn1Reshape[i][j] > q3N1:
                n1_list_number.append(2)
            elif tmpCSIn1Reshape[i][j] < q3N1:
                n1_list_number.append(3)

    qa = []
    qb = []
    qe1 = []
    qe2 = []
    qn1 = []

    # gray码
    for i in range(len(a_list_number)):
        a_list += '{:02b}'.format(graycode.tc_to_gray_code(a_list_number[i]))
    for i in range(len(b_list_number)):
        b_list += '{:02b}'.format(graycode.tc_to_gray_code(b_list_number[i]))
    for i in range(len(e1_list_number)):
        e1_list += '{:02b}'.format(graycode.tc_to_gray_code(e1_list_number[i]))
    for i in range(len(e2_list_number)):
        e2_list += '{:02b}'.format(graycode.tc_to_gray_code(e2_list_number[i]))
    for i in range(len(n1_list_number)):
        n1_list += '{:02b}'.format(graycode.tc_to_gray_code(n1_list_number[i]))

    # print("keys of a:", len(a_list), a_list)
    print("keys of a:", len(a_list_number), a_list_number)
    # print("keys of b:", len(b_list), b_list)
    print("keys of b:", len(b_list_number), b_list_number)
    # print("keys of e:", len(e_list), e_list)
    print("keys of e1:", len(e1_list_number), e1_list_number)
    # print("keys of e:", len(e_list), e_list)
    print("keys of e2:", len(e2_list_number), e2_list_number)
    # print("keys of n1:", len(n1_list), n1_list)
    print("keys of n1:", len(n1_list_number), n1_list_number)

    print(ent.multiscale_entropy(a_list_number / np.max(a_list_number), 3, maxscale=1))

    sum1 = min(len(a_list), len(b_list))
    sum2 = 0
    sum31 = 0
    sum32 = 0
    sum41 = 0
    for i in range(0, sum1):
        sum2 += (a_list[i] == b_list[i])
    for i in range(min(len(a_list), len(e1_list))):
        sum31 += (a_list[i] == e1_list[i])
    for i in range(min(len(a_list), len(e2_list))):
        sum32 += (a_list[i] == e2_list[i])
    for i in range(min(len(a_list), len(n1_list))):
        sum41 += (a_list[i] == n1_list[i])

    print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
    print("a-e1", sum31, sum31 / sum1)
    print("a-e2", sum32, sum32 / sum1)
    print("a-n1", sum41, sum41 / sum1)
    originSum += sum1
    correctSum += sum2
    randomSum1 += sum31
    randomSum2 += sum32
    noiseSum1 += sum41

    originWholeSum += 1
    correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
    randomWholeSum1 = randomWholeSum1 + 1 if sum31 == sum1 else randomWholeSum1
    randomWholeSum2 = randomWholeSum2 + 1 if sum32 == sum1 else randomWholeSum2
    noiseWholeSum1 = noiseWholeSum1 + 1 if sum41 == sum1 else noiseWholeSum1

print("\033[0;32;40ma-b bit agreement rate", correctSum, "/", originSum, "=", round(correctSum / originSum, 10),
      "\033[0m")
print("a-e1 bit agreement rate", randomSum1, "/", originSum, "=", round(randomSum1 / originSum, 10))
print("a-e2 bit agreement rate", randomSum2, "/", originSum, "=", round(randomSum2 / originSum, 10))
print("a-n1 bit agreement rate", noiseSum1, "/", originSum, "=", round(noiseSum1 / originSum, 10))
print("\033[0;32;40ma-b key agreement rate", correctWholeSum, "/", originWholeSum, "=",
      round(correctWholeSum / originWholeSum, 10), "\033[0m")
print("a-e1 key agreement rate", randomWholeSum1, "/", originWholeSum, "=", round(randomWholeSum1 / originWholeSum, 10))
print("a-e2 key agreement rate", randomWholeSum2, "/", originWholeSum, "=", round(randomWholeSum2 / originWholeSum, 10))
print("a-n1 key agreement rate", noiseWholeSum1, "/", originWholeSum, "=", round(noiseWholeSum1 / originWholeSum, 10))
print("times", times)
print("all bits", lossySum)
print(originSum / len(CSIa1Orig))
print(correctSum / len(CSIa1Orig))

print(round(correctSum / originSum, 10), round(correctWholeSum / originWholeSum, 10), originSum / len(CSIa1Orig),
      correctSum / len(CSIa1Orig))
