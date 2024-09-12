import csv
import math
import sys
import time

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


rawData = loadmat("../../data/data_static_indoor_1.mat")

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
# stalking attack
CSIe2Orig = loadmat("../../data/data_static_indoor_1.mat")['A'][:, 0]
dataLen = min(len(CSIe2Orig), len(CSIa1Orig))

segLen = 7  # l=5, lb=6
keyLen = 128 * segLen

lossySum = 0
originSum = 0
correctSum = 0
randomSum1 = 0
randomSum2 = 0
noiseSum1 = 0

originDecSum = 0
correctDecSum = 0
randomDecSum1 = 0
randomDecSum2 = 0
noiseDecSum1 = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum1 = 0
randomWholeSum2 = 0
noiseWholeSum1 = 0

times = 0
overhead = 0

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

    CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
    CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')
    CSIe1Orig = smooth(np.array(CSIe1Orig), window_len=30, window='flat')
    CSIe2Orig = smooth(np.array(CSIe2Orig), window_len=30, window='flat')

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
    tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
    tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]

    # inference attack
    tmpCSIn1 = np.random.random(keyLen)

    # 最后各自的密钥
    o_list = []  # lossy quantization
    a_list = []
    b_list = []
    e1_list = []
    e2_list = []
    n1_list = []

    o_list_number = []
    a_list_number = []
    b_list_number = []
    e1_list_number = []
    e2_list_number = []
    n1_list_number = []

    tmpCSIn1Reshape = tmpCSIn1[:keyLen]
    tmpCSIa1Reshape = tmpCSIa1[:keyLen]
    tmpCSIb1Reshape = tmpCSIb1[:keyLen]
    tmpCSIe1Reshape = tmpCSIe1[:keyLen]
    tmpCSIe2Reshape = tmpCSIe2[:keyLen]

    tmpCSIa1Reshape = np.array(tmpCSIa1Reshape).reshape(int(keyLen / segLen), segLen)
    tmpCSIb1Reshape = np.array(tmpCSIb1Reshape).reshape(int(keyLen / segLen), segLen)
    tmpCSIe1Reshape = np.array(tmpCSIe1Reshape).reshape(int(keyLen / segLen), segLen)
    tmpCSIe2Reshape = np.array(tmpCSIe2Reshape).reshape(int(keyLen / segLen), segLen)
    tmpCSIn1Reshape = np.array(tmpCSIn1Reshape).reshape(int(keyLen / segLen), segLen)

    min_q = sys.maxsize

    start_time = time.time()

    for i in range(int(keyLen / segLen)):
        for j in range(1, segLen):
            min_q = min(min_q, abs(tmpCSIa1Reshape[i][j] - tmpCSIa1Reshape[i][0]))
            deltaA = 1 if tmpCSIa1Reshape[i][j] - tmpCSIa1Reshape[i][0] > 0 else 0
            a_list_number.append(deltaA)
        for j in range(int((segLen - 1) / 2)):
            min_q = min(min_q, abs(abs(tmpCSIa1Reshape[i][j] - tmpCSIa1Reshape[i][0]) - abs(
                tmpCSIa1Reshape[i][j + 1] - tmpCSIa1Reshape[i][0])))
            deltaA = 1 if abs(tmpCSIa1Reshape[i][j] - tmpCSIa1Reshape[i][0]) - abs(
                tmpCSIa1Reshape[i][j + 1] - tmpCSIa1Reshape[i][0]) > 0 else 0
            a_list_number.append(deltaA)

    print("keys of a:", len(a_list_number), a_list_number)

    end_time = time.time() - start_time
    print("time", round(end_time, 9))
    overhead += end_time

print("times", times)
print("overhead", round(overhead / times, 9))
