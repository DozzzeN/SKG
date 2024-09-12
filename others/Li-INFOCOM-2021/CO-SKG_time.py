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

# 5*8为64bit密钥
# 5*8 6*16 7*32 8*64 9*128 10*256
# 64 160 384 896 2048 4608
segLen = 10
keyLen = 256 * segLen

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

    start_time = time.time()

    # transform
    eta = 0.999
    # eta = 0.80
    pca = PCA(n_components=int(eta * len(tmpCSIa1Reshape[0])))
    tmpCSIa1Reshape = pca.fit_transform(tmpCSIa1Reshape)

    # adaptive quantization
    lw = int(keyLen / segLen)
    tmpCSIa1Reshape = tmpCSIa1Reshape.reshape(1, -1)[0]

    tmpCSIa1Reshape = tmpCSIa1Reshape.reshape(int(len(tmpCSIa1Reshape) / lw), lw)

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

    alpha = 0.2
    for i in range(len(tmpCSIa1Reshape)):
        q1A1 = np.mean(tmpCSIa1Reshape[i]) + alpha * np.std(tmpCSIa1Reshape[i])
        q2A1 = np.mean(tmpCSIa1Reshape[i])
        q3A1 = np.mean(tmpCSIa1Reshape[i]) - alpha * np.std(tmpCSIa1Reshape[i])

        guard = 0.08
        drops = []
        for j in range(len(tmpCSIa1Reshape[0])):
            if tmpCSIa1Reshape[i][j] > q1A1 + guard:
                pass
            elif tmpCSIa1Reshape[i][j] > q2A1 + guard and tmpCSIa1Reshape[i][j] < q1A1 - guard:
                pass
            elif tmpCSIa1Reshape[i][j] > q3A1 + guard and tmpCSIa1Reshape[i][j] < q2A1 - guard:
                pass
            elif tmpCSIa1Reshape[i][j] < q3A1 - guard:
                pass
            else:
                drops.append(j)

        for j in range(len(tmpCSIa1Reshape[0])):
            if j in drops:
                continue
            elif tmpCSIa1Reshape[i][j] > q1A1 + guard:
                a_list_number.append(0)
            elif tmpCSIa1Reshape[i][j] > q2A1 + guard and tmpCSIa1Reshape[i][j] < q1A1 - guard:
                a_list_number.append(1)
            elif tmpCSIa1Reshape[i][j] > q3A1 + guard and tmpCSIa1Reshape[i][j] < q2A1 - guard:
                a_list_number.append(2)
            elif tmpCSIa1Reshape[i][j] < q3A1 - guard:
                a_list_number.append(3)

    # gray码
    for i in range(len(a_list_number)):
        a_list += '{:02b}'.format(graycode.tc_to_gray_code(a_list_number[i]))

    print("keys of a:", len(a_list), a_list)
    print("keys of a:", len(a_list_number), a_list_number)

    end_time = time.time() - start_time
    print("time", round(end_time, 9))
    overhead += end_time

print("times", times)
print("overhead", round(overhead / times, 9))
