import csv
import time
from tkinter import messagebox

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.stats import pearsonr


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


filename = "../data/data_static_outdoor_1.mat"
fileName = "data_static_outdoor_1"
rawData = loadmat("../data/" + fileName + ".mat")

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
dataLen = len(CSIa1Orig)

segLen = 5
keyLen = 128 * segLen
addNoise = ""

originSum = 0
correctSum = 0
randomSum = 0
noiseSum = 0

originDecSum = 0
correctDecSum = 0
randomDecSum = 0
noiseDecSum = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum = 0
noiseWholeSum = 0

times = 0
overhead = 0

codings = ""

iterate = 1

for iterate in range(1, 20):
    print(iterate)
    for staInd in range(0, dataLen, int(keyLen / iterate)):
        endInd = staInd + keyLen
        if endInd >= len(CSIa1Orig):
            break

        # np.random.seed(1)
        CSIa1Orig = rawData['A'][:, 0]
        CSIb1Orig = rawData['A'][:, 1]

        CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=len(CSIa1Orig))

        seed = np.random.randint(1000000)
        np.random.seed(seed)

        if addNoise == "add":
            noiseOrig = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=len(CSIa1Orig))
            CSIa1Orig = (CSIa1Orig - np.mean(CSIa1Orig)) + noiseOrig
            CSIb1Orig = (CSIb1Orig - np.mean(CSIb1Orig)) + noiseOrig

            tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
            tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
        else:
            tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
            tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]

            # randomMatrix = np.random.normal(np.mean(tmpCSIa1), np.std(tmpCSIa1), size=(keyLen, keyLen))
            randomMatrix = np.random.uniform(0, 1, size=(keyLen, keyLen))
            tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
            tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
            tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
            tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)

        # 最后各自的密钥
        a_list = []
        b_list = []

        tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
        tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()

        minEpiIndClosenessLsb = np.zeros(int(keyLen / segLen), dtype=int)

        tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)
        permutation = list(range(int(keyLen / segLen)))
        combineMetric = list(zip(tmpCSIa1IndReshape, permutation))
        np.random.seed(staInd)
        np.random.shuffle(combineMetric)
        tmpCSIa1IndReshape, permutation = zip(*combineMetric)
        tmpCSIa1Ind = np.hstack((tmpCSIa1IndReshape))

        for i in range(int(keyLen / segLen)):
            epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]
            epiIndClosenessLsb = np.zeros(int(keyLen / segLen))
            for j in range(int(keyLen / segLen)):
                epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
                epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))
            minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)

        a_list_number = list(permutation)
        b_list_number = list(minEpiIndClosenessLsb)

        # 转成二进制，0填充成0000
        for i in range(len(a_list_number)):
            number = bin(a_list_number[i])[2:].zfill(int(np.log2(len(a_list_number))))
            a_list += number
        for i in range(len(b_list_number)):
            number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
            b_list += number

        coding = ""
        for i in range(len(b_list)):
            coding += b_list[i]
        codings += coding + "\n"

    with open('./key/' + fileName + ".txt", 'a', ) as f:
        f.write(codings)
messagebox.showinfo("提示", "测试结束")