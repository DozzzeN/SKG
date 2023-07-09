from numpy.random import exponential as Exp
from scipy.stats import ortho_group
from scipy.stats import unitary_group

from scipy import linalg
from scipy.ndimage import gaussian_filter1d
# from scipy.stats.stats import pearsonr
from operator import eq
# from pyentrp import entropy as ent  ## Entropy calculation
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
from scipy import signal
from sklearn import preprocessing

import time
import os
import sys
import numpy as np
import random
import math
from matplotlib import pyplot as plt
from scipy import sparse, stats
from scipy.io import loadmat
from scipy.signal import savgol_filter
from scipy.linalg import circulant

def wthresh(data, threshold):
    for i in range(len(data)):
        if data[i] > threshold:
            data[i] = data[i] - threshold
        elif data[i] < -threshold:
            data[i] = data[i] + threshold
        else:
            data[i] = 0
    return data


def l2norm2(data):
    return np.sqrt(np.sum(np.square(data))) ** 2


# 给定A=A0+e1, b=A0h+e2, 求解x: Ax=b, 其中A0是已知的, e1, e2是噪声, h是未知的
def ass_pg_stls_f(A, b, N, K, lam, h, ni):
    # adaptive - step - size proximal - gradient
    AA = np.matmul(np.array(A).T, np.array(A))
    Ab = np.matmul(np.array(A).T, np.array(b))

    er2 = np.zeros(ni)  # error
    er0a = np.zeros(ni)  # missed detections
    er0b = np.zeros(ni)  # wrong detections
    xo = np.zeros(N)  # initialization of solution
    g = -2 * Ab
    mu0 = .2
    x = wthresh(-mu0 * g, mu0 * lam)
    y = 1 / (np.matmul(np.array(x).T, np.array(x)) + 1)
    c = y * l2norm2(np.matmul(A, x) - b)
    muo = mu0

    for nn in range(ni):
        # iterations loop
        # calculate gradient
        go = g  # g0
        co = c  # f1
        g = 2 * y * (np.matmul(AA, x) - Ab - co * x)  # gn

        #  calculate step - size
        if np.matmul(np.array(x - xo).T, (g - go)) == 0:
            mu = muo
        else:
            mus = np.matmul(np.array(x - xo).T, (x - xo)) / np.matmul(np.array(x - xo).T, (g - go))
            mum = np.matmul(np.array(x - xo).T, (g - go)) / np.matmul(np.array(g - go).T, (g - go))
            if mum / mus > .5:
                mu = mum
            else:
                mu = mus - mum / 2
            if mu <= 0:
                mu = muo

        # backtracking line-search
        while True:
            # proximal - gradient
            z = wthresh(x - mu * g, mu * lam)  # xn + 1
            y = 1 / (np.matmul(np.array(z).T, np.array(z)) + 1)
            c = y * l2norm2(np.matmul(A, z) - b)  # fn + 1
            if c <= co + np.matmul(np.array(z - x).T, g) + l2norm2(z - x) / (2 * mu):
                break
            mu = mu / 2
        muo = mu
        xo = x
        x = z

        # calculate errors
        er2[nn] = l2norm2(x - h)
        # ll = length(intersect(find(h), find(x)))
        # er0a(nn) = K - ll
        # er0b(nn) = length(find(x)) - ll
    return er2, er0a, er0b, x

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


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


dataLen = 256
CSIMean = 0
CSIVar = 10

inrArray = np.array(list(range(dataLen)))
freq = np.fft.fftfreq(inrArray.shape[-1])

rawData = loadmat('../../data/data_mobile_indoor_1.mat')
# plt.plot(rawData['A'][0:10000, 0])
# plt.show()
# exit()

matched1 = 0
matched2 = 0
matched3 = 0
matched4 = 0

matchede1 = 0
matchede2 = 0
allbits = 0

key1 = 0
key2 = 0
key3 = 0
key4 = 0
keye1 = 0
keye2 = 0
allkeys = 0
for staInd in range(0, len(rawData['A']), dataLen):
    endInd = staInd + dataLen
    # print("range:", staInd, endInd)
    if endInd >= len(rawData['A']):
        break

    CSIa1Orig = rawData['A'][staInd:endInd, 0]
    CSIb1Orig = rawData['A'][staInd:endInd, 1]

    # CSIOrig = np.random.normal(loc=0, scale=10, size=dataLen)
    # CSIOrig = CSIVar*np.sin(np.linspace(-np.pi*2, np.pi*2, dataLen)) + noiseOrig
    CSIOrig = CSIa1Orig
    CSIOrigP = CSIb1Orig
    CSIOrigE = np.random.normal(np.mean(CSIOrigP), np.std(CSIOrigP, ddof=1), dataLen)

    ## Smoothing denoising
    CSIOrig = savgol_filter(CSIOrig, 7, 1)
    CSIOrigFFT = np.fft.fft(CSIOrig)

    CSIOrigP = savgol_filter(CSIOrigP, 7, 1)
    CSIOrigFFTP = np.fft.fft(CSIOrigP)

    CSIOrigE = savgol_filter(CSIOrigE, 7, 1)
    CSIOrigFFTE = np.fft.fft(CSIOrigE)

    ## Multiplicative perturbation
    noiseOrigMx = np.random.normal(loc=CSIMean, scale=4, size=(dataLen, dataLen))
    # noiseOrigMx = np.random.uniform(0, 1, size=(dataLen,dataLen))

    CSIOrig = np.matmul(CSIOrig - np.mean(CSIOrig), noiseOrigMx)
    CSIOrigP = np.matmul(CSIOrigP - np.mean(CSIOrigP), noiseOrigMx)
    CSIOrigE = np.matmul(CSIOrigE - np.mean(CSIOrigE), noiseOrigMx)

    # CSIPerm = CSIOrig[np.random.permutation(dataLen)]
    # CSIPerm = np.sort(CSIOrig)
    # CSIPermFFT = np.fft.fft(CSIPerm)

    IndPerm = CSIOrig.argsort().argsort()
    IndPermFFT = np.fft.fft(IndPerm - np.mean(IndPerm))

    IndPermP = CSIOrigP.argsort().argsort()
    IndPermFFTP = np.fft.fft(IndPermP - np.mean(IndPermP))

    IndPermE = CSIOrigE.argsort().argsort()
    IndPermFFTE = np.fft.fft(IndPermE - np.mean(IndPermE))

    # plt.subplot(2,  2,  1)
    # plt.plot(NormalizeData(CSIOrig), 'b-')
    # plt.subplot(2,  2,  2)
    # plt.plot(NormalizeData(IndPerm), 'b-')

    # plt.subplot(2,  2,  3)
    # plt.plot(NormalizeData(CSIOrigP), 'r-')
    # plt.subplot(2,  2,  4)
    # plt.plot(NormalizeData(IndPermP), 'r-')
    # plt.show()

    ## Empirial distrbiution correlation;
    # R1 = np.corrcoef(NormalizeData(IndPerm), NormalizeData(IndPermP))
    # print(R1[0, 1])
    # R2 = np.corrcoef(NormalizeData(CSIOrig), NormalizeData(IndPerm))
    # print(R2[0, 1])
    # R3 = np.corrcoef(NormalizeData(CSIOrig), NormalizeData(CSIOrigP))
    # print(R3[0, 1])

    ## Quantization
    # bins = np.array([0.25, 0.5, 0.75])
    # inds1 = np.digitize(NormalizeData(IndPerm), bins)
    # inds2 = np.digitize(NormalizeData(IndPermP), bins)
    # print(inds1)
    # print(inds2)
    # print(inds1 - inds2)

    ## Generated key
    # keyBin = np.random.binomial(n=1, p=0.5, size=dataLen)
    keyBin = np.random.randint(1, 4, size=dataLen)
    # keyRand = np.random.uniform(0, 5, size=dataLen)

    ## Index Modulated key
    IndPermMx = circulant(IndPerm[::-1])
    IndPermPMx = circulant(IndPermP[::-1])
    IndPermEMx = circulant(IndPermE[::-1])
    IndKeyMx = np.dot(IndPermMx, keyBin)
    IndKeyMxP = np.dot(IndPermPMx, keyBin)

    ## CSI Modulated key
    CSIOrigMx = circulant(CSIOrig[::-1])
    CSIOrigPMx = circulant(CSIOrigP[::-1])
    CSIOrigEMx = circulant(CSIOrigE[::-1])
    # CSIKeyMx = np.dot(IndPermMx, keyBin)
    # CSIKeyMxP = np.dot(IndPermPMx, keyBin)
    CSIKeyMx = np.dot(CSIOrigMx, keyBin)
    CSIKeyMxP = np.dot(CSIOrigPMx, keyBin)

    # plt.plot(IndKeyMx)
    # plt.plot(IndKeyMxP)
    # plt.show()

    # plt.plot(CSIKeyMx)
    # plt.plot(CSIKeyMxP)
    # plt.show()

    ## least square solver
    # [_, _, _, lsq1] = ass_pg_stls_f(IndPermMx, IndKeyMx, dataLen, dataLen, 0.02, keyBin, 30)
    # [_, _, _, lsq2] = ass_pg_stls_f(IndPermPMx, IndKeyMxP, dataLen, dataLen, 0.02, keyBin, 30)
    # [_, _, _, lsqe1] = ass_pg_stls_f(IndPermEMx, IndPermE, dataLen, dataLen, 0.02, keyBin, 30)

    # lsq1 = np.linalg.lstsq(IndPermMx, IndKeyMx, rcond=None)
    # lsq2 = np.linalg.lstsq(IndPermPMx, IndKeyMx, rcond=None)
    # lsqe1 = np.linalg.lstsq(IndPermEMx, IndKeyMx, rcond=None)
    # lsq1 = np.linalg.tensorsolve(IndPermMx, IndKeyMxP)
    # lsq2 = np.linalg.tensorsolve(IndPermPMx, IndKeyMx)
    # print(np.around(lsq1[0] - keyBin))
    # print(np.around(lsq2[0] - keyBin))

    matched1 += np.count_nonzero(np.around(lsq1[0] - keyBin))
    matched2 += np.count_nonzero(np.around(lsq2[0] - keyBin))
    matchede1 += np.count_nonzero(np.around(lsqe1[0] - keyBin))

    lsq3 = np.linalg.lstsq(CSIOrigMx, CSIKeyMx, rcond=None)
    lsq4 = np.linalg.lstsq(CSIOrigPMx, CSIKeyMx, rcond=None)
    lsqe2 = np.linalg.lstsq(CSIOrigEMx, CSIKeyMx, rcond=None)
    # lsq3 = np.linalg.tensorsolve(CSIOrigMx, CSIKeyMxP)
    # lsq4 = np.linalg.tensorsolve(CSIOrigPMx, CSIKeyMx)
    # print(np.around(lsq3[0] - keyBin))
    # print(np.around(lsq4[0] - keyBin))

    matched3 += np.count_nonzero(np.around(lsq3[0] - keyBin))
    matched4 += np.count_nonzero(np.around(lsq4[0] - keyBin))
    matchede2 += np.count_nonzero(np.around(lsqe2[0] - keyBin))

    allbits += len(keyBin)

    if matched1 == 0:
        key1 += 1
    if matched2 == 0:
        key2 += 1
    if matched3 == 0:
        key3 += 1
    if matched4 == 0:
        key4 += 1
    if matchede1 == 0:
        keye1 += 1
    if matchede2 == 0:
        keye2 += 1

    allkeys += 1

    # print(np.count_nonzero(np.around(lsq2[0] - keyBin)))
    # print(np.count_nonzero(np.around(lsq4[0] - keyBin)))
    # print("---------------")

print("bit match")
print(1 - matched1 / allbits)
print(1 - matched2 / allbits)
print(1 - matched3 / allbits)
print(1 - matched4 / allbits)

print("eve bit match")
print(1 - matchede1 / allbits)
print(1 - matchede2 / allbits)

print("key match")
print(key1 / allkeys)
print(key2 / allkeys)
print(key3 / allkeys)
print(key4 / allkeys)

print("eve key match")
print(keye1 / allkeys)
print(keye2 / allkeys)
