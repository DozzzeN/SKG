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


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


dataLen = 256
CSIMean = 0
CSIVar = 10

inrArray = np.array(list(range(dataLen)))   
freq = np.fft.fftfreq(inrArray.shape[-1])

rawData = loadmat('../data/data_mobile_indoor_1.mat')
# plt.plot(rawData['A'][0:10000, 0])
# plt.show()
# exit()

for ofst in range(0, 10000, 50):
    print(ofst)
    CSIa1Orig = rawData['A'][ofst:ofst+dataLen, 0]
    CSIb1Orig = rawData['A'][ofst:ofst+dataLen, 1]


    # CSIOrig = np.random.normal(loc=0, scale=10, size=dataLen) 
    # CSIOrig = CSIVar*np.sin(np.linspace(-np.pi*2, np.pi*2, dataLen)) + noiseOrig
    CSIOrig = CSIa1Orig
    CSIOrigP = CSIb1Orig


    ## Smoothing denoising
    CSIOrig = savgol_filter(CSIOrig, 7, 1)
    CSIOrigFFT = np.fft.fft(CSIOrig)

    CSIOrigP = savgol_filter(CSIOrigP, 7, 1)
    CSIOrigFFTP = np.fft.fft(CSIOrigP)


    ## Multiplicative perturbation
    noiseOrigMx = np.random.normal(loc=CSIMean, scale=4, size=(dataLen,dataLen)) 
    # noiseOrigMx = np.random.uniform(0, 1, size=(dataLen,dataLen))

    CSIOrig = np.matmul(CSIOrig - np.mean(CSIOrig), noiseOrigMx)
    CSIOrigP = np.matmul(CSIOrigP - np.mean(CSIOrigP), noiseOrigMx)


    # CSIPerm = CSIOrig[np.random.permutation(dataLen)]
    # CSIPerm = np.sort(CSIOrig)
    # CSIPermFFT = np.fft.fft(CSIPerm)

    IndPerm = CSIOrig.argsort().argsort()
    IndPermFFT = np.fft.fft(IndPerm-np.mean(IndPerm))

    IndPermP = CSIOrigP.argsort().argsort()
    IndPermFFTP = np.fft.fft(IndPermP-np.mean(IndPermP))


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
    bins = np.array([0.25, 0.5, 0.75])
    inds1 = np.digitize(NormalizeData(IndPerm), bins)
    inds2 = np.digitize(NormalizeData(IndPermP), bins)
    # print(inds1)
    # print(inds2)
    print(np.count_nonzero(inds1 - inds2))

    ## Generated key
    keyBin = np.random.binomial(n=1, p=0.5, size=dataLen)
    # keyBin = np.random.randint(1, 4, size=dataLen)
    # keyRand = np.random.uniform(0, 5, size=dataLen)

    ## Index Modulated key
    IndPermMx = circulant(IndPerm[::-1])
    IndPermPMx = circulant(IndPermP[::-1])
    IndKeyMx = np.dot(IndPermMx,keyBin)
    IndKeyMxP = np.dot(IndPermPMx, keyBin)

    ## CSI Modulated key
    CSIOrigMx = circulant(CSIOrig[::-1])
    CSIOrigPMx = circulant(CSIOrigP[::-1])
    CSIKeyMx = np.dot(CSIOrigMx,keyBin)
    CSIKeyMxP = np.dot(CSIOrigPMx, keyBin)

    # plt.plot(IndKeyMx)
    # plt.plot(IndKeyMxP)
    # plt.show()

    # plt.plot(CSIKeyMx)
    # plt.plot(CSIKeyMxP)
    # plt.show()

    ## least square solver
    lsq1 = np.linalg.lstsq(IndPermMx, IndKeyMxP, rcond=None)
    lsq2 = np.linalg.lstsq(IndPermPMx, IndKeyMx, rcond=None)
    # lsq1 = np.linalg.tensorsolve(IndPermMx, IndKeyMxP)
    # lsq2 = np.linalg.tensorsolve(IndPermPMx, IndKeyMx)
    # print(np.around(lsq1[0] - keyBin))
    # print(np.around(lsq2[0] - keyBin))

    print(np.count_nonzero(np.around(lsq1[0] - keyBin)))
    print(np.count_nonzero(np.around(lsq2[0] - keyBin)))

    lsq3 = np.linalg.lstsq(CSIOrigMx, CSIKeyMxP, rcond=None)
    lsq4 = np.linalg.lstsq(CSIOrigPMx, CSIKeyMx, rcond=None)
    # lsq3 = np.linalg.tensorsolve(CSIOrigMx, CSIKeyMxP)
    # lsq4 = np.linalg.tensorsolve(CSIOrigPMx, CSIKeyMx)
    # print(np.around(lsq3[0] - keyBin))
    # print(np.around(lsq4[0] - keyBin))

    print(np.count_nonzero(np.around(lsq3[0] - keyBin)))
    print(np.count_nonzero(np.around(lsq4[0] - keyBin)))
    print('-------------')


