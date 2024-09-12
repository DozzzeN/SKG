from numpy.random import exponential as Exp
from scipy.stats import ortho_group, pearsonr
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

dataLen = 100000
CSIMean = 0
CSIVar = 10

inrArray = np.array(list(range(dataLen)))
freq = np.fft.fftfreq(inrArray.shape[-1])

noiseOrig = np.random.normal(loc=CSIMean, scale=4, size=dataLen)

# CSIOrig = np.random.normal(loc=0, scale=10, size=dataLen)
CSIOrig = CSIVar * np.sin(np.linspace(-np.pi * 2, np.pi * 2, dataLen)) + noiseOrig
CSIOrig = savgol_filter(CSIOrig, 5, 1)
CSIOrigFFT = np.fft.fft(CSIOrig)

# CSIPerm = CSIOrig[np.random.permutation(dataLen)]
# CSIPerm = np.sort(CSIOrig)
# CSIPermFFT = np.fft.fft(CSIPerm)

IndPerm = CSIOrig.argsort().argsort()
IndPermFFT = np.fft.fft(IndPerm - np.mean(IndPerm))

CSIOrig = (CSIOrig - np.min(CSIOrig)) / (np.max(CSIOrig) - np.min(CSIOrig))
IndPerm = (IndPerm - np.min(IndPerm)) / (np.max(IndPerm) - np.min(IndPerm))

plt.figure()
plt.hist(CSIOrig, cumulative=True, density=True)
plt.show()

plt.figure()
plt.hist(IndPerm, cumulative=True, density=True)
plt.show()

print(pearsonr(CSIOrig, IndPerm)[0])
print(pearsonr(abs(CSIOrigFFT), abs(IndPermFFT))[0])

CSIOrig2 = CSIOrig * 2
CSIOrigFFT2 = np.fft.fft(CSIOrig2)
IndPerm2 = CSIOrig2.argsort().argsort()
IndPermFFT2 = np.fft.fft(IndPerm2 - np.mean(IndPerm2))

plt.subplot(1, 2, 1)
plt.plot(CSIOrig)

plt.subplot(1, 2, 2)
plt.plot(IndPerm)
plt.show()

print(pearsonr(CSIOrig2, IndPerm2)[0])
print(pearsonr(abs(CSIOrigFFT2), abs(IndPermFFT2))[0])

print()
print(pearsonr(abs(CSIOrigFFT), abs(CSIOrigFFT2))[0])
exit()

# plt.subplot(2,  2,  1)
# plt.plot(CSIOrig, 'b-') 
# plt.subplot(2,  2,  2)
# plt.plot(CSIPerm, 'b-')


# plt.subplot(2,  2,  3)
# plt.plot(freq, abs(CSIOrigFFT), 'r-') 
# plt.subplot(2,  2,  4)
# plt.plot(freq, abs(CSIPermFFT), 'r-')
# plt.show() 

plt.subplot(2, 2, 1)
plt.plot(CSIOrig, 'b-')
plt.subplot(2, 2, 2)
plt.plot(CSIPerm, IndPerm, 'b-')

# print(IndPerm)

plt.subplot(2, 2, 3)
plt.plot(abs(CSIOrigFFT), 'r-')
plt.subplot(2, 2, 4)
plt.plot(abs(IndPermFFT), 'r-')
plt.show()
