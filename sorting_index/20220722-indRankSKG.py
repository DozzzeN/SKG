# from mwmatching import maxWeightMatching
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

# from tsfresh.feature_extraction.feature_calculators import absolute_sum_of_changes as asc
# from tsfresh.feature_extraction.feature_calculators import abs_energy as ae
# from tsfresh.feature_extraction.feature_calculators import approximate_entropy as aen

# from tsfresh.feature_extraction.feature_calculators import cid_ce as cid

# from tsfresh.feature_extraction.feature_calculators import sample_entropy as se

# from tsfresh.feature_extraction.feature_calculators import agg_autocorrelation as aga
# from tsfresh.feature_extraction.feature_calculators import autocorrelation as ar
# from tsfresh.feature_extraction.feature_calculators import c3 
# from tsfresh.feature_extraction.feature_calculators import count_below_mean as cbm

# from tsfresh.feature_extraction.feature_calculators import skewness
# from tsfresh.feature_extraction.feature_calculators import linear_trend as lt
# from tsfresh.feature_extraction.feature_calculators import mean_abs_change as mac
# from tsfresh.feature_extraction.feature_calculators import mean_change as mc

# from tsfresh.feature_extraction.feature_calculators import mean_second_derivative_central as msdc

# from tsfresh.feature_extraction.feature_calculators import root_mean_square as rms


import time

# import networkx as nx
import os
# import cv2
import sys
import numpy as np
import random
import math
from matplotlib import pyplot as plt
from scipy import sparse, stats
from scipy.io import loadmat
from scipy.signal import savgol_filter


def random_instance(a, b):
    """Generate an bipartite minimum-weight matching
    instance with random Exp(1) edge weights between
    {0, ..., a - 1} and {a, ..., a + b - 1}.
    """
    edges = []
    for ii in range(a):
        for jj in range(a, a + b):
            edges.append([ii, jj, Exp(1.)])

    return edges


def hp_filter(x, lamb=5000):
    w = len(x)
    b = [[1] * w, [-2] * w, [1] * w]
    D = sparse.spdiags(b, [0, 1, 2], w - 2, w)
    I = sparse.eye(w)
    B = (I + lamb * (D.transpose() * D))
    return sparse.linalg.dsolve.spsolve(B, x)


def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    elif window == 'kaiser':
        beta = 5
        w = eval('np.' + window + '(window_len, beta)')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def sumSeries(CSITmp):
    if len(CSITmp) > 1:
        sumCSI = sum(CSITmp) + sumSeries(CSITmp[0:-1])
        return sumCSI
    else:
        return CSITmp[0]


## -----------------------------------
plt.close('all')

rawData = loadmat('../data/data_static_indoor_1.mat')

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
dataLen = len(CSIa1Orig)

# rawData = loadmat('data_mobile_indoor_2.mat')
# CSIb2Orig = rawData['A'][0:dataLen, 1]
CSIb2Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig), size=dataLen)

# CSIa1Orig = CSIa1Orig[1:] - CSIa1Orig[:-1]
# CSIb1Orig = CSIb1Orig[1:] - CSIb1Orig[:-1]
# CSIb2Orig = CSIb2Orig[1:] - CSIb2Orig[:-1]

# # # ----------- Simulated data ---------------
# CSIa1Orig = CSIa1Orig + np.random.normal(loc=0, scale=1, size=dataLen)
# CSIb1Orig = CSIa1Orig + np.random.normal(loc=0, scale=1, size=dataLen)


# # -----------------------------------
# # ---- Smoothing -------------
# ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser']
# CSIa1Orig = smooth(CSIa1Orig, window_len = 5, window = 'flat')
# CSIb1Orig = smooth(CSIb1Orig, window_len = 5, window = 'flat')
# CSIb2Orig = smooth(CSIb2Orig, window_len = 5, window = 'flat')

# CSIa1Orig = smooth(CSIa1Orig, window_len = 9, window = 'bartlett')
# CSIb1Orig = smooth(CSIb1Orig, window_len = 9, window = 'bartlett')
# CSIb2Orig = smooth(CSIb2Orig, window_len = 9, window = 'bartlett')

# CSIa1Orig = hp_filter(CSIa1Orig, lamb=15)
# CSIb1Orig = hp_filter(CSIb1Orig, lamb=15)
# CSIb2Orig = hp_filter(CSIb2Orig, lamb=15)


# CSIa1Orig = savgol_filter(CSIa1Orig, 5, 1)
# CSIb1Orig = savgol_filter(CSIb1Orig, 5, 1)
# CSIb2Orig = savgol_filter(CSIb2Orig, 5, 1)

# -----------------------------------------------------------------------------------
#     ---- Constant Noise Generation ----   
#  Pre-allocated noise, will not change during sorting and matching:
#  Use the following noise, need to comment the ines of "Instant noise generator"
# -----------------------------------------------------------------------------------
CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
CSIb2OrigBack = CSIb2Orig.copy()
dataLen = len(CSIa1Orig)

paraLs = [7, 8, 9]
paraRate = np.zeros((3, len(paraLs)))

for para in paraLs:
    print(para)

    ## ---------------------------------------------------------
    intvl = para
    keyLen = 256
    correctRate = []
    randomRate = []
    noiseRate = []

    # noiseOrigMx = np.random.uniform(0, 3, size=(keyLen*intvl,keyLen*intvl))

    for staInd in range(0, dataLen - keyLen * intvl, keyLen * intvl):
        print(staInd)
        staInd = 0  # fixed start for testing
        endInd = staInd + keyLen * intvl

        # --------------------------------------------
        # BEGIN: Ranking SKG
        # --------------------------------------------
        tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
        tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
        tmpCSIb2 = CSIb2Orig[range(staInd, endInd, 1)]
        epiLen = len(range(staInd, endInd, 1))

        tmpCSIa1Back = tmpCSIa1.copy()
        tmpCSIb1Back = tmpCSIb1.copy()
        tmpCSIb2Back = tmpCSIb2.copy()

        ## --------------------------
        # ## Addtive noise
        # # noiseOrigAdd = np.random.normal(loc=0, scale=np.std(tmpCSIa1), size=epiLen)
        # noiseOrigAdd = np.random.uniform(0, np.std(tmpCSIa1), size=epiLen)

        # tmpCSIa1 = (tmpCSIa1 - np.mean(tmpCSIa1)) + noiseOrigAdd
        # tmpCSIb1 = (tmpCSIb1 - np.mean(tmpCSIb1)) + noiseOrigAdd
        # # tmpCSIb2 = (tmpCSIb2 - np.mean(tmpCSIb2)) + noiseOrigAdd

        # tmpCSIb2 = noiseOrigAdd

        ## Multiplicative noise
        noiseOrigMx = np.random.normal(-1, 1, size=(epiLen, epiLen))
        # noiseOrigMx = np.random.uniform(0, 3, size=(epiLen,epiLen))

        # noiseOrigMx = unitary_group.rvs(epiLen)

        tmpCSIa1 = np.matmul(tmpCSIa1 - np.mean(tmpCSIa1), noiseOrigMx)
        tmpCSIb1 = np.matmul(tmpCSIb1 - np.mean(tmpCSIb1), noiseOrigMx)
        # tmpCSIb2 = np.matmul(tmpCSIb2 - np.mean(tmpCSIb2), noiseOrigMx)
        tmpCSIb2 = np.matmul(np.ones(epiLen), noiseOrigMx)

        ## --------------------------
        ## For indics closeness;
        tmpCSIa1Ind = tmpCSIa1.argsort().argsort()
        tmpCSIb1Ind = tmpCSIb1.argsort().argsort()
        tmpCSIb2Ind = tmpCSIb2.argsort().argsort()

        minEpiIndClosenessLs = np.zeros(keyLen)
        minEpiIndClosenessLsAttack = np.zeros(keyLen)

        for pp in range(0, keyLen, 1):
            epiInda1 = tmpCSIa1Ind[pp * intvl:(pp + 1) * intvl]

            epiIndClosenessLs = np.zeros(keyLen)
            epiIndClosenessLsAttack = np.zeros(keyLen)

            for qq in range(0, keyLen, 1):
                epiIndb1 = tmpCSIb1Ind[qq * intvl:(qq + 1) * intvl]
                epiIndb2 = tmpCSIb2Ind[qq * intvl:(qq + 1) * intvl]

                epiIndClosenessLs[qq] = distance.cityblock(epiInda1, epiIndb1)
                epiIndClosenessLsAttack[qq] = distance.cityblock(epiInda1, epiIndb2)

            minEpiIndClosenessLs[pp] = np.argmin(epiIndClosenessLs)
            minEpiIndClosenessLsAttack[pp] = np.argmin(epiIndClosenessLsAttack)

        ## --------------------------
        ## For value closeness;
        tmpCSIa1 = tmpCSIa1Back
        tmpCSIb1 = tmpCSIb1Back
        tmpCSIb2 = tmpCSIb2Back

        tmpCSIa1 = np.matmul(np.sort(tmpCSIa1) - np.mean(tmpCSIa1), noiseOrigMx)
        tmpCSIb1 = np.matmul(np.sort(tmpCSIb1) - np.mean(tmpCSIb1), noiseOrigMx)
        # tmpCSIb2 = np.matmul(np.sort(tmpCSIb2) - np.mean(tmpCSIb2), noiseOrigMx)
        tmpCSIb2 = np.matmul(np.ones(epiLen), noiseOrigMx)

        # tmpCSIa1 = (tmpCSIa1 - np.min(tmpCSIa1)) / (np.max(tmpCSIa1) - np.min(tmpCSIa1))
        # tmpCSIb1 = (tmpCSIb1 - np.min(tmpCSIb1)) / (np.max(tmpCSIb1) - np.min(tmpCSIb1))
        # tmpCSIb2 = (tmpCSIb2 - np.min(tmpCSIb2)) / (np.max(tmpCSIb2) - np.min(tmpCSIb2))

        # tmpCSIa1 = np.matmul(tmpCSIa1 - np.mean(tmpCSIa1), noiseOrigMx)
        # tmpCSIb1 = np.matmul(tmpCSIb1 - np.mean(tmpCSIb1), noiseOrigMx)
        # # tmpCSIb2 = np.matmul(tmpCSIb2 - np.mean(tmpCSIb2), noiseOrigMx)
        # tmpCSIb2 = np.matmul(np.ones(epiLen), noiseOrigMx)

        minEpiValClosenessLs = np.zeros(keyLen)
        minEpiValClosenessLsAttack = np.zeros(keyLen)

        for pp in range(0, keyLen, 1):
            epiVala1 = range(pp * intvl, (pp + 1) * intvl, 1)

            epiValClosenessLs = np.zeros(keyLen)
            epiValClosenessLsAttack = np.zeros(keyLen)

            for qq in range(0, keyLen, 1):
                epiValb1 = range(qq * intvl, (qq + 1) * intvl, 1)
                epiValb2 = range(qq * intvl, (qq + 1) * intvl, 1)

                epiValClosenessLs[qq] = distance.cityblock(tmpCSIa1[epiVala1], tmpCSIb1[epiValb1])
                epiValClosenessLsAttack[qq] = distance.cityblock(tmpCSIa1[epiVala1], tmpCSIb2[epiValb2])

            minEpiValClosenessLs[pp] = np.argmin(epiValClosenessLs)
            minEpiValClosenessLsAttack[pp] = np.argmin(epiValClosenessLsAttack)

        ## --------------------------
        ## Display results
        indClosenessResults = minEpiIndClosenessLs - np.array(range(0, keyLen, 1))
        print(indClosenessResults)
        indClosenessResultsAttack = minEpiIndClosenessLsAttack - np.array(range(0, keyLen, 1))
        print(indClosenessResultsAttack)
        print(np.mean(abs(indClosenessResultsAttack)))
        # print(minEpiIndClosenessLs)
        # print(minEpiIndClosenessLsAttack)
        # plt.hist(indClosenessResultsAttack, 20)
        # plt.show()
        valClosenessResults = minEpiValClosenessLs - np.array(range(0, keyLen, 1))
        print(valClosenessResults)
        valClosenessResultsAttack = minEpiValClosenessLsAttack - np.array(range(0, keyLen, 1))
        print(valClosenessResultsAttack)
        print(np.mean(abs(valClosenessResultsAttack)))
        print("--------------------------------------------")
