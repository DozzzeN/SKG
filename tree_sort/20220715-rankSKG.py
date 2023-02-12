from numpy.random import exponential as Exp

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
        for jj in range(a,a+b):
            edges.append([ii, jj, Exp(1.)])
 
    return edges
 
def hp_filter(x, lamb=5000):
    w = len(x)
    b = [[1]*w, [-2]*w, [1]*w]
    D = sparse.spdiags(b, [0, 1, 2], w-2, w)
    I = sparse.eye(w)
    B = (I + lamb*(D.transpose()*D))
    return sparse.linalg.dsolve.spsolve(B, x)

def smooth(x, window_len = 11, window = 'hanning'):
    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise(ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser']:
        raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    elif window == 'kaiser':
        beta = 5
        w=eval('np.'+window+'(window_len, beta)')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def sumSeries(CSITmp):

	if len(CSITmp) > 1:
		sumCSI = sum(CSITmp) + sumSeries(CSITmp[0:-1])
		return sumCSI
	else:
		return CSITmp[0]


## -----------------------------------
plt.close('all')
# np.random.seed(0)

rawData = loadmat('../data/data_mobile_indoor_1.mat')
# rawData = loadmat('data_NLOS.mat')
# print(rawData['A'])
# print(rawData['A'][:, 0])
# print(len(rawData['A'][:, 0]))

# CSIa1Orig = rawData['A'][:, 0]
# CSIb1Orig = rawData['A'][:, 1]


CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

dataLen = len(CSIa1Orig)

# dataLen = 100

# rawData = loadmat('data_mobile_indoor_2.mat')
# CSIb2Orig = rawData['A'][0:dataLen, 1]
CSIb2Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig), size=dataLen)


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


# CSIa1Orig = savgol_filter(CSIa1Orig, 11, 1)
# CSIb1Orig = savgol_filter(CSIb1Orig, 11, 1)
# CSIb2Orig = savgol_filter(CSIb2Orig, 11, 1)

# -----------------------------------------------------------------------------------
#     ---- Constant Noise Generation ----   
#  Pre-allocated noise, will not change during sorting and matching:
#  Use the following noise, need to comment the ines of "Instant noise generator"
# -----------------------------------------------------------------------------------
CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
CSIb2OrigBack = CSIb2Orig.copy()
dataLen = len(CSIa1Orig)


paraLs = [8, 9]
paraRate = np.zeros((3, len(paraLs)))

for para in paraLs:
    print(para)

    ## ---------------------------------------------------------
    intvl = para
    keyLen = 100

    correctRate = []
    randomRate = []
    noiseRate = []

    for staInd in range(0, dataLen-keyLen*intvl-1, intvl):
        print(staInd)
        # staInd = 0                               # fixed start for testing
        endInd = staInd + keyLen*intvl

        # --------------------------------------------
        # BEGIN: Ranking SKG
        # --------------------------------------------
        tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
        tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
        tmpCSIb2 = CSIb2Orig[range(staInd, endInd, 1)]

        epiLen = len(range(staInd, endInd, 1))


        ## For indics closeness;
        tmpCSIa1Ind = tmpCSIa1.argsort().argsort()
        tmpCSIb1Ind = tmpCSIb1.argsort().argsort()
        tmpCSIb2Ind = tmpCSIb2.argsort().argsort()
   
        tmpCSIa1Sort = np.sort(tmpCSIa1)
        tmpCSIb1Sort = np.sort(tmpCSIb1)
        tmpCSIb2Sort = np.sort(tmpCSIb2)

        ## For value closeness;
        tmpCSIa1IndP = tmpCSIa1.argsort()
        tmpCSIb1IndP = tmpCSIb1.argsort()
        tmpCSIb2IndP = tmpCSIb2.argsort()
        
        tmpCSIa1SortA = tmpCSIa1[tmpCSIa1IndP]
        tmpCSIb1SortB = tmpCSIb1[tmpCSIa1IndP]


        minEpiIndClosenessLs = np.zeros(keyLen)
        minEpiValClosenessLs = np.zeros(keyLen)
        for pp in range(0, keyLen, 1):
            epiInda1 = tmpCSIa1Ind[pp*intvl:(pp+1)*intvl]

            epiIndClosenessLs = np.zeros(keyLen)
            epiValClosenessLs = np.zeros(keyLen)

            for qq in range(0, keyLen, 1):
                epiIndb1 = tmpCSIb1Ind[qq*intvl:(qq+1)*intvl]

                epiIndClosenessLs[qq] = sum(abs(epiInda1 - epiIndb1))
                # epiIndClosenessLs[qq] = max(abs(epiInda1 - epiIndb1))

                epiValClosenessLs[qq] = sum(abs(tmpCSIa1SortA[epiInda1] - tmpCSIb1SortB[epiIndb1]))

                # epiClosenessLs[qq] = abs(sum(epiInda1) - sum(epiIndb1))

                # print(epiInda1)
                # print(epiIndb1)

            minEpiIndClosenessLs[pp] = np.argmin(epiIndClosenessLs)
            minEpiValClosenessLs[pp] = np.argmin(epiValClosenessLs)

            # print(epiInda1)
            # qq = int(minEpiIndClosenessLs[pp])
            # print(tmpCSIb1Ind[qq*intvl:(qq+1)*intvl])
            # print(tmpCSIb1Ind[qq*intvl:(qq+1)*intvl] - epiInda1)
            # print('-----------------------')

        print(minEpiIndClosenessLs - np.array(range(0, keyLen, 1)))
        print(minEpiValClosenessLs - np.array(range(0, keyLen, 1)))

        plt.plot(tmpCSIa1Sort)
        plt.plot(tmpCSIb1Sort, 'r--')
        plt.show()
        exit()


#         if np.isnan(np.sum(tmpCSIa1)) + np.isnan(np.sum(tmpCSIb1)) == True:
#             print('NaN value after power operation!')
#
#         # # --------------------------------------------
#         # #   END: Ranking SKG
#         # # --------------------------------------------
#
#         # # --------------------------------------------
#         ##           BEGIN: Sorting and matching
#         # # --------------------------------------------
#
#         permLen = len(range(staInd, endInd, intvl))
#         origInd = np.array([xx for xx in range(staInd, endInd, intvl)])
#
#         start_time = time.time()
#
#         sortCSIa1 = np.zeros(permLen)
#         sortCSIb1 = np.zeros(permLen)
#         sortCSIb2 = np.zeros(permLen)
#         sortNoise = np.zeros(permLen)
#
#
#         CSIa1Ls = []
#         CSIb1Ls = []
#         CSIb2Ls = []
#         noiseLs = []
#
#         CSIa1Corr = []
#         CSIb1Corr = []
#         CSIb2Corr = []
#         noiseCorr = []
#
#
#         for ii in range(permLen):
#             indVec = np.array([aa for aa in range(ii, ii+intvl, 1)])
#
#             CSIa1Tmp = tmpCSIa1[indVec]
#             CSIb1Tmp = tmpCSIb1[indVec]
#             CSIb2Tmp = tmpCSIb2[indVec]
#
#             noiseTmp = tmpNoiseb[indVec] # + tmpNoiseb[indVec]
#             # noiseTmp = np.float_power(-1, tmpNoise[indVec])
#
#             CSIa1Ls.append(CSIa1Tmp)
#             CSIb1Ls.append(CSIb1Tmp)
#             CSIb2Ls.append(CSIb2Tmp)
#             noiseLs.append(noiseTmp)
#
#
#             for jj in range(len(CSIa1Ls)-2, len(CSIa1Ls)-1, 1):
#                 if len(CSIa1Ls) > 1:
#                     CSIa1Corr.append(stats.pearsonr(CSIa1Ls[jj], CSIa1Tmp)[0])
#                     CSIb1Corr.append(stats.pearsonr(CSIb1Ls[jj], CSIb1Tmp)[0])
#                     CSIb2Corr.append(stats.pearsonr(CSIb2Ls[jj], CSIb2Tmp)[0])
#                     noiseCorr.append(stats.pearsonr(noiseLs[jj], noiseTmp)[0])
#
#
#
#                 distpara = 'cityblock'
#                 # distpara = 'canberra'
#                 # distpara = 'braycurtis'
#                 # distpara = 'cosine'
#                 # distpara = 'euclidean'
#                 CSIa1Corr.append( pdist(np.vstack((CSIa1Ls[jj], CSIa1Tmp)), distpara)[0])
#                 CSIb1Corr.append( pdist(np.vstack((CSIb1Ls[jj], CSIb1Tmp)), distpara)[0])
#                 CSIb2Corr.append( pdist(np.vstack((CSIb2Ls[jj], CSIb2Tmp)), distpara)[0])
#                 noiseCorr.append( pdist(np.vstack((noiseLs[jj], noiseTmp)), distpara)[0])
#
#             # print(CSIa1Corr)
#             # print(CSIb1Corr)
#             # time.sleep(1)
#
#             # noiseTmp = np.random.normal(loc=0, scale=1, size=len(CSIa1Tmp))
#             # noiseTmpAdd = np.random.normal(loc=para, scale=np.std(CSIa1Tmp), size=len(CSIa1Tmp))
#
#             # CSIa1Tmp = (CSIa1Tmp ) * noiseTmp                                          ## Method 1: addition
#             # CSIb1Tmp = (CSIb1Tmp ) * noiseTmp
#             # CSIb2Tmp = (CSIb2Tmp ) * noiseTmp
#
#             # # ----------------------------------------------
#             # #    Sorting with different metrics
#             # ## Indoor outperforms outdoor;  indoor with msdc feature performs better; outdoor feature unclear, mean seems better.
#             # # ----------------------------------------------
#
#             sortCSIa1[ii] = np.mean(CSIa1Tmp)                           ## Metric 1: Mean
#             sortCSIb1[ii] = np.mean(CSIb1Tmp)
#             sortCSIb2[ii] = np.mean(CSIb2Tmp)
#             sortNoise[ii] = np.mean(noiseTmp)
#
#             # sortCSIa1[ii] = np.max(CSIa1Tmp)                          ## Metric 1: Mean
#             # sortCSIb1[ii] = np.max(CSIb1Tmp)
#             # sortCSIb2[ii] = np.max(CSIb2Tmp)
#             # sortNoise[ii] = np.max(noiseTmp)
#
#             # sortCSIa1[ii] = sumSeries(CSIa1Tmp)                       ## Metric 2: Sum
#             # sortCSIb1[ii] = sumSeries(CSIb1Tmp)
#             # sortCSIb2[ii] = sumSeries(CSIb2Tmp)
#             # sortNoise[ii] = sumSeries(noiseTmp)
#
#             # sortCSIa1[ii] = msdc(CSIa1Tmp)                            ## Metric 3: tsfresh.msdc:  the metrics, msdc and mc, seem better
#             # sortCSIb1[ii] = msdc(CSIb1Tmp)
#             # sortCSIb2[ii] = msdc(CSIb2Tmp)
#             # sortNoise[ii] = msdc(noiseTmp)
#
#             # sortCSIa1[ii] = mc(CSIa1Tmp)                              ## Metric 4: tsfresh.mc
#             # sortCSIb1[ii] = mc(CSIb1Tmp)
#             # sortCSIb2[ii] = mc(CSIb2Tmp)
#             # sortNoise[ii] = mc(noiseTmp)
#
#             # sortCSIa1[ii] = cid(CSIa1Tmp, 1)                          ## Metric 5: tsfresh.cid_ie,
#             # sortCSIb1[ii] = cid(CSIb1Tmp, 1)
#             # sortCSIb2[ii] = cid(CSIb2Tmp, 1)
#             # sortNoise[ii] = cid(noiseTmp, 1)
#
#
#         # plt.plot(sortCSIa1)
#         # # plt.plot(sortCSIb1)
#         # plt.plot(sortCSIb2)
#         # plt.show()
#
#         ## Matching outcomes
#         print('----------------------------')
#         sortInda = np.argsort(sortCSIa1)
#         sortIndb1 = np.argsort(sortCSIb1)
#         sortIndb2 = np.argsort(sortCSIb2)
#         sortIndn = np.argsort(sortNoise)
#
#         # print(permLen, (sortInda-sortIndb1 == 0).sum())
#         # print(permLen, (sortInda-sortIndb2 == 0).sum())
#         # print(permLen, (sortInda-sortIndn == 0).sum())
#
#         # print(sortInda.tolist())
#         # print(sortIndb1.tolist())
#         # print(sortIndb2.tolist())
#         # print(sortIndn.tolist())
#         # time.sleep(1)
#         # continue
#
#
#         hamDist = []
#         hamDiste = []
#         hamDistn = []
#         for kk in sortInda:
#             indexa = np.where(sortInda == kk)[0][0]
#             indexb1 = np.where(sortIndb1 == kk)[0][0]
#             indexb2 = np.where(sortIndb2 == kk)[0][0]
#             indexn = np.where(sortIndn == kk)[0][0]
#
#             hamDist.append(np.abs(indexa-indexb1))
#             hamDiste.append(np.abs(indexa-indexb2))
#             hamDistn.append(np.abs(indexa-indexn))
#
#         print(hamDist)
#         print(max(hamDist))
#         print(hamDiste)
#         print(max(hamDiste))
#         print(hamDistn)
#         print(max(hamDistn))
#         time.sleep(1)
#
#
#
#
#         correctRate.append((sortInda-sortIndb1 == 0).sum())
#         randomRate.append((sortInda-sortIndb2 == 0).sum())
#         noiseRate.append((sortInda-sortIndn == 0).sum())
#
#
#
#         # sortCorrInda = np.argsort(CSIa1Corr)
#         # sortCorrIndb1 = np.argsort(CSIb1Corr)
#         # sortCorrIndb2 = np.argsort(CSIb2Corr)
#         # sortCorrIndn = np.argsort(noiseCorr)
#
#         # print(len(sortCorrInda), (sortCorrInda-sortCorrIndb1 == 0).sum())
#         # print(len(sortCorrInda), (sortCorrInda-sortCorrIndb2 == 0).sum())
#         # print(len(sortCorrInda), (sortCorrInda-sortCorrIndn == 0).sum())
#
#         # # --------------------------------------------
#         ##         END: Sorting and matching
#         # # --------------------------------------------
#
#         if (sortInda-sortIndb1 == 0).sum() == permLen:
#             np.save('tmpNoise.npy', tmpNoise)
#             # sys.exit()
#
#
#     paraRate[0, paraLs.index(para)] = sum(correctRate)/len(correctRate)
#     paraRate[1, paraLs.index(para)] = sum(randomRate)/len(randomRate)
#     paraRate[2, paraLs.index(para)] = sum(noiseRate)/len(noiseRate)
#
# # print(paraRate)
#
# plt.plot(paraRate[0,:])
# plt.plot(paraRate[1,:])
# plt.plot(paraRate[2,:])
# plt.show()
#
# sys.exit("Stop.")
#


