from mwmatching import maxWeightMatching
from numpy.random import exponential as Exp

from scipy import linalg
from scipy.ndimage import gaussian_filter1d
from scipy.stats.stats import pearsonr
from operator import eq
# from pyentrp import entropy as ent  ## Entropy calculation
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
from scipy import signal
from sklearn import preprocessing
# import partition   ## e.g., partition.greedy.greedy([1,2,3,4,5], 2); partition.kk.kk([1,2,3,4,5], 2); partition.dp.dp([1,2,3,4,5], 2)

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


import networkx as nx
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

def entropyPerm(CSIa1Orig, CSIb1Orig, dataLen, entropyThres):
    ts = CSIa1Orig/np.max(CSIa1Orig)
    # shanon_entropy = ent.shannon_entropy(ts)
    # perm_entropy = ent.permutation_entropy(ts, order=3, delay=1, normalize=True)
    # mulperm_entropy = ent.multiscale_permutation_entropy(ts, 3, 1, 1)
    mul_entropy = ent.multiscale_entropy(ts, 3, maxscale = 1)
    # print(mul_entropy)

    cnts = 0
    while mul_entropy < entropyThres and cnts < 10:
    # while mul_entropy < 2.510
        shuffleInd = np.random.permutation(dataLen)  
        CSIa1Orig = CSIa1Orig[shuffleInd] 
        CSIb1Orig = CSIb1Orig[shuffleInd]
        # CSIa2Orig = CSIa2Orig[shuffleInd]
        # CSIb2Orig = CSIb2Orig[shuffleInd]

        ts = CSIa1Orig/np.max(CSIa1Orig)
        mul_entropy = ent.multiscale_entropy(ts, 4, maxscale = 1)
        cnts += 1
        # print(mul_entropy[0])

    return CSIa1Orig, CSIb1Orig

def sumSeries(CSITmp):

    if len(CSITmp) > 1:
        sumCSI = sum(CSITmp) + sumSeries(CSITmp[0:-1])
        return sumCSI
    else:
        return CSITmp[0]


## -----------------------------------
plt.close('all')
# np.random.seed(0)

# rawData = loadmat('data_static_indoor_1.mat')
# rawData = loadmat('data_mobile_outdoor_1.mat')
rawData = loadmat('../data/data_static_indoor_1.mat')
# print(rawData['A'])
# print(rawData['A'][:, 0])
# print(len(rawData['A'][:, 0]))

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
dataLen = len(CSIa1Orig)
print(dataLen)

# # Fake data
# rawData = loadmat('data_mobile_outdoor_1.mat')
# CSIb1Orig = rawData['A'][0:dataLen, 1]


# # # ----------- Simulated data ---------------
# CSIa1Orig =  np.random.normal(loc=-60, scale=7, size=dataLen)
# CSIb1Orig =  np.random.normal(loc=-60, scale=7, size=dataLen)

# CSIa1Orig = np.random.uniform(-80, -40, size=dataLen)
# # CSIb1Orig = np.random.uniform(-80, -40, size=dataLen)
# CSIb1Orig = CSIa1Orig + np.random.normal(0, 1, size=dataLen)


# # print(type(CSIa1Orig))
# # plt.plot(CSIa1Orig)
# # plt.show()
# # sys.exit()

# ---------------------------------
# data1 = loadmat('RSSa.mat')
# data2 = loadmat('RSSa.mat')
# data3 = loadmat('RSSb.mat')
# data4 = loadmat('RSSb.mat')

# CSIa1Orig = data1['RSSa'][0] 
# CSIa2Orig = data2['RSSa'][0]
# CSIb1Orig = data3['RSSb'][0]
# CSIb2Orig = data4['RSSb'][0]

# data1 = loadmat('RSSa.mat')
# # data2 = loadmat('CSIa.mat')
# data3 = loadmat('RSSe.mat')
# # data4 = loadmat('CSIb.mat')

# CSIa1Orig = data1['RSSa'][0] 
# # CSIa2Orig = data2['RSSa'][0]
# CSIb1Orig = data3['RSSe'][0]
# # CSIb2Orig = data4['RSSe'][0]

# data1 = loadmat('CSIa.mat')
# # data2 = loadmat('CSIa.mat')
# data3 = loadmat('CSIb.mat')
# # data4 = loadmat('CSIb.mat')

# CSIa1Orig = data1['CSIa'][0] 
# # CSIa2Orig = data2['CSIa'][0]
# CSIb1Orig = data3['CSIb'][0]
# # CSIb2Orig = data4['CSIb'][0]

# print(CSIa1Orig)
# sys.exit()


# CSIa1Orig = CSIa1Orig[1:] - CSIa1Orig[0:-1]
# CSIb1Orig = CSIb1Orig[1:] - CSIb1Orig[0:-1]


# # -----------------------------------------------------
# # Normalization
# CSIb1Orig = CSIa1Orig - (np.mean(CSIa1Orig) - np.mean(CSIb1Orig))
# CSIa1Orig = preprocessing.normalize([CSIa1Orig], norm='max')[0]
# CSIb1Orig = preprocessing.normalize([CSIb1Orig], norm='max')[0]

# if np.max(CSIa1Orig) > np.max(CSIb1Orig):
#     CSIa1Orig = CSIa1Orig - np.max(CSIa1Orig) 
#     CSIb1Orig = CSIb1Orig - np.max(CSIa1Orig) 
# else:
#     CSIa1Orig = CSIa1Orig - np.max(CSIb1Orig) 
#     CSIb1Orig = CSIb1Orig - np.max(CSIb1Orig) 

# CSIa1Orig = CSIa1Orig + 10
# CSIb1Orig = CSIb1Orig + 10

# # # -----------------------------------
# # # ---- Sorting ----
# sortInda = CSIa1Orig.argsort().argsort()
# sortCSIOrigb = np.sort(CSIb1Orig)
# CSIb1Orig = sortCSIOrigb[sortInda]


# plt.plot(CSIa1Orig)
# plt.plot(CSIb1Orig)
# plt.show()
# sys.exit()

# # -----------------------------------
# # ---- Entropy-based permutation ----
# entropyThres = 2
# CSIa1Orig, CSIb1Orig = entropyPerm(CSIa1Orig, CSIb1Orig, dataLen, entropyThres)


# # -----------------------------------
# # ---- Smoothing -------------
# ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser']
# CSIa1Orig = smooth(CSIa1Orig, window_len = 11, window = 'hamming')
# CSIb1Orig = smooth(CSIb1Orig, window_len = 11, window = 'hamming')

# CSIa1Orig = hp_filter(CSIa1Orig, lamb=500)
# CSIb1Orig = hp_filter(CSIb1Orig, lamb=500)

# CSIa1Orig = savgol_filter(CSIa1Orig, 7, 1)
# CSIb1Orig = savgol_filter(CSIb1Orig, 7, 1)

dataLen = len(CSIa1Orig)

# -----------------------------------
#     ---- Noise Insertion ----   
###  Most important:  the sign of noise values play critical role on determining the matching performance;
# -----------------------------------
CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
noiseOrig = np.random.uniform(5, 6, size=dataLen)      ## Multiplication item normal distribution 
noiseOrigBack = noiseOrig.copy()

# noise = np.asarray([random.randint(-20, 20) for iter in range(dataLen)])   ## Integer Uniform Distribution
# noise = np.round(np.random.normal(loc=0, scale=np.std(CSIa1Orig), size=dataLen))   ## Integer Normal Distribution

# noise = np.random.normal(loc=0, scale=0.1, size=dataLen)      ## Multiplication item normal distribution 
# noiseAdd = np.random.normal(loc=0, scale=10, size=dataLen)   ## Addition item normal distribution

# CSIa1Orig = CSIa1Orig + noise
# CSIb1Orig = CSIb1Orig + noise

# CSIa1Orig = np.float_power(CSIa1Orig, noise)
# CSIb1Orig = np.float_power(CSIb1Orig, noise)

# noiseBack = noise.copy()

# plt.subplot(1, 2, 1)
# plt.plot(CSIa1Orig[:300])
# plt.plot(CSIb1Orig[:300])
# plt.subplot(1, 2, 2)
# plt.plot(CSIa1OrigBack[:300])
# plt.plot(CSIb1OrigBack[:300])
# plt.plot(noise[:300])
# plt.show()
# sys.exit()

## ---------------------------------------------------------
sft = 3
intvl = 2*sft+1
keyLen = 128

misRate = []
errRate = []
noiseLs = np.empty(1)
obsvLs = np.empty(1)

noiseCtrl = 1
errCnt = 0

for staInd in range(0, dataLen-keyLen*intvl-1, keyLen*intvl):
    print(staInd)
    # staInd = 0                               # fixed test
    endInd = staInd + keyLen*intvl

    CSIa1Orig = CSIa1OrigBack.copy()
    CSIb1Orig = CSIb1OrigBack.copy()
    noiseOrig = noiseOrigBack.copy()

    permLen = len(range(staInd, endInd, intvl))
    origInd = np.array([xx for xx in range(staInd, endInd, 1)])

    # # ----------------------------------
    if noiseCtrl == 0:
        # # Long-term Noising
        CSIa1Orig = CSIa1Orig ** 2 * noiseOrig
        CSIb1Orig = CSIb1Orig ** 2 * noiseOrig

        # CSIa1Orig = np.float_power(np.abs(CSIa1Orig), noiseOrig) 
        # CSIb1Orig = np.float_power(np.abs(CSIb1Orig), noiseOrig)

        # CSIa1Orig = np.float_power(noiseOrig, np.abs(CSIa1Orig)) 
        # CSIb1Orig = np.float_power(noiseOrig, np.abs(CSIb1Orig))

        CSIa1Epi = CSIa1Orig[origInd]
        CSIb1Epi = CSIb1Orig[origInd]

        noisePrd = noiseOrig[origInd]

    else:
        # # Instant Noising
        CSIa1Epi = CSIa1Orig[origInd] 
        CSIb1Epi = CSIb1Orig[origInd] 
        # CSIb1Epi = CSIa1Epi - (np.mean(CSIa1Epi) - np.mean(CSIb1Epi))

        # sortInda = CSIa1Epi.argsort().argsort()
        # sortCSIEpib = np.sort(CSIb1Epi)
        # CSIb1Epi = sortCSIEpib[sortInda]

        # CSIa1Epi = preprocessing.normalize([CSIa1Epi], norm='l2')[0]
        # CSIb1Epi = preprocessing.normalize([CSIb1Epi], norm='l2')[0]

        # CSIa1Epi = CSIa1Epi - np.mean(CSIa1Epi)
        # CSIb1Epi = CSIb1Epi - np.mean(CSIb1Epi)

        # entropyThres = 2
        # CSIa1Epi, CSIb1Epi = entropyPerm(CSIa1Epi, CSIb1Epi, len(CSIb1Epi), entropyThres)

        # noisePrd = np.random.uniform(6, 7, size=len(CSIa1Epi))
        noisePrd = np.random.uniform(4, 5, size=len(CSIa1Epi))
        noiseExp = np.random.uniform(4, 5, size=len(CSIa1Epi))

        noisePrd = np.random.normal(0, 1, size=len(CSIa1Epi))
        noisePrd = np.random.uniform(0, 1, size=len(CSIa1Epi))

        CSIa1EpiTmp = CSIa1Epi.copy()
        CSIb1EpiTmp = CSIb1Epi.copy()

        CSIa1Epi =  CSIa1Epi * noisePrd 
        CSIb1Epi =  CSIb1Epi * noisePrd

        tmpInd = np.array(range(0,len(origInd)))
        tmpPermInd = np.random.permutation(tmpInd)
        CSIa1EpiPerm = CSIa1Epi[tmpPermInd]
        CSIb1EpiPerm = CSIb1Epi[tmpPermInd]


        

        # CSIa1EpiPerm  = CSIa1EpiPerm / np.linalg.norm(CSIa1EpiPerm)
        # CSIb1EpiPerm  = CSIb1EpiPerm / np.linalg.norm(CSIb1EpiPerm)

        # plt.plot(CSIa1EpiPerm)
        # plt.plot(CSIb1EpiPerm)
        # plt.show()
        # sys.exit()
        # CSIa1Epi = CSIa1EpiPerm * CSIa1EpiTmp
        # CSIb1Epi = CSIb1EpiPerm * CSIb1EpiTmp

        # CSIa1Epi = CSIa1EpiPerm + CSIa1EpiTmp
        # CSIb1Epi = CSIb1EpiPerm + CSIb1EpiTmp


        # CSIa1Epi =  np.arctan(CSIa1Epi/noisePrd)/(2*np.pi) - 0.5 
        # CSIb1Epi =  np.arctan(CSIb1Epi/noisePrd)/(2*np.pi) - 0.5 

        # CSIa1Epi =  np.arctan(CSIa1Epi/noisePrd)
        # CSIb1Epi =  np.arctan(CSIb1Epi/noisePrd)

        # CSIa1Epi = (CSIa1Epi + noiseExp) * noisePrd
        # CSIb1Epi = (CSIb1Epi + noiseExp) * noisePrd    

        # CSIa1Epi = np.float_power(np.abs(CSIa1Epi) * noisePrd, noiseExp) 
        # CSIb1Epi = np.float_power(np.abs(CSIb1Epi) * noisePrd, noiseExp)

        # CSIa1Epi = np.float_power(np.abs(CSIa1Epi) + noisePrd, noiseExp) 
        # CSIb1Epi = np.float_power(np.abs(CSIb1Epi) + noisePrd, noiseExp)

        # CSIa1Epi = np.float_power(np.abs(CSIa1Epi), noiseExp) 
        # CSIb1Epi = np.float_power(np.abs(CSIb1Epi), noiseExp)

        # entropyThres = 2
        # CSIa1Epi, CSIb1Epi = entropyPerm(CSIa1Epi, CSIb1Epi, len(CSIb1Epi), entropyThres)

    CSIa1Orig[origInd] = CSIa1Epi               
    CSIb1Orig[origInd] = CSIb1Epi 

    # print(CSIa1Epi)
    # print(CSIb1Epi)
    # plt.plot(CSIa1Epi)
    # plt.plot(CSIb1Epi)
    # plt.plot(noisePrd)
    # plt.show()

    noiseLs = np.append(noiseLs, noisePrd)
    # noiseLs = np.append(noiseLs, np.arctan(np.mean(CSIa1OrigBack)/noisePrd))
    # noiseLs = np.append(noiseLs, np.float_power(np.abs(np.mean(CSIa1Orig)), noiseExp) * noisePrd)
    # noiseLs = np.append(noiseLs, np.mean(CSIa1Epi)  * noisePrd)
    # noiseLs = np.append(noiseLs, np.float_power(noisePrd, np.abs(np.mean(CSIa1Orig))))
    # noiseLs = np.append(noiseLs, np.float_power(np.abs(np.mean(CSIa1Orig))*noisePrd, noiseExp))
    # noiseLs = np.append(noiseLs, np.float_power(np.abs(np.mean(CSIa1Epi)) + noisePrd, noiseExp))
    # noiseLs = np.append(noiseLs, np.float_power(noisePrd, noiseExp))
    # noiseLs = np.append(noiseLs, (np.abs(np.mean(CSIa1Epi)) + noiseExp) * noisePrd)

    obsvLs = np.append(obsvLs, CSIa1Epi)

    noiseLs = noiseLs[1:]
    obsvLs = obsvLs[1:]

    # --------------------------------------------
    # Random permutation --- Begin
    # --------------------------------------------
    newOrigInd = np.array([xx for xx in range(staInd, endInd, intvl)])
    permInd = np.random.permutation(permLen)   ## KEY
    permOrigInd = newOrigInd[permInd]

    # --------------------------------------------
    # Random permutation --- End
    # --------------------------------------------

    ## Main: Weighted biparitite maximum matching
    edges = []
    edgesn = []
    start_time = time.time()
    matchSort = []

    # epiCSIa1 = np.zeros((permLen, intvl))
    # epiCSIb1 = np.zeros((permLen, intvl))
    # epiCorr = np.zeros((permLen, permLen))

    for ii in range(permLen):
        coefLs = []
        aIndVec = np.array([aa for aa in range(permOrigInd[ii], permOrigInd[ii]+intvl, 1)])      ## for permuted CSIa1
        # aIndVec = np.array([aa for aa in range(newOrigInd[ii], newOrigInd[ii]+intvl, 1)])            ## for non-permuted CSIa1 
        
        distLs = []
        edgesTmp = []

        for jj in range(permLen, permLen*2):                       
            bIndVec = np.array([bb for bb in range(newOrigInd[jj-permLen], newOrigInd[jj-permLen]+intvl, 1)])
            
            CSIa1Tmp = CSIa1Orig[aIndVec] 
            CSIb1Tmp = CSIb1Orig[bIndVec]
            # noiseTmp = noiseOrig[aIndVec]                                                    ## only for long-term noising

            # epiCorr[ii, jj-permLen] = np.correlate(CSIb1Tmp, noiseTmp)
            # epiCorr[ii, jj-permLen] = np.corrcoef(CSIb1Tmp, noiseTmp)[1,0]

            # print(CSIa1Tmp)
           
            # # ----------------------------------------------

            # epiCSIa1[ii,:] = CSIa1Tmp
            # epiCSIb1[jj-permLen,:] = CSIb1Tmp
         
            distpara = 'cityblock'
            # distpara = 'canberra'
            # distpara = 'braycurtis'
            # distpara = 'cosine'
            # distpara = 'euclidean'
            X = np.vstack((CSIa1Tmp, CSIb1Tmp))
            dist = pdist(X, distpara)
            # edges.append([ ii, jj, dist[0]])

            distLs.append(dist[0])
            edgesTmp.append([ ii, jj, dist[0]])


            # Xn = np.vstack((noiseTmp, CSIb1Tmp))
            # distn = pdist(Xn, distpara)
            # edgesn.append([ ii, jj, distn[0]])

        topNum = 16
        sortInd = np.argsort(distLs)    ## Increasing order
        topInd = sortInd[0:topNum]
        for kk in topInd:
            edges.append(edgesTmp[kk])

        if topNum == 1:
            matchSort.append(topInd[0])

           
    #####################################################################
    #####################################################################
    #####################################################################
    #####################################################################
    #####################################################################

    # print(permInd.tolist())
    # print(matchSort)
    # mismatchInd = permInd-matchSort
    # nonzeroIndSort = [ii for ii in range(permLen) if mismatchInd[ii] != 0]

    # print('------ Error Key and Statistics-------')
    # print(nonzeroIndSort)


    # key agreement
    neg_edges = [(i,j,-wt) for i,j,wt in edges]
    match = maxWeightMatching(neg_edges, maxcardinality=True)

    matchb = [j-permLen for (i,j,wt) in neg_edges if match[i] == j]
    print("--- %s seconds ---" % (time.time() - start_time))
    # sys.exit()

    ## Mismatch check
    # print(permInd.tolist())
    # print(matchb)
    mismatchInd = permInd-matchb
    nonzeroInd = [ii for ii in range(permLen) if mismatchInd[ii] != 0]

    print('------ Error Key and Statistics-------')
    print(nonzeroInd)
    
    if nonzeroInd:
        errCnt = errCnt + len(nonzeroInd)
        print(errCnt/((staInd/(keyLen*intvl)+1)*keyLen))
    print('--------------------------------------')
    
    # # noise agreement
    # neg_edgesn = [(i,j,-wt) for i,j,wt in edgesn]
    # match = maxWeightMatching(neg_edgesn, maxcardinality=True)
    # matchn = [j-permLen for (i,j,wt) in neg_edgesn if match[i] == j]

    # print('------ Noise Match Statistics-------')
    # mismatchIndn = permInd-matchn
    # nonzeroIndn = [ii for ii in range(permLen) if mismatchIndn[ii] != 0]
    # print(nonzeroIndn)
    # print('--------------------------------------')

    # # # ------------------------------------------------
    # # # correlation
    # print(epiCorr)
    # plt.matshow(epiCorr)
    # plt.colorbar()
    # plt.show()

    continue                       #### Breaking point for debugging;

    if nonzeroInd:
        print(staInd)
        print(nonzeroInd)   # Groundtruth

        # Find mismatched index
        randin = np.asarray([random.randint(0, 255) for iter in range(permLen)])
        # CSIa1 = np.transpose(CSIa1)[0]
        print(type(CSIa1))
        print(type(randin))
        print(type(permInd))
        randim = randin ^ CSIa1
        randouta = randin ^ permInd

        tmpIndb = [ matchb.index(iter) for iter in range(0, permLen) ]
        # CSIa1Prime = np.transpose(CSIa1Perm[tmpIndb])[0]
        CSIa1Prime = CSIa1Perm[tmpIndb]
        randoutb = randouta ^ matchb ^ CSIa1Prime

        mismatchInd1 = [iter for iter in range(0, permLen) if randoutb[iter] != randim[iter] ]
        print(mismatchInd1) 

        CSIa1Prime[mismatchInd1] = 60
        CSIa1Mod = CSIa1
        CSIa1Mod[mismatchInd1] = 60

        randim = randin ^ CSIa1Mod
        randoutb = randouta ^ matchb ^ CSIa1Prime

        mismatchInd2 = [iter for iter in range(0, permLen) if randoutb[iter] != randim[iter] ]
        print(mismatchInd2) 


    # # For randomness test
    # xTest = [permInd[ii]*abs(CSIa1[ii]) for ii in range(0, permLen)]
    # permIndStr = format(xTest[0], 'b')
    # for ii in range(1, permLen):
    #     permIndStr += format(xTest[ii], 'b')
    # print(permIndStr)
    # sys.exit()


    # # Coding:
    # keyLenNew = math.floor(math.log2(math.log2(keyLen)*keyLen))
    # grayInd = np.random.permutation(keyLenNew*permLen)

    if nonzeroInd:
        diffLen = 0
        for p in range(0, len(nonzeroInd)-1):
            tmp1 = bin(permInd[nonzeroInd[p]])
            tmp2 = bin(matchb[nonzeroInd[p]])

            # tmp1 = bin(grayInd[permInd[nonzeroInd[p]]])
            # tmp2 = bin(grayInd[matchb[nonzeroInd[p]]])

            diffLen = diffLen + sum(1 for a, b in zip(tmp1, tmp2) if a != b)

        misRate.append(diffLen/(math.log2(keyLen)*keyLen))
        errRate.append(1)
    else:
        misRate.append(0)
        errRate.append(0)
        # print(misRate)
    print(np.mean(misRate))
    print(np.mean(errRate))


print(errCnt/((staInd/(keyLen*intvl)+1)*keyLen))
plt.plot(noiseLs, obsvLs, 'bo')
plt.show()

sys.exit("Stop.")



## To correct the error

# mismatchInd1 = []
# mismatchInd2 = []


# for ii in range(0, permLen):
#     # print('---------')
    
#     # print(CSIa1Perm[tmpInd])
#     # print(CSIa1[ii])
#     randin = random.randint(0, 255)
#     # print(randin)
#     randim = randin ^ abs(CSIa1[ii])
#     # print(randim)
#     randout = randin ^ permInd[ii]
#     # print(randout)

#     tmpIndb = matchb.index(ii)
#     randout = randout ^ matchb[ii] ^ abs(CSIa1Perm[tmpIndb])
#     # print(randout)

    
#     # if permInd[ii] != matchb[ii] and CSIa1[ii] == CSIa1Perm[tmpIndb]:
#     #     print('1-----')
#     #     print([ii, permInd[ii], matchb[ii], CSIa1[ii], CSIa1Perm[tmpIndb]])
#     #     print(randim)
#     #     print(randout)

    
#     # if permInd[ii] == matchb[ii] and CSIa1[ii] != CSIa1Perm[tmpIndb]:
#     #     print('2-----')
#     #     print([ii, permInd[ii], matchb[ii], CSIa1[ii], CSIa1Perm[tmpIndb]])
#     #     print(tmpIndb)

    
#     # if permInd[ii] != matchb[ii] and CSIa1[ii] != CSIa1Perm[tmpIndb]:
#     #     print('3-----')
#     #     print([ii, permInd[ii], matchb[ii], CSIa1[ii], CSIa1Perm[tmpIndb]])
#     #     print(tmpIndb)

#     if randout != randim:
#         mismatchInd1.append(ii)

#     if randout != randim:
#         mismatchInd1.append(ii)

# print(staInd)
# print(nonzeroInd)
# print('unmatched')
# print(mismatchInd1)
# # print(mismatchInd2)

# noverLs1 = []
# noverLs2 = []
# weimismatch1 = list(permInd[mismatchInd1])
# print(weimismatch1)
# for p in mismatchInd1:
#     if p not in weimismatch1:
#         noverLs1.append(p)

# matchb = np.array(matchb)
# weimismatch2 = list(matchb[mismatchInd1])
# print(weimismatch2)
# for p in mismatchInd1:
#     if p not in weimismatch2:
#         noverLs2.append(p)

# print(noverLs1)
# print(noverLs2)



plt.figure(1)
plt.subplot(411)
plt.plot(CSIa1[matchb])
plt.plot(CSIb1[matchb])
plt.subplot(412)
plt.plot(CSIa1)
plt.plot(CSIb1)
plt.subplot(413)
plt.plot(weiLs)
plt.subplot(414)
plt.plot(CSIa2[matchb]-CSIa2Perm)
plt.show()

# diffCSI = CSIb1[permInd] - CSIa1[permInd]
# diffCSIPerm = CSIb1[matchb] - CSIa1[permInd] 
# plt.figure(1)
# plt.subplot(211)
# plt.plot(diffCSI)
# plt.subplot(212)
# plt.plot(diffCSIPerm)
# plt.show()


# mismatchInd = permInd-matchb
# nonzeroInd = [ii for ii in range(permLen) if mismatchInd[ii] != 0]
# zeroInd = [ii for ii in range(permLen) if mismatchInd[ii] == 0]
# print(nonzeroInd)
# print(len(nonzeroInd))

# print(CSIa1Perm[nonzeroInd])

# plt.figure(3)
# plt.subplot(211)
# plt.plot(CSIa1Perm[nonzeroInd])
# plt.subplot(212)
# plt.plot(CSIa1Perm[zeroInd])
# plt.show()