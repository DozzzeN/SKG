import sys
import time

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import exponential as Exp
from scipy import stats
from scipy.io import loadmat
from scipy.spatial.distance import pdist


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


def whiten(X, method='zca'):
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    W = None

    if method in ['zca', 'pca', 'cholesky']:
        U, Lambda, _ = np.linalg.svd(Sigma)
        if method == 'zca':
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
        elif method == 'pca':
            W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
        elif method == 'cholesky':
            W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
    elif method in ['zca_cor', 'pca_cor']:
        V_sqrt = np.diag(np.std(X, axis=0))
        P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
        G, Theta, _ = np.linalg.svd(P)
        if method == 'zca_cor':
            W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)), np.linalg.inv(V_sqrt))
        elif method == 'pca_cor':
            W = np.dot(np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T), np.linalg.inv(V_sqrt))
    else:
        raise Exception('Whitening method not found.')

    return np.dot(X_centered, W.T)


## -----------------------------------
plt.close('all')
# np.random.seed(0)

rawData = loadmat('../data/data_static_indoor_1.mat')
# rawData = loadmat('data_NLOS.mat')
# print(rawData['A'])
# print(rawData['A'][:, 0])
# print(len(rawData['A'][:, 0]))

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
dataLen = len(CSIa1Orig)

# dataLen = 100

rawData = loadmat('../data/data_mobile_indoor_2.mat')
CSIb2Orig = rawData['A'][0:dataLen, 1]
# CSIb2Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig), size=dataLen)


# # # ----------- Simulated data ---------------
# CSIa1Orig = CSIa1Orig + np.random.normal(loc=0, scale=1, size=dataLen)
# CSIb1Orig = CSIa1Orig + np.random.normal(loc=0, scale=1, size=dataLen)

# CSIa1Orig = CSIa1Orig - np.mean(CSIa1Orig)
# CSIb1Orig = CSIb1Orig - np.mean(CSIb1Orig) 
# CSIb2Orig = CSIb2Orig - np.mean(CSIb2Orig)

# -----------------------------------------------------------------------------------
#     ---- Constant Noise Generation ----   
#  Pre-allocated noise, will not change during sorting and matching:
#  Use the following noise, need to comment the ines of "Instant noise generator"
# -----------------------------------------------------------------------------------

# noise = np.asarray([random.randint(-1, 1) for iter in range(dataLen)])   ## Integer Uniform Distribution
# noise = np.round(np.random.normal(loc=0, scale=1, size=dataLen))   ## Integer Normal Distribution

avg = 0
std = 1
noise = np.random.normal(loc=avg, scale=std, size=dataLen)  ## Multiplication item normal distribution
noiseAdd = np.random.normal(loc=avg, scale=std, size=dataLen)  ## Addition item normal distribution

# noise = np.random.uniform(-1, 1, size=dataLen)   ## float Uniform distribution
# noiseAdd = np.random.uniform(-1,1, size=dataLen)   ## Addition item normal distribution


win = 13
for p in range(0, dataLen, win):
    # print(p)
    stdSeg = 1 / np.std(noise[p:p + win])
    # print(varSeg)
    noise[p] = np.random.normal(loc=avg, scale=stdSeg, size=1)
    noiseAdd[p] = np.random.normal(loc=avg, scale=stdSeg, size=1)

# plt.plot(CSIa1Orig)
# plt.show()

# plt.plot(noise*CSIa1Orig)
# plt.show()
# sys.exit()


# # -----------------------------------
# # ---- Smoothing -------------
# ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser']
winNam = 'flat'
CSIa1Orig = smooth(CSIa1Orig, window_len=win, window=winNam)
CSIb1Orig = smooth(CSIb1Orig, window_len=win, window=winNam)
CSIb2Orig = smooth(CSIb2Orig, window_len=win, window=winNam)

# CSIa1Orig = smooth(CSIa1Orig, window_len = 9, window = 'bartlett')
# CSIb1Orig = smooth(CSIb1Orig, window_len = 9, window = 'bartlett').;;
# CSIb2Orig = smooth(CSIb2Orig, window_len = 9, window = 'bartlett')

# CSIa1Orig = hp_filter(CSIa1Orig, lamb=15)
# CSIb1Orig = hp_filter(CSIb1Orig, lamb=15)
# CSIb2Orig = hp_filter(CSIb2Orig, lamb=15)


# CSIa1Orig = savgol_filter(CSIa1Orig, win, 1)
# CSIb1Orig = savgol_filter(CSIb1Orig, win, 1)
# CSIb2Orig = savgol_filter(CSIb2Orig, win, 1)


## save the noises:
CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
CSIb2OrigBack = CSIb2Orig.copy()

noiseBack = noise.copy()
noiseAddBack = noiseAdd.copy()

# paraLs = [0 - 5 - 25, -50, -75, -100, -150, -200]  ## RSS mean shift
# paraLs = [-10, -8, -6, -4, -2, -1, 0] #, 1, 2, 4, 6, 8, 10]  ## noise mean
# paraLs = [3, 1, 2, 3, 4]
paraLs = [3]
paraRate = np.zeros((3, len(paraLs)))

for para in paraLs:
    print(para)

    ## ---------------------------------------------------------
    sft = para
    intvl = 2 * sft + 1
    keyLen = 256

    correctRate = []
    randomRate = []
    noiseRate = []

    for staInd in range(0, dataLen - keyLen * intvl - 1, intvl):
        print(staInd)
        # staInd = 0                               # fixed start for testing
        endInd = staInd + keyLen * intvl
        if endInd >= len(CSIb1Orig) / 4 or endInd >= len(CSIb2Orig) / 4:
            break

        # --------------------------------------------
        # BEGIN: Noise-assisted channel manipulation
        # --------------------------------------------
        tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
        tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
        tmpCSIb2 = CSIb2Orig[range(staInd, endInd, 1)]

        tmpNoise = noise[range(staInd, endInd, 1)]
        tmpNoiseAdd = noiseAdd[range(staInd, endInd, 1)]
        epiLen = len(range(staInd, endInd, 1))

        ## remove mean-value
        tmpCSIa1p = np.sort(tmpCSIa1)
        tmpCSIb1p = np.sort(tmpCSIb1)
        tmpCSIb2p = np.sort(tmpCSIb2)

        # -----------------------------------
        # Instant noise generator (Noise changes for every segment of channel measurements)

        # tmpNoise = np.random.normal(loc=0, scale=2, size=len(tmpNoise)) 
        tmpNoiseb = np.random.normal(loc=0, scale=1, size=len(tmpNoiseAdd))

        # tmpNoise = np.random.uniform(-5, 5, size=len(tmpNoise))
        # tmpNoiseb = np.random.uniform(-5, 5, size=len(tmpNoise)) 

        # # permutate itself as noise
        tmpInd = np.array(range(0, len(tmpCSIa1)))
        tmpPermInd = np.random.permutation(tmpInd)

        tmpNoisePa = tmpCSIa1[tmpPermInd]
        tmpNoisePb = tmpCSIb1[tmpPermInd]

        # tmpNoisePa = CSIa1Orig[range(staInd+intvl, endInd+intvl, 1)]
        # tmpNoisePb = CSIb1Orig[range(staInd+intvl, endInd+intvl, 1)]

        # tmpNoisePa  = tmpNoisePa  / np.linalg.norm(tmpNoisePa)
        # tmpNoisePb  = tmpNoisePb  / np.linalg.norm(tmpNoisePb)

        # Permutation as noise
        tmpInd = np.array(range(0, len(tmpCSIa1)))
        tmpPermInd = np.random.permutation(tmpInd)
        tmpCSIa1Perm = tmpCSIa1[tmpPermInd]
        tmpCSIb1Perm = tmpCSIb1[tmpPermInd]
        tmpCSIb2Perm = tmpCSIb2[tmpPermInd]

        # tmpNoisePa = tmpCSIa1Perm
        # tmpNoisePb = tmpCSIb1Perm
        # tmpNoisePb2 = tmpCSIb2Perm

        ## ----------------- BEGIN: Noise-assisted channel manipulation ---------------------------

        # for p in range(3):

        #     tmpNoise = tmpNoise * np.random.normal(loc=p, scale=np.std(tmpCSIa1), size=len(tmpNoise)) 

        # tmpCSIa1 = np.float_power(tmpCSIa1 + tmpNoise, para)                                              ## Method 0: X
        # tmpCSIb1 = np.float_power(tmpCSIb1 + tmpNoise, para)
        # tmpCSIb2 = np.float_power(tmpCSIb2 + tmpNoise, para)

        # tmpCSIa1 = np.arctan(tmpCSIa1/tmpNoise)                                                  ## Method 0: X
        # tmpCSIb1 = np.arctan(tmpCSIb1/tmpNoise) 
        # tmpCSIb2 = np.arctan(tmpCSIb2/tmpNoise) 

        # tmpCSIa1 = tmpCSIa1 + tmpNoise #+ tmpNoiseb                                                 ## Method 1: addition 
        # tmpCSIb1 = tmpCSIb1 + tmpNoise # + tmpNoiseb
        # tmpCSIb2 = tmpCSIb2 + tmpNoise # + tmpNoiseb

        # tmpCSIa1 = tmpCSIa1 * tmpNoise                                         ## Method 2: polynomial product better than addition
        # tmpCSIb1 = tmpCSIb1 * tmpNoise  
        # tmpCSIb2 = tmpCSIb2 * tmpNoise 

        # x1 = whiten(tmpCSIa1.reshape((int(len(tmpCSIa1)/2), 2)), method='zca')
        # y1 = whiten(tmpCSIb1.reshape((int(len(tmpCSIb1)/2), 2)), method='zca')
        # y2 = whiten(tmpCSIb2.reshape((int(len(tmpCSIb2)/2), 2)), method='zca')

        # x = x1.reshape(-1)
        # y = y1.reshape(-1)
        # z = y2.reshape(-1)

        # tmpCSIa1 = np.abs(np.fft.fft(x))
        # tmpCSIb1 = np.abs(np.fft.fft(y))
        # tmpCSIb2 = np.abs(np.fft.fft(z))

        # fig, axs = plt.subplots(2)
        # axs[0].plot(np.abs(np.fft.fft(np.sign(tmpCSIa1) * np.log(np.abs(tmpCSIa1-np.mean(tmpCSIa1))))))

        # tmpCSIa1 = tmpCSIa1  * tmpNoise   * tmpNoiseAdd                                  ## Method 2: polynomial product better than addition
        # tmpCSIb1 = tmpCSIb1  * tmpNoise   * tmpNoiseAdd
        # tmpCSIb2 = tmpCSIb2 * tmpCSIa1 * tmpCSIb1 * tmpNoise * tmpNoiseAdd
        # tmpCSIb2 = tmpCSIb2 * tmpNoise

        tmpCSIa1 = tmpCSIa1 * tmpNoise  ## Method 2: polynomial product better than addition
        tmpCSIb1 = tmpCSIb1 * tmpNoise
        tmpCSIb2 = tmpCSIb2 * tmpNoise

        # tmpCSIa1 =  np.float_power(np.abs(tmpCSIa1), tmpNoise)               
        # tmpCSIb1 =  np.float_power(np.abs(tmpCSIb1), tmpNoise) 
        # tmpCSIb2 =  np.float_power(np.abs(tmpCSIb2), tmpNoise)
        # tmpNoise =  np.float_power(np.mean(tmpCSIa1) * np.ones((len(tmpCSIa1), 1)), tmpNoise)

        x1 = whiten(tmpCSIa1.reshape((int(len(tmpCSIa1) / 2), 2)), method='pca')
        y1 = whiten(tmpCSIb1.reshape((int(len(tmpCSIb1) / 2), 2)), method='pca')
        y2 = whiten(tmpCSIb2.reshape((int(len(tmpCSIb2) / 2), 2)), method='pca')

        x = x1.reshape(-1)
        y = y1.reshape(-1)
        z = y2.reshape(-1)

        tmpCSIa1 = np.abs(np.fft.fft(x))
        tmpCSIb1 = np.abs(np.fft.fft(y))
        tmpCSIb2 = np.abs(np.fft.fft(z))
        tmpNoise = np.abs(np.fft.fft(tmpNoise))

        # tmpCSIa1 = tmpCSIa1 * tmpNoisePa                                           ## Method 2: polynomial product better than addition
        # tmpCSIb1 = tmpCSIb1 * tmpNoisePb  
        # tmpCSIb2 = tmpCSIb2 * tmpNoise 

        # tmpCSIa1 = tmpCSIa1 + tmpNoisePa                                          
        # tmpCSIb1 = tmpCSIb1 + tmpNoisePb  
        # tmpCSIb2 = tmpCSIb2 + tmpNoise 

        # tmpCSIa1 = np.abs(np.fft.fft(tmpCSIa1))
        # tmpCSIb1 = np.abs(np.fft.fft(tmpCSIb1))
        # tmpCSIb2 = np.abs(np.fft.fft(tmpCSIb2))

        # tmpCSIa1 =  np.float_power(np.abs(tmpCSIa1), tmpNoise)               
        # tmpCSIb1 =  np.float_power(np.abs(tmpCSIb1), tmpNoise) 
        # tmpCSIb2 =  np.float_power(np.abs(tmpCSIb2), tmpNoise)  

        # tmpCSIa1 = np.convolve(tmpCSIa1, tmpNoiseb, 'same')                                               ## Method 1: addition 
        # tmpCSIb1 = np.convolve(tmpCSIb1, tmpNoiseb, 'same')
        # tmpCSIb2 = np.convolve(tmpCSIb2, tmpNoiseb, 'same')

        # tmpCSIa1 = np.polymul(tmpCSIa1, tmpNoiseb)
        # tmpCSIb1 = np.polymul(tmpCSIb1, tmpNoiseb)
        # tmpCSIb2 = np.polymul(tmpCSIb2, tmpNoiseb)

        # tmpCSIa1 = np.polymul(tmpCSIa1, tmpNoisePa)
        # tmpCSIb1 = np.polymul(tmpCSIb1, tmpNoisePb)
        # tmpCSIb2 = np.polymul(tmpCSIb2, tmpNoiseb)

        # tmpCSIa1 = signal.fftconvolve(tmpCSIa1, tmpNoise)                                               ## Method 1: addition
        # tmpCSIb1 = signal.fftconvolve(tmpCSIb1, tmpNoise)
        # tmpCSIb2 = signal.fftconvolve(tmpCSIb2, tmpNoise)

        # tmpCSIa1 = np.convolve(tmpCSIa1, tmpNoiseb, 'same')                                               ## Method 1: addition
        # tmpCSIb1 = np.convolve(tmpCSIb1, tmpNoiseb, 'same')
        # tmpCSIb2 = np.convolve(tmpCSIb2, tmpNoiseb, 'same')

        # tmpCSIa1 = np.exp(tmpCSIa1 * tmpNoise)                                                   ## Method 2: polynomial product better than addition
        # tmpCSIb1 = np.exp(tmpCSIb1 * tmpNoise)
        # tmpCSIb2 = np.exp(tmpCSIb2 * tmpNoise) 

        # tmpCSIa1 =  np.float_power(np.abs(tmpCSIa1 * tmpNoiseAdd), tmpNoise)              
        # tmpCSIb1 =  np.float_power(np.abs(tmpCSIb1 * tmpNoiseAdd), tmpNoise)  
        # tmpCSIb2 =  np.float_power(np.abs(tmpCSIb2 * tmpNoiseAdd), tmpNoise)  

        # tmpCSIa1 =  np.float_power(np.abs(tmpCSIa1), tmpNoise) * tmpNoiseAdd              
        # tmpCSIb1 =  np.float_power(np.abs(tmpCSIb1), tmpNoise) * tmpNoiseAdd
        # tmpCSIb2 =  np.float_power(np.abs(tmpCSIb2), tmpNoise) * tmpNoiseAdd

        # tmpCSIa1 =  np.log(np.float_power(np.abs(tmpCSIa1), tmpNoise))
        # tmpCSIb1 =  np.log(np.float_power(np.abs(tmpCSIb1), tmpNoise))
        # tmpCSIb2 =  np.log(np.float_power(np.abs(tmpCSIb2), tmpNoise)) 

        # tmpCSIa1 =  np.float_power(np.abs(tmpCSIa1), tmpNoise) + tmpNoiseAdd               ## Method 3: Power better than polynomial 
        # tmpCSIb1 =  np.float_power(np.abs(tmpCSIb1), tmpNoise) + tmpNoiseAdd
        # tmpCSIb2 =  np.float_power(np.abs(tmpCSIb2), tmpNoise) + tmpNoiseAdd

        # tmpCSIa1 =  np.float_power(np.abs(tmpCSIa1), tmpNoise)   +   tmpCSIa1 * tmpNoise            ## Method 3: Power better than polynomial 
        # tmpCSIb1 =  np.float_power(np.abs(tmpCSIb1), tmpNoise)   +   tmpCSIb1 * tmpNoise 
        # tmpCSIb2 =  np.float_power(np.abs(tmpCSIb2), tmpNoise)   +   tmpCSIb2 * tmpNoise 

        ## ----------------- END: Noise-assisted channel manipulation ---------------------------

        # tmpCSIa1 = smooth(tmpCSIa1, window_len = 9, window = 'flat')
        # tmpCSIb1 = smooth(tmpCSIb1, window_len = 9, window = 'flat')
        # tmpCSIb2 = smooth(tmpCSIb2, window_len = 9, window = 'flat')

        # plt.plot(tmpCSIa1)
        # plt.plot(tmpCSIb2)
        # plt.show()
        # continue

        if np.isnan(np.sum(tmpCSIa1)) + np.isnan(np.sum(tmpCSIb1)) == True:
            print('NaN value after power operation!')

        # # --------------------------------------------
        # #   END: Noise-assisted channel manipulation
        # # --------------------------------------------

        # # --------------------------------------------
        ##           BEGIN: Sorting and matching 
        # # --------------------------------------------

        permLen = len(range(staInd, endInd, intvl))
        origInd = np.array([xx for xx in range(staInd, endInd, intvl)])

        start_time = time.time()

        sortCSIa1 = np.zeros(permLen)
        sortCSIb1 = np.zeros(permLen)
        sortCSIb2 = np.zeros(permLen)
        sortNoise = np.zeros(permLen)

        CSIa1Ls = []
        CSIb1Ls = []
        CSIb2Ls = []
        noiseLs = []

        CSIa1Corr = []
        CSIb1Corr = []
        CSIb2Corr = []
        noiseCorr = []

        CSIa1Array = np.zeros((permLen, permLen))
        CSIb1Array = np.zeros((permLen, permLen))
        CSIb2Array = np.zeros((permLen, permLen))
        noiseArray = np.zeros((permLen, permLen))

        for ii in range(permLen):
            indVec = np.array([aa for aa in range(ii, ii + intvl, 1)])

            CSIa1Tmp = tmpCSIa1[indVec]
            CSIb1Tmp = tmpCSIb1[indVec]
            CSIb2Tmp = tmpCSIb2[indVec]

            noiseTmp = tmpNoise[indVec]  # + tmpNoiseb[indVec]
            # noiseTmp = np.float_power(-1, tmpNoise[indVec])

            CSIa1Ls.append(CSIa1Tmp)
            CSIb1Ls.append(CSIb1Tmp)
            CSIb2Ls.append(CSIb2Tmp)
            noiseLs.append(noiseTmp)

            for jj in range(len(CSIa1Ls) - 2, len(CSIa1Ls) - 1, 1):
                if len(CSIa1Ls) > 1:
                    CSIa1Corr.append(stats.pearsonr(CSIa1Ls[jj], CSIa1Tmp)[0])
                    CSIb1Corr.append(stats.pearsonr(CSIb1Ls[jj], CSIb1Tmp)[0])
                    CSIb2Corr.append(stats.pearsonr(CSIb2Ls[jj], CSIb2Tmp)[0])
                    noiseCorr.append(stats.pearsonr(noiseLs[jj], noiseTmp)[0])

                distpara = 'cityblock'
                # distpara = 'canberra'
                # distpara = 'braycurtis'
                # distpara = 'cosine'
                # distpara = 'euclidean'
                CSIa1Corr.append(pdist(np.vstack((CSIa1Ls[jj], CSIa1Tmp)), distpara)[0])
                CSIb1Corr.append(pdist(np.vstack((CSIb1Ls[jj], CSIb1Tmp)), distpara)[0])
                CSIb2Corr.append(pdist(np.vstack((CSIb2Ls[jj], CSIb2Tmp)), distpara)[0])
                noiseCorr.append(pdist(np.vstack((noiseLs[jj], noiseTmp)), distpara)[0])

            # print(CSIa1Corr)
            # print(CSIb1Corr)
            # time.sleep(1)

            # noiseTmp = np.random.normal(loc=0, scale=1, size=len(CSIa1Tmp)) 
            # noiseTmpAdd = np.random.normal(loc=para, scale=np.std(CSIa1Tmp), size=len(CSIa1Tmp))

            # CSIa1Tmp = (CSIa1Tmp ) * noiseTmp                                          ## Method 1: addition 
            # CSIb1Tmp = (CSIb1Tmp ) * noiseTmp 
            # CSIb2Tmp = (CSIb2Tmp ) * noiseTmp 

            # # ----------------------------------------------
            # #    Sorting with different metrics 
            # ## Indoor outperforms outdoor;  indoor with msdc feature performs better; outdoor feature unclear, mean seems better.
            # # ----------------------------------------------

            sortCSIa1[ii] = np.mean(CSIa1Tmp)  ## Metric 1: Mean
            sortCSIb1[ii] = np.mean(CSIb1Tmp)
            sortCSIb2[ii] = np.mean(CSIb2Tmp)
            sortNoise[ii] = np.mean(noiseTmp)

            # sortCSIa1[ii] = np.max(CSIa1Tmp)                          ## Metric 1: Mean 
            # sortCSIb1[ii] = np.max(CSIb1Tmp) 
            # sortCSIb2[ii] = np.max(CSIb2Tmp) 
            # sortNoise[ii] = np.max(noiseTmp) 

            # sortCSIa1[ii] = sumSeries(CSIa1Tmp)                       ## Metric 2: Sum
            # sortCSIb1[ii] = sumSeries(CSIb1Tmp)
            # sortCSIb2[ii] = sumSeries(CSIb2Tmp)
            # sortNoise[ii] = sumSeries(noiseTmp) 

            # sortCSIa1[ii] = msdc(CSIa1Tmp)                            ## Metric 3: tsfresh.msdc:  the metrics, msdc and mc, seem better
            # sortCSIb1[ii] = msdc(CSIb1Tmp)
            # sortCSIb2[ii] = msdc(CSIb2Tmp)
            # sortNoise[ii] = msdc(noiseTmp) 

            # sortCSIa1[ii] = mc(CSIa1Tmp)                              ## Metric 4: tsfresh.mc
            # sortCSIb1[ii] = mc(CSIb1Tmp)
            # sortCSIb2[ii] = mc(CSIb2Tmp)
            # sortNoise[ii] = mc(noiseTmp) 

            # sortCSIa1[ii] = cid(CSIa1Tmp, 1)                          ## Metric 5: tsfresh.cid_ie,               
            # sortCSIb1[ii] = cid(CSIb1Tmp, 1)
            # sortCSIb2[ii] = cid(CSIb2Tmp, 1)
            # sortNoise[ii] = cid(noiseTmp, 1)

        # plt.plot(sortCSIa1)
        # # plt.plot(sortCSIb1)
        # plt.plot(sortCSIb2)
        # plt.show()

        ## Matching outcomes
        print('----------------------------')
        sortInda = np.argsort(sortCSIa1)
        sortIndb1 = np.argsort(sortCSIb1)
        sortIndb2 = np.argsort(sortCSIb2)
        sortIndn = np.argsort(sortNoise)

        # print(permLen, (sortInda-sortIndb1 == 0).sum())
        # print(permLen, (sortInda-sortIndb2 == 0).sum())
        # print(permLen, (sortInda-sortIndn == 0).sum())  

        # print(sortInda.tolist())
        # print(sortIndb1.tolist())
        # print(sortIndb2.tolist())       
        # print(sortIndn.tolist())
        # time.sleep(1)
        # continue

        hamDist = []
        hamDiste = []
        hamDistn = []
        for kk in sortInda:
            indexa = np.where(sortInda == kk)[0][0]
            indexb1 = np.where(sortIndb1 == kk)[0][0]
            indexb2 = np.where(sortIndb2 == kk)[0][0]
            indexn = np.where(sortIndn == kk)[0][0]

            # sortInda和sortIndb1的对应索引元素距离：(1,2,3)和(1,2,3)返回(0,0,0)，(1,2,3)和(3,2,1)返回(2,0,2)
            hamDist.append(np.abs(indexa - indexb1))
            hamDiste.append(np.abs(indexa - indexb2))
            hamDistn.append(np.abs(indexa - indexn))

        print("hamDist", hamDist)
        print("hamDist_mean", np.mean(hamDist))
        print("hamDist_max", max(hamDist))
        print("hamDiste", hamDiste)
        print("hamDiste_mean", np.mean(hamDiste))
        print("hamDiste_max", max(hamDiste))
        print("hamDistn", hamDistn)
        print("hamDistn_mean", np.mean(hamDistn))
        print("hamDistn_max", max(hamDistn))
        # time.sleep(1)

        correctRate.append((sortInda - sortIndb1 == 0).sum())
        randomRate.append((sortInda - sortIndb2 == 0).sum())
        noiseRate.append((sortInda - sortIndn == 0).sum())

        # sortCorrInda = np.argsort(CSIa1Corr)
        # sortCorrIndb1 = np.argsort(CSIb1Corr)
        # sortCorrIndb2 = np.argsort(CSIb2Corr)
        # sortCorrIndn = np.argsort(noiseCorr)

        # print(len(sortCorrInda), (sortCorrInda-sortCorrIndb1 == 0).sum())
        # print(len(sortCorrInda), (sortCorrInda-sortCorrIndb2 == 0).sum())
        # print(len(sortCorrInda), (sortCorrInda-sortCorrIndn == 0).sum())

        # # --------------------------------------------
        ##         END: Sorting and matching
        # # --------------------------------------------

        if (sortInda - sortIndb1 == 0).sum() == permLen:
            np.save('../experiments/tmpNoise.npy', tmpNoise)
            # sys.exit()

    paraRate[0, paraLs.index(para)] = sum(correctRate) / len(correctRate)
    paraRate[1, paraLs.index(para)] = sum(randomRate) / len(randomRate)
    paraRate[2, paraLs.index(para)] = sum(noiseRate) / len(noiseRate)

print(paraRate)

plt.plot(paraRate[0, :])
plt.plot(paraRate[1, :])
plt.plot(paraRate[2, :])
plt.show()

sys.exit("Stop.")
