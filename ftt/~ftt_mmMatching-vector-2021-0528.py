import numpy as np
from numpy.random import exponential as Exp
from scipy.io import loadmat
from scipy import signal


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
rawData = loadmat('../data/data_static_indoor_1.mat')
CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
dataLen = len(CSIa1Orig)

# CSIb2Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)
rawData = loadmat('../data/data_mobile_indoor_2.mat')
CSIb2Orig = rawData['A'][0:dataLen, 1]

avg = 0
std = 1
noise = np.random.normal(loc=avg, scale=std, size=dataLen)  ## Multiplication item normal distribution
noiseAdd = np.random.normal(loc=avg, scale=std, size=dataLen)  ## Addition item normal distribution

win = 13
for p in range(0, dataLen, win):
    stdSeg = 1 / np.std(noise[p:p + win])
    noise[p] = np.random.normal(loc=avg, scale=stdSeg, size=1)
    noiseAdd[p] = np.random.normal(loc=avg, scale=stdSeg, size=1)

# # -----------------------------------
# # ---- Smoothing -------------
winNam = 'flat'
CSIa1Orig = smooth(CSIa1Orig, window_len=win, window=winNam)
CSIb1Orig = smooth(CSIb1Orig, window_len=win, window=winNam)
CSIb2Orig = smooth(CSIb2Orig, window_len=win, window=winNam)

paraLs = [3]
paraRate = np.zeros((3, len(paraLs)))
hamRate = []

for para in paraLs:
    sft = para
    intvl = 2 * sft + 1
    keyLen = 256

    correctRate = []
    randomRate = []
    noiseRate = []

    totalHamDist = []
    totalHamDiste = []
    totalHamDistn = []

    for staInd in range(0, dataLen - keyLen * intvl - 1, intvl):
        print(staInd)
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

        tmpCSIa1 = tmpCSIa1 * tmpNoise  ## Method 2: polynomial product better than addition
        tmpCSIb1 = tmpCSIb1 * tmpNoise
        tmpCSIb2 = tmpCSIb2 * tmpNoise

        # tmpPulse = signal.square(
        #     2 * np.pi * 1 / intvl * np.linspace(0, np.pi * 0.5 * keyLen, keyLen * intvl))  ## Rectangular pulse

        # tmpCSIa1 = tmpPulse * (np.float_power(np.abs(tmpCSIa1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIa1))
        # tmpCSIb1 = tmpPulse * (np.float_power(np.abs(tmpCSIb1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIb1))
        # tmpCSIb2 = tmpPulse * (np.float_power(np.abs(tmpCSIb2), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIb2))

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

        if np.isnan(np.sum(tmpCSIa1)) + np.isnan(np.sum(tmpCSIb1)) == True:
            print('NaN value after power operation!')

        # # --------------------------------------------
        ##           BEGIN: Sorting and matching 
        # # --------------------------------------------

        permLen = len(range(staInd, endInd, intvl))

        sortCSIa1 = np.zeros(permLen)
        sortCSIb1 = np.zeros(permLen)
        sortCSIb2 = np.zeros(permLen)
        sortNoise = np.zeros(permLen)

        for ii in range(permLen):
            indVec = np.array([aa for aa in range(ii, ii + intvl, 1)])

            CSIa1Tmp = tmpCSIa1[indVec]
            CSIb1Tmp = tmpCSIb1[indVec]
            CSIb2Tmp = tmpCSIb2[indVec]
            noiseTmp = tmpNoise[indVec]  # + tmpNoiseb[indVec]

            sortCSIa1[ii] = np.mean(CSIa1Tmp)  ## Metric 1: Mean
            sortCSIb1[ii] = np.mean(CSIb1Tmp)
            sortCSIb2[ii] = np.mean(CSIb2Tmp)
            sortNoise[ii] = np.mean(noiseTmp)

        ## Matching outcomes
        print('----------------------------')
        sortInda = np.argsort(sortCSIa1)
        sortIndb1 = np.argsort(sortCSIb1)
        sortIndb2 = np.argsort(sortCSIb2)
        sortIndn = np.argsort(sortNoise)

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
        print("\033[0;32;40mhamDist_max", max(hamDist), "\033[0m")
        print("hamDiste", hamDiste)
        print("hamDiste_mean", np.mean(hamDiste))
        print("\033[0;33;40mhamDiste_max", max(hamDiste), "\033[0m")
        print("hamDistn", hamDistn)
        print("hamDistn_mean", np.mean(hamDistn))
        print("\033[0;33;40mhamDistn_max", max(hamDistn),"\033[0m")
        # time.sleep(1)

        hamDist = np.array(hamDist)
        hamDiste = np.array(hamDiste)
        hamDistn = np.array(hamDistn)

        correctRate.append((sortInda - sortIndb1 == 0).sum())
        randomRate.append((sortInda - sortIndb2 == 0).sum())
        noiseRate.append((sortInda - sortIndn == 0).sum())

        totalHamDist.append(np.mean(hamDist))
        totalHamDiste.append(np.mean(hamDiste))
        totalHamDistn.append(np.mean(hamDistn))

    paraRate[0, paraLs.index(para)] = sum(correctRate) / len(correctRate) / 256  # 平均正确的个数
    paraRate[1, paraLs.index(para)] = sum(randomRate) / len(randomRate) / 256
    paraRate[2, paraLs.index(para)] = sum(noiseRate) / len(noiseRate) / 256

    hamRate.append(sum(totalHamDist) / len(totalHamDist))  # 平均汉明距离
    hamRate.append(sum(totalHamDiste) / len(totalHamDiste))
    hamRate.append(sum(totalHamDistn) / len(totalHamDistn))

print(paraRate)
print(hamRate)
