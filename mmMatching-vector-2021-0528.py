# from mwmatching import maxWeightMatching
# import cv2
import sys
import time

import numpy as np
from numpy.random import exponential as Exp
from scipy import signal
from scipy import sparse
from scipy.io import loadmat
from tsfresh.feature_extraction.feature_calculators import mean_second_derivative_central as msdc


# 指数分布
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


def sumSeries(CSITmp):
    if len(CSITmp) > 1:
        sumCSI = sum(CSITmp) + sumSeries(CSITmp[0:-1])
        return sumCSI
    else:
        return CSITmp[0]


## -----------------------------------
# plt.close('all')
# np.random.seed(0)

rawData = loadmat('data/data_mobile_outdoor_1.mat')
print(rawData['A'])
# 取rawData['A']的第一个元素
print(rawData['A'][:, 0])
print(len(rawData['A'][:, 0]))  # 6745

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
dataLen = len(CSIa1Orig)  # 6745

# # # ----------- Simulated data ---------------
# CSIa1Orig = CSIa1Orig + np.random.normal(loc=0, scale=1, size=dataLen)
# CSIb1Orig = CSIa1Orig + np.random.normal(loc=0, scale=1, size=dataLen)


# # -----------------------------------
# # ---- Smoothing -------------
# ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser']
CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')
CSIb1Orig = smooth(CSIb1Orig, window_len=15, window='flat')

# CSIa1Orig = hp_filter(CSIa1Orig, lamb=500)
# CSIb1Orig = hp_filter(CSIb1Orig, lamb=500)

# CSIa1Orig = savgol_filter(CSIa1Orig, 11, 1)
# CSIb1Orig = savgol_filter(CSIb1Orig, 11, 1)

# -----------------------------------------------------------------------------------
#     ---- Constant Noise Generation ----
#  Pre-allocated noise, will not change during sorting and matching:
#  Use the following noise, need to comment the ines of "Instant noise generator"
# -----------------------------------------------------------------------------------
CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
# noise = np.asarray([random.randint(-1, 1) for iter in range(dataLen)])   ## Integer Uniform Distribution
# noise = np.round(np.random.normal(loc=0, scale=1, size=dataLen))   ## Integer Normal Distribution

noise = np.random.normal(loc=-1, scale=1, size=dataLen)  ## Multiplication item normal distribution
noiseAdd = np.random.normal(loc=0, scale=10, size=dataLen)  ## Addition item normal distribution

# conNoise = np.concatenate((np.random.normal(-1, 10, int(np.ceil(dataLen*0.5))), np.random.normal(mb, vb, int(np.floor(dataLen*0.5)))))
# noise = conNoise[np.random.permutation(dataLen)  ]                                                                                           ## Bimodel distribution

# noise = np.random.uniform(-1, 1, size=dataLen)   ## float Uniform distribution

## save the noises:
noiseBack = noise.copy()
noiseAddBack = noiseAdd.copy()

## ---------------------------------------------------------
sft = 2
intvl = 2 * sft + 1
keyLen = 128

misRate = []
errRate = []
errCnt = 0

for staInd in range(0, dataLen - keyLen * intvl - 1, intvl):
    print("staInd", staInd)
    # staInd = 0                               # fixed start for testing
    endInd = staInd + keyLen * intvl

    CSIa1Orig = CSIa1OrigBack.copy()
    CSIb1Orig = CSIb1OrigBack.copy()

    # --------------------------------------------
    # BEGIN: Noise-assisted channel manipulation
    # --------------------------------------------
    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
    tmpNoise = noise[range(staInd, endInd, 1)]
    tmpNoiseAdd = noiseAdd[range(staInd, endInd, 1)]
    epiLen = len(range(staInd, endInd, 1))

    tmpCSIa1 = tmpCSIa1 - (np.mean(tmpCSIa1) - np.mean(tmpCSIb1))  # Mean value consistency

    # tmpPulse = np.sin(np.linspace(0, np.pi*0.5*keyLen, keyLen*intvl))                              ## Sine pulse
    # linspace函数生成元素为50的等间隔数列，可以指定第三个参数为元素个数
    # signal.square返回周期性的方波波形
    tmpPulse = signal.square(
        2 * np.pi * 1 / intvl * np.linspace(0, np.pi * 0.5 * keyLen, keyLen * intvl))  ## Rectangular pulse

    # # -----------------------------------
    # # Instant noise generator (Noise changes for every segment of channel measurements)
    # tmpNoise = np.random.normal(loc=-1*np.std(tmpCSIa1), scale=np.std(tmpCSIa1), size=len(tmpNoise))
    # tmpNoiseAdd = np.random.normal(loc=0, scale=10, size=len(tmpNoiseAdd))

    ## ----------------- BEGIN: Noise-assisted channel manipulation ---------------------------

    # tmpCSIa1 = (tmpCSIa1 + tmpNoiseAdd)                                               ## Method 1: addition
    # tmpCSIb1 = (tmpCSIb1 + tmpNoiseAdd)

    # tmpCSIa1 = tmpCSIa1 * tmpNoise + tmpNoiseAdd                                       ## Method 2: polynomial product better than addition
    # tmpCSIb1 = tmpCSIb1 * tmpNoise + tmpNoiseAdd

    # np.float_power将数组A的每个元素求数据B对应元素的幂
    tmpCSIa1 = tmpPulse * np.float_power(np.abs(tmpCSIa1), tmpNoise)  ## Method 3: Power better than polynomial
    tmpCSIb1 = tmpPulse * np.float_power(np.abs(tmpCSIb1), tmpNoise)

    ## ----------------- END: Noise-assisted channel manipulation ---------------------------

    CSIa1Orig[range(staInd, endInd, 1)] = tmpCSIa1
    CSIb1Orig[range(staInd, endInd, 1)] = tmpCSIb1
    noise[range(staInd, endInd, 1)] = tmpNoise

    # np.isnan判断是不是空值
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
    sortNoise = np.zeros(permLen)

    for ii in range(permLen):
        coefLs = []
        aIndVec = np.array([aa for aa in range(origInd[ii], origInd[ii] + intvl, 1)])  ## for non-permuted CSIa1

        for jj in range(permLen, permLen * 2):
            bIndVec = np.array([bb for bb in range(origInd[jj - permLen], origInd[jj - permLen] + intvl, 1)])

            CSIa1Tmp = CSIa1Orig[aIndVec]
            CSIb1Tmp = CSIb1Orig[bIndVec]
            noiseaTmp = noise[aIndVec]
            noisebTmp = noise[bIndVec]
            # noiseaTmpAdd = noiseAdd[aIndVec]
            noiseaTmpAdd = noise[aIndVec]
            noisebTmpAdd = noise[bIndVec]

            CSIapTmp = CSIa1Orig[aIndVec]
            CSIbpTmp = CSIb1Orig[bIndVec]

            # # ----------------------------------------------
            # #    Sorting with different metrics
            # ## Indoor outperforms outdoor;  indoor with msdc feature performs better; outdoor feature unclear, mean seems better.
            # # ----------------------------------------------

            # sortCSIa1[ii] = np.mean(CSIa1Tmp)  ## Metric 1: Mean
            # sortCSIb1[jj - permLen] = np.mean(CSIb1Tmp)  # 只赋值一次
            # sortNoise[ii] = np.mean(noiseaTmp)

            # sortCSIa1[ii] = sumSeries(CSIa1Tmp)                       ## Metric 2: Sum
            # sortCSIb1[jj-permLen] = sumSeries(CSIb1Tmp)
            # sortNoise[ii] = sumSeries(noiseaTmp)

            sortCSIa1[ii] = msdc(CSIa1Tmp)  ## Metric 3: tsfresh.msdc:  the metrics, msdc and mc, seem better
            sortCSIb1[jj - permLen] = msdc(CSIb1Tmp)
            sortNoise[ii] = msdc(noisebTmp)

            # sortCSIa1[ii] = mc(CSIa1Tmp)                              ## Metric 4: tsfresh.mc
            # sortCSIb1[jj-permLen] = mc(CSIb1Tmp)
            # sortNoise[ii] = mc(noisebTmp)

            # sortCSIa1[ii] = cid(CSIa1Tmp, 1)                          ## Metric 5: tsfresh.cid_ie,
            # sortCSIb1[jj-permLen] = cid(CSIb1Tmp, 1)
            # sortNoise[ii] = cid(noisebTmp, 1)

    ## Matching outcomes
    # np.argsort函数是将x中的元素从小到大排列，提取其对应的索引，然后输出到y
    sortInda = np.argsort(sortCSIa1)
    sortIndb = np.argsort(sortCSIb1)
    sortIndn = np.argsort(sortNoise)

    print(permLen, (sortInda - sortIndb == 0).sum())  # sortInda和sortIndb对应位置相等的个数
    print(permLen, (sortInda - sortIndn == 0).sum())
    print(permLen, (sortIndb - sortIndn == 0).sum())

    # print(sortInda)
    # print(sortIndb)
    # print(sortIndn)

    # print("sortCSIa1", sortCSIa1)
    # print("sortCSIb1", sortCSIb1)

    sortCSIa1Back = sortCSIa1.copy()
    sortCSIb1Back = sortCSIb1.copy()

    sortCSIa1Back.sort()
    sortCSIb1Back.sort()
    # print("sortCSIa1Back", sortCSIa1Back)
    # print("sortCSIb1Back", sortCSIb1Back)

    diff = list(map(lambda x, y: x - y, sortInda, sortIndb))
    print("diff", diff)

    for a in range(len(diff)):
        if diff[a] != 0:
            print("except", a)
            break

    step = []

    for a in range(len(sortCSIa1Back) - 1):
        step.append(sortCSIa1Back[a + 1] - sortCSIa1Back[a])
    # print("step", step)

    step = []

    for a in range(len(sortCSIb1Back) - 1):
        step.append(sortCSIb1Back[a + 1] - sortCSIb1Back[a])
    # print("step", step)

    # # --------------------------------------------
    ##         END: Sorting and matching
    # # --------------------------------------------

    if (sortInda - sortIndb == 0).sum() == permLen:
        np.save('experiments/tmpNoise.npy', tmpNoise)
        # sys.exit()

sys.exit("Stop.")
