import numpy as np
from numpy.random import exponential as Exp
from scipy.io import loadmat

print(np.argsort([1, 3, 5, 2, 4]))

part = 3


def CSIPartialMean(CSITmp):
    anotherCSITmp = []
    for i in range(len(CSITmp)):
        partialSum = 0
        for j in range(0, part):
            partialSum += (CSITmp[(i + j) % len(CSITmp)])
        anotherCSITmp.append(partialSum / part)
    return anotherCSITmp


print(CSIPartialMean([1, 3, 5, 2, 4]))


def double(CSITmp):
    anotherCSITmp = []
    for i in range(len(CSITmp)):
        anotherCSITmp.append(CSITmp[i] * 2)
    return anotherCSITmp


print(double([1, 2, 3, 4, 5]))


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


def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser'")

    # 14 + 6745 + 14
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    print(len(s))  # 6773

    if window == 'flat':  # moving average

        w = np.ones(window_len, 'd')
    elif window == 'kaiser':
        beta = 5
        w = eval('np.' + window + '(window_len, beta)')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


# print(random_instance(2, 4))

rawData = loadmat('data/data_mobile_outdoor_1.mat')
# print(rawData['A'])
# print(rawData['A'][:, 0])
# print(len(rawData['A'][:, 0]))

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
dataLen = len(CSIa1Orig)

print("CSIa1Orig", CSIa1Orig)
print("CSIb1Orig", CSIb1Orig)

CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')
CSIb1Orig = smooth(CSIb1Orig, window_len=15, window='flat')

noise = np.random.normal(loc=-1, scale=1, size=dataLen)
print(noise)
