import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr

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

CSIa1Orig = list(loadmat("testdata(2).mat")['testdata'][:, 0])
CSIb1Orig = list(loadmat("testdata(2).mat")['testdata'][:, 1])
# CSIa1Orig = list(loadmat("testdata(1).mat")['testdata'][:, 0])
# CSIb1Orig = list(loadmat("testdata(1).mat")['testdata'][:, 1])
# CSIa1Orig = list(loadmat("testdata.mat")['testdata'][:, 0])
# CSIb1Orig = list(loadmat("testdata.mat")['testdata'][:, 1])
# CSIa1Orig = list(loadmat("edit_distance/predictable/CSI1.mat")['CSI1'][:, 0])
# CSIb1Orig = list(loadmat("edit_distance/predictable/CSI2.mat")['CSI2'][:, 0])

CSIa = []
CSIb = []
print(len(CSIa1Orig), len(CSIb1Orig))
for i in range(min(len(CSIa1Orig), len(CSIb1Orig))):
    if abs(CSIa1Orig[i] - CSIb1Orig[i]) <= 0.5:
        CSIa.append(CSIa1Orig[i])
        CSIb.append(CSIb1Orig[i])

print(len(CSIa), len(CSIb))

# rawDataA = loadmat("data/CSIa_r.mat")
# rawDataB = loadmat("data/CSIb_r.mat")

# CSIa1Orig = rawDataA['CSIa'][0][0:100]
# CSIb1Orig = rawDataB['CSIb'][0][0:100]

print(pearsonr(CSIa, CSIb)[0])
k = 0
CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=3, window='flat')
CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=3, window='flat')

print(pearsonr(CSIa1Orig[k:], CSIb1Orig[:len(CSIa1Orig) - k])[0])

# savemat('testdata_r.mat', {"testdata": np.array([CSIa, CSIb]).T})

plt.figure()
plt.plot(CSIa)
plt.plot(CSIb)
plt.show()

plt.figure()
plt.plot(CSIa1Orig)
plt.plot(CSIb1Orig)
plt.show()