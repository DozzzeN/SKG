import math
import time

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from scipy import signal
from scipy.fft import dct
from scipy.io import loadmat


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


# 数组第二维的所有内容求和
def sumEachDim(list, index):
    res = 0
    for i in range(len(list[index])):
        res += (list[index][i][0] + list[index][i][1])
        # res += (list[index][i][0] * list[index][i][1])
    return round(res, 8)


def projection(l1, l2, p):
    v1 = [p[0] - l1[0], p[1] - l1[1]]
    v2 = [l2[0] - l1[0], l2[1] - l1[1]]
    v1v2 = v1[0] * v2[0] + v1[1] * v2[1]
    k = v1v2 / (math.pow(l2[0] - l1[0], 2) + math.pow(l2[1] - l1[1], 2))
    p0 = [l1[0] + k * (l2[0] - l1[0]), l1[1] + k * (l2[1] - l1[1])]
    return p0


def normal2uniform(data):
    data_reshape = np.array(data[0: 2 * int(len(data) / 2)])
    data_reshape = data_reshape.reshape(int(len(data_reshape) / 2), 2)
    x_list = []
    for i in range(len(data_reshape)):
        r = np.sum(np.square(data_reshape[i]))
        x_list.append(np.exp(-0.5 * r))

    # plt.figure()
    # plt.hist(x_list)
    # plt.show()

    return x_list


l1 = [1, 0]
l2 = [0, 1]

start_time = time.time()
fileName = "../data/data_static_indoor_1_r_m.mat"
rawData = loadmat(fileName)

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

CSIa1Orig = np.array(CSIa1Orig)
CSIb1Orig = np.array(CSIb1Orig)

dataLen = len(CSIa1Orig)

CSIn1Orig = np.random.normal(loc=-100, scale=1000, size=dataLen)

CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)

# 固定随机置换的种子
np.random.seed(1)  # 8 1024 8; 4 128 4
combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig, CSIe1Orig, CSIn1Orig))
np.random.shuffle(combineCSIx1Orig)
CSIa1Orig, CSIb1Orig, CSIe1Orig, CSIn1Orig = zip(*combineCSIx1Orig)

CSIa1Orig = np.array(CSIa1Orig)
CSIb1Orig = np.array(CSIb1Orig)
CSIe1Orig = np.array(CSIe1Orig)
CSIn1Orig = np.array(CSIn1Orig)

CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')
CSIb1Orig = smooth(CSIb1Orig, window_len=15, window='flat')
CSIe1Orig = smooth(CSIe1Orig, window_len=15, window="flat")
CSIn1Orig = smooth(CSIn1Orig, window_len=15, window="flat")

noise = np.random.normal(loc=-1, scale=1, size=dataLen)  ## Multiplication item normal distribution
noiseAdd = np.random.normal(loc=0, scale=10, size=dataLen)  ## Addition item normal distribution

segLen = 7
keyLen = 64 * segLen

ratio = 1
# rawOp = ""
rawOp = "coxbox-uniform"

originSum = 0
correctSum = 0
randomSum = 0
noiseSum = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum = 0
noiseWholeSum = 0

times = 0

for staInd in range(0, len(CSIa1Orig), keyLen):
    endInd = staInd + keyLen
    print("range:", staInd, endInd)
    if endInd >= len(CSIa1Orig):
        break
    times += 1

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
    tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
    tmpNoise = CSIn1Orig[range(staInd, endInd, 1)]

    # 去除直流分量
    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
    tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
    tmpNoise = tmpNoise - np.mean(tmpNoise)

    scale = 10
    offset = -200
    if rawOp == "fft":
        sortCSIa1 = np.abs(np.fft.fft(tmpCSIa1))
        sortCSIb1 = np.abs(np.fft.fft(tmpCSIb1))
        sortCSIe1 = np.abs(np.fft.fft(tmpCSIe1))
        sortNoise = np.abs(np.fft.fft(tmpNoise))

        sortCSIa1 = sortCSIa1 - np.mean(sortCSIa1)
        sortCSIb1 = sortCSIb1 - np.mean(sortCSIb1)
        sortCSIe1 = sortCSIe1 - np.mean(sortCSIe1)
        sortNoise = sortNoise - np.mean(sortNoise)
    elif rawOp == "dct":
        sortCSIa1 = np.abs(dct(tmpCSIa1))
        sortCSIb1 = np.abs(dct(tmpCSIb1))
        sortCSIe1 = np.abs(dct(tmpCSIe1))
        sortNoise = np.abs(dct(tmpNoise))

        sortCSIa1 = sortCSIa1 - np.mean(sortCSIa1)
        sortCSIb1 = sortCSIb1 - np.mean(sortCSIb1)
        sortCSIe1 = sortCSIe1 - np.mean(sortCSIe1)
        sortNoise = sortNoise - np.mean(sortNoise)
    elif rawOp == "zca":
        step = 1
        tmpCSIa12D = tmpCSIa1.reshape(int(len(tmpCSIa1) / step), step)
        trf = ZCA().fit(tmpCSIa12D)
        sortCSIa1 = trf.transform(np.abs(tmpCSIa12D)).reshape(1, -1)[0]
        tmpCSIb12D = tmpCSIb1.reshape(int(len(tmpCSIb1) / step), step)
        sortCSIb1 = trf.transform(np.abs(tmpCSIb12D)).reshape(1, -1)[0]
        tmpCSIe12D = tmpCSIe1.reshape(int(len(tmpCSIe1) / step), step)
        sortCSIe1 = trf.transform(np.abs(tmpCSIe12D)).reshape(1, -1)[0]
        tmpNoise2D = tmpNoise.reshape(int(len(tmpNoise) / step), step)
        sortNoise = trf.transform(np.abs(tmpNoise2D)).reshape(1, -1)[0]
    elif rawOp == "uniform":
        sortCSIa1 = normal2uniform(tmpCSIa1)
        sortCSIb1 = normal2uniform(tmpCSIb1)
        sortCSIe1 = normal2uniform(tmpCSIe1)
        sortNoise = normal2uniform(tmpNoise)
    elif rawOp == "coxbox":
        sortCSIa1 = scipy.stats.boxcox(np.abs(tmpCSIa1))[0]
        sortCSIb1 = scipy.stats.boxcox(np.abs(tmpCSIb1))[0]
        sortCSIe1 = scipy.stats.boxcox(np.abs(tmpCSIe1))[0]
        sortNoise = scipy.stats.boxcox(np.abs(tmpNoise))[0]
    elif rawOp == "coxbox-uniform":
        sortCSIa1 = scipy.stats.boxcox(np.abs(tmpCSIa1))[0]
        sortCSIb1 = scipy.stats.boxcox(np.abs(tmpCSIb1))[0]
        sortCSIe1 = scipy.stats.boxcox(np.abs(tmpCSIe1))[0]
        sortNoise = scipy.stats.boxcox(np.abs(tmpNoise))[0]
        sortCSIa1 = normal2uniform(sortCSIa1)
        sortCSIb1 = normal2uniform(sortCSIb1)
        sortCSIe1 = normal2uniform(sortCSIe1)
        sortNoise = normal2uniform(sortNoise)
    else:
        if rawOp is not None and rawOp != "":
            raise Exception("error rawOp")
        sortCSIa1 = tmpCSIa1
        sortCSIb1 = tmpCSIb1
        sortCSIe1 = tmpCSIe1
        sortNoise = tmpNoise

    sortCSIa1 = smooth(np.array(sortCSIa1), window_len=15, window='flat')
    sortCSIb1 = smooth(np.array(sortCSIb1), window_len=15, window='flat')
    sortCSIe1 = smooth(np.array(sortCSIe1), window_len=15, window='flat')
    sortNoise = smooth(np.array(sortNoise), window_len=15, window='flat')

    # 归一化
    # _max = max(max(sortCSIa1), max(sortCSIb1), max(sortCSIe1), max(sortNoise))
    # _min = min(min(sortCSIa1), min(sortCSIb1), min(sortCSIe1), min(sortNoise))

    # sortCSIa1 = sortCSIa1 / (_max - _min) - _min / (_max - _min)
    # sortCSIb1 = sortCSIb1 / (_max - _min) - _min / (_max - _min)
    # sortCSIe1 = sortCSIe1 / (_max - _min) - _min / (_max - _min)
    # sortNoise = sortNoise / (_max - _min) - _min / (_max - _min)

    # sortCSIa1是原始算法中排序前的数据
    sortCSIa1 = np.log10(np.abs(sortCSIa1))
    sortCSIb1 = np.log10(np.abs(sortCSIb1))
    sortCSIe1 = np.log10(np.abs(sortCSIe1))
    sortNoise = np.log10(np.abs(sortNoise))

    # 形成三维数组，其中第三维是一对坐标值
    # 数组的长度由param调节
    param = 0
    step = int(math.pow(2, param))
    sortCSIa1 = sortCSIa1.reshape(int(len(sortCSIa1) / step / 2), 2)
    sortCSIb1 = sortCSIb1.reshape(int(len(sortCSIb1) / step / 2), 2)
    sortCSIe1 = sortCSIe1.reshape(int(len(sortCSIe1) / step / 2), 2)
    sortNoise = sortNoise.reshape(int(len(sortNoise) / step / 2), 2)

    plt.figure()
    plt.plot(sortCSIa1[:, 0], sortCSIa1[:, 1], color="red", linewidth=1, label="a")
    # plt.plot(CSIb1Orig, color="blue", linewidth=.05, label="b")
    plt.legend(loc='upper left')
    # plt.show()

    # 最后各自的密钥
    a_list = []
    b_list = []
    e_list = []
    n_list = []

    projCSIa1XY = []
    projCSIb1XY = []
    projCSIe1XY = []
    projCSIn1XY = []

    projCSIa1X = []
    projCSIb1X = []
    projCSIe1X = []
    projCSIn1X = []

    for i in range(len(sortCSIa1)):
        projCSIa1XY.append(projection(l1, l2, sortCSIa1[i]))
        projCSIb1XY.append(projection(l1, l2, sortCSIb1[i]))
        projCSIe1XY.append(projection(l1, l2, sortCSIe1[i]))
        projCSIn1XY.append(projection(l1, l2, sortNoise[i]))

        projCSIa1X.append(projection(l1, l2, sortCSIa1[i])[0])
        projCSIb1X.append(projection(l1, l2, sortCSIb1[i])[0])
        projCSIe1X.append(projection(l1, l2, sortCSIe1[i])[0])
        projCSIn1X.append(projection(l1, l2, sortNoise[i])[0])

    a_list = np.argsort(projCSIa1X)
    b_list = np.argsort(projCSIb1X)
    e_list = np.argsort(projCSIe1X)
    n_list = np.argsort(projCSIn1X)

    # 绘图
    plt.figure()
    xa, ya = zip(*projCSIa1XY)
    xb, yb = zip(*projCSIb1XY)
    xe, ye = zip(*projCSIe1XY)
    xn, yn = zip(*projCSIn1XY)
    plt.scatter(xa, ya, color="red", linewidth=.5, label="a")
    plt.scatter(xb, yb, color="blue", linewidth=.5, label="b")
    # plt.plot(xn, yn, color="yellow", linewidth=2.5, label="n") # 数量级差别太大，不方便显示
    plt.legend(loc='upper left')
    # plt.show()

    plt.scatter(xa, ya, color="red", linewidth=.5, label="a")
    plt.scatter(xe, ye, color="black", linewidth=.5, label="e")
    plt.legend(loc='upper left')
    # plt.show()

    print("keys of a:", a_list)
    print("keys of b:", b_list)
    print("keys of e:", e_list)
    print("keys of n:", n_list)

    sum1 = len(a_list)
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for i in range(0, sum1):
        sum2 += (a_list[i] - b_list[i] == 0)
        sum3 += (a_list[i] - e_list[i] == 0)
        sum4 += (a_list[i] - n_list[i] == 0)

    print("a-b", sum2 / sum1)
    print("a-e", sum3 / sum1)
    print("a-n", sum4 / sum1)
    print("----------------------")
    originSum += sum1
    correctSum += sum2
    randomSum += sum3
    noiseSum += sum4

    originWholeSum += 1
    correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
    randomWholeSum = randomWholeSum + 1 if sum3 == sum1 else randomWholeSum
    noiseWholeSum = noiseWholeSum + 1 if sum4 == sum1 else noiseWholeSum

print("a-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10))
print("a-e all", randomSum, "/", originSum, "=", round(randomSum / originSum, 10))
print("a-n all", noiseSum, "/", originSum, "=", round(noiseSum / originSum, 10))
print("a-b whole match", correctWholeSum, "/", originWholeSum, "=", round(correctWholeSum / originWholeSum, 10))
print("a-e whole match", randomWholeSum, "/", originWholeSum, "=", round(randomWholeSum / originWholeSum, 10))
print("a-n whole match", noiseWholeSum, "/", originWholeSum, "=", round(noiseWholeSum / originWholeSum, 10))
print("times", times)
print("测试结束，耗时" + str(round(time.time() - start_time, 3)), "s")
