import csv
import math
import time
from tkinter import messagebox

import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import dct
from scipy.io import loadmat
from scipy.stats import pearsonr, boxcox

from zca import ZCA


# 定义计算离散点积分的函数
def integral_from_start(x, y):
    import scipy
    from scipy.integrate import simps  # 用于计算积分
    integrals = []
    for i in range(len(y)):  # 计算梯形的面积，由于是累加，所以是切片"i+1"
        integrals.append(scipy.integrate.trapz(y[:i + 1], x[:i + 1]))
    return integrals


# 定义计算离散点导数的函数
def derivative(x, y):  # x, y的类型均为列表
    diff_x = []  # 用来存储x列表中的两数之差
    for i, j in zip(x[0::], x[1::]):
        diff_x.append(j - i)

    diff_y = []  # 用来存储y列表中的两数之差
    for i, j in zip(y[0::], y[1::]):
        diff_y.append(j - i)

    slopes = []  # 用来存储斜率
    for i in range(len(diff_y)):
        slopes.append(diff_y[i] / diff_x[i])

    deriv = []  # 用来存储一阶导数
    for i, j in zip(slopes[0::], slopes[1::]):
        deriv.append((0.5 * (i + j)))  # 根据离散点导数的定义，计算并存储结果
    deriv.insert(0, slopes[0])  # (左)端点的导数即为与其最近点的斜率
    deriv.append(slopes[-1])  # (右)端点的导数即为与其最近点的斜率
    return deriv  # 返回存储一阶导数结果的列表


def integral_sq_derivative_increment(data, noise):
    index = list(range(len(data)))
    intgrl = integral_from_start(index, data)
    # square = np.power(intgrl, 2)
    square = intgrl + noise
    diff = derivative(index, square)
    return diff


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


def normal2uniform(data):
    data1 = data[:int(len(data) / 2)]
    data2 = data[int(len(data) / 2):]
    data_reshape = np.array(data[0: 2 * int(len(data) / 2)])
    data_reshape = data_reshape.reshape(int(len(data_reshape) / 2), 2)
    x_list = []
    for i in range(len(data_reshape)):
        # r = np.sum(np.square(data_reshape[i]))
        r = np.sum(data1[i] * data1[i] + data2[i] * data2[i])
        x_list.append(np.exp(-0.5 * r))
        x_list.append(np.exp(-0.5 * r))
        # r = data2[i] / data1[i]
        # x_list.append(math.atan(r) / math.pi + 0.5)
        # x_list.append(math.atan(r) / math.pi + 0.5)

    return x_list


# fileName = ["../data/data_mobile_indoor_1.mat",
#             "../data/data_mobile_outdoor_1.mat",
#             "../data/data_static_outdoor_1.mat",
#             "../data/data_static_indoor_1.mat"
#             ]

fileName = ["../data/data_mobile_indoor_1.mat"]

# fileName = ["../data/data_static_indoor_1.mat"]
# fileName = ["../data/data_static_outdoor_1_r.mat"]

# fileName = ["../data/data_NLOS.mat"]  # "../data/data3_upto5.mat"就是si

# fileName = ["../csi/csi_mobile_indoor_1_r",
#             "../csi/csi_static_indoor_1_r",
#             "../csi/csi_mobile_outdoor_r",
#             "../csi/csi_static_outdoor_r"]

# data BMR BGR BGR-with-no-error
# segLen = 2
# mi1 0.7801432292 0.0 5.0 3.9007161458333335
# si1 0.7910058594 0.0 5.0 3.955029296875
# mo1 0.7801432292 0.0 5.0 3.9007161458333335
# so1 0.6795605469 0.0 5.0 3.397802734375

# segLen = 3
# mi1 0.995686849 0.1666666667 3.3333333333333335 3.3189561631944446
# si1 0.9569839015 0.0 3.3333333333333335 3.1899463383838387
# mo1 0.9861816406 0.0 3.3333333333333335 3.2872721354166665
# so1-r 0.9740652902 0.7142857143 3.3333333333333335 3.2468843005952377
# NLOS 0.7683558873 0.0714285714 3.3333333333333335 2.561186290922619
# NLOS shuffling 0.8006766183 0.0 3.3333333333333335 2.6689220610119047
# NLOS shuffling 0.8044658954 0.0 3.3333333333333335 2.681552984775641
# csi mi1 0.9924153646 0.0 3.3333333333333335 3.308051215277778
# csi si1 0.9934570312 0.0 3.3333333333333335 3.3115234375
# csi mo1 1.0 1.0 3.3333333333333335 3.3333333333333335
# csi so1 0.9891927083 0.6666666667 3.3333333333333335 3.297309027777778

# segLen = 4
# mi1 1.0 1.0 2.5 2.5
# si1 0.9681640625 0.76 2.5 2.42041015625
# mo1 1.0 1.0 2.5 2.5
# so1-r 0.999296875 0.8 2.5 2.4982421875
# NLOS 0.8022460938 0.380952381 2.5 2.005615234375
# NLOS shuffling 0.9429268973 0.0 2.5 2.3573172433035716
# NLOS shuffling 0.9467197516 0.0 2.5 2.36679937900641
# csi mi1 1.0 1.0 2.5 2.5
# csi mi1 0.9999556108 0.9090909091 2.5 2.4998890269886362
# csi si1 0.9998779297 0.75 2.5 2.49969482421875
# csi mo1 1.0 1.0 2.5 2.5
# csi so1 0.9993326823 0.6666666667 2.5 2.4983317057291665

# segLen = 5
# mi1 1.0 1.0 0.6666666666666667 0.6666666666666667
# si1 1.0 1.0 2.0 2.0
# mo1 1.0 1.0 2.0 2.0
# so1 1.0 1.0 1.5 1.5
# NLOS 0.8243278952 0.4705882353 2.0 1.6486557904411765
# NLOS shuffling 0.9901941636 0.0 2.0 1.9803883272058822
# NLOS shuffling 0.9916614163 0.0 2.0 1.9833228326612904
# csi mi1 1.0 1.0 2.0 2.0
# csi si1 1.0 1.0 2.0 2.0
# csi mo1 1.0 1.0 2.0 2.0
# csi so1 1.0 1.0 2.0 2.0

# segLen = 6
# mi1 1.0 1.0 1.6666666666666667 1.6666666666666667
# si1 1.0 1.0 1.6666666666666667 1.6666666666666667
# mo1 1.0 1.0 1.6666666666666667 1.6666666666666667
# so1 0.9998535156 0.8888888889 1.6666666666666667 1.6664225260416667
# NLOS 0.8511439732 0.5714285714 1.6666666666666667 1.4185732886904763
# NLOS shuffling 0.9988769531 0.0 1.6666666666666667 1.664794921875
# NLOS shuffling 0.9988506611 0.1923076923 1.6666666666666667 1.6647511017628205
# csi mi1 1.0 1.0 1.6666666666666667 1.6666666666666667
# csi si1 1.0 1.0 1.6666666666666667 1.6666666666666667
# csi mo1 1.0 1.0 1.6666666666666667 1.6666666666666667
# csi so1 1.0 1.0 1.6666666666666667 1.6666666666666667

# segLen = 6
# so1 0.9998535156 0.8888888889 1.6666666666666667 1.6664225260416667
# NLOS 0.9987454928 0.0769230769 1.6666666666666667 1.6645758213141024

# segLen = 7
# so1 1.0 1.0 1.4285714285714286 1.4285714285714286
# NLOS shuffling 0.9999023437 0.8333333333 1.4285714285714286 1.4284319196428572
# NLOS shuffling 1.0 1.0 1.25 1.25

# data BMR BGR BGR-with-no-error
# keyLen = 1280
# mi1 1.0 1.0 2.04 2.04
# si1 0.9974961439 0.8620689655 2.039956896551724 2.0348491379310345
# mo1 1.0 1.0 2.04 2.04
# so1 0.8605445155 0.2857142857 2.0398214285714285 1.7553571428571428
# so1_r shuffling 1.0 1.0 2.04 2.04
# so1_r 0.9998978758 0.6666666667 2.04 2.0397916666666664

# data BMR BGR BGR-with-no-error
# keyLen = 1636
# mi1 1.0 1.0 2.0748166259168705 2.0748166259168705
# si1 0.9686356789 0.7391304348 2.0747741043903476 2.0097002232380143
# mo1 1.0000441924 0.5 2.0747249388753057 2.0748166259168705
# so1 0.9061940228 0.4 2.074718826405868 1.8800977995110026
# so1_r 0.954307094 0.5 2.0748166259168705 1.9800122249388754
# so1_r shuffling 1.0 1.0 2.0748166259168705 2.0748166259168705

# data BMR BGR BGR-with-no-error
# keyLen = 1892
# mi1 1.0000126341 0.75 2.091728329809725 2.091754756871036
# si1 0.9732608295 0.8 2.091696617336152 2.0357663847780127
# mo1 1.0 1.0 2.091754756871036 2.091754756871036
# so1 0.7623307055 0.5 2.091754756871036 1.5946088794926003
# so1_r 1.0 1.0 2.091754756871036 2.091754756871036
# so1_r shuffling 1.0 1.0 2.091754756871036 2.091754756871036

# data BMR BGR BGR-with-no-error
# keyLen = 2048
# mi1 1.0 1.0 2.2 2.2
# si1 0.9812040958 0.6956521739 2.0747900499627936 2.035792494950569
# mo1 0.9998964252 0.6666666667 2.2 2.1997721354166666
# so1 0.9995783026 0.5 2.2 2.199072265625
# so1 0.9994451349 0.5 2.2 2.198779296875
# so1_r 1.0 1.0 2.2 2.2
# so1_r shuffling 1.0 1.0 2.2 2.2
# so1_r segLen = 4 0.9920543324 0.5 2.75 2.7281494140625

# segLen = 4
# mi1 no sorting 1.0 1.0 2.5 2.5
# si1 no sorting 0.8060465495 0.375 2.5 2.0151163736979165
# mo1 no sorting 1.0 1.0 2.5 2.5
# so1 no sorting 0.9857226562 0.2 2.5 2.464306640625
# mi1 no perturbation no sorting 0.7253173828 0.0 2.5 1.81329345703125
# si1 no perturbation no sorting 0.6102579753 0.0 2.5 1.5256449381510417
# mo1 no perturbation no sorting 0.6185546875 0.0 2.5 1.54638671875
# so1 no perturbation no sorting 0.6678710937 0.0 2.5 1.669677734375
# mi1 no perturbation 0.7419433594 0.0 2.5 1.8548583984375

# si1 no perturbation 0.6137003581 0.0 2.5 1.5342508951822917
# si1 no perturbation 0.8920728601 0.0 2.5 2.2301821501358696
# si1 no perturbation +U(0,0.5) 0.8097911005 0.0 2.5 2.024477751358696

# mo1 no perturbation 0.6438476562 0.0 2.5 1.609619140625
# so1 no perturbation 0.6687109375 0.0 2.5 1.67177734375

# segLen = 5
# mi1 no sorting 1.0 1.0 2.0 2.0
# si1 no sorting 0.8498869243 0.4736842105 2.0 1.6997738486842107
# mo1 no sorting 1.0 1.0 2.0 2.0
# so1 no sorting 0.9927246094 0.0 2.0 1.98544921875
# mi1 no perturbation no sorting 0.7609049479 0.0 2.0 1.5218098958333335
# si1 no perturbation no sorting 0.6935601128 0.0 2.0 1.3871202256944444
# mo1 no perturbation no sorting 0.5977539063 0.0 2.0 1.1955078125
# so1 no perturbation no sorting 0.7032714844 0.0 2.0 1.40654296875
# mi1 no perturbation 0.7958007812 0.0 2.0 1.5916015625

# si1 no perturbation 0.7477756076 0.0 2.0 1.495551215277778
# si1 no perturbation 0.9292209201 0.0 2.0 1.8584418402777778
# si1 no perturbation +U(0,0.5) 0.8436306424 0.0 2.0 1.6872612847222221

# mo1 no perturbation 0.6625976562 0.0 2.0 1.3251953125
# so1 no perturbation 0.7209960937 0.0 2.0 1.4419921875

# segLen = 6
# mi1 no sorting 1.0 1.0 1.6666666666666667 1.6666666666666667
# si1 no sorting 0.89296875 0.5 1.6666666666666667 1.48828125
# mo1 no sorting 1.0 1.0 1.6666666666666667 1.6666666666666667
# so1 no sorting 1.0 1.0 1.6666666666666667 1.6666666666666667
# mi1 no perturbation no sorting 0.7798828125 0.0 1.6666666666666667 1.2998046875
# si1 no perturbation no sorting 0.7087565104 0.0 1.6666666666666667 1.1812608506944444
# mo1 no perturbation no sorting 0.6196289062 0.0 1.6666666666666667 1.03271484375
# so1 no perturbation no sorting 0.7317382812 0.0 1.6666666666666667 1.2195638020833333
# mi1 no perturbation 0.7868164063 0.0 1.6666666666666667 1.3113606770833333

# si1 no perturbation 0.7232552083 0.0 1.6666666666666667 1.2054253472222223
# si1 no perturbation 0.8699414063 0.0 1.6666666666666667 1.44990234375
# si1 no perturbation +U(0,0.5) 0.8705013021 0.0 1.6666666666666667 1.450835503472222

# mo1 no perturbation 0.6747070312 0.0 1.6666666666666667 1.12451171875
# so1 no perturbation 0.7338541667 0.0 1.6666666666666667 1.223090277777778

# segLen = 7
# mi1 no sorting 1.0 1.0 1.4285714285714286 1.4285714285714286
# si1 no sorting 0.9072265625 0.5714285714 1.4285714285714286 1.2960379464285714
# mo1 no sorting 0.626171875 0.0 1.4285714285714286 0.89453125
# so1 no sorting 0.7520833333 0.0 1.4285714285714286 1.0744047619047619
# mi1 no perturbation no sorting 0.7743652344 0.0 1.4285714285714286 1.1062360491071428
# si1 no perturbation no sorting 0.7203500601 0.0 1.4285714285714286 1.0290715144230769
# mo1 no perturbation no sorting 0.626171875 0.0 1.4285714285714286 0.89453125
# so1 no perturbation no sorting 0.7520833333 0.0 1.4285714285714286 1.0744047619047619
# mi1 no perturbation 0.7861328125 0.0 1.4285714285714286 1.123046875

# si1 no perturbation 0.7326397236 0.0 1.4285714285714286 1.046628176510989
# si1 no perturbation 0.9635967548 0.0 1.4285714285714286 1.3765667925824177
# si1 no perturbation +U(0,0.5) 0.8891225962 0.0 1.4285714285714286 1.2701751373626373

# mo1 no perturbation 0.6768798828 0.0 1.4285714285714286 0.9669712611607143
# so1 no perturbation 0.7558789062 0.0 1.4285714285714286 1.0798270089285713

for f in fileName:
    print(f)
    rawData = loadmat(f)

    # CSIa1Orig = np.tile(rawData['A'][:, 0], 5)
    # CSIb1Orig = np.tile(rawData['A'][:, 1], 5)
    CSIa1Orig = rawData['A'][:, 0]
    CSIb1Orig = rawData['A'][:, 1]
    dataLen = len(CSIa1Orig)
    print("dataLen", dataLen)

    segLen = 4
    keyLen = 1024 * segLen
    rec = True
    tell = True

    print("segLen", segLen)
    print("keyLen", keyLen / segLen)

    originSum = 0
    correctSum = 0
    randomSum1 = 0
    randomSum2 = 0
    noiseSum1 = 0
    noiseSum2 = 0
    noiseSum3 = 0
    noiseSum4 = 0

    originDecSum = 0
    correctDecSum = 0
    randomDecSum1 = 0
    randomDecSum2 = 0
    noiseDecSum1 = 0
    noiseDecSum2 = 0
    noiseDecSum3 = 0
    noiseDecSum4 = 0

    originWholeSum = 0
    correctWholeSum = 0
    randomWholeSum1 = 0
    randomWholeSum2 = 0
    noiseWholeSum1 = 0
    noiseWholeSum2 = 0
    noiseWholeSum3 = 0
    noiseWholeSum4 = 0

    times = 0
    overhead = 0

    # no perturbation
    withoutSort = False
    addNoise = "mul"
    codings = ""
    if withoutSort:
        if addNoise == "mul":
            print("no sorting")
    if withoutSort:
        if addNoise == "":
            print("no sorting and no perturbation")
    if withoutSort is False:
        if addNoise == "":
            print("no perturbation")
        if  addNoise == "mul":
            print("normal")

    # static indoor
    # if f == "../data/data_static_indoor_1.mat":
    #     dataLen = int(dataLen / 5)
    #     keyLen = int(keyLen / 5)
    #     print(dataLen, keyLen)
    for staInd in range(0, int(dataLen / 5.5), int(keyLen / 5)):
        # for staInd in range(0, dataLen, keyLen):
        start = time.time()
        endInd = staInd + keyLen
        print("range:", staInd, endInd)
        if endInd >= len(CSIa1Orig):
            break

        times += 1

        # np.random.seed(1)
        # CSIa1Orig = np.tile(rawData['A'][:, 0], 5)
        # CSIb1Orig = np.tile(rawData['A'][:, 1], 5)
        CSIa1Orig = rawData['A'][:, 0]
        CSIb1Orig = rawData['A'][:, 1]

        # imitation attack
        CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=len(CSIa1Orig))
        # stalking attack
        CSIe2Orig = loadmat("../skyglow/Scenario2-Office-LoS-eve_NLoS/data_eave_LOS_EVE_NLOS.mat")['A'][:, 0]

        tmpNoise1 = []
        tmpNoise2 = []
        tmpNoise3 = []
        tmpNoise4 = []

        # noiseOrig = np.random.normal(np.mean(CSIa1Orig), np.std(CSIa1Orig), size=len(CSIa1Orig))
        # noiseOrig = np.random.normal(0, np.std(CSIa1Orig), size=len(CSIa1Orig))
        # np.random.seed(int(seeds[times - 1][0]))
        seed = np.random.randint(100000)
        np.random.seed(seed)

        if addNoise == "add":
            CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
            CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')

            # noiseOrig = np.random.normal(0, np.std(CSIa1Orig) * 4, size=len(CSIa1Orig))
            # CSIa1Orig = CSIa1Orig + noiseOrig
            # CSIb1Orig = CSIb1Orig + noiseOrig
            # CSIe1Orig = CSIe1Orig + noiseOrig
            # CSIe2Orig = CSIe2Orig + noiseOrig
            # CSIn1Orig = noiseOrig

            tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
            tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
            tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
            tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]

            noiseOrig = np.random.uniform(0, 0.5, size=keyLen)
            tmpCSIa1 = (tmpCSIa1 - np.mean(tmpCSIa1)) + noiseOrig
            tmpCSIb1 = (tmpCSIb1 - np.mean(tmpCSIb1)) + noiseOrig
            tmpCSIe1 = (tmpCSIe1 - np.mean(tmpCSIe1)) + noiseOrig
            tmpCSIe2 = (tmpCSIe2 - np.mean(tmpCSIe2)) + noiseOrig
            tmpNoise = noiseOrig
        elif addNoise == "mul":
            # 静态数据需要置换
            # 固定随机置换的种子
            # np.random.seed(0)
            # combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig))
            # np.random.shuffle(combineCSIx1Orig)
            # CSIa1Orig, CSIb1Orig = zip(*combineCSIx1Orig)

            CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
            CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')
            CSIe1Orig = smooth(np.array(CSIe1Orig), window_len=30, window='flat')
            CSIe2Orig = smooth(np.array(CSIe2Orig), window_len=30, window='flat')

            tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
            tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
            tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
            tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]

            randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
            tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
            tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
            tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
            tmpCSIe2 = tmpCSIe2 - np.mean(tmpCSIe2)
            tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
            tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)
            tmpCSIe1 = np.matmul(tmpCSIe1, randomMatrix)
            tmpCSIe2 = np.matmul(tmpCSIe2, randomMatrix)
            # tmpCSIa1 = np.matmul(tmpCSIa1, np.outer(tmpCSIa1, tmpCSIa1))
            # tmpCSIb1 = np.matmul(tmpCSIb1, np.outer(tmpCSIb1, tmpCSIb1))
            # tmpCSIe1 = np.matmul(tmpCSIe1, np.outer(tmpCSIe1, tmpCSIe1))
            # tmpCSIe2 = np.matmul(tmpCSIe2, np.outer(tmpCSIe2, tmpCSIe2))
            # inference attack
            tmpNoise1 = randomMatrix.mean(axis=0)  # 按列求均值
            tmpNoise2 = randomMatrix.mean(axis=1)  # 按行求均值
            tmpNoise3 = np.matmul(np.ones(keyLen), randomMatrix)
            tmpNoise4 = np.random.normal(loc=np.mean(tmpCSIa1), scale=np.std(tmpCSIa1, ddof=1), size=len(tmpCSIa1))
        else:
            # np.random.seed(0)
            # combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig))
            # np.random.shuffle(combineCSIx1Orig)
            # CSIa1Orig, CSIb1Orig = zip(*combineCSIx1Orig)

            CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
            CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')

            tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
            tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
            tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
            tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]

            tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
            tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
            tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
            tmpCSIe2 = tmpCSIe2 - np.mean(tmpCSIe2)
            tmpNoise = np.random.normal(0, np.std(CSIa1Orig), size=keyLen)

        # tmpCSIa1 = np.array(integral_sq_derivative_increment(tmpCSIa1, tmpNoise)) * tmpCSIa1
        # tmpCSIb1 = np.array(integral_sq_derivative_increment(tmpCSIb1, tmpNoise)) * tmpCSIb1
        # tmpCSIe1 = np.array(integral_sq_derivative_increment(tmpCSIe1, tmpNoise)) * tmpCSIe1
        # print("correlation a-e1", pearsonr(tmpCSIa1, tmpCSIe1)[0])
        # print("correlation a-e2", pearsonr(tmpCSIa1, tmpCSIe2)[0])
        # print("correlation a-n", pearsonr(tmpCSIa1, tmpNoise)[0])
        # tmpNoise = np.array(integral_sq_derivative_increment(np.ones(keyLen), tmpNoise)) * np.ones(keyLen)
        # print("correlation a-n'", pearsonr(tmpCSIa1, tmpNoise)[0])

        # tmpCSIe1 = np.random.normal(loc=np.mean(tmpCSIe1), scale=np.std(tmpCSIe1, ddof=1), size=len(tmpCSIe1))

        # noise = np.random.uniform(0, 1, keyLen)
        # tmpCSIa1 = np.power(np.abs(tmpCSIa1), noise)
        # tmpCSIb1 = np.power(np.abs(tmpCSIb1), noise)
        # tmpCSIe1 = np.power(np.abs(tmpCSIe1), noise)
        # tmpNoise = np.power(np.abs(tmpNoise), noise)

        # box-muller
        # tmpCSIa1 = boxcox(np.abs(tmpCSIa1))[0]
        # tmpCSIb1 = boxcox(np.abs(tmpCSIb1))[0]
        # tmpCSIe1 = boxcox(np.abs(tmpCSIe1))[0]
        # tmpNoise = boxcox(np.abs(tmpNoise))[0]
        # tmpCSIa1 = normal2uniform(tmpCSIa1)
        # tmpCSIb1 = normal2uniform(tmpCSIb1)
        # tmpCSIe1 = normal2uniform(tmpCSIe1)
        # tmpNoise = normal2uniform(tmpNoise)

        # plt.figure()
        # plt.plot(np.sort(tmpCSIa1))
        # plt.show()

        # 最后各自的密钥
        a_list = []
        b_list = []
        e1_list = []
        e2_list = []
        n1_list = []
        n2_list = []
        n3_list = []
        n4_list = []

        # without sorting
        if withoutSort:
            tmpCSIa1Ind = np.array(tmpCSIa1)
            tmpCSIb1Ind = np.array(tmpCSIb1)
            tmpCSIe1Ind = np.array(tmpCSIe1)
            tmpCSIe2Ind = np.array(tmpCSIe2)
            tmpCSIn1Ind = np.array(tmpNoise1)
            tmpCSIn2Ind = np.array(tmpNoise2)
            tmpCSIn3Ind = np.array(tmpNoise3)
            tmpCSIn4Ind = np.array(tmpNoise4)
        else:
            tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
            tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()
            tmpCSIe1Ind = np.array(tmpCSIe1).argsort().argsort()
            tmpCSIe2Ind = np.array(tmpCSIe2).argsort().argsort()
            tmpCSIn1Ind = np.array(tmpNoise1).argsort().argsort()
            tmpCSIn2Ind = np.array(tmpNoise2).argsort().argsort()
            tmpCSIn3Ind = np.array(tmpNoise3).argsort().argsort()
            tmpCSIn4Ind = np.array(tmpNoise4).argsort().argsort()

        minEpiIndClosenessLsb = np.zeros(int(keyLen / segLen), dtype=int)
        minEpiIndClosenessLse1 = np.zeros(int(keyLen / segLen), dtype=int)
        minEpiIndClosenessLse2 = np.zeros(int(keyLen / segLen), dtype=int)
        minEpiIndClosenessLsn = np.zeros(int(keyLen / segLen), dtype=int)
        minEpiIndClosenessLsn1 = np.zeros(int(keyLen / segLen), dtype=int)
        minEpiIndClosenessLsn2 = np.zeros(int(keyLen / segLen), dtype=int)
        minEpiIndClosenessLsn3 = np.zeros(int(keyLen / segLen), dtype=int)
        minEpiIndClosenessLsn4 = np.zeros(int(keyLen / segLen), dtype=int)

        tmpCSIa1IndReshape = np.array(tmpCSIa1Ind).reshape(int(keyLen / segLen), segLen)
        permutation = list(range(int(keyLen / segLen)))
        combineMetric = list(zip(tmpCSIa1IndReshape, permutation))
        np.random.seed(staInd)
        np.random.shuffle(combineMetric)
        tmpCSIa1IndReshape, permutation = zip(*combineMetric)
        tmpCSIa1Ind = np.hstack((tmpCSIa1IndReshape))

        for i in range(int(keyLen / segLen)):
            epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]

            epiIndClosenessLsb = np.zeros(int(keyLen / segLen))
            epiIndClosenessLse1 = np.zeros(int(keyLen / segLen))
            epiIndClosenessLse2 = np.zeros(int(keyLen / segLen))
            epiIndClosenessLsn1 = np.zeros(int(keyLen / segLen))
            epiIndClosenessLsn2 = np.zeros(int(keyLen / segLen))
            epiIndClosenessLsn3 = np.zeros(int(keyLen / segLen))
            epiIndClosenessLsn4 = np.zeros(int(keyLen / segLen))

            for j in range(int(keyLen / segLen)):
                epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
                epiInde1 = tmpCSIe1Ind[j * segLen: (j + 1) * segLen]
                epiInde2 = tmpCSIe2Ind[j * segLen: (j + 1) * segLen]
                epiIndn1 = tmpCSIn1Ind[j * segLen: (j + 1) * segLen]
                epiIndn2 = tmpCSIn2Ind[j * segLen: (j + 1) * segLen]
                epiIndn3 = tmpCSIn3Ind[j * segLen: (j + 1) * segLen]
                epiIndn4 = tmpCSIn4Ind[j * segLen: (j + 1) * segLen]

                epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))
                epiIndClosenessLse1[j] = sum(abs(epiInde1 - np.array(epiInda1)))
                epiIndClosenessLse2[j] = sum(abs(epiInde2 - np.array(epiInda1)))
                epiIndClosenessLsn1[j] = sum(abs(epiIndn1 - np.array(epiInda1)))
                epiIndClosenessLsn2[j] = sum(abs(epiIndn2 - np.array(epiInda1)))
                epiIndClosenessLsn3[j] = sum(abs(epiIndn3 - np.array(epiInda1)))
                epiIndClosenessLsn4[j] = sum(abs(epiIndn4 - np.array(epiInda1)))
                # epiIndClosenessLsb[j] = abs(epiIndb1[0] - epiInda1[0]) + abs(epiIndb1[2] - epiInda1[2]) + abs(epiIndb1[4] - epiInda1[4]) + abs(epiIndb1[6] - epiInda1[6])
                # epiIndClosenessLse1[j] = abs(epiInde1[0] - epiInda1[0]) + abs(epiInde1[2] - epiInda1[2]) + abs(epiInde1[4] - epiInda1[4]) + abs(epiInde1[6] - epiInda1[6])
                # epiIndClosenessLse2[j] = abs(epiInde2[0] - epiInda1[0]) + abs(epiInde2[2] - epiInda1[2]) + abs(epiInde2[4] - epiInda1[4]) + abs(epiInde2[6] - epiInda1[6])
                # epiIndClosenessLsn[j] = abs(epiIndn1[0] - epiInda1[0]) + abs(epiIndn1[2] - epiInda1[2]) + abs(epiIndn1[4] - epiInda1[4]) + abs(epiIndn1[6] - epiInda1[6])
                # epiIndClosenessLsb[j] = abs(epiIndb1[1] - epiInda1[1]) + abs(epiIndb1[3] - epiInda1[3]) + abs(epiIndb1[5] - epiInda1[5]) + abs(epiIndb1[7] - epiInda1[7])
                # epiIndClosenessLse1[j] = abs(epiInde1[1] - epiInda1[1]) + abs(epiInde1[3] - epiInda1[3]) + abs(epiInde1[5] - epiInda1[5]) + abs(epiInde1[7] - epiInda1[7])
                # epiIndClosenessLse2[j] = abs(epiInde2[1] - epiInda1[1]) + abs(epiInde2[3] - epiInda1[3]) + abs(epiInde2[5] - epiInda1[5]) + abs(epiInde2[7] - epiInda1[7])
                # epiIndClosenessLsn[j] = abs(epiIndn1[1] - epiInda1[1]) + abs(epiIndn1[3] - epiInda1[3]) + abs(epiIndn1[5] - epiInda1[5]) + abs(epiIndn1[7] - epiInda1[7])

            minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)
            minEpiIndClosenessLse1[i] = np.argmin(epiIndClosenessLse1)
            minEpiIndClosenessLse2[i] = np.argmin(epiIndClosenessLse2)
            minEpiIndClosenessLsn1[i] = np.argmin(epiIndClosenessLsn1)
            minEpiIndClosenessLsn2[i] = np.argmin(epiIndClosenessLsn2)
            minEpiIndClosenessLsn3[i] = np.argmin(epiIndClosenessLsn3)
            minEpiIndClosenessLsn4[i] = np.argmin(epiIndClosenessLsn4)

        # a_list_number = list(range(int(keyLen / segLen)))
        a_list_number = list(permutation)
        b_list_number = list(minEpiIndClosenessLsb)
        e1_list_number = list(minEpiIndClosenessLse1)
        e2_list_number = list(minEpiIndClosenessLse2)
        n1_list_number = list(minEpiIndClosenessLsn1)
        n2_list_number = list(minEpiIndClosenessLsn2)
        n3_list_number = list(minEpiIndClosenessLsn3)
        n4_list_number = list(minEpiIndClosenessLsn4)

        # 转成二进制，0填充成0000
        for i in range(len(a_list_number)):
            number = bin(a_list_number[i])[2:].zfill(int(np.log2(len(a_list_number))))
            a_list += number
        for i in range(len(b_list_number)):
            number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
            b_list += number
        for i in range(len(e1_list_number)):
            number = bin(e1_list_number[i])[2:].zfill(int(np.log2(len(e1_list_number))))
            e1_list += number
        for i in range(len(e2_list_number)):
            number = bin(e2_list_number[i])[2:].zfill(int(np.log2(len(e2_list_number))))
            e2_list += number
        for i in range(len(n1_list_number)):
            number = bin(n1_list_number[i])[2:].zfill(int(np.log2(len(n1_list_number))))
            n1_list += number
        for i in range(len(n2_list_number)):
            number = bin(n2_list_number[i])[2:].zfill(int(np.log2(len(n2_list_number))))
            n2_list += number
        for i in range(len(n3_list_number)):
            number = bin(n3_list_number[i])[2:].zfill(int(np.log2(len(n3_list_number))))
            n3_list += number
        for i in range(len(n4_list_number)):
            number = bin(n4_list_number[i])[2:].zfill(int(np.log2(len(n4_list_number))))
            n4_list += number

        # 对齐密钥，随机补全
        for i in range(len(a_list) - len(e1_list)):
            e1_list += str(np.random.randint(0, 2))
        for i in range(len(a_list) - len(e2_list)):
            e2_list += str(np.random.randint(0, 2))
        for i in range(len(a_list) - len(n1_list)):
            n1_list += str(np.random.randint(0, 2))
        for i in range(len(a_list) - len(n2_list)):
            n2_list += str(np.random.randint(0, 2))
        for i in range(len(a_list) - len(n3_list)):
            n3_list += str(np.random.randint(0, 2))
        for i in range(len(a_list) - len(n4_list)):
            n4_list += str(np.random.randint(0, 2))

            # print("keys of a:", len(a_list), a_list)
            # print("keys of a:", len(a_list_number), a_list_number)
            # print("keys of b:", len(b_list), b_list)
            # print("keys of b:", len(b_list_number), b_list_number)
            # print("keys of e:", len(e_list), e_list)
            # print("keys of e1:", len(e1_list_number), e1_list_number)
            # print("keys of e:", len(e_list), e_list)
            # print("keys of e2:", len(e2_list_number), e2_list_number)
            # print("keys of n1:", len(n1_list), n1_list)
            # print("keys of n1:", len(n1_list_number), n1_list_number)
            # print("keys of n2:", len(n2_list), n2_list)
            # print("keys of n2:", len(n2_list_number), n2_list_number)
            # print("keys of n3:", len(n3_list), n3_list)
            # print("keys of n3:", len(n3_list_number), n3_list_number)
            # print("keys of n4:", len(n4_list), n4_list)
            # print("keys of n4:", len(n4_list_number), n4_list_number)

        sum1 = min(len(a_list), len(b_list))
        sum2 = 0
        sum31 = 0
        sum32 = 0
        sum41 = 0
        sum42 = 0
        sum43 = 0
        sum44 = 0
        for i in range(0, sum1):
            sum2 += (a_list[i] == b_list[i])
        for i in range(min(len(a_list), len(e1_list))):
            sum31 += (a_list[i] == e1_list[i])
        for i in range(min(len(a_list), len(e2_list))):
            sum32 += (a_list[i] == e2_list[i])
        for i in range(min(len(a_list), len(n1_list))):
            sum41 += (a_list[i] == n1_list[i])
        for i in range(min(len(a_list), len(n2_list))):
            sum42 += (a_list[i] == n2_list[i])
        for i in range(min(len(a_list), len(n3_list))):
            sum43 += (a_list[i] == n3_list[i])
        for i in range(min(len(a_list), len(n4_list))):
            sum44 += (a_list[i] == n4_list[i])

        end = time.time()
        overhead += end - start
        print("time:", end - start)

        # 自适应纠错
        if sum1 != sum2 and rec:
            if tell:
                # a告诉b哪些位置出错，b对其纠错
                for i in range(len(a_list_number)):
                    if a_list_number[i] != b_list_number[i]:
                        epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]

                        epiIndClosenessLsb = np.zeros(int(keyLen / segLen))

                        for j in range(int(keyLen / segLen)):
                            epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
                            epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))

                        min_b = np.argmin(epiIndClosenessLsb)
                        epiIndClosenessLsb[min_b] = keyLen * segLen
                        b_list_number[i] = np.argmin(epiIndClosenessLsb)

                        b_list = []

                        for i in range(len(b_list_number)):
                            number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
                            b_list += number

                        # print("keys of b:", len(b_list_number), b_list_number)

                        sum2 = 0
                        for i in range(0, min(len(a_list), len(b_list))):
                            sum2 += (a_list[i] == b_list[i])
            else:
                # 正式纠错
                trueError = []
                for i in range(len(a_list_number)):
                    if a_list_number[i] != b_list_number[i]:
                        trueError.append(i)
                # print("true error", trueError)
                # print("a-b", sum2, sum2 / sum1)
                reconciliation = b_list_number.copy()
                reconciliation.sort()

                repeatInd = []
                # 检查两个候选
                closeness = []
                for i in range(len(reconciliation) - 1):
                    # 相等的索引就是密钥出错的地方
                    if reconciliation[i] == reconciliation[i + 1]:
                        repeatInd.append(reconciliation[i])
                repeatNumber = []
                for i in range(len(repeatInd)):
                    tmp = []
                    for j in range(len(b_list_number)):
                        if repeatInd[i] == b_list_number[j]:
                            tmp.append(j)
                    repeatNumber.append(tmp)
                for i in range(len(repeatNumber)):
                    tmp = []
                    for j in range(len(repeatNumber[i])):
                        epiInda1 = tmpCSIa1Ind[repeatNumber[i][j] * segLen:(repeatNumber[i][j] + 1) * segLen]

                        epiIndClosenessLsb = np.zeros(int(keyLen / segLen))

                        for k in range(int(keyLen / segLen)):
                            epiIndb1 = tmpCSIb1Ind[k * segLen: (k + 1) * segLen]

                            epiIndClosenessLsb[k] = sum(abs(epiIndb1 - np.array(epiInda1)))

                        min_b = np.argmin(epiIndClosenessLsb)
                        tmp.append(epiIndClosenessLsb[min_b])

                    closeness.append(tmp)

                errorInd = []

                for i in range(len(closeness)):
                    for j in range(len(closeness[i]) - 1):
                        if closeness[i][j] < closeness[i][j + 1]:
                            errorInd.append(repeatNumber[i][j + 1])
                        else:
                            errorInd.append(repeatNumber[i][j])
                # print(errorInd)
                b_list_number1 = b_list_number.copy()
                for i in errorInd:
                    epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]
                    epiIndClosenessLsb = np.zeros(int(keyLen / segLen))

                    for j in range(int(keyLen / segLen)):
                        epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
                        epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))

                    min_b = np.argmin(epiIndClosenessLsb)
                    while min_b in b_list_number:
                        epiIndClosenessLsb[min_b] = keyLen * segLen
                        min_b = np.argmin(epiIndClosenessLsb)
                    b_list_number1[i] = min_b

                b_list = []

                for i in range(len(b_list_number1)):
                    number = bin(b_list_number1[i])[2:].zfill(int(np.log2(len(b_list_number1))))
                    b_list += number

                # print("keys of b:", len(b_list_number1), b_list_number1)

                sum2 = 0
                for i in range(0, min(len(a_list), len(b_list))):
                    sum2 += (a_list[i] == b_list[i])

                if sum1 == sum2:
                    b_list_number = b_list_number1

                # 二次纠错
                if sum1 != sum2:
                    for r in range(len(repeatNumber)):
                        tmp = list(set(repeatNumber[r]) - set(errorInd))
                        errorInd = tmp
                        # print(errorInd)
                        b_list_number2 = b_list_number.copy()
                        for i in errorInd:
                            epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]
                            epiIndClosenessLsb = np.zeros(int(keyLen / segLen))

                            for j in range(int(keyLen / segLen)):
                                epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
                                epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))

                            min_b = np.argmin(epiIndClosenessLsb)
                            while min_b in b_list_number:
                                epiIndClosenessLsb[min_b] = keyLen * segLen
                                min_b = np.argmin(epiIndClosenessLsb)
                            b_list_number2[i] = min_b

                        b_list = []

                        for i in range(len(b_list_number2)):
                            number = bin(b_list_number2[i])[2:].zfill(int(np.log2(len(b_list_number2))))
                            b_list += number

                        # print("keys of b:", len(b_list_number2), b_list_number2)

                        sum2 = 0
                        for i in range(0, min(len(a_list), len(b_list))):
                            sum2 += (a_list[i] == b_list[i])

                        if sum1 == sum2:
                            b_list_number = b_list_number2
                # 正式纠错 end

        # maxSum2 = max(sum2, maxSum2)
        # print("sum2", maxSum2, sum1)

        print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
        print("a-e1", sum31, sum31 / sum1)
        print("a-e2", sum32, sum32 / sum1)
        print("a-n1", sum41, sum41 / sum1)
        print("a-n2", sum42, sum42 / sum1)
        print("a-n3", sum43, sum43 / sum1)
        print("a-n4", sum44, sum44 / sum1)
        originSum += sum1
        correctSum += sum2
        randomSum1 += sum31
        randomSum2 += sum32
        noiseSum1 += sum41
        noiseSum2 += sum42
        noiseSum3 += sum43
        noiseSum4 += sum44

        # decSum1 = min(len(a_list_number), len(b_list_number))
        # decSum2 = 0
        # decSum31 = 0
        # decSum32 = 0
        # decSum41 = 0
        # decSum42 = 0
        # decSum43 = 0
        # decSum44 = 0
        # for i in range(0, decSum1):
        #     decSum2 += (a_list_number[i] == b_list_number[i])
        # for i in range(min(len(a_list_number), len(e1_list_number))):
        #     decSum31 += (a_list_number[i] == e1_list_number[i])
        # for i in range(min(len(a_list_number), len(e2_list_number))):
        #     decSum32 += (a_list_number[i] == e2_list_number[i])
        # for i in range(min(len(a_list_number), len(n1_list_number))):
        #     decSum41 += (a_list_number[i] == n1_list_number[i])
        # for i in range(min(len(a_list_number), len(n2_list_number))):
        #     decSum42 += (a_list_number[i] == n2_list_number[i])
        # for i in range(min(len(a_list_number), len(n3_list_number))):
        #     decSum43 += (a_list_number[i] == n3_list_number[i])
        # for i in range(min(len(a_list_number), len(n4_list_number))):
        #     decSum44 += (a_list_number[i] == n4_list_number[i])
        # if decSum1 == 0:
        #     continue
        # if decSum2 == decSum1:
        #     print("\033[0;32;40ma-b dec", decSum2, decSum2 / decSum1, "\033[0m")
        # else:
        #     print("\033[0;31;40ma-b dec", "bad", decSum2, decSum2 / decSum1, "\033[0m")
        # print("a-e1", decSum31, decSum31 / decSum1)
        # print("a-e2", decSum32, decSum32 / decSum1)
        # print("a-n1", decSum41, decSum41 / decSum1)
        # print("a-n2", decSum42, decSum42 / decSum1)
        # print("a-n3", decSum43, decSum43 / decSum1)
        # print("a-n4", decSum44, decSum44 / decSum1)
        # print("----------------------")
        # originDecSum += decSum1
        # correctDecSum += decSum2
        # randomDecSum1 += decSum31
        # randomDecSum2 += decSum32
        # noiseDecSum1 += decSum41
        # noiseDecSum2 += decSum42
        # noiseDecSum3 += decSum43
        # noiseDecSum4 += decSum44

        originWholeSum += 1
        correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
        randomWholeSum1 = randomWholeSum1 + 1 if sum31 == sum1 else randomWholeSum1
        randomWholeSum2 = randomWholeSum2 + 1 if sum32 == sum1 else randomWholeSum2
        noiseWholeSum1 = noiseWholeSum1 + 1 if sum41 == sum1 else noiseWholeSum1
        noiseWholeSum2 = noiseWholeSum2 + 1 if sum42 == sum1 else noiseWholeSum2
        noiseWholeSum3 = noiseWholeSum3 + 1 if sum43 == sum1 else noiseWholeSum3
        noiseWholeSum4 = noiseWholeSum4 + 1 if sum44 == sum1 else noiseWholeSum4

    # coding = ""
    # for i in range(len(a_list)):
    #     coding += a_list[i]
    # codings += coding + "\n"
    #
    # with open('./key/' + fileName + ".txt", 'a', ) as f:
    #     f.write(codings)

    print("\033[0;34;40ma-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10), "\033[0m")
    print("a-e1 all", randomSum1, "/", originSum, "=", round(randomSum1 / originSum, 10))
    print("a-e2 all", randomSum2, "/", originSum, "=", round(randomSum2 / originSum, 10))
    print("a-n1 all", noiseSum1, "/", originSum, "=", round(noiseSum1 / originSum, 10))
    print("a-n2 all", noiseSum2, "/", originSum, "=", round(noiseSum2 / originSum, 10))
    print("a-n3 all", noiseSum3, "/", originSum, "=", round(noiseSum3 / originSum, 10))
    print("a-n4 all", noiseSum4, "/", originSum, "=", round(noiseSum4 / originSum, 10))
    # print("a-b dec", correctDecSum, "/", originDecSum, "=", round(correctDecSum / originDecSum, 10))
    # print("a-e1 dec", randomDecSum1, "/", originDecSum, "=", round(randomDecSum1 / originDecSum, 10))
    # print("a-e2 dec", randomDecSum2, "/", originDecSum, "=", round(randomDecSum2 / originDecSum, 10))
    # print("a-n dec", noiseDecSum, "/", originDecSum, "=", round(noiseDecSum / originDecSum, 10))
    print("\033[0;34;40ma-b whole match", correctWholeSum, "/", originWholeSum, "=",
          round(correctWholeSum / originWholeSum, 10), "\033[0m")
    print("a-e1 whole match", randomWholeSum1, "/", originWholeSum, "=", round(randomWholeSum1 / originWholeSum, 10))
    print("a-e2 whole match", randomWholeSum2, "/", originWholeSum, "=", round(randomWholeSum2 / originWholeSum, 10))
    print("a-n1 whole match", noiseWholeSum1, "/", originWholeSum, "=", round(noiseWholeSum1 / originWholeSum, 10))
    print("a-n2 whole match", noiseWholeSum2, "/", originWholeSum, "=", round(noiseWholeSum2 / originWholeSum, 10))
    print("a-n3 whole match", noiseWholeSum3, "/", originWholeSum, "=", round(noiseWholeSum3 / originWholeSum, 10))
    print("a-n4 whole match", noiseWholeSum4, "/", originWholeSum, "=", round(noiseWholeSum4 / originWholeSum, 10))
    print("times", times)
    print(overhead / originWholeSum)

    print(round(correctSum / originSum, 10), round(correctWholeSum / originWholeSum, 10), originSum / times / keyLen,
          correctSum / times / keyLen)
messagebox.showinfo("提示", "测试结束")
