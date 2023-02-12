import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
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


# 添加数组头元素以构成封闭的多边形
def makePolygon(list):
    listPolygon = []
    for i in range(len(list)):
        listTmp = []
        for j in range(len(list[i])):
            listTmp.append(list[i][j])
        listTmp.append(list[i][0])
        listPolygon.append(listTmp)
    return listPolygon


# 数组第二维的所有内容求和
def sumEachDim(list, index):
    res = 0
    for i in range(len(list[index])):
        res += (list[index][i][0] + list[index][i][1])
        # res += (list[index][i][0] * list[index][i][1])
    return round(res, 8)


def projection(A, B, C, D, x):
    abc = A * A + B * B + C * C
    x0 = [((B * B + C * C) * x[0] - A * (B * x[1] + C * x[2] + D)) / abc,
          ((A * A + C * C) * x[1] - B * (A * x[0] + C * x[2] + D)) / abc,
          ((A * A + B * B) * x[2] - C * (A * x[0] + B * x[1] + D)) / abc]
    return x0


D = 1
A = 1
B = 1
C = 1

rawData = loadmat('../data/data_mobile_indoor_1.mat')

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
# CSIa1OrigRaw = rawData['A'][:, 0]
# CSIb1OrigRaw = rawData['A'][:, 1]
#
# CSIa1Orig = []
# CSIb1Orig = []
# for i in range(2000):
#     CSIa1Orig.append(CSIa1OrigRaw[i])
#     CSIb1Orig.append(CSIb1OrigRaw[i])
# for i in range(5000):
#     CSIa1Orig.append(CSIa1OrigRaw[i + 20000])
#     CSIb1Orig.append(CSIb1OrigRaw[i + 20000])

CSIa1Orig = np.array(CSIa1Orig)
CSIb1Orig = np.array(CSIb1Orig)

# plt.figure()
# plt.plot(CSIa1Orig, color="red", linewidth=.05, label="a")
# # plt.plot(CSIb1Orig, color="blue", linewidth=.05, label="b")
# plt.legend(loc='upper left')
# plt.show()

dataLen = len(CSIa1Orig)  # 6745

CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=dataLen)

CSIa1Orig = smooth(CSIa1Orig, window_len=15, window='flat')
CSIb1Orig = smooth(CSIb1Orig, window_len=15, window='flat')
CSIe1Orig = smooth(CSIe1Orig, window_len=15, window="flat")

CSIa1OrigBack = CSIa1Orig.copy()
CSIb1OrigBack = CSIb1Orig.copy()
CSIe1OrigBack = CSIe1Orig.copy()

noise = np.random.normal(loc=-1, scale=1, size=dataLen)  ## Multiplication item normal distribution
noiseAdd = np.random.normal(loc=0, scale=10, size=dataLen)  ## Addition item normal distribution

sft = 2
intvl = 2 * sft + 1
keyLen = 243
addNoise = True

originSum = 0
correctSum = 0
randomSum = 0
noiseSum = 0

codings = ""
# for ii in range(0, 5):
for staInd in range(0, 10 * intvl + 1, intvl):
    endInd = staInd + keyLen * intvl
    print("range:", staInd, endInd)
    if endInd > len(CSIa1Orig):
        break

    CSIa1Orig = CSIa1OrigBack.copy()
    CSIb1Orig = CSIb1OrigBack.copy()
    CSIe1Orig = CSIe1OrigBack.copy()

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
    tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]

    tmpNoise = noise[range(staInd, endInd, 1)]
    tmpNoiseAdd = noiseAdd[range(staInd, endInd, 1)]

    tmpCSIa1 = tmpCSIa1 - (np.mean(tmpCSIa1) - np.mean(tmpCSIb1))  # Mean value consistency

    # linspace函数生成元素为50的等间隔数列，可以指定第三个参数为元素个数
    # signal.square返回周期性的方波波形
    tmpPulse = signal.square(
        2 * np.pi * 1 / intvl * np.linspace(0, np.pi * 0.5 * keyLen, keyLen * intvl))  ## Rectangular pulse

    if addNoise:
        tmpCSIa1 = tmpPulse * (np.float_power(np.abs(tmpCSIa1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIa1))
        tmpCSIb1 = tmpPulse * (np.float_power(np.abs(tmpCSIb1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIb1))
        tmpCSIe1 = tmpPulse * (np.float_power(np.abs(tmpCSIe1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIe1))
    else:
        tmpCSIa1 = tmpPulse * tmpCSIa1
        tmpCSIb1 = tmpPulse * tmpCSIb1
        tmpCSIe1 = tmpPulse * tmpCSIe1

    CSIa1Orig[range(staInd, endInd, 1)] = tmpCSIa1
    CSIb1Orig[range(staInd, endInd, 1)] = tmpCSIb1
    CSIe1Orig[range(staInd, endInd, 1)] = tmpCSIe1

    permLen = len(range(staInd, endInd, intvl))
    origInd = np.array([xx for xx in range(staInd, endInd, intvl)])

    sortCSIa1 = np.zeros(permLen)
    sortCSIb1 = np.zeros(permLen)
    sortCSIe1 = np.zeros(permLen)
    sortNoise = np.zeros(permLen)

    for ii in range(permLen):
        aIndVec = np.array([aa for aa in range(origInd[ii], origInd[ii] + intvl, 1)])  ## for non-permuted CSIa1

        for jj in range(permLen, permLen * 2):
            bIndVec = np.array([bb for bb in range(origInd[jj - permLen], origInd[jj - permLen] + intvl, 1)])

            CSIa1Tmp = CSIa1Orig[aIndVec]
            CSIb1Tmp = CSIb1Orig[bIndVec]
            CSIe1Tmp = CSIe1Orig[bIndVec]
            noiseTmp = noise[aIndVec]

            sortCSIa1[ii] = np.mean(CSIa1Tmp)  ## Metric 1: Mean
            sortCSIb1[jj - permLen] = np.mean(CSIb1Tmp)  # 只赋值一次
            sortCSIe1[jj - permLen] = np.mean(CSIe1Tmp)
            sortNoise[ii] = np.mean(noiseTmp)

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
    step = int(math.pow(3, param))
    sortCSIa1 = sortCSIa1.reshape(int(len(sortCSIa1) / step / 3), 3)
    sortCSIb1 = sortCSIb1.reshape(int(len(sortCSIb1) / step / 3), 3)
    sortCSIe1 = sortCSIe1.reshape(int(len(sortCSIe1) / step / 3), 3)
    sortNoise = sortNoise.reshape(int(len(sortNoise) / step / 3), 3)

    # 在数组a后面加上a[0]使之成为一个首尾封闭的多面体
    sortCSIa1Add = makePolygon(sortCSIa1)
    sortCSIb1Add = makePolygon(sortCSIb1)
    sortCSIe1Add = makePolygon(sortCSIe1)
    sortCSIn1Add = makePolygon(sortNoise)

    # 最后各自的密钥
    a_list = []
    b_list = []
    e_list = []
    n_list = []

    projCSIa1XYZ = []
    projCSIb1XYZ = []
    projCSIe1XYZ = []
    projCSIn1XYZ = []

    projCSIa1X = []
    projCSIb1X = []
    projCSIe1X = []
    projCSIn1X = []

    for i in range(len(sortCSIa1)):
        projCSIa1XYZ.append(projection(A, B, C, D, sortCSIa1[i]))
        projCSIb1XYZ.append(projection(A, B, C, D, sortCSIb1[i]))
        projCSIe1XYZ.append(projection(A, B, C, D, sortCSIe1[i]))
        projCSIn1XYZ.append(projection(A, B, C, D, sortNoise[i]))

        projCSIa1X.append(projection(A, B, C, D, sortCSIa1[i])[0])
        projCSIb1X.append(projection(A, B, C, D, sortCSIb1[i])[0])
        projCSIe1X.append(projection(A, B, C, D, sortCSIe1[i])[0])
        projCSIn1X.append(projection(A, B, C, D, sortNoise[i])[0])

    a_list = np.argsort(projCSIa1X)
    b_list = np.argsort(projCSIb1X)
    e_list = np.argsort(projCSIe1X)
    n_list = np.argsort(projCSIn1X)

    fig = plt.figure()
    xa, ya, za = zip(*projCSIa1XYZ)
    xb, yb, zb = zip(*projCSIb1XYZ)
    xe, ye, ze = zip(*projCSIe1XYZ)
    xn, yn, zn = zip(*projCSIn1XYZ)
    ax = plt.axes(projection='3d')
    ax.set_title("3D_Curve")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.plot(xa, ya, za, color='red', linewidth=.5, label="a")
    ax.plot(xb, yb, zb, color="blue", linewidth=.5, label="b")
    # plt.show()

    ax = plt.axes(projection='3d')
    ax.set_title("3D_Curve")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.plot(xa, ya, za, color='red', linewidth=.5, label="a")
    ax.plot(xe, ye, ze, color="black", linewidth=.5, label="e")
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

    # 编码密钥
    # char_weights = []
    # weights = Counter(a_list)  # 得到list中元素出现次数
    # for i in range(len(a_list)):
    #     char_weights.append((a_list[i], weights[a_list[i]]))
    # tree = HuffmanTree(char_weights)
    # tree.get_code()
    # HuffmanTree.codings += "\n"

    # for i in range(len(a_list)):
    #     codings += bin(a_list[i]) + "\n"

# with open('../experiments/key.txt', 'a', ) as f:
#     f.write(codings)

print("a-b all", correctSum, "/", originSum, "=", correctSum / originSum)
print("a-e all", randomSum, "/", originSum, "=", randomSum / originSum)
print("a-n all", noiseSum, "/", originSum, "=", noiseSum / originSum)
