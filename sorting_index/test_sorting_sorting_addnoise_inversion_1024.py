import csv
import math
import time
from tkinter import messagebox

import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import dct
from scipy.io import loadmat
from scipy.stats import pearsonr, boxcox


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

def search(data, p):
    for i in range(len(data)):
        if p == data[i]:
            return i

def iterAttack(x, r):
    x = x / np.linalg.norm(x, ord=2)

    d = []
    for j in range(len(r) - 1):
        for k in range(j + 1, len(r)):
            if np.dot(r[j], x) > np.dot(r[k], x):
                d.append((r[j] - r[k]) / np.linalg.norm(r[j] - r[k], ord=2))
            else:
                d.append((r[k] - r[j]) / np.linalg.norm(r[k] - r[j], ord=2))

    xe = np.mean(d, axis=0).T

    for e in range(10):
        y = np.matmul(r, xe)
        yMin = np.min(y)
        yMinIndex = 0
        for i in range(len(y)):
            if np.isclose(y[i], yMin):
                yMinIndex = i
        kStar = r[yMinIndex]
        # xe = np.array(xe.T - np.dot(kStar, xe) * np.array(kStar).T / np.linalg.norm(kStar, ord=2)).T
        xe = np.array(xe.T - np.dot(kStar, xe) * np.array(kStar).T).T
        xe = xe / np.linalg.norm(xe, ord=2)

    return xe


def dpAttack(x, r):
    import gurobipy as gp
    L = len(x)

    y = np.matmul(r, x)
    rir = np.matmul(r, np.linalg.pinv(r))
    M = rir - np.array(np.eye(L))

    model = gp.Model()
    inputs = []
    for i in range(L):
        inputs.append(model.addVar(lb=-100, ub=100, name=f'x{i}'))
    obj = sum(np.dot(M, inputs) ** 2)
    model.setObjective(obj, sense=gp.GRB.MINIMIZE)

    ra = np.argsort(y)
    dk = []
    for j in range(len(r) - 1, 0, -1):
        # 将y的不等式转为向量
        dktmp = np.zeros(L)
        dktmp[ra[j]] = 1
        dktmp[ra[j - 1]] = -1
        dk.append(dktmp)

    for i in range(len(dk)):
        model.addConstr(np.dot(dk[i], np.array(inputs)) >= 0, name=f'c{i}')

    params = {"NonConvex": 2, "OutputFlag": 0, "LogToConsole": 0}
    # set OutputFlag to 0 to suppress solver output
    for key, value in params.items():
        model.setParam(key, value)
    model.optimize()

    if model.status == gp.GRB.OPTIMAL:
        xe = []
        for v in model.getVars():
            xe.append(v.x)
        xe = np.array(xe)
    else:
        xe = np.random.normal(0, 1, L)

    return xe


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
    keyLen = 64 * segLen
    rec = True

    print("segLen", segLen)
    print("keyLen", keyLen / segLen)

    originSum = 0
    correctSum = 0
    randomSum1 = 0
    randomSum2 = 0

    originDecSum = 0
    correctDecSum = 0
    randomDecSum1 = 0
    randomDecSum2 = 0

    originWholeSum = 0
    correctWholeSum = 0
    randomWholeSum1 = 0
    randomWholeSum2 = 0

    times = 0

    # static indoor
    # if f == "../data/data_static_indoor_1.mat":
    #     dataLen = int(dataLen / 5)
    #     keyLen = int(keyLen / 5)
    #     print(dataLen, keyLen)
    # for staInd in range(0, int(dataLen / 5.5), int(keyLen / 5)):

    distance_e1 = 0
    distance_e2 = 0

    mhd_e1_dist = 0
    mhd_e2_dist = 0

    # runs = 0
    for staInd in range(0, dataLen, keyLen):
        # runs += 1
        # if runs > 2:
        #     break
        endInd = staInd + keyLen
        print("range:", staInd, endInd)
        if endInd >= len(CSIa1Orig):
            break
        times += 1

        tmpCSIe11 = []
        tmpCSIe12 = []
        for repeated in range(2):
            # np.random.seed(1)
            # CSIa1Orig = np.tile(rawData['A'][:, 0], 5)
            # CSIb1Orig = np.tile(rawData['A'][:, 1], 5)
            CSIa1Orig = rawData['A'][:, 0]
            CSIb1Orig = rawData['A'][:, 1]

            seed = np.random.randint(100000)
            np.random.seed(seed)

            # 静态数据需要置换
            # 固定随机置换的种子
            # np.random.seed(0)
            # combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig))
            # np.random.shuffle(combineCSIx1Orig)
            # CSIa1Orig, CSIb1Orig = zip(*combineCSIx1Orig)

            CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
            CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')

            tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
            tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]

            randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
            tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
            tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)

            tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
            tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)

            if repeated == 0:
                # iterative attack
                tmpCSIe11 = iterAttack(tmpCSIa1, randomMatrix)
                # solving dp problem
                # tmpCSIe12 = dpAttack(tmpCSIa1, randomMatrix)
                tmpCSIe12 = np.random.normal(0, np.std(CSIa1Orig) * 4, keyLen)
            if repeated == 1:
                tmpCSIe1 = np.matmul(tmpCSIe11 - np.mean(tmpCSIe11), randomMatrix)
                tmpCSIe2 = np.matmul(tmpCSIe12 - np.mean(tmpCSIe12), randomMatrix)

            # 最后各自的密钥
            a_list = []
            b_list = []

            tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
            tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()

            minEpiIndClosenessLsb = np.zeros(int(keyLen / segLen), dtype=int)

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

                for j in range(int(keyLen / segLen)):
                    epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]

                    epiIndClosenessLsb[j] = sum(abs(epiIndb1 - np.array(epiInda1)))

                minEpiIndClosenessLsb[i] = np.argmin(epiIndClosenessLsb)

            a_list_number = list(permutation)
            b_list_number = list(minEpiIndClosenessLsb)

            # 转成二进制，0填充成0000
            for i in range(len(a_list_number)):
                number = bin(a_list_number[i])[2:].zfill(int(np.log2(len(a_list_number))))
                a_list += number
            for i in range(len(b_list_number)):
                number = bin(b_list_number[i])[2:].zfill(int(np.log2(len(b_list_number))))
                b_list += number

            sum1 = min(len(a_list), len(b_list))
            sum2 = 0

            for i in range(0, sum1):
                sum2 += (a_list[i] == b_list[i])

            # 自适应纠错
            if sum1 != sum2 and rec:
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

            if repeated == 1:
                e1_list = []
                e2_list = []

                tmpCSIe1Ind = np.array(tmpCSIe11).argsort().argsort()
                tmpCSIe2Ind = np.array(tmpCSIe12).argsort().argsort()

                minEpiIndClosenessLse1 = np.zeros(int(keyLen / segLen), dtype=int)
                minEpiIndClosenessLse2 = np.zeros(int(keyLen / segLen), dtype=int)

                for i in range(int(keyLen / segLen)):
                    epiInda1 = tmpCSIa1Ind[i * segLen:(i + 1) * segLen]

                    epiIndClosenessLse1 = np.zeros(int(keyLen / segLen))
                    epiIndClosenessLse2 = np.zeros(int(keyLen / segLen))

                    for j in range(int(keyLen / segLen)):
                        epiIndb1 = tmpCSIb1Ind[j * segLen: (j + 1) * segLen]
                        epiInde1 = tmpCSIe1Ind[j * segLen: (j + 1) * segLen]
                        epiInde2 = tmpCSIe2Ind[j * segLen: (j + 1) * segLen]

                        epiIndClosenessLse1[j] = sum(abs(epiInde1 - np.array(epiInda1)))
                        epiIndClosenessLse2[j] = sum(abs(epiInde2 - np.array(epiInda1)))

                    minEpiIndClosenessLse1[i] = np.argmin(epiIndClosenessLse1)
                    minEpiIndClosenessLse2[i] = np.argmin(epiIndClosenessLse2)

                e1_list_number = list(minEpiIndClosenessLse1)
                e2_list_number = list(minEpiIndClosenessLse2)

                for i in range(len(e1_list_number)):
                    number = bin(e1_list_number[i])[2:].zfill(int(np.log2(len(e1_list_number))))
                    e1_list += number
                for i in range(len(e2_list_number)):
                    number = bin(e2_list_number[i])[2:].zfill(int(np.log2(len(e2_list_number))))
                    e2_list += number

                # 对齐密钥，随机补全
                for i in range(len(a_list) - len(e1_list)):
                    e1_list += str(np.random.randint(0, 2))
                for i in range(len(a_list) - len(e2_list)):
                    e2_list += str(np.random.randint(0, 2))

                sum31 = 0
                sum32 = 0

                for i in range(min(len(a_list), len(e1_list))):
                    sum31 += (a_list[i] == e1_list[i])
                for i in range(min(len(a_list), len(e2_list))):
                    sum32 += (a_list[i] == e2_list[i])

                print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
                print("a-e1", sum31, sum31 / sum1)
                print("a-e2", sum32, sum32 / sum1)

                originSum += sum1
                correctSum += sum2
                randomSum1 += sum31
                randomSum2 += sum32

                originWholeSum += 1
                correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
                randomWholeSum1 = randomWholeSum1 + 1 if sum31 == sum1 else randomWholeSum1
                randomWholeSum2 = randomWholeSum2 + 1 if sum32 == sum1 else randomWholeSum2

                # mhd：绝对值距离之和除以密钥长度
                mhd_e1_dist += sum(abs(np.array(a_list_number) - np.array(e1_list_number))) / int(keyLen / segLen)
                mhd_e2_dist += sum(abs(np.array(a_list_number) - np.array(e2_list_number))) / int(keyLen / segLen)

                # distance：与真实值索引偏移距离之和除以密钥长度
                tmp_e1_dist = 0
                for i in range(len(e1_list_number)):
                    real_pos = search(a_list_number, e1_list_number[i])
                    guess_pos = i
                    tmp_e1_dist += abs(real_pos - guess_pos)
                distance_e1 += (tmp_e1_dist / int(keyLen / segLen))

                tmp_e2_dist = 0
                for i in range(len(e2_list_number)):
                    real_pos = search(a_list_number, e2_list_number[i])
                    guess_pos = i
                    tmp_e2_dist += abs(real_pos - guess_pos)
                distance_e2 += (tmp_e2_dist / int(keyLen / segLen))

    print("\033[0;34;40ma-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10), "\033[0m")
    print("a-e1 all", randomSum1, "/", originSum, "=", round(randomSum1 / originSum, 10))
    print("a-e2 all", randomSum2, "/", originSum, "=", round(randomSum2 / originSum, 10))

    print("\033[0;34;40ma-b whole match", correctWholeSum, "/", originWholeSum, "=",
          round(correctWholeSum / originWholeSum, 10), "\033[0m")
    print("a-e1 whole match", randomWholeSum1, "/", originWholeSum, "=", round(randomWholeSum1 / originWholeSum, 10))
    print("a-e2 whole match", randomWholeSum2, "/", originWholeSum, "=", round(randomWholeSum2 / originWholeSum, 10))
    print("times", times)

    print(round(correctSum / originSum, 10), round(correctWholeSum / originWholeSum, 10), originSum / times / keyLen,
          correctSum / times / keyLen)

    print("与真实值索引偏移距离之和除以密钥长度")
    print("e1", round(distance_e1 / times, 8))
    print("e2", round(distance_e2 / times, 8))

    print("绝对值距离之和除以密钥长度")
    print("e1", round(mhd_e1_dist / times, 8))
    print("e2", round(mhd_e2_dist / times, 8))
messagebox.showinfo("提示", "测试结束")
