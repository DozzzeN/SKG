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


fileName = ["../data/data_mobile_indoor_1.mat",
            "../data/data_mobile_outdoor_1.mat",
            "../data/data_static_outdoor_1.mat",
            "../data/data_static_indoor_1.mat"
            ]

# fileName = ["../data/data_mobile_indoor_1.mat"]

# segLen = 4    keyLen = 256 * segLen
# 有纠错
# si1 normal                        0.9997612847 0.9444444444 2.0 1.9995225694444445
# si1 no sorting with perturbation  0.9996310764 0.9444444444 2.0 1.9992621527777779
# si1 no perturbation with sorting  0.7866644965 0.0 2.0 1.5733289930555556
# si1 no perturbation no sorting    0.7680501302 0.0 2.0 1.5361002604166667
# 无纠错
# si1 normal                        0.9953938802 0.4555555556 2.0 1.9907877604166666
# si1 no sorting with perturbation  0.9964029948 0.4888888889 2.0 1.9928059895833334
# si1 no perturbation with sorting  0.7307725694 0.0 2.0 1.461545138888889
# si1 no perturbation no sorting    0.7161241319 0.0 2.0 1.432248263888889

# segLen = 4    keyLen = 256 * segLen
# 有纠错
# so1 normal                        0.9923037574 0.4761904762 2.0 1.9846075148809523
# so1 no sorting with perturbation  0.9946754092 0.4285714286 2.0 1.989350818452381
# so1 no perturbation with sorting  0.7449544271 0.0 2.0 1.4899088541666667
# so1 no perturbation no sorting    0.7433035714 0.0 2.0 1.4866071428571428
# 无纠错
# so1 normal                        0.9681222098 0.0952380952 2.0 1.9362444196428572
# so1 no sorting with perturbation  0.972749256 0.0952380952 2.0 1.9454985119047619
# so1 no perturbation with sorting  0.6937313988 0.0 2.0 1.3874627976190477
# so1 no perturbation no sorting    0.6915922619 0.0 2.0 1.3831845238095237

# segLen = 4    keyLen = 256 * segLen
# 有纠错
# mo1 normal                        0.9951171875 0.5 2.0 1.990234375
# mo1 no sorting with perturbation  0.995686849 0.5 2.0 1.9913736979166667
# mo1 no perturbation with sorting  0.7184244792 0.0 2.0 1.4368489583333333
# mo1 no perturbation no sorting    0.6780598958 0.0 2.0 1.3561197916666667
# 无纠错
# mo1 normal                        0.9825032552 0.3333333333 2.0 1.9650065104166667
# mo1 no sorting with perturbation  0.9738769531 0.1666666667 2.0 1.94775390625
# mo1 no perturbation with sorting  0.6794433594 0.0 2.0 1.35888671875
# mo1 no perturbation no sorting    0.6411132812 0.0 2.0 1.2822265625

# segLen = 4    keyLen = 256 * segLen
# 有纠错
# mi1 normal                        0.9979926215 0.8333333333 2.0 1.9959852430555556
# mi1 no sorting with perturbation  0.9981553819 0.8333333333 2.0 1.9963107638888888
# mi1 no perturbation with sorting  0.8315972222 0.0 2.0 1.6631944444444444
# mi1 no perturbation no sorting    0.814046224 0.0 2.0 1.6280924479166667
# 无纠错
# mi1 normal                        0.9881998698 0.4444444444 2.0 1.9763997395833333
# mi1 no sorting with perturbation  0.9900444878 0.5555555556 2.0 1.9800889756944444
# mi1 no perturbation with sorting  0.7734917535 0.0 2.0 1.5469835069444444
# mi1 no perturbation no sorting    0.7630479601 0.0 2.0 1.5260959201388888
##############################################################################################################

# segLen = 5    keyLen = 1024 * segLen
# 有纠错
# si1 normal                        1.0 1.0 2.0 2.0
# si1 no sorting with perturbation  1.0 1.0 2.0 2.0
# si1 no perturbation with sorting
# si1 no perturbation no sorting
# 无纠错
# si1 normal                        0.9994411892 0.3333333333 2.0 1.998882378472222
# si1 no sorting with perturbation  0.9996365017 0.5 2.0 1.9992730034722221
# si1 no perturbation with sorting  0.6685167101 0.0 2.0 1.337033420138889
# si1 no perturbation no sorting    0.6597981771 0.0 2.0 1.3195963541666667

# segLen = 5    keyLen = 1024 * segLen
# 有纠错
# so1 normal                        0.9999267578 0.75 2.0 1.999853515625
# so1 no sorting with perturbation  0.9997802734 0.75 2.0 1.999560546875
# so1 no perturbation with sorting  0.7000732422 0.0 2.0 1.400146484375
# so1 no perturbation no sorting    0.7032714844 0.0 2.0 1.40654296875
# 无纠错
# so1 normal                        0.9971679688 0.0 2.0 1.9943359375
# so1 no sorting with perturbation  0.997265625 0.0 2.0 1.99453125
# so1 no perturbation with sorting  0.6572265625 0.0 2.0 1.314453125
# so1 no perturbation no sorting    0.6598144531 0.0 2.0 1.31962890625

# segLen = 5    keyLen = 1024 * segLen
# 有纠错
# mo1 normal                        1.0 1.0 2.0 2.0
# mo1 no sorting with perturbation  1.0 1.0 2.0 2.0
# mo1 no perturbation with sorting  0.6625976562 0.0 2.0 1.3251953125
# mo1 no perturbation no sorting    0.5977539063 0.0 2.0 1.1955078125
# 无纠错
# mo1 normal                        1.0 1.0 2.0 2.0
# mo1 no sorting with perturbation  0.9982421875 0.0 2.0 1.996484375
# mo1 no perturbation with sorting  0.6296875 0.0 2.0 1.259375
# mo1 no perturbation no sorting    0.589453125 0.0 2.0 1.17890625

# segLen = 5    keyLen = 1024 * segLen
# 有纠错
# mi1 normal                        1.0 1.0 2.0 2.0
# mi1 no sorting with perturbation  1.0 1.0 2.0 2.0
# mi1 no perturbation with sorting  0.7686523438 0.0 2.0 1.5373046875
# mi1 no perturbation no sorting    0.7609049479 0.0 2.0 1.5218098958333335
# 无纠错
# mi1 normal                        1.0 1.0 2.0 2.0
# mi1 no sorting with perturbation  1.0 1.0 2.0 2.0
# mi1 no perturbation with sorting  0.7202148438 0.0 2.0 1.4404296875
# mi1 no perturbation no sorting    0.7165690104 0.0 2.0 1.4331380208333333
##############################################################################################################
# 无纠错，无滤波
# si1 normal                        0.6776745855 0.0 2.0 1.3553491709183674
# si1 no sorting with perturbation  0.6848941725 0.0 2.0 1.3697883450255102
# si1 no perturbation with sorting  0.6711126834 0.0 2.0 1.3422253667091837
# si1 no perturbation no sorting    0.6224938217 0.0 2.0 1.2449876434948979

# 无纠错，无滤波，所有数据
# si1 normal                        0.8535807292 0.0 2.0 1.7071614583333334
# si1 no sorting with perturbation  0.8688259549 0.0 2.0 1.7376519097222223
# si1 no perturbation with sorting  0.7351453993 0.0 2.0 1.4702907986111111
# si1 no perturbation no sorting    0.7502712674 0.0 2.0 1.5005425347222223

# segLen = 5    keyLen = 1024 * segLen
# 无纠错，无滤波
# mi1 normal                        0.9639322917 0.0 2.0 1.9278645833333332
# mi1 no sorting with perturbation  0.9724283854 0.0 2.0 1.9448567708333333
# mi1 no perturbation with sorting  0.7467447917 0.0 2.0 1.4934895833333335
# mi1 no perturbation no sorting    0.7791015625 0.0 2.0 1.558203125

# 无纠错，有滤波
# mi1 normal                        1.0 1.0 2.0 2.0
# mi1 no sorting with perturbation  1.0 1.0 2.0 2.0
# mi1 no perturbation with sorting  0.7202148438 0.0 2.0 1.4404296875
# mi1 no perturbation no sorting    0.7165690104 0.0 2.0 1.4331380208333333

# 是否纠错
rec = False

# 是否排序
withoutSorts = [True, False]
# 是否添加噪声
addNoises = ["mul", ""]

for f in fileName:
    for addNoise in addNoises:
        for withoutSort in withoutSorts:
            print(f)
            rawData = loadmat(f)

            CSIa1Orig = rawData['A'][:, 0]
            CSIb1Orig = rawData['A'][:, 1]
            dataLen = len(CSIa1Orig)
            print("dataLen", dataLen)

            segLen = 5
            keyLen = 1024 * segLen
            tell = True

            print("segLen", segLen)
            print("keyLen", keyLen / segLen)

            originSum = 0
            correctSum = 0

            originDecSum = 0
            correctDecSum = 0

            originWholeSum = 0
            correctWholeSum = 0

            times = 0
            overhead = 0

            if withoutSort:
                if addNoise == "mul":
                    print("no sorting")
            if withoutSort:
                if addNoise == "":
                    print("no sorting and no perturbation")
            if withoutSort is False:
                if addNoise == "":
                    print("no perturbation")
                if addNoise == "mul":
                    print("normal")

            dataLenLoop = dataLen
            keyLenLoop = keyLen
            if f == "../data/data_static_indoor_1.mat":
                dataLenLoop = int(dataLen / 5.5)
                keyLenLoop = int(keyLen / 5)
            for staInd in range(0, dataLenLoop, keyLenLoop):
                start = time.time()
                endInd = staInd + keyLen
                # print("range:", staInd, endInd)
                if endInd >= len(CSIa1Orig):
                    break

                times += 1

                CSIa1Orig = rawData['A'][:, 0]
                CSIb1Orig = rawData['A'][:, 1]

                seed = np.random.randint(100000)
                np.random.seed(seed)

                # 固定随机置换的种子
                # np.random.seed(0)
                # combineCSIx1Orig = list(zip(CSIa1Orig, CSIb1Orig))
                # np.random.shuffle(combineCSIx1Orig)
                # CSIa1Orig, CSIb1Orig = zip(*combineCSIx1Orig)
                # CSIa1Orig = np.array(CSIa1Orig)
                # CSIb1Orig = np.array(CSIb1Orig)

                CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
                CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')

                tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
                tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]

                if addNoise == "mul":
                    randomMatrix = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(keyLen, keyLen))
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
                    tmpCSIa1 = np.matmul(tmpCSIa1, randomMatrix)
                    tmpCSIb1 = np.matmul(tmpCSIb1, randomMatrix)
                else:
                    tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
                    tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)

                # 最后各自的密钥
                a_list = []
                b_list = []

                # without sorting
                if withoutSort:
                    tmpCSIa1Ind = np.array(tmpCSIa1)
                    tmpCSIb1Ind = np.array(tmpCSIb1)
                else:
                    tmpCSIa1Ind = np.array(tmpCSIa1).argsort().argsort()
                    tmpCSIb1Ind = np.array(tmpCSIb1).argsort().argsort()

                    # plt.figure()
                    # plt.plot(tmpCSIa1Ind[0:100])
                    # plt.show()
                    # plt.figure()
                    # plt.plot(tmpCSIa1[0:100])
                    # plt.show()
                    # exit()

                np.random.seed(0)
                combineCSIx1Orig = list(zip(tmpCSIa1Ind, tmpCSIb1Ind))
                np.random.shuffle(combineCSIx1Orig)
                tmpCSIa1Ind, tmpCSIb1Ind = zip(*combineCSIx1Orig)
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

                end = time.time()
                overhead += end - start
                # print("time:", end - start)

                # 自适应纠错
                if sum1 != sum2 and rec:
                    if tell:
                        # print("correction")
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

                                sum2 = 0
                                for i in range(0, min(len(a_list), len(b_list))):
                                    sum2 += (a_list[i] == b_list[i])

                # print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
                originSum += sum1
                correctSum += sum2

                originWholeSum += 1
                correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum

            print("\033[0;34;40ma-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10), "\033[0m")
            print("\033[0;34;40ma-b whole match", correctWholeSum, "/", originWholeSum, "=",
                  round(correctWholeSum / originWholeSum, 10), "\033[0m")
            # print("times", times)
            # print(overhead / originWholeSum)

            print(round(correctSum / originSum, 10), round(correctWholeSum / originWholeSum, 10), originSum / times / keyLen,
                  correctSum / times / keyLen)
            if withoutSort:
                print("withoutSort")
            else:
                print("withSort")
            print("\n")