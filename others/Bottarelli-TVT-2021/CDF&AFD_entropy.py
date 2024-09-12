import math
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct
from scipy.io import loadmat


# 根据样本数据估计概率分布
def frequency(samples):
    samples = np.array(samples)
    total_samples = len(samples)

    # 使用字典来记录每个数值出现的次数
    frequency_count = {}
    for sample in samples:
        if sample in frequency_count:
            frequency_count[sample] += 1
        else:
            frequency_count[sample] = 1

    # 计算每个数值的频率，即概率分布
    frequency = []
    for sample in frequency_count:
        frequency.append(frequency_count[sample] / total_samples)

    return frequency


def minEntropy(probabilities):
    return -np.log2(np.max(probabilities) + 1e-12)


def normalize(data):
    if np.max(data) == np.min(data):
        return (data - np.min(data)) / np.max(data)
    else:
        return (np.array(data) - np.min(data)) / (np.max(data) - np.min(data))


def cdf_thresholding(q_minus, sigma):
    return np.sqrt(2) * sigma * np.sqrt(np.log(1 / (1 - np.exp(-q_minus ** 2 / (2 * sigma ** 2)))))


def afd_thresholding(q_minus, sigma):
    return q_minus / (np.exp(q_minus ** 2 / (2 * sigma ** 2)) - 1)


def quantize(data, q_minus, q_plus):
    quantization = []
    for i in range(len(data)):
        if data[i] > max(q_plus, q_minus):
            quantization.append(1)
        elif data[i] < min(q_minus, q_plus):
            quantization.append(0)
    return quantization


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


fileNames = ["../../data/data_mobile_indoor_1.mat",
             "../../data/data_static_indoor_1.mat",
             "../../data/data_mobile_outdoor_1.mat",
             "../../data/data_static_outdoor_1.mat"
             ]

# fileNames = ["../../csi/csi_mobile_indoor_1_r",
#             "../../csi/csi_static_indoor_1_r",
#             "../../csi/csi_mobile_outdoor_r",
#             "../../csi/csi_static_outdoor_r"]

# RSS security strength cdf afd
# mi1 0.5595224387889343    0.4741440904085227
# si1 0.37590394615243455   0.40115349526558214
# mo1 0.5542483252560813    0.532494650284643
# so1 0.4971921503776873    0.4752773678514461

# CSI security strength
# mi1 0.5025708837566587    0.383768637364391
# si1 0.602522370296804     0.5081614325284459
# mo1 0.5364195494800698    0.3819660184204586
# so1 0.5609708706990487    0.4824669678738464

for f in fileNames:
    rawData = loadmat(f)
    print(f)

    if f.find("csi") != -1:
        CSIa1Orig = rawData['testdata'][:, 0]
        CSIb1Orig = rawData['testdata'][:, 1]
    else:
        CSIa1Orig = rawData['A'][:, 0]
        CSIb1Orig = rawData['A'][:, 1]

    # stalking attack
    CSIe2Orig = loadmat("../../data/data_static_indoor_1.mat")['A'][:, 0]
    dataLen = min(len(CSIe2Orig), len(CSIa1Orig))

    CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=3, window='flat')
    CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=3, window='flat')

    segLen = 1
    keyLen = 8 * segLen

    bit_len = int(keyLen / segLen)

    originSum = 0
    correctSum = 0
    randomSum1 = 0
    randomSum2 = 0
    noiseSum1 = 0

    originDecSum = 0
    correctDecSum = 0
    randomDecSum1 = 0
    randomDecSum2 = 0
    noiseDecSum1 = 0

    originWholeSum = 0
    correctWholeSum = 0
    randomWholeSum1 = 0
    randomWholeSum2 = 0
    noiseWholeSum1 = 0

    times = 0

    sigma_C = []

    keys = []

    # 至少测试100*bit_len次，保证每次结果均出现
    for staInd in range(0, 2 ** bit_len * 100, keyLen):
        endInd = staInd + keyLen
        # print("range:", staInd, endInd)
        if endInd >= len(CSIa1Orig):
            print("too long")
            break
        times += 1

        origInd = np.array([xx for xx in range(staInd, endInd, 1)])

        CSIa1Epi = CSIa1Orig[origInd]
        CSIb1Epi = CSIb1Orig[origInd]

        CSIa1Orig[origInd] = CSIa1Epi
        CSIb1Orig[origInd] = CSIb1Epi

        np.random.seed(0)

        # imitation attack
        CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=len(CSIa1Orig))

        tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
        tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
        tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]
        tmpCSIe2 = CSIe2Orig[range(staInd, endInd, 1)]

        # inference attack
        tmpCSIn1 = np.random.random(keyLen)

        # 不能均值化，否则不符合原始的瑞利分布
        # tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
        # tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
        # tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
        # tmpCSIe2 = tmpCSIe2 - np.mean(tmpCSIe2)

        tmpCSIa1 = normalize(tmpCSIa1)
        tmpCSIb1 = normalize(tmpCSIb1)
        tmpCSIe1 = normalize(tmpCSIe1)
        tmpCSIe2 = normalize(tmpCSIe2)

        # 绘制直方图
        # plt.figure()
        # plt.hist(CSIa1Orig, bins=100, color='r', alpha=0.7, label='a')
        # plt.show()

        sigma_C.append(np.sum(np.square(tmpCSIa1 - tmpCSIb1)) / 2 / len(tmpCSIa1))

        alpha_STD = 0.5
        method = "afd"
        # 只能用原始的阈值量化，新方法产生的阈值不可用
        if method == "cdf":
            qa_minus = np.mean(tmpCSIa1) - alpha_STD * np.std(tmpCSIa1)
            qa_plus = cdf_thresholding(qa_minus, np.std(tmpCSIa1))
            qb_minus = np.mean(tmpCSIb1) - alpha_STD * np.std(tmpCSIb1)
            qb_plus = cdf_thresholding(qb_minus, np.std(tmpCSIb1))
            qe1_minus = np.mean(tmpCSIe1) - alpha_STD * np.std(tmpCSIe1)
            qe1_plus = cdf_thresholding(qe1_minus, np.std(tmpCSIe1))
            qe2_minus = np.mean(tmpCSIe2) - alpha_STD * np.std(tmpCSIe2)
            qe2_plus = cdf_thresholding(qe2_minus, np.std(tmpCSIe2))
            qn1_minus = np.mean(tmpCSIn1) - alpha_STD * np.std(tmpCSIn1)
            qn1_plus = cdf_thresholding(qn1_minus, np.std(tmpCSIn1))
        elif method == "afd":
            qa_minus = np.mean(tmpCSIa1) - alpha_STD * np.std(tmpCSIa1)
            qa_plus = afd_thresholding(qa_minus, np.std(tmpCSIa1))
            qb_minus = np.mean(tmpCSIb1) - alpha_STD * np.std(tmpCSIb1)
            qb_plus = afd_thresholding(qb_minus, np.std(tmpCSIb1))
            qe1_minus = np.mean(tmpCSIe1) - alpha_STD * np.std(tmpCSIe1)
            qe1_plus = afd_thresholding(qe1_minus, np.std(tmpCSIe1))
            qe2_minus = np.mean(tmpCSIe2) - alpha_STD * np.std(tmpCSIe2)
            qe2_plus = afd_thresholding(qe2_minus, np.std(tmpCSIe2))
            qn1_minus = np.mean(tmpCSIn1) - alpha_STD * np.std(tmpCSIn1)
            qn1_plus = afd_thresholding(qn1_minus, np.std(tmpCSIn1))
        else:
            qa_minus = np.mean(tmpCSIa1) - alpha_STD * np.std(tmpCSIa1)
            qa_plus = np.mean(tmpCSIa1) + alpha_STD * np.std(tmpCSIa1)
            qb_minus = np.mean(tmpCSIb1) - alpha_STD * np.std(tmpCSIb1)
            qb_plus = np.mean(tmpCSIb1) + alpha_STD * np.std(tmpCSIb1)
            qe1_minus = np.mean(tmpCSIe1) - alpha_STD * np.std(tmpCSIe1)
            qe1_plus = np.mean(tmpCSIe1) + alpha_STD * np.std(tmpCSIe1)
            qe2_minus = np.mean(tmpCSIe2) - alpha_STD * np.std(tmpCSIe2)
            qe2_plus = np.mean(tmpCSIe2) + alpha_STD * np.std(tmpCSIe2)
            qn1_minus = np.mean(tmpCSIn1) - alpha_STD * np.std(tmpCSIn1)
            qn1_plus = np.mean(tmpCSIn1) + alpha_STD * np.std(tmpCSIn1)

        a_list = quantize(tmpCSIa1, qa_minus, qa_plus)
        b_list = quantize(tmpCSIb1, qb_minus, qb_plus)
        e1_list = quantize(tmpCSIe1, qe1_minus, qe1_plus)
        e2_list = quantize(tmpCSIe2, qe2_minus, qe2_plus)
        n1_list = quantize(tmpCSIn1, qn1_minus, qn1_plus)

        # 对齐密钥，随机补全
        for i in range(len(a_list) - len(e1_list)):
            e1_list += str(np.random.randint(0, 2))
        for i in range(len(a_list) - len(e2_list)):
            e2_list += str(np.random.randint(0, 2))
        for i in range(len(a_list) - len(n1_list)):
            n1_list += str(np.random.randint(0, 2))

        # print("keys of a:", len(a_list), a_list)
        # print("keys of b:", len(b_list), b_list)
        # print("keys of e1:", len(e1_list), e1_list)
        # print("keys of e2:", len(e2_list), e2_list)
        # print("keys of n1:", len(n1_list), n1_list)

        sum1 = min(len(a_list), len(b_list))
        sum2 = 0
        sum31 = 0
        sum32 = 0
        sum41 = 0
        for i in range(0, sum1):
            sum2 += (a_list[i] == b_list[i])
        for i in range(min(len(a_list), len(e1_list))):
            sum31 += (a_list[i] == e1_list[i])
        for i in range(min(len(a_list), len(e2_list))):
            sum32 += (a_list[i] == e2_list[i])
        for i in range(min(len(a_list), len(n1_list))):
            sum41 += (a_list[i] == n1_list[i])

        # print("\033[0;32;40ma-b", sum2, sum2 / sum1, "\033[0m")
        # print("a-e1", sum31, sum31 / sum1)
        # print("a-e2", sum32, sum32 / sum1)
        # print("a-n1", sum41, sum41 / sum1)
        originSum += sum1
        correctSum += sum2
        randomSum1 += sum31
        randomSum2 += sum32
        noiseSum1 += sum41

        # decSum1 = min(len(a_list_number), len(b_list_number))
        # decSum2 = 0
        # decSum31 = 0
        # decSum32 = 0
        # decSum41 = 0
        # for i in range(0, decSum1):
        #     decSum2 += (a_list_number[i] == b_list_number[i])
        # for i in range(min(len(a_list_number), len(e1_list_number))):
        #     decSum31 += (a_list_number[i] == e1_list_number[i])
        # for i in range(min(len(a_list_number), len(e2_list_number))):
        #     decSum32 += (a_list_number[i] == e2_list_number[i])
        # for i in range(min(len(a_list_number), len(n1_list_number))):
        #     decSum41 += (a_list_number[i] == n1_list_number[i])
        # if decSum1 == 0:
        #     continue
        # if decSum2 == decSum1:
        #     print("\033[0;32;40ma-b dec", decSum2, decSum2 / decSum1, "\033[0m")
        # else:
        #     print("\033[0;31;40ma-b dec", "bad", decSum2, decSum2 / decSum1, "\033[0m")
        # print("a-e1", decSum31, decSum31 / decSum1)
        # print("a-e2", decSum32, decSum32 / decSum1)
        # print("a-n1", decSum41, decSum41 / decSum1)
        # print("----------------------")
        # originDecSum += decSum1
        # correctDecSum += decSum2
        # randomDecSum1 += decSum31
        # randomDecSum2 += decSum32
        # noiseDecSum1 += decSum41

        originWholeSum += 1
        correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
        randomWholeSum1 = randomWholeSum1 + 1 if sum31 == sum1 else randomWholeSum1
        randomWholeSum2 = randomWholeSum2 + 1 if sum32 == sum1 else randomWholeSum2
        noiseWholeSum1 = noiseWholeSum1 + 1 if sum41 == sum1 else noiseWholeSum1

        keys.append("".join(map(str, a_list)))

    distribution = frequency(keys)
    print("minEntropy", minEntropy(distribution) / bit_len, "bit_len", bit_len, "keyLen", len(keys))

    print("\033[0;32;40ma-b bit agreement rate", correctSum, "/", originSum, "=", round(correctSum / originSum, 10),
          "\033[0m")
    print("a-e1 bit agreement rate", randomSum1, "/", originSum, "=", round(randomSum1 / originSum, 10))
    print("a-e2 bit agreement rate", randomSum2, "/", originSum, "=", round(randomSum2 / originSum, 10))
    print("a-n1 bit agreement rate", noiseSum1, "/", originSum, "=", round(noiseSum1 / originSum, 10))
    # print("\033[0;32;40ma-b dec agreement rate", correctDecSum, "/", originDecSum, "=",
    #       round(correctDecSum / originDecSum, 10), "\033[0m")
    # print("a-e1 dec agreement rate", randomDecSum1, "/", originDecSum, "=", round(randomDecSum1 / originDecSum, 10))
    # print("a-e2 dec agreement rate", randomDecSum2, "/", originDecSum, "=", round(randomDecSum2 / originDecSum, 10))
    # print("a-n1 dec agreement rate", noiseDecSum1, "/", originDecSum, "=", round(noiseDecSum1 / originDecSum, 10))
    print("\033[0;32;40ma-b key agreement rate", correctWholeSum, "/", originWholeSum, "=",
          round(correctWholeSum / originWholeSum, 10), "\033[0m")
    print("a-e1 key agreement rate", randomWholeSum1, "/", originWholeSum, "=",
          round(randomWholeSum1 / originWholeSum, 10))
    print("a-e2 key agreement rate", randomWholeSum2, "/", originWholeSum, "=",
          round(randomWholeSum2 / originWholeSum, 10))
    print("a-n1 key agreement rate", noiseWholeSum1, "/", originWholeSum, "=",
          round(noiseWholeSum1 / originWholeSum, 10))
    print("times", times)
    print(originSum / len(CSIa1Orig))
    print(correctSum / len(CSIa1Orig))

    # 更高一些
    print(round(correctSum / originSum, 10), round(correctWholeSum / originWholeSum, 10),
          originSum / times / keyLen,
          correctSum / times / keyLen)

    # 算上了密钥长度的取整
    print(round(correctSum / originSum, 10), round(correctWholeSum / originWholeSum, 10),
          originSum / len(CSIa1Orig),
          correctSum / len(CSIa1Orig))

    print("sigma_C", np.mean(sigma_C))
    print()
