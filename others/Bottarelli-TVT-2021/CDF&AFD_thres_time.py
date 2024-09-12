import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct
from scipy.io import loadmat


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

method = "cdf"

rawData = loadmat("../../data/data_mobile_indoor_1.mat")
CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
# stalking attack
CSIe2Orig = loadmat("../../data/data_static_indoor_1.mat")['A'][:, 0]
dataLen = min(len(CSIe2Orig), len(CSIa1Orig))

INT_SUCCESS = 5
INT_FAIL = 5

CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=3, window='flat')
CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=3, window='flat')

segLen = 1
keyLen = 128 * segLen

times = 0
overhead = 0

successCounts = 0
failCounts = 0

for staInd in range(0, dataLen, keyLen):
    endInd = staInd + keyLen
    # print("range:", staInd, endInd)
    if endInd >= len(CSIa1Orig):
        break
    times += 1

    origInd = np.array([xx for xx in range(staInd, endInd, 1)])

    CSIa1Epi = CSIa1Orig[origInd]
    CSIb1Epi = CSIb1Orig[origInd]

    CSIa1Orig[origInd] = CSIa1Epi
    CSIb1Orig[origInd] = CSIb1Epi

    np.random.seed(0)

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]

    tmpCSIa1 = normalize(tmpCSIa1)
    tmpCSIb1 = normalize(tmpCSIb1)

    alpha_STD = 0.5
    # 只能用原始的阈值量化，新方法产生的阈值不可用
    if times == 1:
        if method == "cdf":
            start_time = time.time()
            qa2_minus = np.mean(tmpCSIa1) - alpha_STD * np.std(tmpCSIa1)
            qa2_plus = cdf_thresholding(qa2_minus, np.std(tmpCSIa1))
            overhead += time.time() - start_time
            qb2_minus = np.mean(tmpCSIb1) - alpha_STD * np.std(tmpCSIb1)
            qb2_plus = cdf_thresholding(qb2_minus, np.std(tmpCSIb1))
        elif method == "afd":
            start_time = time.time()
            qa2_minus = np.mean(tmpCSIa1) - alpha_STD * np.std(tmpCSIa1)
            qa2_plus = afd_thresholding(qa2_minus, np.std(tmpCSIa1))
            overhead += time.time() - start_time
            qb2_minus = np.mean(tmpCSIb1) - alpha_STD * np.std(tmpCSIb1)
            qb2_plus = afd_thresholding(qb2_minus, np.std(tmpCSIb1))
        else:
            qa2_minus = np.mean(tmpCSIa1) - alpha_STD * np.std(tmpCSIa1)
            qa2_plus = np.mean(tmpCSIa1) + alpha_STD * np.std(tmpCSIa1)
            qb2_minus = np.mean(tmpCSIb1) - alpha_STD * np.std(tmpCSIb1)
            qb2_plus = np.mean(tmpCSIb1) + alpha_STD * np.std(tmpCSIb1)

        delta = (qa2_plus - qa2_minus) / 10
        start_time = time.time()
        qa1_minus = qa2_minus + delta
        qa3_minus = qa2_minus - delta
        overhead += time.time() - start_time
        qb1_minus = qb2_minus + delta
        qb3_minus = qb2_minus - delta
        if method == "cdf":
            start_time = time.time()
            qa1_plus = cdf_thresholding(qa1_minus, np.std(tmpCSIa1))
            qa3_plus = cdf_thresholding(qa3_minus, np.std(tmpCSIa1))
            overhead += time.time() - start_time
            qb1_plus = cdf_thresholding(qb1_minus, np.std(tmpCSIb1))
            qb3_plus = cdf_thresholding(qb3_minus, np.std(tmpCSIb1))
        elif method == "afd":
            start_time = time.time()
            qa1_plus = afd_thresholding(qa1_minus, np.std(tmpCSIa1))
            qa3_plus = afd_thresholding(qa3_minus, np.std(tmpCSIa1))
            overhead += time.time() - start_time
            qb1_plus = afd_thresholding(qb1_minus, np.std(tmpCSIb1))
            qb3_plus = afd_thresholding(qb3_minus, np.std(tmpCSIb1))
        else:
            qa1_plus = qa2_plus + delta
            qa3_plus = qa2_plus - delta
            qb1_plus = qb2_plus + delta
            qb3_plus = qb2_plus - delta

    qa_minus = qa2_minus
    qa_plus = qa2_plus
    qb_minus = qb2_minus
    qb_plus = qb2_plus

    start_time = time.time()
    a_list = quantize(tmpCSIa1, qa_minus, qa_plus)
    overhead += time.time() - start_time
    b_list = quantize(tmpCSIb1, qb_minus, qb_plus)

    sum1 = min(len(a_list), len(b_list))
    sum2 = 0
    for i in range(0, sum1):
        sum2 += (a_list[i] == b_list[i])

    start_time = time.time()
    if sum2 == sum1:
        if qa_minus == qa1_minus and qa_plus == qa1_plus:
            successCounts += 1
            failCounts = 0
            if successCounts > INT_SUCCESS:
                qa2_minus = qa1_minus
                qb2_minus = qb1_minus

                qa1_minus = qa2_minus + delta
                qb1_minus = qb2_minus + delta

                qa3_minus = qa2_minus - delta
                qb3_minus = qb2_minus - delta

                if method == "cdf":
                    qa1_plus = cdf_thresholding(qa1_minus, np.std(tmpCSIa1))
                    qa3_plus = cdf_thresholding(qa3_minus, np.std(tmpCSIa1))
                    qb1_plus = cdf_thresholding(qb1_minus, np.std(tmpCSIb1))
                    qb3_plus = cdf_thresholding(qb3_minus, np.std(tmpCSIb1))
                elif method == "afd":
                    qa1_plus = afd_thresholding(qa1_minus, np.std(tmpCSIa1))
                    qa3_plus = afd_thresholding(qa3_minus, np.std(tmpCSIa1))
                    qb1_plus = afd_thresholding(qb1_minus, np.std(tmpCSIb1))
                    qb3_plus = afd_thresholding(qb3_minus, np.std(tmpCSIb1))
                else:
                    qa1_plus = qa2_plus + delta
                    qa3_plus = qa2_plus - delta
                    qb1_plus = qb2_plus + delta
                    qb3_plus = qb2_plus - delta
        elif qa_minus == qa2_minus and qa_plus == qa2_plus:
            successCounts = 0
            failCounts = 0
        else:
            failCounts += 1
            successCounts = 0

            if failCounts > INT_FAIL:
                qa2_minus = qa3_minus
                qb2_minus = qb3_minus

                qa1_minus = qa2_minus + delta
                qb1_minus = qb2_minus + delta

                qa3_minus = qa2_minus - delta
                qb3_minus = qb2_minus - delta

                if method == "cdf":
                    qa1_plus = cdf_thresholding(qa1_minus, np.std(tmpCSIa1))
                    qa3_plus = cdf_thresholding(qa3_minus, np.std(tmpCSIa1))
                    qb1_plus = cdf_thresholding(qb1_minus, np.std(tmpCSIb1))
                    qb3_plus = cdf_thresholding(qb3_minus, np.std(tmpCSIb1))
                elif method == "afd":
                    qa1_plus = afd_thresholding(qa1_minus, np.std(tmpCSIa1))
                    qa3_plus = afd_thresholding(qa3_minus, np.std(tmpCSIa1))
                    qb1_plus = afd_thresholding(qb1_minus, np.std(tmpCSIb1))
                    qb3_plus = afd_thresholding(qb3_minus, np.std(tmpCSIb1))
                else:
                    qa1_plus = qa2_plus + delta
                    qa3_plus = qa2_plus - delta
                    qb1_plus = qb2_plus + delta
                    qb3_plus = qb2_plus - delta
    else:
        failCounts += 1
        successCounts = 0

        if failCounts > INT_FAIL:
            qa2_minus = qa3_minus
            qb2_minus = qb3_minus

            qa1_minus = qa2_minus + delta
            qb1_minus = qb2_minus + delta

            qa3_minus = qa2_minus - delta
            qb3_minus = qb2_minus - delta

            if method == "cdf":
                qa1_plus = cdf_thresholding(qa1_minus, np.std(tmpCSIa1))
                qa3_plus = cdf_thresholding(qa3_minus, np.std(tmpCSIa1))
                qb1_plus = cdf_thresholding(qb1_minus, np.std(tmpCSIb1))
                qb3_plus = cdf_thresholding(qb3_minus, np.std(tmpCSIb1))
            elif method == "afd":
                qa1_plus = afd_thresholding(qa1_minus, np.std(tmpCSIa1))
                qa3_plus = afd_thresholding(qa3_minus, np.std(tmpCSIa1))
                qb1_plus = afd_thresholding(qb1_minus, np.std(tmpCSIb1))
                qb3_plus = afd_thresholding(qb3_minus, np.std(tmpCSIb1))
            else:
                qa1_plus = qa2_plus + delta
                qa3_plus = qa2_plus - delta
                qb1_plus = qb2_plus + delta
                qb3_plus = qb2_plus - delta

    overhead += time.time() - start_time

print("times", times)
print("overhead", round(overhead / times, 9) * 1000, "ms")