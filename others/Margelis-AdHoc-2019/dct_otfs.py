import time
from tkinter import messagebox

import matplotlib.pyplot as plt
import numpy as np
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


fileNames = ["../../data/otfs/codedOFDM.mat",
             "../../data/otfs/codedOTFS.mat",
             "../../data/otfs/uncodedOFDM.mat",
             "../../data/otfs/uncodedOTFS.mat"
             ]

bmr = []

start = time.time()
for f in fileNames:
    rawData = loadmat(f)
    print(f)

    # Range of energy/bit to noise power ratio
    EbNo = int(len(rawData[f[f.find("otfs/") + 5: f.find(".mat")]]) / 2)

    bmr_temp = []

    for snr in range(EbNo):
        print("EbNo:", snr, 2 * snr, 2 * snr + 1)
        CSIa1Orig = rawData[f[f.find("otfs/") + 5: f.find(".mat")]][2 * snr][:2000]
        CSIb1Orig = rawData[f[f.find("otfs/") + 5: f.find(".mat")]][2 * snr + 1][:2000]

        print("bit error rate", (CSIa1Orig != CSIb1Orig).sum() / len(CSIa1Orig), (CSIa1Orig != CSIb1Orig).sum(), len(CSIa1Orig))

        dataLen = len(CSIa1Orig)

        # CSIa1Orig = smooth(np.array(CSIa1Orig), window_len=30, window='flat')
        # CSIb1Orig = smooth(np.array(CSIb1Orig), window_len=30, window='flat')

        segLen = 7
        keyLen = 256 * segLen

        originSum = 0
        correctSum = 0
        randomSum1 = 0
        noiseSum1 = 0

        originDecSum = 0
        correctDecSum = 0
        randomDecSum1 = 0
        noiseDecSum1 = 0

        originWholeSum = 0
        correctWholeSum = 0
        randomWholeSum1 = 0
        noiseWholeSum1 = 0

        times = 0

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

            # imitation attack
            CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=len(CSIa1Orig))

            tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
            tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
            tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]

            # inference attack
            tmpCSIn1 = np.random.random(keyLen)
            # tmpCSIa1 = tmpCSIa1 - np.mean(tmpCSIa1)
            # tmpCSIb1 = tmpCSIb1 - np.mean(tmpCSIb1)
            # tmpCSIe1 = tmpCSIe1 - np.mean(tmpCSIe1)
            # tmpCSIn1 = tmpCSIn1 - np.mean(tmpCSIn1)

            tmpCSIa1 = smooth(np.array(tmpCSIa1), window_len=3, window='flat')
            tmpCSIb1 = smooth(np.array(tmpCSIb1), window_len=3, window='flat')

            dctCSIa1 = dct(tmpCSIa1, n=int(len(tmpCSIa1) / 2))
            dctCSIb1 = dct(tmpCSIb1, n=int(len(tmpCSIb1) / 2))
            dctCSIe1 = dct(tmpCSIe1, n=int(len(tmpCSIe1) / 2))
            dctCSIn1 = dct(tmpCSIn1, n=int(len(tmpCSIn1) / 2))

            # dctCSIa1 = dct(tmpCSIa1)
            # dctCSIb1 = dct(tmpCSIb1)
            # dctCSIe1 = dct(tmpCSIe1)
            # dctCSIe2 = dct(tmpCSIe2)
            # dctCSIn1 = dct(tmpCSIn1)

            mean_a = np.mean(dctCSIa1)
            mean_b = np.mean(dctCSIb1)
            mean_e1 = np.mean(dctCSIe1)
            mean_n1 = np.mean(dctCSIn1)

            std_a = np.std(dctCSIa1)
            std_b = np.std(dctCSIb1)
            std_e1 = np.std(dctCSIe1)
            std_n1 = np.std(dctCSIn1)

            a_list = []
            b_list = []
            e1_list = []
            n1_list = []

            a_list_number = []
            b_list_number = []
            e1_list_number = []
            n1_list_number = []

            for i in range(len(dctCSIe1)):
                if dctCSIa1[i] > mean_a + std_a:
                    a_list_number.append(3)
                elif dctCSIa1[i] <= mean_a + std_a and dctCSIa1[i] > mean_a:
                    a_list_number.append(2)
                elif dctCSIa1[i] <= mean_a and dctCSIa1[i] > mean_a - std_a:
                    a_list_number.append(1)
                elif dctCSIa1[i] <= mean_a - std_a:
                    a_list_number.append(0)

            for i in range(len(dctCSIb1)):
                if dctCSIb1[i] > mean_b + std_b:
                    b_list_number.append(3)
                elif dctCSIb1[i] <= mean_b + std_b and dctCSIb1[i] > mean_b:
                    b_list_number.append(2)
                elif dctCSIb1[i] <= mean_b and dctCSIb1[i] > mean_b - std_b:
                    b_list_number.append(1)
                elif dctCSIb1[i] <= mean_b - std_b:
                    b_list_number.append(0)

            for i in range(len(dctCSIe1)):
                if dctCSIe1[i] > mean_e1 + std_e1:
                    e1_list_number.append(3)
                elif dctCSIe1[i] <= mean_e1 + std_e1 and dctCSIe1[i] > mean_e1:
                    e1_list_number.append(2)
                elif dctCSIe1[i] <= mean_e1 and dctCSIe1[i] > mean_e1 - std_e1:
                    e1_list_number.append(1)
                elif dctCSIe1[i] <= mean_e1 - std_e1:
                    e1_list_number.append(0)

            for i in range(len(dctCSIn1)):
                if dctCSIn1[i] >= mean_n1 + std_n1:
                    n1_list_number.append(3)
                elif dctCSIn1[i] < mean_n1 + std_n1 and dctCSIn1[i] >= mean_n1:
                    n1_list_number.append(2)
                elif dctCSIn1[i] < mean_n1 and dctCSIn1[i] > mean_n1 - std_n1:
                    n1_list_number.append(1)
                elif dctCSIn1[i] <= mean_n1 - std_n1:
                    n1_list_number.append(0)

            # 转成二进制，0填充成00
            for i in range(len(a_list_number)):
                number = bin(a_list_number[i])[2:].zfill(2)
                a_list += number
            for i in range(len(b_list_number)):
                number = bin(b_list_number[i])[2:].zfill(2)
                b_list += number
            for i in range(len(e1_list_number)):
                number = bin(e1_list_number[i])[2:].zfill(2)
                e1_list += number
            for i in range(len(n1_list_number)):
                number = bin(n1_list_number[i])[2:].zfill(2)
                n1_list += number

            # 对齐密钥，随机补全
            for i in range(len(a_list) - len(e1_list)):
                e1_list += str(np.random.randint(0, 2))
            for i in range(len(a_list) - len(n1_list)):
                n1_list += str(np.random.randint(0, 2))

            sum1 = min(len(a_list), len(b_list))
            sum2 = 0
            sum31 = 0
            sum41 = 0
            for i in range(0, sum1):
                sum2 += (a_list[i] == b_list[i])
            for i in range(min(len(a_list), len(e1_list))):
                sum31 += (a_list[i] == e1_list[i])
            for i in range(min(len(a_list), len(n1_list))):
                sum41 += (a_list[i] == n1_list[i])

            originSum += sum1
            correctSum += sum2
            randomSum1 += sum31
            noiseSum1 += sum41

            originWholeSum += 1
            correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
            randomWholeSum1 = randomWholeSum1 + 1 if sum31 == sum1 else randomWholeSum1
            noiseWholeSum1 = noiseWholeSum1 + 1 if sum41 == sum1 else noiseWholeSum1

        bmr_temp.append([snr, correctSum / originSum])

        print("\033[0;32;40ma-b bit agreement rate", correctSum, "/", originSum, "=", round(correctSum / originSum, 10),
              "\033[0m")
        print("a-e1 bit agreement rate", randomSum1, "/", originSum, "=", round(randomSum1 / originSum, 10))
        print("a-n1 bit agreement rate", noiseSum1, "/", originSum, "=", round(noiseSum1 / originSum, 10))
        print("\033[0;32;40ma-b key agreement rate", correctWholeSum, "/", originWholeSum, "=",
              round(correctWholeSum / originWholeSum, 10), "\033[0m")
        print("a-e1 key agreement rate", randomWholeSum1, "/", originWholeSum, "=",
              round(randomWholeSum1 / originWholeSum, 10))
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
        print(round(correctSum / originSum, 10), round(correctWholeSum / originWholeSum, 10), originSum / len(CSIa1Orig),
              correctSum / len(CSIa1Orig))

        print()

    bmr.append(bmr_temp)

plt.figure()
for i in range(len(bmr)):
    plt.plot([x[0] for x in bmr[i]], [x[1] for x in bmr[i]], label=fileNames[i]
    [fileNames[i].find("otfs/") + 5: fileNames[i].find(".mat")])
plt.legend()
plt.savefig("dct_ofdm_otfs.png", dpi=1200, bbox_inches='tight')
plt.show()
print("time:", time.time() - start, "s")
messagebox.showinfo("提示", "测试结束")
