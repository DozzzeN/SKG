import math
import random

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


def genRandomStep(len, lowBound, highBound):
    length = 0
    randomStep = []
    # 少于三则无法分，因为至少要划分出一个三角形
    while len - length >= lowBound:
        step = random.randint(lowBound, highBound)
        randomStep.append(step)
        length += step
    return randomStep


# 将三维数组转为一维数组
def toOneDim(list):
    oneDim = []
    for i in range(len(list)):
        tmp = 0
        for j in range(len(list[i])):
            tmp += (list[i][j][0] + list[i][j][1])
            # tmp += (list[i][j][0] * list[i][j][1])
        oneDim.append(round(tmp, 10))
    return oneDim


def genAlign(base, strB):
    align = []
    j = 0
    for i in range(len(base)):
        if base[i] == "=":
            align.append(strB[j])
            j += 1
        elif base[i] == "-":
            j += 1
    return align


def align(score, strA, strB):
    m = len(strA)
    n = len(strB)
    aux = [[]] * (m + 1)
    rule = [[]] * (m + 1)
    for i in range(len(aux)):
        aux[i] = [0] * (n + 1)
        rule[i] = [""] * (n + 1)
    rule[0][0] = ""
    for i in range(1, m + 1):
        rule[i][0] = rule[i - 1][0] + "-"
        aux[i][0] = aux[i - 1][0] + score["-"]
    for i in range(1, n + 1):
        rule[0][i] = rule[0][i - 1] + "+"
        aux[0][i] = aux[0][i - 1] + score["+"]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if strA[i - 1] == strB[j - 1]:
                aux[i][j] = aux[i - 1][j - 1] + score["="]
                rule[i][j] = rule[i - 1][j - 1] + "="
            else:
                aux[i][j] = max(aux[i - 1][j] + score["-"], aux[i][j - 1] + score["+"], aux[i - 1][j - 1] + score["~"])
                if aux[i][j] == aux[i - 1][j - 1] + score["~"]:
                    rule[i][j] = rule[i - 1][j - 1] + "~"
                elif aux[i][j] == aux[i - 1][j] + score["-"]:
                    rule[i][j] = rule[i - 1][j] + "-"
                elif aux[i][j] == aux[i][j - 1] + score["+"]:
                    rule[i][j] = rule[i][j - 1] + "+"
    return rule[m][n]


rawData = loadmat('../data/data_mobile_indoor_1.mat')

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

CSIa1Orig = np.array(CSIa1Orig)
CSIb1Orig = np.array(CSIb1Orig)

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
keyLen = 128
addNoise = True
opNums = 16
rule = {"=": 3, "+": 1, "-": 1, "~": 0}

originSum = 0
correctSum = 0
randomSum = 0
noiseSum = 0

codings = ""
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
        # tmpCSIa1 = tmpPulse * np.float_power(np.abs(tmpCSIa1), tmpNoise)
        # tmpCSIb1 = tmpPulse * np.float_power(np.abs(tmpCSIb1), tmpNoise)
        # tmpCSIe1 = tmpPulse * np.float_power(np.abs(tmpCSIe1), tmpNoise)
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

    # sortCSIa1是原始算法中排序前的数据
    # sortCSIa1 = np.log10(np.abs(sortCSIa1))
    # sortCSIb1 = np.log10(np.abs(sortCSIb1))
    # sortCSIe1 = np.log10(np.abs(sortCSIe1))
    # sortNoise = np.log10(np.abs(sortNoise))

    # 形成三维数组，其中第三维是一对坐标值
    # 数组的长度由param调节
    param = 2
    step = int(math.pow(2, param))

    sortCSIa1 = np.argsort(sortCSIa1)
    sortCSIb1 = np.argsort(sortCSIb1)
    sortCSIe1 = np.argsort(sortCSIe1)
    sortNoise = np.argsort(sortNoise)

    sortCSIa1Copy = sortCSIa1.copy()
    sortCSIb1Copy = sortCSIb1.copy()
    sortCSIe1Copy = sortCSIe1.copy()
    sortCSIn1Copy = sortNoise.copy()

    sortCSIa1Copy = sortCSIa1Copy.reshape(int(len(sortCSIa1) / 2), 2)
    sortCSIb1Copy = sortCSIb1Copy.reshape(int(len(sortCSIb1) / 2), 2)
    sortCSIe1Copy = sortCSIe1Copy.reshape(int(len(sortCSIe1) / 2), 2)
    sortCSIn1Copy = sortCSIn1Copy.reshape(int(len(sortNoise) / 2), 2)

    sortCSIa1Split = []
    sortCSIb1Split = []
    sortCSIe1Split = []
    sortNoiseSplit = []

    randomStep = genRandomStep(int(len(sortCSIa1) / 2), step, step)
    print("randomStep", randomStep)
    startIndex = 0

    for i in range(len(randomStep)):
        # 由于随机产生step的算法不一定刚好满足step之和等于keyLen/2，故在每次复制值的时候需要判断
        if startIndex >= len(sortCSIa1Copy) or len(sortCSIa1Copy) - startIndex < 3:
            break
        sortCSIa1Split.append(sortCSIa1Copy[startIndex:startIndex + randomStep[i]])
        sortCSIb1Split.append(sortCSIb1Copy[startIndex:startIndex + randomStep[i]])
        sortCSIe1Split.append(sortCSIe1Copy[startIndex:startIndex + randomStep[i]])
        sortNoiseSplit.append(sortCSIn1Copy[startIndex:startIndex + randomStep[i]])
        startIndex = startIndex + randomStep[i]

    sortCSIa1 = sortCSIa1Split
    sortCSIb1 = sortCSIb1Split
    sortCSIe1 = sortCSIe1Split
    sortNoise = sortNoiseSplit

    CSIa1Back = toOneDim(sortCSIa1)
    CSIb1Back = toOneDim(sortCSIb1)
    CSIe1Back = toOneDim(sortCSIe1)
    CSIn1Back = toOneDim(sortNoise)

    # 最后各自的密钥
    a_list = []
    b_list = []
    e_list = []
    n_list = []

    strA = np.argsort(CSIa1Back)
    strB = np.argsort(CSIb1Back)
    strE = np.argsort(CSIe1Back)
    strN = np.argsort(CSIn1Back)
    print("strA", strA)
    print("strB", strB)
    print("strE", strE)
    print("strN", strN)

    charA = []
    charB = []
    charE = []
    charN = []
    for i in range(len(strA)):
        charA.append(chr(65 + int(strA[i])))
        charB.append(chr(65 + int(strB[i])))
        charE.append(chr(65 + int(strE[i])))
        charN.append(chr(65 + int(strN[i])))
    chrA = ''.join(charA)
    chrB = ''.join(charB)
    chrE = ''.join(charE)
    chrN = ''.join(charN)

    chrPA = chrA
    chrPB = chrB
    maxChrA = 'A'
    maxChrB = 'A'
    for i in range(len(chrPA)):
        maxChrA = max(maxChrA, chrPA[i])
    for i in range(len(chrPB)):
        maxChrB = max(maxChrB, chrPB[i])

    arrayIndexA = []
    allOpsIndexA = []
    arrayIndexB = []
    allOpsIndexB = []
    for i in range(opNums):
        arrayIndexA.append(random.randint(0, len(chrPA) - 1))
        arrayIndexB.append(random.randint(0, len(chrPB) - 1))
        flag = random.randint(0, 2)
        if flag == 0:
            allOpsIndexA.append(-1)
            allOpsIndexB.append(-1)
        elif flag == 1:
            allOpsIndexA.append(-2)
            allOpsIndexB.append(-2)
        else:
            allOpsIndexA.append(random.randint(65, ord(maxChrA)))
            allOpsIndexB.append(random.randint(65, ord(maxChrB)))
    # B的随机操作行为与A的不同
    # for i in range(opNums):
    #     arrayIndexB.append(random.randint(0, len(chrPB) - 1))
    #     flag = random.randint(0, 2)
    #     if flag == 0:
    #         allOpsIndexB.append(-1)
    #     elif flag == 1:
    #         allOpsIndexB.append(-2)
    #     else:
    #         allOpsIndexB.append(random.randint(65, ord(maxChrB)))
    random.shuffle(arrayIndexA)
    random.shuffle(arrayIndexB)
    for i in range(opNums):
        if allOpsIndexA[i] == -1:
            chrPList = list(chrPA)[:arrayIndexA[i]] + list(chrPA)[arrayIndexA[i] + 1:]
            chrP = ''.join(chrPList)
        elif allOpsIndexA[i] == -2:
            l = list(chrPA)
            if arrayIndexA[i] + 2 >= len(l):
                continue
            chrPList = l[:arrayIndexA[i]] + l[arrayIndexA[i] + 1:arrayIndexA[i] + 2] + \
                       l[arrayIndexA[i]:arrayIndexA[i] + 1] + l[min(len(l) - 1, arrayIndexA[i] + 2):]
            chrPA = ''.join(chrPList)
        else:
            chrPList = list(chrPA)
            chrPList.insert(arrayIndexA[i], chr(allOpsIndexA[i]))
            chrPA = ''.join(chrPList)

        if allOpsIndexB[i] == -1:
            chrPList = list(chrPB)[:arrayIndexB[i]] + list(chrPB)[arrayIndexB[i] + 1:]
            chrP = ''.join(chrPList)
        elif allOpsIndexB[i] == -2:
            l = list(chrPB)
            if arrayIndexB[i] + 2 >= len(l):
                continue
            chrPList = l[:arrayIndexB[i]] + l[arrayIndexB[i] + 1:arrayIndexB[i] + 2] + \
                       l[arrayIndexB[i]:arrayIndexB[i] + 1] + l[min(len(l) - 1, arrayIndexB[i] + 2):]
            chrPB = ''.join(chrPList)
        else:
            chrPList = list(chrPB)
            chrPList.insert(arrayIndexB[i], chr(allOpsIndexB[i]))
            chrPB = ''.join(chrPList)

    print("chrPA", chrPA)
    print("chrPB", chrPB)
    print("chrA", chrA)
    print("chrB", chrB)
    print("chrE", chrE)
    print("chrN", chrN)

    # 用A匹配P
    ruleStr1 = align(rule, chrA, chrPA)
    # ruleStr1 = align(rule, chrA, chrPB)
    alignStr1 = genAlign(ruleStr1, chrA)
    # 用B匹配P
    ruleStr2 = align(rule, chrB, chrPA)
    alignStr2 = genAlign(ruleStr2, chrB)
    # 用E匹配P
    ruleStr3 = align(rule, chrE, chrPA)
    alignStr3 = genAlign(ruleStr3, chrE)
    # 用N匹配P
    ruleStr4 = align(rule, chrN, chrPA)
    alignStr4 = genAlign(ruleStr4, chrN)

    a_list = alignStr1
    b_list = alignStr2
    e_list = alignStr3
    n_list = alignStr4

    print("keys of a:", a_list)
    print("keys of b:", b_list)
    print("keys of e:", e_list)
    print("keys of n:", n_list)

    sum1 = min(len(a_list), len(b_list))
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for i in range(0, sum1):
        sum2 += (a_list[i] == b_list[i])
    for i in range(min(len(a_list), len(e_list))):
        sum3 += (a_list[i] == e_list[i])
    for i in range(min(len(a_list), len(n_list))):
        sum4 += (a_list[i] == n_list[i])

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

with open('./evaluations/key.txt', 'a', ) as f:
    f.write(codings)

print("a-b all", correctSum, "/", originSum, "=", correctSum / originSum)
print("a-e all", randomSum, "/", originSum, "=", randomSum / originSum)
print("a-n all", noiseSum, "/", originSum, "=", noiseSum / originSum)
