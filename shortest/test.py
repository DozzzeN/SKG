# from mwmatching import maxWeightMatching
import math
import time

import dijkstra as dj
import numpy as np
from numpy.random import exponential as Exp
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


# 输入数组[a, b, c]，计算a+b+c + a+b + a
def sumSeries(CSITmp):
    if len(CSITmp) > 1:
        sumCSI = sum(CSITmp) + sumSeries(CSITmp[0:-1])
        return sumCSI
    else:
        return CSITmp[0]


def calculate(CSITmp):
    anotherCSITmp = []
    CSIPivot = np.min(CSITmp)
    CSIMean = np.mean(CSITmp)
    for i in range(len(CSITmp)):
        anotherCSITmp.append(np.abs(CSITmp[i] - CSIPivot))
    return anotherCSITmp


def CSIHash(CSITmp):
    anotherCSITmp = []
    for i in range(len(CSITmp)):
        anotherCSITmp.append(hash(str(CSITmp[i])))
    return anotherCSITmp


def CSIMod(CSITmp):
    anotherCSITmp = []
    for i in range(len(CSITmp)):
        temp1 = math.floor(10000 * np.abs(CSITmp[i]))
        anotherCSITmp.append(math.floor(temp1 / 10000))
    return anotherCSITmp


def CSIMean(CSITmp):
    anotherCSITmp = []
    CSIMin = np.min(CSITmp)
    for i in range(len(CSITmp)):
        anotherCSITmp.append(np.abs(CSITmp[i] + CSITmp[i] / CSIMin))
    return anotherCSITmp


def CSIInc(CSITmp):
    anotherCSITmp = []
    for i in range(len(CSITmp)):
        anotherCSITmp.append(-70)
    return anotherCSITmp


# plt.close('all')
# np.random.seed(0)

rawData = loadmat('../data/data_mobile_outdoor_1.mat')
# print(rawData['A'])
# 取rawData['A']的第一个元素
# print(rawData['A'][:, 0])
# print(len(rawData['A'][:, 0]))  # 6745

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
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

# for step in range(10, 320, 10):
# for step in range(1, 10, 1):
sft = 2
intvl = 2 * sft + 1
keyLen = 128

sum1 = 0
sum2 = 0
sum3 = 0
sum4 = 0

# for ii in range(0, 5):
for staInd in range(0, 10 * intvl + 1, intvl):
    endInd = staInd + keyLen * intvl

    CSIa1Orig = CSIa1OrigBack.copy()
    CSIb1Orig = CSIb1OrigBack.copy()
    CSIe1Orig = CSIe1OrigBack.copy()

    tmpCSIa1 = CSIa1Orig[range(staInd, endInd, 1)]
    tmpCSIb1 = CSIb1Orig[range(staInd, endInd, 1)]
    tmpCSIe1 = CSIe1Orig[range(staInd, endInd, 1)]

    tmpNoise = noise[range(staInd, endInd, 1)]
    tmpNoiseAdd = noiseAdd[range(staInd, endInd, 1)]

    tmpCSIa1 = tmpCSIa1 - (np.mean(tmpCSIa1) - np.mean(tmpCSIb1))  # Mean value consistency

    # tmpPulse = np.sin(np.linspace(0, np.pi * 0.5 * keyLen, keyLen * intvl))  ## Sine pulse
    # linspace函数生成元素为50的等间隔数列，可以指定第三个参数为元素个数
    # signal.square返回周期性的方波波形
    tmpPulse = signal.square(
        2 * np.pi * 1 / intvl * np.linspace(0, np.pi * 0.5 * keyLen, keyLen * intvl))  ## Rectangular pulse

    # np.std计算标准差，需要参数ddof=1
    # tmpNoise = np.random.normal(loc=-1 * np.std(tmpCSIa1, ddof=1), scale=np.std(tmpCSIa1, ddof=1), size=endInd - staInd)
    # tmpNoiseAdd = np.random.normal(loc=0, scale=10, size=endInd - staInd)

    # tmpCSIa1 = (tmpCSIa1 + tmpNoiseAdd)  ## Method 1: addition
    # tmpCSIb1 = (tmpCSIb1 + tmpNoiseAdd)
    # tmpCSIe1 = (tmpCSIe1 + tmpNoiseAdd)

    # tmpCSIa1 = tmpCSIa1 * tmpNoise + tmpNoiseAdd  ## Method 2: polynomial product better than addition
    # tmpCSIb1 = tmpCSIb1 * tmpNoise + tmpNoiseAdd
    # tmpCSIe1 = tmpCSIe1 * tmpNoise + tmpNoiseAdd

    # tmpCSIa1 = tmpPulse * np.float_power(np.abs(tmpCSIa1), tmpNoise)  ## Method 3: Power better than polynomial
    # tmpCSIb1 = tmpPulse * np.float_power(np.abs(tmpCSIb1), tmpNoise)
    # tmpCSIe1 = tmpPulse * np.float_power(np.abs(tmpCSIe1), tmpNoise)

    # tmpCSIa1 = tmpPulse * np.float_power(np.abs(np.mean(tmpCSIa1)), tmpNoise)  ## Method 4: Statistical Characteristic
    # tmpCSIb1 = tmpPulse * np.float_power(np.abs(np.mean(tmpCSIb1)), tmpNoise)

    # tmpCSIa1 = tmpPulse * (np.abs(np.mean(tmpCSIa1)) + tmpNoise)  ## Method 5: Statistical Characteristic
    # tmpCSIb1 = tmpPulse * (np.abs(np.mean(tmpCSIb1)) + tmpNoise)

    # tmpCSIa1 = tmpPulse * np.float_power(np.abs(tmpNoise), tmpCSIa1)  ## Method 6
    # tmpCSIb1 = tmpPulse * np.float_power(np.abs(tmpNoise), tmpCSIb1)
    # tmpCSIe1 = tmpPulse * np.float_power(np.abs(tmpNoise), tmpCSIe1)

    # tmpCSIa1Cal = CISMean(tmpCSIa1)
    # tmpCSIb1Cal = CISMean(tmpCSIb1)
    # tmpCSIe1Cal = CISMean(tmpCSIe1)

    # tmpCSIa1 = tmpPulse * tmpCSIa1 * np.float_power(step, np.abs(tmpNoise))
    # tmpCSIb1 = tmpPulse * tmpCSIb1 * np.float_power(step, np.abs(tmpNoise))
    # tmpCSIe1 = tmpPulse * tmpCSIe1 * np.float_power(step, np.abs(tmpNoise))

    # tmpCSIa1 = tmpPulse * tmpCSIa1Cal * tmpNoise
    # tmpCSIb1 = tmpPulse * tmpCSIb1Cal * tmpNoise
    # tmpCSIe1 = tmpPulse * tmpCSIe1Cal * tmpNoise

    # print("mean of a", np.abs(np.mean(tmpCSIa1)))
    # print("mean of b", np.abs(np.mean(tmpCSIb1)))
    #
    # print("median of a", np.abs(np.median(tmpCSIa1)))
    # print("median of b", np.abs(np.median(tmpCSIb1)))

    # part = step

    # def CSIPartialMean(CSITmp):
    #     anotherCSITmp = []
    #     for i in range(len(CSITmp)):
    #         partialSum = 0
    #         for j in range(0, part):
    #             partialSum += (CSITmp[(i + j) % len(CSITmp)])
    #         anotherCSITmp.append(partialSum / part)
    #     return anotherCSITmp
    #
    #
    # tmpCSIa1 = CSIPartialMean(tmpCSIa1)
    # tmpCSIb1 = CSIPartialMean(tmpCSIb1)
    # tmpCSIe1 = CSIPartialMean(tmpCSIe1)

    # tmpCSIa1 = tmpPulse * np.float_power(np.abs(tmpCSIa1), tmpNoise)
    # tmpCSIb1 = tmpPulse * np.float_power(np.abs(tmpCSIb1), tmpNoise)
    # tmpCSIe1 = tmpPulse * np.float_power(np.abs(tmpCSIe1), tmpNoise)

    tmpCSIa1 = tmpPulse * (np.float_power(np.abs(tmpCSIa1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIa1))
    tmpCSIb1 = tmpPulse * (np.float_power(np.abs(tmpCSIb1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIb1))
    tmpCSIe1 = tmpPulse * (np.float_power(np.abs(tmpCSIe1), tmpNoise) * np.float_power(np.abs(tmpNoise), tmpCSIe1))

    CSIa1Orig[range(staInd, endInd, 1)] = tmpCSIa1
    CSIb1Orig[range(staInd, endInd, 1)] = tmpCSIb1
    CSIe1Orig[range(staInd, endInd, 1)] = tmpCSIe1

    permLen = len(range(staInd, endInd, intvl))
    origInd = np.array([xx for xx in range(staInd, endInd, intvl)])

    start_time = time.time()

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

            # sortCSIa1[ii] = sumSeries(CSIa1Tmp)  ## Metric 2: Sum
            # sortCSIb1[jj - permLen] = sumSeries(CSIb1Tmp)
            # sortCSIe1[jj - permLen] = sumSeries(CSIe1Tmp)
            # sortNoise[ii] = sumSeries(noiseTmp)

            # sortCSIa1[ii] = msdc(CSIa1Tmp)  ## Metric 3: tsfresh.msdc:  the metrics, msdc and mc, seem better
            # sortCSIb1[jj - permLen] = msdc(CSIb1Tmp)
            # sortCSIe1[jj - permLen] = msdc(CSIe1Tmp)
            # sortNoise[ii] = msdc(noiseTmp)

            # sortCSIa1[ii] = mc(CSIa1Tmp)  ## Metric 4: tsfresh.mc
            # sortCSIb1[jj - permLen] = mc(CSIb1Tmp)
            # sortCSIe1[jj - permLen] = mc(CSIe1Tmp)
            # sortNoise[ii] = mc(noiseTmp)

            # sortCSIa1[ii] = cid(CSIa1Tmp, True)  ## Metric 5: tsfresh.cid_ie
            # sortCSIb1[jj - permLen] = cid(CSIb1Tmp, True)
            # sortCSIe1[jj - permLen] = cid(CSIe1Tmp, True)
            # sortNoise[ii] = cid(noiseTmp, True)


    ## Matching outcomes
    # np.argsort函数是将x中的元素从小到大排列，提取其对应的索引，然后输出到y
    # sortInda = np.argsort(sortCSIa1)
    # sortIndb = np.argsort(sortCSIb1)
    # sortInde = np.argsort(sortCSIe1)
    # sortIndn = np.argsort(sortNoise)

    # episode中的点除去均值，然后计算几何平均（算术平均）
    def CSIEpisode(CSITmp, step):
        episode = []
        for i in range(0, len(CSITmp), step):
            sum = 1
            for j in range(0, step):
                if i + j >= len(CSITmp):
                    break
                sum *= (CSITmp[i + j] / np.mean(CSITmp))
            episode.append(math.pow(np.abs(sum), 1 / step))
        return episode


    # episode中的点计算算术平均
    def CSIEpisode2(CSITmp, step):
        episode = []
        for i in range(0, len(CSITmp), step):
            sum = 1
            for j in range(0, step):
                if i + j >= len(CSITmp):
                    break
                sum += (CSITmp[i + j])
            episode.append(np.abs(sum) / step)
        return episode


    # episode中的点先进行求差，然后求总和（平均值）
    def CSIEpisode3(CSITmp, step):
        episode = []
        for i in range(0, len(CSITmp), step):
            sum = 0
            for j in range(0, step):
                if i + j + step >= len(CSITmp):
                    break
                sum += round(np.abs((CSITmp[i + j + step] - CSITmp[i + j])), 8)
            episode.append(sum)
        return episode


    def getGraph(CSITmp):
        CSILen = len(CSITmp)
        vertexSet = []

        for i in range(0, CSILen):
            vertexSet.append(i)
        return vertexSet


    # 使用整个episode的处理结果进行构建图，边是episode的差的绝对值
    def constructGraph(graph, vertex, data):
        for i in range(len(vertex)):
            for j in range(len(vertex)):
                # print("i", data[vertex[i]], "---j", data[vertex[j]])
                # print(round(np.abs(np.log(data[vertex[i]])-np.log(data[vertex[j]])), 8))
                if data[vertex[i]] == 0 and data[vertex[j]] == 0:
                    graph.add_edge(vertex[i], vertex[j], 0)
                elif data[vertex[i]] != 0 and data[vertex[j]] == 0:
                    graph.add_edge(vertex[i], vertex[j],
                                   round(np.abs(np.log(data[vertex[i]])), 8))
                elif data[vertex[i]] == 0 and data[vertex[j]] != 0:
                    graph.add_edge(vertex[i], vertex[j],
                                   round(np.abs(np.log(data[vertex[j]])), 8))
                else:
                    graph.add_edge(vertex[i], vertex[j],
                                   round(np.abs(np.log(data[vertex[i]]) - np.log(data[vertex[j]])), 8))

        return graph


    def getPath(djGraph, vertex):
        path = {}
        for i in range(0, len(vertex)):
            path[vertex[i]] = djGraph.get_path(vertex[i])
        return path


    episodeCSIa1 = CSIEpisode3(sortCSIa1, 3)
    episodeCSIb1 = CSIEpisode3(sortCSIb1, 3)
    episodeCSIe1 = CSIEpisode3(sortCSIe1, 3)

    vertexCSIa1 = getGraph(episodeCSIa1)
    vertexCSIb1 = getGraph(episodeCSIb1)
    vertexCSIe1 = getGraph(episodeCSIe1)

    a1Graph = dj.Graph()
    b1Graph = dj.Graph()
    e1Graph = dj.Graph()

    a1Graph = constructGraph(a1Graph, vertexCSIa1, episodeCSIa1)
    b1Graph = constructGraph(b1Graph, vertexCSIb1, episodeCSIb1)
    e1Graph = constructGraph(e1Graph, vertexCSIe1, episodeCSIe1)

    djA1 = dj.DijkstraSPF(a1Graph, 0)
    djB1 = dj.DijkstraSPF(b1Graph, 0)
    djE1 = dj.DijkstraSPF(e1Graph, 0)

    pathA1 = getPath(djA1, vertexCSIa1)
    pathB1 = getPath(djB1, vertexCSIb1)
    pathE1 = getPath(djE1, vertexCSIe1)

    print("WEIGHT-A", a1Graph._Graph__edge_weights)
    print("WEIGHT-B", b1Graph._Graph__edge_weights)
    print("WEIGHT-E", e1Graph._Graph__edge_weights)

    print("PATH-A", pathA1)
    print("PATH-B", pathB1)
    print("PATH-E", pathE1)

    # print(" -> ".join('%s' % id for id in djA1.get_path(10)))
    # print(" -> ".join('%s' % id for id in djB1.get_path(10)))
    # print(" -> ".join('%s' % id for id in djE1.get_path(10)))
    #
    # for u in vertexCSIa1:
    #     print("%-5s %8d" % (u, dj.DijkstraSPF(a1Graph, 0).get_distance(u)))
    #
    # for u in vertexCSIb1:
    #     print("%-5s %8d" % (u, dj.DijkstraSPF(b1Graph, 0).get_distance(u)))
    #
    # for u in vertexCSIe1:
    #     print("%-5s %8d" % (u, dj.DijkstraSPF(e1Graph, 0).get_distance(u)))

    # print("----------------------")
    # print(permLen, (sortInda - sortIndb == 0).sum())  # sortInda和sortIndb对应位置相等的个数
    # print(permLen, (sortInda - sortIndn == 0).sum())
    # print(permLen, (sortIndb - sortIndn == 0).sum())
    # print(permLen, (sortIndb - sortInde == 0).sum())

    # sum1 += permLen
    # sum2 += (sortInda - sortIndb == 0).sum()
    # sum3 += (sortInda - sortIndn == 0).sum()
    # sum4 += (sortIndb - sortInde == 0).sum()

    # print(sortInda)
    # print(sortIndb)
    # print(sortIndn)

    # print("sortCSIa1", sortCSIa1)
    # print("sortCSIb1", sortCSIb1)

    # sortCSIa1Back = sortCSIa1.copy()
    # sortCSIb1Back = sortCSIb1.copy()
    #
    # sortCSIa1Back.sort()
    # sortCSIb1Back.sort()
    # print("sortCSIa1Back", sortCSIa1Back)
    # print("sortCSIb1Back", sortCSIb1Back)

    # diff = list(map(lambda x, y: x - y, sortInda, sortIndb))
    # print("diff", diff)
    #
    # for a in range(len(diff)):
    #     if diff[a] != 0:
    #         print("except", a)
    #         break

    # step = []
    #
    # for a in range(len(sortCSIa1Back) - 1):
    #     step.append(sortCSIa1Back[a + 1] - sortCSIa1Back[a])
    # print("step", step)
    #
    # step = []
    #
    # for a in range(len(sortCSIb1Back) - 1):
    #     step.append(sortCSIb1Back[a + 1] - sortCSIb1Back[a])
    # print("step", step)
# print("----------------------")
# print("step", step)
# print(sum2 / sum1)
# print(sum4 / sum1)
# print(sum3 / sum1)
