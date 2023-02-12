# from mwmatching import maxWeightMatching
import math
import time

import dijkstra as dj
import numpy as np
from numpy.random import exponential as Exp
from scipy import signal
from scipy.io import loadmat


# CSIEpisode求滑动平均值

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

hamRate = []
paraRate = []

correctRate = []
randomRate = []
noiseRate = []

totalHamDist = []
totalHamDiste = []
totalHamDistn = []

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
    sortInda = np.argsort(sortCSIa1)
    sortIndb = np.argsort(sortCSIb1)
    sortInde = np.argsort(sortCSIe1)
    sortIndn = np.argsort(sortNoise)


    # episode中的点先进行求差，然后求总和（平均值）
    def CSIEpisode1(CSITmp, step):
        episode = []
        for i in range(0, len(CSITmp), step):
            sum = 0
            for j in range(0, step):
                if i + j + step >= len(CSITmp):
                    break
                sum += round(np.abs((CSITmp[i + j + step] - CSITmp[i + j])), 8)
            episode.append(sum)
        return episode


    # episode中的点计算滑动算术平均
    def CSIEpisode2(CSITmp, step):
        episode = []
        for i in range(0, len(CSITmp), int(step / 2)):
            sum = 0
            for j in range(0, step):
                if i + j >= len(CSITmp):
                    break
                sum += (CSITmp[i + j])
            episode.append(round(np.abs(sum) / step, 8))
        return episode


    def getGraph(CSITmp):
        CSILen = len(CSITmp)
        vertexSet = []

        for i in range(0, CSILen):
            vertexSet.append(i)
        return vertexSet

        # 使用整个episode的处理结果进行构建图，边是episode的差的绝对值


    # 直接求差效果不好
    # def constructGraph(graph, vertex, data):
    #     for i in range(len(vertex)):
    #         for j in range(len(vertex)):
    #             # print("i", data[vertex[i]], "---j", data[vertex[j]])
    #             # print(round(np.abs(data[vertex[i]] - data[vertex[j]]), 8))
    #             graph.add_edge(vertex[i], vertex[j],
    #                            round(np.abs(data[vertex[i]] - data[vertex[j]]), 8))
    #     return graph

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


    # 分块
    episodeCSIa1 = CSIEpisode2(sortInda, 4)
    episodeCSIb1 = CSIEpisode2(sortIndb, 4)
    episodeCSIe1 = CSIEpisode2(sortInde, 4)
    episodeCSIn1 = CSIEpisode2(sortIndn, 4)

    # CSIEpisode1
    # step = 1 hamRate [6.3828125, 33.67116477272727, 51.75994318181818]
    # step = 2 hamRate [7.771306818181818, 27.755681818181817, 31.524147727272727]
    # step = 3 hamRate [6.961945031712474, 17.841437632135307, 20.818181818181817]
    # step = 5 hamRate [7.860139860139859, 12.758741258741258, 13.143356643356645]
    # step = 10 hamRate [5.125874125874125, 6.79020979020979, 6.944055944055944]

    # CSIEpisode2
    # step = 2 hamRate [10.372159090909092, 35.54190340909091, 59.38494318181818]
    # step = 4 hamRate [10.826704545454545, 45.86292613636363, 59.43110795454545]
    # step = 8 hamRate [16.553267045454547, 50.93821022727273, 61.87002840909091]

    # 不进行分块
    # episodeCSIa1 = sortInda
    # episodeCSIb1 = sortIndb
    # episodeCSIe1 = sortInde
    # episodeCSIn1 = sortIndn

    # [0.25, 1.8252840909090908, 42.72301136363637]

    # print("sortCSIa1", sortCSIa1)
    # print("sortCSIb1", sortCSIb1)
    # print("sortCSIe1", sortCSIe1)
    # print("sortNoise", sortNoise)
    #
    print("sortInda", sortInda)
    print("sortIndb", sortIndb)
    print("sortInde", sortInde)
    print("sortIndn", sortIndn)

    print("episodeCSIa1", episodeCSIa1)
    print("episodeCSIb1", episodeCSIb1)
    print("episodeCSIe1", episodeCSIe1)
    print("episodeCSIn1", episodeCSIn1)

    # hamDist = []
    # hamDiste = []
    # hamDistn = []
    # for kk in sortInda:
    #     indexa = np.where(sortInda == kk)[0][0]
    #     indexb = np.where(sortIndb == kk)[0][0]
    #     indexe = np.where(sortInde == kk)[0][0]
    #     indexn = np.where(sortIndn == kk)[0][0]
    #
    #     # sortInda和sortIndb1的对应索引元素距离：(1,2,3)和(1,2,3)返回(0,0,0)，(1,2,3)和(3,2,1)返回(2,0,2)
    #     hamDist.append(np.abs(indexa - indexb))
    #     hamDiste.append(np.abs(indexa - indexe))
    #     hamDistn.append(np.abs(indexa - indexn))
    #
    # print("hamDist", hamDist)
    # print("\033[0;32;40mhamDist_mean", np.mean(hamDist), "\033[0m")
    # print("hamDist_max", max(hamDist))
    # print("hamDiste", hamDiste)
    # print("\033[0;32;40mhamDiste_mean", np.mean(hamDiste), "\033[0m")
    # print("hamDiste_max", max(hamDiste))
    # print("hamDistn", hamDistn)
    # print("\033[0;32;40mhamDistn_mean", np.mean(hamDistn), "\033[0m")
    # print("hamDistn_max", max(hamDistn))

    hamDist = []
    hamDiste = []
    hamDistn = []
    for ll in episodeCSIa1:
        indexa = 0
        indexb = 0
        indexe = 0
        indexn = 0
        if len(np.where(episodeCSIa1 == ll)[0]) != 0:
            indexa = np.where(episodeCSIa1 == ll)[0][0]
        else:
            indexa = len(episodeCSIa1)
        if len(np.where(episodeCSIb1 == ll)[0]) != 0:
            indexb = np.where(episodeCSIb1 == ll)[0][0]
        else:
            indexb = len(episodeCSIa1)
        if len(np.where(episodeCSIe1 == ll)[0]) != 0:
            indexe = np.where(episodeCSIe1 == ll)[0][0]
        else:
            indexe = len(episodeCSIa1)
        if len(np.where(episodeCSIn1 == ll)[0]) != 0:
            indexn = np.where(episodeCSIn1 == ll)[0][0]
        else:
            indexn = len(episodeCSIa1)

        # sortInda和sortIndb1的对应索引元素距离：(1,2,3)和(1,2,3)返回(0,0,0)，(1,2,3)和(3,2,1)返回(2,0,2)
        hamDist.append(np.abs(indexa - indexb))
        hamDiste.append(np.abs(indexa - indexe))
        hamDistn.append(np.abs(indexa - indexn))

    print("hamDist", hamDist)
    print("\033[0;32;40mhamDist_mean", np.mean(hamDist), "\033[0m")
    print("hamDist_max", max(hamDist))
    print("hamDiste", hamDiste)
    print("\033[0;32;40mhamDiste_mean", np.mean(hamDiste), "\033[0m")
    print("hamDiste_max", max(hamDiste))
    print("hamDistn", hamDistn)
    print("\033[0;32;40mhamDistn_mean", np.mean(hamDistn), "\033[0m")
    print("hamDistn_max", max(hamDistn))

    hamDist = np.array(hamDist)
    hamDiste = np.array(hamDiste)
    hamDistn = np.array(hamDistn)

    totalHamDist.append(np.mean(hamDist))
    totalHamDiste.append(np.mean(hamDiste))
    totalHamDistn.append(np.mean(hamDistn))

    vertexCSIa1 = getGraph(episodeCSIa1)
    vertexCSIb1 = getGraph(episodeCSIb1)
    vertexCSIe1 = getGraph(episodeCSIe1)
    vertexCSIn1 = getGraph(episodeCSIn1)

    a1Graph = dj.Graph()
    b1Graph = dj.Graph()
    e1Graph = dj.Graph()
    n1Graph = dj.Graph()

    a1Graph = constructGraph(a1Graph, vertexCSIa1, sortInda)
    b1Graph = constructGraph(b1Graph, vertexCSIb1, sortIndb)
    e1Graph = constructGraph(e1Graph, vertexCSIe1, sortInde)
    n1Graph = constructGraph(n1Graph, vertexCSIn1, sortIndn)

    djA1 = dj.DijkstraSPF(a1Graph, 1)
    djB1 = dj.DijkstraSPF(b1Graph, 1)
    djE1 = dj.DijkstraSPF(e1Graph, 1)
    djN1 = dj.DijkstraSPF(n1Graph, 1)

    pathA1 = getPath(djA1, vertexCSIa1)
    pathB1 = getPath(djB1, vertexCSIb1)
    pathE1 = getPath(djE1, vertexCSIe1)
    pathN1 = getPath(djN1, vertexCSIn1)

    # print("WEIGHT-A", a1Graph._Graph__edge_weights)
    # print("WEIGHT-B", b1Graph._Graph__edge_weights)
    # print("WEIGHT-E", e1Graph._Graph__edge_weights)

    print("PATH-A", pathA1)
    print("PATH-B", pathB1)
    print("PATH-E", pathE1)
    print("PATH-N", pathN1)

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

    sum1 = len(pathA1)
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for i in range(0, sum1):
        sum2 += (len(pathA1[i]) - len(pathB1[i]) == 0)
        sum3 += (len(pathA1[i]) - len(pathE1[i]) == 0)
        sum4 += (len(pathA1[i]) - len(pathN1[i]) == 0)

    print(sum2 / sum1)
    print(sum3 / sum1)
    print(sum4 / sum1)
    print("----------------------")

    correctRate.append(sum2 / sum1)
    randomRate.append(sum3 / sum1)
    noiseRate.append(sum4 / sum1)

hamRate.append(sum(totalHamDist) / len(totalHamDist))  # 平均汉明距离
hamRate.append(sum(totalHamDiste) / len(totalHamDiste))
hamRate.append(sum(totalHamDistn) / len(totalHamDistn))

paraRate.append(sum(correctRate) / len(correctRate))
paraRate.append(sum(randomRate) / len(randomRate))
paraRate.append(sum(noiseRate) / len(noiseRate))

print("hamRate", hamRate)
print("paraRate", paraRate)
# sortCSIa1 [ 1.69158026e+046 -8.52511024e-004 -1.90920077e+034  3.26485671e+079
#  -3.78190803e+010 -3.68405556e+050  1.19696497e+050  1.08132325e+038
#  -5.26179743e+025 -2.16697913e+050  1.56639967e+005 -8.33889963e+054
#  -3.17970104e+013  4.39000268e+023 -4.76385956e+004 -9.01687162e+029
#   1.53963915e+027  2.36209833e+055 -1.00988976e+014  2.84052243e+013
#  -7.39535546e+024 -5.29708652e+025 -1.24939867e+019  1.40425933e+127
#  -9.98080136e+038 -4.59125081e+072  2.22089983e+046 -2.82311752e+018
#  -2.06820473e+033  1.10743542e+061  2.80211730e+017 -2.70011116e+041
#   3.47539815e+038 -8.70761702e+022 -6.37797325e+005  5.00272494e+037
#  -4.83331524e+116 -3.02465596e+034  1.11596079e+002 -7.39430045e+047
#  -1.14476270e-002  9.22685235e+059  1.74176021e+096 -1.67206823e+049
#   2.95301623e+039  2.08031829e+041 -7.88688553e+014 -9.22272876e+068
#   3.85323446e+018  1.18979457e+015 -6.43741811e+048  1.54115618e+010
#  -4.90337738e+045 -1.72228665e+083  8.56562949e+212 -4.15972794e-001
#  -2.64977025e+083  2.62980620e+044  3.55341706e+031 -9.56713473e+084
#  -1.37586148e+063  2.72466594e-008 -3.04284450e+019 -2.03167374e+033
#   4.45285125e+039 -2.11097932e+020 -5.70974975e+010  3.31811080e+075
#  -7.55533145e+014 -2.55180377e+040  2.97841295e+011  1.28459841e+023
#  -3.36876941e+019  1.24577143e+016  6.58686045e+055 -1.31043654e+056
#   1.28690007e-009  9.08688792e+002 -1.83662073e+068 -4.30248304e+013
#   5.22967574e+015 -3.92990747e+021 -7.23418155e+036  1.04667614e+038
#  -7.38052088e+033 -1.13885049e+004  7.37076047e+019 -4.54143087e+016
#  -1.12295213e+035  5.58125192e+068  3.33657498e+006 -4.22295297e+077
#   3.04622515e+069  7.88139990e+038 -2.57667855e+011 -1.53902080e+023
#   1.77339219e+086 -1.15434981e+052 -1.78033566e+071  5.72575344e+019
#  -3.47958113e+000 -8.40334892e+048  2.76485544e+092  1.02102189e+075
#  -2.10469662e+042  1.37468990e+067 -8.30792744e+052 -2.97540012e+065
#   3.45985596e+006  1.22246803e+035 -7.72751216e-009  4.59658261e+026
#   6.34445134e+049 -1.90222638e+035 -3.04918891e+051  5.82961425e+006
#   5.44544576e+057  1.76879784e+056  2.63964476e+012 -1.33334284e+064
#   2.81811078e+004  4.50890639e+072 -2.82487029e+109 -1.43386909e+028
#   1.38024430e+040  2.44602089e+039 -1.35691993e+014  1.24317941e+050]
# sortCSIb1 [ 5.11320538e+047 -8.20667496e-004 -1.81383023e+035  7.31414788e+078
#  -2.70988246e+010 -8.00638587e+049  3.45878718e+049  6.34950982e+037
#  -3.38560471e+025 -9.42653366e+049  1.48296945e+005 -3.74950881e+054
#  -2.29865457e+013  1.91614802e+023 -3.99930461e+004 -4.01736648e+029
#   8.58231060e+026  6.11095149e+054 -6.18411623e+013  1.58819388e+013
#  -2.70493109e+024 -2.00311390e+025 -4.51016019e+018  1.09031453e+123
#  -1.57155641e+038 -5.78561035e+071  1.76143076e+046 -4.15312869e+018
#  -1.40753019e+033  7.69082267e+060  2.08870044e+017 -4.22939528e+041
#   3.10014203e+038 -5.74754314e+022 -5.92541106e+005  2.96573221e+037
#  -1.67685179e+116 -3.07669606e+034  1.11796201e+002 -2.76632732e+048
#  -1.14509147e-002  4.06375116e+060  1.38853512e+097 -9.34009715e+048
#   1.43851186e+039  2.11867988e+041 -1.89685751e+015 -1.93395895e+070
#   8.73438133e+018  1.26515361e+015 -4.15198754e+048  1.60311752e+010
#  -4.39195245e+045 -1.41610372e+083  5.17747718e+212 -4.30178785e-001
#  -1.57172140e+084  9.19860297e+044  7.86414702e+031 -3.56165773e+085
#  -2.50015112e+063  2.43383831e-008 -5.23206006e+019 -3.79070040e+033
#   6.52073675e+039 -2.58170441e+020 -6.51102922e+010  1.34571032e+076
#  -8.37791699e+014 -3.72312765e+040  3.22512786e+011  1.49194387e+023
#  -3.83840897e+019  1.31663735e+016  1.58387079e+056 -4.53518733e+056
#   1.09150604e-009  9.68226310e+002 -1.97342631e+069 -7.20882169e+013
#   6.85286369e+015 -4.56012979e+021 -6.58119824e+036  1.19489093e+038
#  -7.50401813e+033 -1.16602166e+004  8.87746628e+019 -4.80161112e+016
#  -9.33752880e+034  4.73181128e+068  3.36725211e+006 -1.06007379e+078
#   5.71170237e+069  1.13289835e+039 -2.95092850e+011 -1.66444965e+023
#   1.12507946e+086 -7.56031717e+051 -1.22363721e+071  5.78099510e+019
#  -3.55548935e+000 -1.74765533e+049  2.50159457e+093  4.89572744e+075
#  -4.59635857e+042  2.11253966e+067 -7.28528750e+052 -2.53222877e+065
#   3.31232332e+006  1.24347985e+035 -7.25245660e-009  5.95267364e+026
#   1.17885768e+050 -3.64564522e+035 -9.25472631e+051  6.51767886e+006
#   5.58711817e+057  1.35150913e+056  2.55951506e+012 -6.80041985e+063
#   2.60744824e+004  5.78347087e+071 -1.27714402e+108 -7.41549843e+027
#   4.82399264e+039  1.27463213e+039 -9.73675058e+013  9.77800087e+049]
# sortCSIe1 [ 3.15129700e+055 -5.06172296e-004 -7.74002367e+043  5.58802680e+100
#  -9.52807605e+012 -1.80640094e+065  4.24872492e+066  1.15168761e+052
#  -1.87066002e+036 -1.44932010e+064  4.72768024e+006 -3.64963716e+070
#  -1.01117970e+018  1.46238509e+035 -1.82427131e+007 -1.29232845e+039
#   1.75846724e+035  1.16548929e+076 -5.29927430e+021  4.01918072e+020
#  -3.57340450e+033 -1.58592480e+033 -1.34586815e+025  2.37287600e+162
#  -3.30521842e+050 -3.25647003e+088  1.19629024e+056 -4.52808852e+021
#  -3.00265890e+043  7.02310134e+092  6.32163220e+026 -2.77364129e+063
#   5.26586368e+059 -1.26964653e+037 -1.97357620e+010  3.73672209e+061
#  -1.46832510e+167 -3.82662665e+043  1.25632423e+003 -3.88491464e+060
#  -1.19938171e-002  1.61994316e+074  1.45250736e+120 -2.23269847e+067
#   5.80346897e+052  2.88544111e+051 -6.91613668e+016 -1.73537739e+080
#   6.82811641e+023  3.23830739e+021 -8.13902922e+064  3.34558785e+013
#  -8.85312257e+060 -1.51893864e+111  6.62175548e+292 -1.36371929e+000
#  -1.51142301e+113  1.68672323e+061  3.24120629e+046 -5.18595568e+127
#  -1.94433926e+093  2.04818942e-010 -1.60794033e+027 -1.65578129e+046
#   2.53208439e+054 -7.58156408e+027 -7.16101654e+013  1.86407813e+101
#  -1.03311674e+021 -3.33952483e+055  1.36812271e+015  1.34970075e+030
#  -3.24888378e+027  5.30782762e+022  3.21285332e+080 -2.17656927e+075
#   7.18487011e-012  8.46340367e+003 -3.99802316e+100 -7.58462682e+020
#   9.05236279e+023 -9.92537532e+031 -7.32354667e+052  2.49763710e+052
#  -1.47496263e+046 -1.93252096e+005  2.13975928e+026 -3.25544029e+021
#  -6.11945923e+043  9.07647245e+089  5.24418375e+009 -3.63753275e+104
#   1.13890037e+095  4.90895405e+050 -1.46779294e+015 -4.28138944e+029
#   2.28473431e+108 -1.63727077e+065 -4.23268599e+093  2.00130090e+026
#  -1.71819688e+001 -2.22560682e+065  3.14765812e+125  2.29117710e+103
#  -5.12756889e+058  4.48770036e+092 -1.75141815e+076 -1.65689252e+090
#   1.41936982e+009  1.91149764e+046 -1.62399728e-010  1.47723075e+035
#   1.54098472e+068 -1.38173098e+048 -2.10252369e+071  2.05439041e+008
#   1.06140604e+069  1.07297183e+066  1.04699185e+016 -8.11670398e+083
#   6.86803360e+005  2.60975155e+096 -5.70206044e+144 -4.74379507e+038
#   4.73969791e+055  3.96746773e+051 -6.67170697e+016  5.18039082e+056]
# sortNoise [-1.11468530e+00 -1.98689072e+00 -9.42213928e-01 -7.36188392e-01
#  -1.19215235e+00 -8.59282522e-01 -1.01126663e+00 -1.22946598e+00
#  -7.83748685e-01 -1.07001329e+00 -1.37513462e+00 -5.71806310e-01
#  -1.10681373e+00 -1.64391326e+00 -8.58404915e-01 -3.29664207e-01
#  -1.64441677e+00 -1.04790160e+00 -6.18374662e-01 -1.27927429e+00
#  -1.20636040e+00 -1.38115215e+00 -6.51404731e-01 -1.71313993e-01
#  -1.18731433e+00 -6.09902043e-01 -4.34335923e-01 -7.40292023e-01
#  -1.20360883e+00 -7.97315669e-01 -9.46779883e-01 -5.02877034e-01
#  -1.66395045e+00 -4.35948694e-01 -1.51041533e+00 -5.14807024e-01
#  -9.06234720e-01 -9.51382423e-01 -1.04298462e+00 -2.20618811e-01
#  -1.60534903e+00 -1.15514206e+00 -1.64482110e+00 -8.65877092e-01
#  -5.45591203e-01 -8.53309941e-01 -1.01670711e+00 -9.41361005e-01
#  -1.06454006e+00 -7.00081167e-01 -1.31132584e+00 -9.42967361e-01
#  -1.12458574e+00 -6.69554469e-01 -8.82273548e-01 -1.25938695e+00
#  -1.05479728e+00 -5.99101434e-01 -1.11993875e+00  1.54099733e-03
#  -5.49168046e-01 -1.73758866e+00 -3.87251194e-01 -3.71560676e-01
#  -1.28737803e-01 -1.03628119e+00 -1.52396554e+00 -9.32312355e-01
#  -6.23717480e-01 -8.33182818e-01 -1.13337461e+00 -9.68650317e-01
#  -9.82199053e-01 -7.42556691e-01 -1.44913116e+00 -9.31808661e-01
#  -1.77942807e+00 -1.22293838e+00 -9.92495129e-01 -1.35522547e+00
#  -1.77014766e+00 -1.01057181e+00 -7.56641354e-01 -5.37662951e-01
#  -7.07691561e-01 -9.10213312e-01 -7.77149213e-01 -7.01822080e-01
#  -8.98210484e-01 -1.38856428e+00 -9.37680300e-01 -6.15841089e-01
#  -1.46535034e+00 -1.02830622e+00 -1.46096360e+00 -4.16149111e-01
#  -6.58908770e-01 -7.80466836e-01 -1.01400083e+00 -1.19040908e+00
#  -1.60278200e+00 -1.17995554e+00 -1.22898157e+00 -4.78476967e-01
#  -1.05295875e+00 -8.38380637e-01 -5.89996539e-01 -7.35350297e-01
#  -1.52324709e+00 -7.54068305e-01 -2.00614000e+00 -1.15494449e+00
#  -1.87504797e-01 -4.16522902e-01 -4.51844476e-01 -1.51382989e+00
#  -8.15132059e-01 -5.97119385e-01 -1.25624808e+00 -2.15487136e-01
#  -1.35835905e+00 -1.02618283e+00 -8.31232167e-01 -7.11046971e-01
#  -1.03983454e+00 -1.18589832e+00 -9.22387861e-01 -9.37123270e-01]
# sortInda [ 36 122  59  56  53  91  25  98  47  78 107 119  60  75  11 106  97 114
#    5   9  43 101  50  39  52 104  31  69  24  82 113  88  37   2  84  28
#   63  15 123  21   8  20  95  33  81  65  72  62  22  27  87  46  68 126
#   18  79  12  94  66   4  34  14  85 100  55  40   1 110  76  61  38  77
#  120  10  90 108 115  51  70 118  19  49  80  73  30  48  99  86  71  13
#  111  16  58 109  35  83   7  32  93 125  44  64 124  45  57   0  26 112
#    6 127  17  74 117 116  41  29 105  89  92 121 103  67   3  96 102  42
#   23  54]
# sortIndb [ 36 122  59  56  53  91  25  98  47  78 107 119  60  75  11 106 114  97
#    9   5 101  43  50  39  52 104  31  69  24  82 113   2  88  37  84  63
#   28  15 123   8  21  20  95  33  81  65  62  72  22  27  87  46  68 126
#   79  18  12  94  66   4  34  14  85 100  55  40   1 110  76  61  38  77
#  120  10 108  90 115  51  70 118  19  49  80  73  30  48  99  86  71  13
#  111  16  58 109  35   7  83  32  93 125  44 124  64  45  57  26   0   6
#  127 112  17 117  74 116  41  29 105  89  92 121 103  67   3  96 102  42
#   23  54]
# sortInde [ 36 122  59  56  53  91  78  98  60 107  25 119  47 106  75 114  11  43
#  101   5  97  50   9  31  52  39 104  69  82  24 113  63  84   2  88  37
#   28  15 123  33   8  20  21  81  95  65  72  62  22  18  27  87  68  79
#   12  46 126  94  66   4  34  14  85 100  55  40   1 110  76  61  38  77
#  120  10 115 108  90  51  70 118  19  49  73  48  80  99  86  30  71  13
#  111  16 109  58  93  45 125   7  83  44  64   0 124  26 127  32  57  35
#  117   6 112 116  41  17  74  89 105  29  92 121   3  67 103  96  42 102
#   23  54]
# sortIndn [110   1  76  80  61  32  42  16  13  40 100  66 108 115  34  92  94  74
#   89  21  10 120  79  50  19  55 118   7 102  77  20  28   4  99  24 125
#  101  41 111  70  52  58   0  12   9  48  56 104  17  38 124  65  93 121
#   46  98   6  81  78  72  71  37  30  51   2  47  90 127  67  75 126  85
#   36  88  54  43   5  14  45 105  69 122 116  29   8  97  86  82 109  73
#   27   3 107 123  84  87  49  53  96  22  68  18  91  25  57 117 106  11
#   60  44  83  35  31 103 114  33  26 113  95  62  63  15  39 119 112  23
#   64  59]
# episodeCSIa1 [86, 63, 3, 3, 38, 66, 73, 51, 31, 29, 12, 59, 15, 64, 95, 9, 17, 109, 4, 34, 58, 51, 11, 13, 52, 73, 38, 45, 58, 31, 25, 51, 35, 82, 56, 35, 48, 108, 102, 13, 12, 75, 62, 48, 16, 7, 10, 40, 5, 60, 41, 22, 58, 108, 61, 67, 82, 28, 62, 30, 20, 71, 15, 45, 15, 39, 109, 34, 15, 23, 39, 43, 110, 80, 18, 7, 64, 19, 48, 99, 30, 31, 7, 43, 18, 51, 13, 15, 58, 98, 95, 42, 51, 74, 48, 76, 25, 61, 32, 81, 20, 60, 79, 12, 57, 26, 86, 106, 121, 110, 57, 43, 1, 75, 12, 76, 16, 3, 29, 18, 36, 64, 93, 6, 60, 19, 31, 0]
# episodeCSIb1 [86, 63, 3, 3, 38, 66, 73, 51, 31, 29, 12, 59, 15, 64, 95, 8, 17, 88, 4, 96, 58, 7, 11, 13, 52, 73, 38, 45, 58, 31, 111, 86, 51, 47, 21, 35, 13, 108, 115, 13, 1, 75, 62, 48, 16, 3, 10, 50, 5, 60, 41, 22, 58, 47, 61, 6, 82, 28, 62, 30, 20, 71, 15, 45, 15, 39, 109, 34, 15, 23, 39, 43, 110, 98, 18, 25, 64, 19, 48, 99, 30, 31, 7, 43, 18, 51, 13, 15, 58, 98, 95, 42, 51, 74, 28, 76, 51, 61, 32, 81, 80, 60, 19, 12, 31, 26, 6, 121, 15, 95, 100, 43, 42, 75, 12, 76, 16, 3, 29, 18, 36, 64, 93, 6, 60, 19, 31, 0]
# episodeCSIe1 [86, 63, 3, 3, 38, 13, 20, 38, 47, 82, 94, 72, 59, 31, 39, 103, 32, 58, 96, 92, 47, 41, 22, 21, 13, 65, 35, 13, 58, 89, 50, 21, 82, 86, 51, 9, 13, 108, 90, 25, 12, 1, 60, 14, 30, 7, 10, 40, 4, 9, 60, 19, 11, 67, 34, 80, 32, 28, 62, 30, 20, 71, 15, 45, 15, 39, 109, 34, 15, 23, 39, 43, 110, 105, 7, 18, 39, 19, 48, 99, 30, 24, 25, 32, 19, 13, 56, 41, 58, 98, 95, 93, 51, 35, 48, 80, 118, 76, 39, 20, 64, 124, 98, 101, 95, 25, 22, 82, 111, 106, 4, 75, 24, 57, 15, 16, 76, 63, 29, 118, 64, 36, 7, 54, 60, 79, 31, 0]
# episodeCSIn1 [109, 75, 4, 19, 29, 10, 26, 3, 27, 60, 34, 42, 7, 81, 58, 2, 20, 15, 68, 11, 110, 41, 29, 31, 36, 63, 111, 95, 25, 57, 8, 24, 95, 75, 101, 24, 60, 70, 41, 18, 6, 58, 12, 3, 39, 8, 48, 87, 21, 86, 59, 28, 28, 75, 52, 92, 75, 3, 6, 1, 34, 7, 21, 49, 45, 43, 37, 60, 8, 51, 41, 49, 52, 34, 11, 38, 9, 31, 60, 36, 53, 6, 87, 21, 89, 11, 4, 27, 36, 46, 24, 104, 16, 39, 3, 38, 4, 43, 74, 46, 50, 73, 66, 32, 60, 11, 95, 49, 16, 39, 48, 4, 72, 11, 81, 7, 87, 18, 33, 1, 48, 24, 80, 7, 89, 41, 5, 0]
# PATH-A {0: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0], 1: [1], 2: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2], 3: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 3], 4: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4], 5: [1, 119, 11, 17, 67, 93, 120, 124, 123, 5], 6: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6], 7: [1, 119, 11, 17, 67, 93, 120, 124, 7], 8: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 8], 9: [1, 72, 76, 10, 116, 63, 42, 117, 87, 9], 10: [1, 72, 76, 10], 11: [1, 119, 11], 12: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12], 13: [1, 72, 76, 10, 116, 63, 42, 117, 87, 29, 13], 14: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41, 80, 54, 110, 91, 89, 14], 15: [1, 72, 76, 15], 16: [1, 119, 11, 17, 67, 93, 120, 124, 16], 17: [1, 119, 11, 17], 18: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41, 80, 54, 110, 91, 89, 14, 18], 19: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41, 80, 54, 110, 56, 19], 20: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20], 21: [1, 72, 76, 10, 116, 21], 22: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22], 23: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 23], 24: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24], 25: [1, 119, 11, 17, 67, 93, 25], 26: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 43, 26], 27: [1, 72, 76, 10, 116, 63, 42, 117, 87, 55, 46, 27], 28: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28], 29: [1, 72, 76, 10, 116, 63, 42, 117, 87, 29], 30: [1, 119, 11, 112, 113, 30], 31: [1, 119, 11, 17, 67, 93, 120, 124, 7, 31], 32: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32], 33: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41, 80, 54, 110, 91, 89, 14, 33], 34: [1, 72, 76, 10, 116, 63, 42, 57, 74, 34], 35: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 35], 36: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 36], 37: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41, 80, 54, 110, 37], 38: [1, 38], 39: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39], 40: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41, 80, 54, 110, 91, 89, 14, 40], 41: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41], 42: [1, 72, 76, 10, 116, 63, 42], 43: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 43], 44: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44], 45: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45], 46: [1, 72, 76, 10, 116, 63, 42, 117, 87, 55, 46], 47: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47], 48: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 126, 48], 49: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49], 50: [1, 72, 76, 10, 116, 63, 42, 50], 51: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51], 52: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 52], 53: [1, 53], 54: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41, 80, 54], 55: [1, 72, 76, 10, 116, 63, 42, 117, 87, 55], 56: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41, 80, 54, 110, 56], 57: [1, 72, 76, 10, 116, 63, 42, 57], 58: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58], 59: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41, 80, 54, 110, 91, 89, 14, 73, 59], 60: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60], 61: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41, 80, 54, 110, 61], 62: [1, 72, 76, 10, 116, 63, 42, 57, 74, 62], 63: [1, 72, 76, 10, 116, 63], 64: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 64], 65: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 65], 66: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41, 80, 54, 110, 91, 89, 14, 66], 67: [1, 119, 11, 17, 67], 68: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68], 69: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 69], 70: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 65, 70], 71: [1, 72, 76, 10, 116, 63, 42, 117, 87, 71], 72: [1, 72], 73: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41, 80, 54, 110, 91, 89, 14, 73], 74: [1, 72, 76, 10, 116, 63, 42, 57, 74], 75: [1, 119, 11, 112, 113, 75], 76: [1, 72, 76], 77: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 64, 77], 78: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78], 79: [1, 79], 80: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41, 80], 81: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 64, 77, 81], 82: [1, 72, 76, 10, 116, 63, 42, 117, 87, 82], 83: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 83], 84: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 84], 85: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85], 86: [1, 119, 11, 17, 67, 93, 120, 124, 86], 87: [1, 72, 76, 10, 116, 63, 42, 117, 87], 88: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88], 89: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41, 80, 54, 110, 91, 89], 90: [1, 119, 11, 17, 107, 90], 91: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41, 80, 54, 110, 91], 92: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92], 93: [1, 119, 11, 17, 67, 93], 94: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 65, 94], 95: [1, 72, 76, 10, 116, 63, 42, 57, 95], 96: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41, 80, 54, 110, 91, 89, 14, 96], 97: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 97], 98: [1, 119, 11, 17, 67, 93, 120, 124, 98], 99: [1, 99], 100: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 100], 101: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 101], 102: [1, 102], 103: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 103], 104: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104], 105: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41, 80, 54, 110, 91, 89, 14, 105], 106: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 43, 106], 107: [1, 119, 11, 17, 107], 108: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41, 80, 54, 110, 91, 89, 14, 73, 108], 109: [1, 109], 110: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41, 80, 54, 110], 111: [1, 72, 76, 10, 116, 63, 42, 117, 87, 111], 112: [1, 119, 11, 112], 113: [1, 119, 11, 112, 113], 114: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 103, 114], 115: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115], 116: [1, 72, 76, 10, 116], 117: [1, 72, 76, 10, 116, 63, 42, 117], 118: [1, 119, 11, 17, 67, 93, 120, 124, 7, 118], 119: [1, 119], 120: [1, 119, 11, 17, 67, 93, 120], 121: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 121], 122: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 39, 41, 80, 54, 110, 91, 89, 14, 96, 122], 123: [1, 119, 11, 17, 67, 93, 120, 124, 123], 124: [1, 119, 11, 17, 67, 93, 120, 124], 125: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 125], 126: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 104, 4, 24, 22, 85, 51, 20, 32, 0, 60, 115, 49, 6, 28, 126], 127: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 47, 12, 2, 92, 127]}
# PATH-B {0: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0], 1: [1], 2: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2], 3: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 3], 4: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4], 5: [1, 119, 11, 16, 67, 93, 120, 124, 123, 5], 6: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6], 7: [1, 119, 11, 16, 67, 93, 120, 124, 7], 8: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 8], 9: [1, 72, 76, 10, 116, 63, 42, 117, 87, 9], 10: [1, 72, 76, 10], 11: [1, 119, 11], 12: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12], 13: [1, 72, 76, 10, 116, 63, 42, 117, 87, 29, 13], 14: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41, 80, 55, 110, 91, 89, 14], 15: [1, 72, 76, 15], 16: [1, 119, 11, 16], 17: [1, 119, 11, 16, 67, 93, 120, 124, 17], 18: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41, 80, 55, 110, 56, 18], 19: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41, 80, 55, 110, 91, 89, 14, 19], 20: [1, 72, 76, 10, 116, 20], 21: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21], 22: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22], 23: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 23], 24: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24], 25: [1, 119, 11, 16, 67, 93, 25], 26: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 43, 26], 27: [1, 72, 76, 10, 116, 63, 42, 117, 87, 54, 47, 27], 28: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28], 29: [1, 72, 76, 10, 116, 63, 42, 117, 87, 29], 30: [1, 119, 11, 111, 113, 30], 31: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41, 80, 55, 110, 91, 89, 14, 31], 32: [1, 119, 11, 16, 67, 93, 120, 124, 7, 32], 33: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33], 34: [1, 72, 76, 10, 116, 63, 42, 57, 75, 34], 35: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 35], 36: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 36], 37: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41, 80, 55, 110, 37], 38: [1, 38], 39: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41, 80, 55, 110, 91, 89, 14, 39], 40: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40], 41: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41], 42: [1, 72, 76, 10, 116, 63, 42], 43: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 43], 44: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44], 45: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45], 46: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46], 47: [1, 72, 76, 10, 116, 63, 42, 117, 87, 54, 47], 48: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 126, 48], 49: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49], 50: [1, 72, 76, 10, 116, 63, 42, 50], 51: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51], 52: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 52], 53: [1, 53], 54: [1, 72, 76, 10, 116, 63, 42, 117, 87, 54], 55: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41, 80, 55], 56: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41, 80, 55, 110, 56], 57: [1, 72, 76, 10, 116, 63, 42, 57], 58: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58], 59: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41, 80, 55, 110, 91, 89, 14, 73, 59], 60: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60], 61: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41, 80, 55, 110, 61], 62: [1, 72, 76, 10, 116, 63, 42, 57, 75, 62], 63: [1, 72, 76, 10, 116, 63], 64: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 64], 65: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 65], 66: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41, 80, 55, 110, 91, 89, 14, 66], 67: [1, 119, 11, 16, 67], 68: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68], 69: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 69], 70: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 65, 70], 71: [1, 72, 76, 10, 116, 63, 42, 117, 87, 71], 72: [1, 72], 73: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41, 80, 55, 110, 91, 89, 14, 73], 74: [1, 119, 11, 111, 113, 74], 75: [1, 72, 76, 10, 116, 63, 42, 57, 75], 76: [1, 72, 76], 77: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 64, 77], 78: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78], 79: [1, 79], 80: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41, 80], 81: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 64, 77, 81], 82: [1, 72, 76, 10, 116, 63, 42, 117, 87, 82], 83: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 83], 84: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 84], 85: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85], 86: [1, 119, 11, 16, 67, 93, 120, 124, 86], 87: [1, 72, 76, 10, 116, 63, 42, 117, 87], 88: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88], 89: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41, 80, 55, 110, 91, 89], 90: [1, 119, 11, 16, 109, 90], 91: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41, 80, 55, 110, 91], 92: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92], 93: [1, 119, 11, 16, 67, 93], 94: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 65, 94], 95: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41, 80, 55, 110, 91, 89, 14, 95], 96: [1, 72, 76, 10, 116, 63, 42, 57, 96], 97: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 97], 98: [1, 119, 11, 16, 67, 93, 120, 124, 98], 99: [1, 99], 100: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 100], 101: [1, 101], 102: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 102], 103: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 103], 104: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104], 105: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 43, 105], 106: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41, 80, 55, 110, 91, 89, 14, 106], 107: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41, 80, 55, 110, 91, 89, 14, 73, 107], 108: [1, 108], 109: [1, 119, 11, 16, 109], 110: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41, 80, 55, 110], 111: [1, 119, 11, 111], 112: [1, 72, 76, 10, 116, 63, 42, 117, 87, 112], 113: [1, 119, 11, 111, 113], 114: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 103, 114], 115: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115], 116: [1, 72, 76, 10, 116], 117: [1, 72, 76, 10, 116, 63, 42, 117], 118: [1, 119, 11, 16, 67, 93, 120, 124, 7, 118], 119: [1, 119], 120: [1, 119, 11, 16, 67, 93, 120], 121: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 121], 122: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 40, 41, 80, 55, 110, 91, 89, 14, 95, 122], 123: [1, 119, 11, 16, 67, 93, 120, 124, 123], 124: [1, 119, 11, 16, 67, 93, 120, 124], 125: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 125], 126: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 104, 4, 24, 22, 85, 51, 21, 33, 0, 60, 115, 49, 6, 28, 126], 127: [1, 72, 76, 10, 116, 63, 42, 117, 87, 44, 68, 88, 78, 58, 45, 46, 12, 2, 92, 127]}
# PATH-E {0: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0], 1: [1], 2: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2], 3: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 3], 4: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4], 5: [1, 119, 11, 15, 67, 92, 122, 125, 123, 5], 6: [1, 72, 74, 9, 116, 63, 44, 115, 86, 6], 7: [1, 119, 11, 15, 67, 92, 122, 125, 7], 8: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8], 9: [1, 72, 74, 9], 10: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10], 11: [1, 119, 11], 12: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 12], 13: [1, 72, 74, 13], 14: [1, 72, 74, 9, 116, 63, 44, 115, 86, 28, 14], 15: [1, 119, 11, 15], 16: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41, 80, 49, 113, 91, 89, 16], 17: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17], 18: [1, 72, 74, 9, 116, 18], 19: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41, 80, 49, 113, 91, 89, 16, 19], 20: [1, 119, 11, 15, 67, 92, 122, 125, 20], 21: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21], 22: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41, 80, 49, 113, 54, 22], 23: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 39, 23], 24: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24], 25: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 25], 26: [1, 119, 11, 15, 67, 92, 26], 27: [1, 72, 74, 9, 116, 63, 44, 115, 86, 53, 46, 27], 28: [1, 72, 74, 9, 116, 63, 44, 115, 86, 28], 29: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29], 30: [1, 119, 11, 108, 111, 30], 31: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 31], 32: [1, 72, 74, 9, 116, 63, 44, 57, 76, 32], 33: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41, 80, 49, 113, 91, 89, 16, 33], 34: [1, 119, 11, 15, 67, 92, 122, 125, 7, 34], 35: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35], 36: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 36], 37: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41, 80, 49, 113, 37], 38: [1, 38], 39: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 39], 40: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41, 80, 49, 113, 91, 89, 16, 40], 41: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41], 42: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42], 43: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43], 44: [1, 72, 74, 9, 116, 63, 44], 45: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45], 46: [1, 72, 74, 9, 116, 63, 44, 115, 86, 53, 46], 47: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47], 48: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 126, 48], 49: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41, 80, 49], 50: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50], 51: [1, 72, 74, 9, 116, 63, 44, 51], 52: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 52], 53: [1, 72, 74, 9, 116, 63, 44, 115, 86, 53], 54: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41, 80, 49, 113, 54], 55: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55], 56: [1, 56], 57: [1, 72, 74, 9, 116, 63, 44, 57], 58: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58], 59: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41, 80, 49, 113, 91, 89, 16, 73, 59], 60: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60], 61: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41, 80, 49, 113, 61], 62: [1, 72, 74, 9, 116, 63, 44, 57, 76, 62], 63: [1, 72, 74, 9, 116, 63], 64: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 64], 65: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 65], 66: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41, 80, 49, 113, 91, 89, 16, 66], 67: [1, 119, 11, 15, 67], 68: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68], 69: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 69], 70: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 65, 70], 71: [1, 72, 74, 9, 116, 63, 44, 115, 86, 71], 72: [1, 72], 73: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41, 80, 49, 113, 91, 89, 16, 73], 74: [1, 72, 74], 75: [1, 119, 11, 108, 111, 75], 76: [1, 72, 74, 9, 116, 63, 44, 57, 76], 77: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 64, 77], 78: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78], 79: [1, 79], 80: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41, 80], 81: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 64, 77, 81], 82: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 82], 83: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83], 84: [1, 72, 74, 9, 116, 63, 44, 115, 86, 84], 85: [1, 119, 11, 15, 67, 92, 122, 125, 85], 86: [1, 72, 74, 9, 116, 63, 44, 115, 86], 87: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 87], 88: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88], 89: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41, 80, 49, 113, 91, 89], 90: [1, 119, 11, 15, 110, 90], 91: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41, 80, 49, 113, 91], 92: [1, 119, 11, 15, 67, 92], 93: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93], 94: [1, 119, 11, 15, 67, 92, 122, 125, 94], 95: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 95], 96: [1, 96], 97: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41, 80, 49, 113, 91, 89, 16, 97], 98: [1, 72, 74, 9, 116, 63, 44, 57, 98], 99: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 99], 100: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 100], 101: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41, 80, 49, 113, 91, 89, 16, 101], 102: [1, 102], 103: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 39, 103], 104: [1, 104], 105: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 105], 106: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106], 107: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 65, 107], 108: [1, 119, 11, 108], 109: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41, 80, 49, 113, 91, 89, 16, 73, 109], 110: [1, 119, 11, 15, 110], 111: [1, 119, 11, 108, 111], 112: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 95, 112], 113: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41, 80, 49, 113], 114: [1, 72, 74, 9, 116, 63, 44, 115, 86, 114], 115: [1, 72, 74, 9, 116, 63, 44, 115], 116: [1, 72, 74, 9, 116], 117: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117], 118: [1, 119, 11, 15, 67, 92, 122, 125, 7, 118], 119: [1, 119], 120: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 42, 41, 80, 49, 113, 91, 89, 16, 97, 120], 121: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 121], 122: [1, 119, 11, 15, 67, 92, 122], 123: [1, 119, 11, 15, 67, 92, 122, 125, 123], 124: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 124], 125: [1, 119, 11, 15, 67, 92, 122, 125], 126: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 106, 4, 24, 21, 83, 55, 17, 35, 0, 60, 117, 50, 10, 29, 126], 127: [1, 72, 74, 9, 116, 63, 44, 115, 86, 43, 68, 88, 78, 58, 45, 47, 8, 2, 93, 127]}
# PATH-N {0: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 0], 1: [1], 2: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 60, 2], 3: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 17, 3], 4: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4], 5: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 5], 6: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6], 7: [1, 76, 44, 7], 8: [1, 76, 44, 8], 9: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9], 10: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 10], 11: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 11], 12: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 12], 13: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 92, 114, 13], 14: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14], 15: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 15], 16: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 16], 17: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 17], 18: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 18], 19: [1, 76, 44, 43, 48, 101, 24, 30, 19], 20: [1, 20], 21: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 92, 114, 13, 21], 22: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22], 23: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23], 24: [1, 76, 44, 43, 48, 101, 24], 25: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25], 26: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 92, 114, 82, 105, 26], 27: [1, 27], 28: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 98, 28], 29: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 29], 30: [1, 76, 44, 43, 48, 101, 24, 30], 31: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 31], 32: [1, 32], 33: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 33], 34: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34], 35: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 92, 114, 82, 105, 26, 53, 81, 93, 35], 36: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36], 37: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 37], 38: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 12, 38], 39: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 39], 40: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40], 41: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41], 42: [1, 42], 43: [1, 76, 44, 43], 44: [1, 76, 44], 45: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 37, 45], 46: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 74, 46], 47: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 47], 48: [1, 76, 44, 43, 48], 49: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49], 50: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 92, 114, 82, 105, 26, 53, 81, 93, 50], 51: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51], 52: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52], 53: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 92, 114, 82, 105, 26, 53], 54: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 37, 54], 55: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 55], 56: [1, 56], 57: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57], 58: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 17, 58], 59: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59], 60: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 60], 61: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 61], 62: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62], 63: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 63], 64: [1, 64], 65: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65], 66: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 94, 66], 67: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 92, 114, 82, 105, 26, 53, 81, 93, 50, 67], 68: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 68], 69: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 69], 70: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 92, 114, 82, 105, 26, 53, 81, 93, 70], 71: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 71], 72: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72], 73: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 73], 74: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 74], 75: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 75], 76: [1, 76], 77: [1, 76, 44, 77], 78: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 37, 78], 79: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79], 80: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80], 81: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 92, 114, 82, 105, 26, 53, 81], 82: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 92, 114, 82], 83: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83], 84: [1, 84], 85: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 85], 86: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86], 87: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 87], 88: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 88], 89: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 60, 89], 90: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 90], 91: [1, 91], 92: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 92], 93: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 92, 114, 82, 105, 26, 53, 81, 93], 94: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 94], 95: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95], 96: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 37, 96], 97: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97], 98: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 98], 99: [1, 76, 44, 43, 48, 101, 24, 30, 99], 100: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 11, 100], 101: [1, 76, 44, 43, 48, 101], 102: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 102], 103: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103], 104: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104], 105: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 92, 114, 82, 105], 106: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106], 107: [1, 107], 108: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 74, 46, 108], 109: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 109], 110: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 17, 58, 110], 111: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 111], 112: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 31, 112], 113: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 113], 114: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 92, 114], 115: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115], 116: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116], 117: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 117], 118: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 118], 119: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 119], 120: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120], 121: [1, 76, 44, 121], 122: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 61, 122], 123: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 92, 114, 82, 105, 123], 124: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 4, 120, 51, 80, 59, 22, 57, 86, 95, 52, 36, 79, 106, 12, 124], 125: [1, 76, 44, 43, 48, 101, 24, 30, 19, 125], 126: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 25, 104, 41, 126], 127: [1, 76, 44, 43, 48, 101, 24, 30, 19, 34, 103, 116, 83, 62, 115, 14, 72, 49, 9, 6, 65, 23, 40, 97, 74, 46, 127]}
# 0.765625
# 0.46875
# 0.015625


# sortInda [125 100 123 109  84  87 114  62  24 104  82  68   5  97   2  81 101 107
#   98  49  53  91 110  21 126  33  47  69  50 119  40  44  79 113  43  65
#   46  72  30  56 116  59  17  75  18  14   8  37  88  31 103 117  27  95
#   28  34  78  85  66  15  63  94 115  13  54  96  57  67  29  77 120  19
#   45  64  74  10 118 102  36  22 121  51   3  76   0  61  80   6  42  39
#    9 112 106  26  55 108  32  41  60 122  90  89  93  58  48  35  71  83
#   92 105 111  86 124  11   4  16   1  99  25  23  73 127  12  70  20  52
#   38   7]
# sortIndb [125 100 123 109  84  87 114  62  24 104  82  68   5  97  81   2 101 107
#   98  49  91  53 110  21 126  33  47  69  50  40 119  44  79 113  43  46
#   65  72  56  30  59 116  75  17  18   8  14  37  88  31 103 117  27  95
#   28  34  78  85  66  15  63  94 115  13  54  96  57  67  29  77 120  19
#   45  64  74  10 118 102  36  22 121  51   3  76   0  61  80   6  42  39
#    9 112 106  26  55 108  32  41 122  60  90  89  93  58  48  35  71  83
#   92 105 124 111  86  11   4  16  99   1  25  23 127  73  12  70  20  52
#   38   7]
# sortInde [125 100 123 109  84  87  62 104 114  68  24  82  97   5  49  81   2  98
#  107  33 101  53 110  21  91  47 126  69 119  79  50  40 113  44  43  30
#   56  65  17  46  18  14  75  72  59 116   8  31  37  88 103  95  27 117
#   28  34  78  85  66  15  63  94 115  13  54  96  57  67  29  45  77 120
#   19  64  74  10 118 102  51   3  22 121  36   0   6  61  76  39  42  80
#    9 106 112  26 108  55  41  60 122  90  32  89  71  93  58  83  92  48
#   35 105 124   4 111  86   1  11  99  25  23 127  73  16  70  52  20  12
#   38   7]
# sortIndn [119  73  57  40 111  24 121  63  94  43  31  66  98  14  74  19 125  96
#    6  78  85  17 115  67  76  56 120  54  37  29 116  89  53   8  49 124
#   75  61  59  72  48 113 117  97  38  95  58 126  16 103  77  88 108  18
#   83  39 112  79  42  28  13  30  11  25  23 102  86  71 101 107  44  65
#   50   3  34  55  90  99  26 114  21  87   5  20  27   1 104 105  46  62
#   64  47  15 109  12  52  69  22  91 118  51  81  68 110 122 123  36  33
#    4   0  92   2 106  41   7 100  80  93  32  10 127  84  35   9  45  70
#   82  60]
# episodeCSIa1 [114.25, 100.75, 86.75, 76.0, 69.5, 63.0, 46.25, 72.75, 88.75, 72.75, 68.75, 72.5, 68.75, 71.25, 63.25, 69.0, 75.0, 56.5, 51.0, 65.25, 66.75, 31.0, 19.25, 41.0, 84.75, 85.5, 46.0, 56.25, 61.0, 59.5, 71.25, 69.5, 68.5, 57.5, 61.25, 62.0, 48.25, 76.0, 69.5, 57.5, 62.75, 35.0, 36.75, 41.75, 50.5, 63.25, 73.75, 59.0, 63.75, 90.25, 82.5, 58.5, 59.25, 87.75, 98.5, 83.0, 38.75, 30.0, 37.0, 62.0, 70.5, 38.5, 29.25, 11.25]
# episodeCSIb1 [114.25, 100.75, 86.75, 76.0, 69.5, 63.0, 46.25, 72.75, 88.75, 72.75, 68.75, 72.5, 68.75, 51.5, 63.25, 88.75, 70.25, 56.5, 55.75, 65.25, 66.75, 29.5, 19.25, 42.5, 84.75, 85.5, 46.0, 56.25, 61.0, 59.5, 71.25, 69.5, 68.5, 57.5, 61.25, 62.0, 48.25, 76.0, 69.5, 57.5, 62.75, 35.0, 36.75, 41.75, 50.5, 63.25, 73.75, 59.0, 63.75, 90.25, 82.5, 58.5, 59.25, 87.75, 108.0, 83.0, 29.25, 30.0, 37.0, 62.0, 70.5, 38.5, 29.25, 11.25]
# episodeCSIe1 [114.25, 100.75, 84.25, 87.0, 72.0, 52.0, 58.0, 57.5, 60.0, 73.5, 71.25, 67.25, 83.25, 98.25, 72.0, 61.75, 57.5, 48.5, 46.0, 23.75, 44.75, 80.5, 53.5, 41.0, 80.75, 85.5, 51.5, 56.25, 61.0, 59.5, 71.25, 69.5, 68.5, 49.5, 67.75, 70.0, 41.75, 76.0, 68.5, 49.25, 44.75, 25.75, 45.5, 59.25, 59.25, 63.25, 75.25, 66.0, 78.25, 83.25, 71.25, 76.25, 70.25, 70.0, 67.0, 81.25, 52.25, 34.0, 68.5, 59.75, 52.75, 38.5, 19.25, 11.25]
