import math
import time
from itertools import permutations

import numpy as np


def euclidean_metric(data1, data2):
    if isinstance(data1, list) is False:
        return abs(data1 - data2)
    res = 0
    for i in range(len(data1)):
        res += abs(data1[i] - data2[i])
    return res
    # for i in range(len(data1)):
    #     res += math.pow(data1[i] - data2[i], 2)
    # return math.sqrt(res)


def genDiffMatrix(data1, data2, perm_threshold):
    if len(data1) == 0:
        return [], []
    matrix = []
    for i in range(len(data1)):
        row = []
        for j in range(len(data2)):
            row.append(euclidean_metric(data1[i], data2[j]))
        matrix.append(row)

    fit_seq = []
    index_seq = []
    selectMaxDiffBacktrack(matrix, perm_threshold, [], [], index_seq, fit_seq)

    max_diff = 0
    fittest_seq = []
    perm = []
    # 排序后首尾相差最大的是fittest
    for i in range(len(fit_seq)):
        seq_back = list(fit_seq[i].copy())
        seq_back.sort()
        diff = seq_back[len(seq_back) - 1] - seq_back[0]
        if diff > max_diff:
            max_diff = diff
            fittest_seq = fit_seq[i]
            perm = index_seq[i]

    # print(fittest_seq)
    # print(np.array(data2)[perm])

    if len(fittest_seq) == 0:
        shuffle = np.random.permutation(list(range(len(data2))))
        seq = []
        for i in range(len(shuffle)):
            seq.append(matrix[i][shuffle[i]])
        return shuffle, seq

    return perm, fittest_seq


def befitting(seq, threshold, input):
    for s in seq:
        metric = euclidean_metric(s, input)
        if metric < threshold or math.isclose(metric, threshold, rel_tol=1e-5):
            return False
    return True


def selectMaxDiffBacktrack(matrix, threshold, cur_seq, index_temp, index_seq, fit_seq):
    if len(cur_seq) == len(matrix):
        fit_seq.append(cur_seq.copy())
        index_seq.append(index_temp.copy())
    else:
        cur_index = len(cur_seq)
        for i in range(len(matrix[cur_index])):
            # 必须不同列
            if i not in index_temp:
                # 必须和以前检查过的相比，不能选取bad metric导致差值接近
                if befitting(cur_seq, threshold, matrix[cur_index][i]):
                    cur_seq.append(matrix[cur_index][i])
                    index_temp.append(i)
                    selectMaxDiffBacktrack(matrix, threshold, cur_seq, index_temp, index_seq, fit_seq)
                    cur_seq.pop()
                    index_temp.pop()


# a-b all 21985 / 22045 = 0.9972782944
# a-e all 11339 / 22045 = 0.5143569971
# a-n all 11342 / 22045 = 0.5144930823
# a-b whole match 719 / 741 = 0.9703103914
# a-e whole match 0 / 741 = 0.0
# a-n whole match 1 / 741 = 0.0013495277
# times 741
# a-b all 19889 / 19895 = 0.9996984167
# a-e all 10611 / 19895 = 0.533350088
# a-n all 10527 / 19895 = 0.5291279216
# a-b whole match 738 / 741 = 0.995951417
# a-e whole match 1 / 741 = 0.0013495277
# a-n whole match 3 / 741 = 0.004048583
# times 741
# a-b all 21426 / 21466 = 0.9981365881
# a-e all 11220 / 21466 = 0.52268704
# a-n all 11194 / 21466 = 0.5214758222
# a-b whole match 728 / 741 = 0.9824561404
# a-e whole match 1 / 741 = 0.0013495277
# a-n whole match 1 / 741 = 0.0013495277
# times 741
# a-b all 20781 / 20801 = 0.9990385078
# a-e all 10997 / 20801 = 0.5286765059
# a-n all 10806 / 20801 = 0.5194942551
# a-b whole match 734 / 741 = 0.9905533063
# a-e whole match 1 / 741 = 0.0013495277
# a-n whole match 0 / 741 = 0.0
# times 741

# [1345]-[689 10]=[5555] => [1345]-[10 986]=[9641] perm=[3210]
# [1453]-[69 10 8]=[5555] => [1453]-[10 869]=[9641] perm=[2301]
def genDiffPerm(origin, data):
    sort_perm = list(np.argsort(data))

    perm = list(range(len(data) - 1, -1, -1))
    reverse_sort_perm = list(np.argsort(np.argsort(origin)))
    # reverse_sort_perm = list(np.argsort(np.argsort(data)))  # 不考虑origin的顺序，假设为增序
    combine_perm = np.array(sort_perm)[perm][reverse_sort_perm]
    # return list(range(len(data)))
    return combine_perm


# print(np.array([6, 8, 9, 10])[genDiffPerm([1, 3, 4, 5], [6, 8, 9, 10])])
# start = time.time()
# origin = [2, 4, 1, 5]
# data = np.array([6, 9, 10, 8])
#
# res = []
# for i in range(len(origin)):
#     res.append(abs(origin[i] - data[i]))
# print(res, np.argsort(res))
# perm = genDiffPerm(data)
# print(perm, data[perm])
# data = data[perm]
# res = []
# for i in range(len(origin)):
#     res.append(abs(origin[i] - data[i]))
# print(res, np.argsort(res))
# print(genDiffMatrix([1, 2, 3], [1, 3, 4], 1))
# print(genDiffMatrix([-0.55, -0.63, -0.68, -0.69, -0.71, -0.73, -0.52],
#                     [-0.31, -0.22, -0.20, -0.16, -0.12, -0.09, -0.35]))
# print(time.time() - start)


def genMaxDiffPerm(data):
    perms = [p for p in permutations(list(range(len(data))))]
    perms = np.array(perms) + 1
    maxDiff = -1
    seg = []
    for i in range(len(perms)):
        tmp = []
        for j in range(0, len(perms[i]), 2):
            tmp.append(perms[i][j] + perms[i][j + 1])
        tmp.sort()
        seg.append(tmp)
    for i in range(len(seg)):
        y = []
        isZero = False
        for j in range(len(seg[i]) - 1):
            yj = seg[i][j + 1] - seg[i][j]
            if yj == 0:
                isZero = True
            y.append(yj)
        if isZero:
            continue
        tmp = 0
        # for j in range(len(y)):
        #     tmp += y[j]
        for j in range(len(y) - 1):
            tmp += abs(y[j + 1] - y[j])
        maxDiff = max(maxDiff, tmp)
        if tmp == maxDiff:
            print(tmp, seg[i], perms[i])

# genMaxDiffPerm([1, 2, 3, 4, 5, 6, 7, 8])
