import sys
from math import fabs

import numpy as np
import numpy.linalg


def genAlign(base):
    align = []
    equalNum = 0
    insNum = 0
    delNum = 0
    updNum = 0
    swpNum = 0
    for i in range(len(base)):
        if base[i] == "=":
            align.append(i)
            equalNum += 1
        elif base[i] == "+":
            delNum += 1
        elif base[i] == "-":
            insNum += 1
        elif base[i] == "~":
            updNum += 1
        elif base[i] == "^":
            swpNum += 1
    # print("equalNum:", equalNum, "insNum:", insNum, "delNum:", delNum, "updNum:", updNum, "swpNum:", swpNum)
    return align


# 所有的编辑操作都进行编码成密钥
def genAlign2(base):
    align = []
    equalNum = 0
    insNum = 0
    delNum = 0
    updNum = 0
    swpNum = 0
    for i in range(len(base)):
        if base[i] == "=":
            align.append('=' + str(i))  # 加上一个操作符前缀以区分不同操作
            equalNum += 1
        elif base[i] == "+":
            align.append('+' + str(i))
            delNum += 1
        elif base[i] == "-":
            align.append('-' + str(i))
            insNum += 1
        elif base[i] == "~":
            align.append('~' + str(i))
            updNum += 1
        elif base[i] == "^":
            align.append('^' + str(i))
            swpNum += 1
    print("equalNum:", equalNum, "insNum:", insNum, "delNum:", delNum, "updNum:", updNum, "swpNum:", swpNum)
    return align


# 只匹配不相等的
def genAlign3(base):
    align = []
    equalNum = 0
    insNum = 0
    delNum = 0
    updNum = 0
    swpNum = 0
    for i in range(len(base)):
        if base[i] == "=":
            equalNum += 1
        elif base[i] == "+":
            align.append('+' + str(i))
            delNum += 1
        elif base[i] == "-":
            align.append('-' + str(i))
            insNum += 1
        elif base[i] == "~":
            align.append('~' + str(i))
            updNum += 1
        elif base[i] == "^":
            align.append('^' + str(i))
            swpNum += 1
    print("equalNum:", equalNum, "insNum:", insNum, "delNum:", delNum, "updNum:", updNum, "swpNum:", swpNum)
    return align


# 只匹配不相等的
def genAlignInsDel(base):
    align = []
    equalNum = 0
    insNum = 0
    delNum = 0
    for i in range(len(base)):
        if base[i] == "=":
            equalNum += 1
        elif base[i] == "+":
            align.append('+' + str(i))
            delNum += 1
        elif base[i] == "-":
            align.append('-' + str(i))
            insNum += 1
    # print("equalNum:", equalNum, "insNum:", insNum, "delNum:", delNum)
    return align


# def genAlign(base, strB):
#     align = []
#     j = 0
#     for i in range(len(base)):
#         if base[i] == "=":
#             align.append(strB[j])
#             j += 1
#         elif base[i] == "-":
#             j += 1
#         elif base[i] == "~":
#             j += 1
#         elif base[i] == "^":
#             align.append(strB[j + 1])
#             align.append(strB[j])
#             j += 2
#     return align


def generate(base, strB):
    align = []
    j = 0
    for i in range(len(base)):
        if base[i] == "=":
            align.append(strB[j])
            j += 1
        elif base[i] == "-":
            j += 1
        elif base[i] == "^":
            align.append(strB[j + 1])
            align.append(strB[j])
            j += 2
    return align


# def align(score, strA, strB):
#     m = len(strA)
#     n = len(strB)
#     aux = [[]] * (m + 1)
#     rule = [[]] * (m + 1)
#     for i in range(len(aux)):
#         aux[i] = [0] * (n + 1)
#         rule[i] = [""] * (n + 1)
#     rule[0][0] = ""
#     for i in range(1, m + 1):
#         rule[i][0] = rule[i - 1][0] + "-"
#         aux[i][0] = aux[i - 1][0] + score["-"]
#     for i in range(1, n + 1):
#         rule[0][i] = rule[0][i - 1] + "+"
#         aux[0][i] = aux[0][i - 1] + score["+"]
#     for i in range(1, m + 1):
#         for j in range(1, n + 1):
#             if strA[i - 1] == strB[j - 1]:
#                 aux[i][j] = aux[i - 1][j - 1] + score["="]
#                 rule[i][j] = rule[i - 1][j - 1] + "="
#             else:
#                 aux[i][j] = max(aux[i - 1][j] + score["-"], aux[i][j - 1] + score["+"], aux[i - 1][j - 1] + score["~"])
#                 if aux[i][j] == aux[i - 1][j - 1] + score["~"]:
#                     rule[i][j] = rule[i - 1][j - 1] + "~"
#                 elif aux[i][j] == aux[i - 1][j] + score["-"]:
#                     rule[i][j] = rule[i - 1][j] + "-"
#                 elif aux[i][j] == aux[i][j - 1] + score["+"]:
#                     rule[i][j] = rule[i][j - 1] + "+"
#     return rule[m][n]


def alignFloat(score, arrayA, arrayB, threshold):
    # threshold = max(2, second_diffAB)
    # print("阈值", threshold)

    def equal(f1, f2):
        return fabs(f1 - f2) <= threshold

    m = len(arrayA)
    n = len(arrayB)

    sortA = arrayA.copy()
    sortB = arrayB.copy()
    sortA.sort()
    sortB.sort()
    diffA = sys.maxsize
    diffB = sys.maxsize
    second_diffA = sys.maxsize
    second_diffB = sys.maxsize
    diffAB = sys.maxsize
    second_diffAB = sys.maxsize
    for i in range(m - 1):
        diff = abs(sortA[i] - sortA[i + 1])
        if second_diffA > diff:
            if diffA < diff:
                second_diffA = diff
            else:
                second_diffA = diffA
                diffA = diff
    for i in range(n - 1):
        diff = abs(sortB[i] - sortB[i + 1])
        if second_diffB > diff:
            if diffB < diff:
                second_diffB = diff
            else:
                second_diffB = diffB
                diffB = diff
    for i in range(min(m, n)):
        diff = abs(sortA[i] - sortB[i])
        if second_diffAB > diff:
            if diffAB < diff:
                second_diffAB = diff
            else:
                second_diffAB = diffAB
                diffAB = diff
    # print("min diff of A        ", diffA)
    # print("min diff of B        ", diffB)
    print("min diff of AB       ", diffAB)
    # print("second min diff of A ", second_diffA)
    # print("second min diff of B ", second_diffB)
    print("second min diff of AB", second_diffAB)

    diffA = 0
    diffB = 0
    second_diffA = 0
    second_diffB = 0
    diffAB = 0
    second_diffAB = 0
    for i in range(m - 1):
        diff = abs(sortA[i] - sortA[i + 1])
        if second_diffA < diff:
            if diffA > diff:
                second_diffA = diff
            else:
                second_diffA = diffA
                diffA = diff
    for i in range(n - 1):
        diff = abs(sortB[i] - sortB[i + 1])
        if second_diffB < diff:
            if diffB > diff:
                second_diffB = diff
            else:
                second_diffB = diffB
                diffB = diff
    for i in range(min(m, n)):
        diff = abs(sortA[i] - sortB[i])
        if second_diffAB < diff:
            if diffAB > diff:
                second_diffAB = diff
            else:
                second_diffAB = diffAB
                diffAB = diff
    # print("max diff of A        ", diffA)
    # print("max diff of B        ", diffB)
    print("max diff of AB       ", diffAB)
    # print("second max diff of A ", second_diffA)
    # print("second max diff of B ", second_diffB)
    print("second max diff of AB", second_diffAB)

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
            if equal(arrayA[i - 1], arrayB[j - 1]):
                aux[i][j] = aux[i - 1][j - 1] + score["="]
                rule[i][j] = rule[i - 1][j - 1] + "="
            elif i > 1 and j > 1 and equal(arrayA[i - 1], arrayB[j - 2]) and equal(arrayA[i - 2], arrayB[j - 1]):
                aux[i][j] = min(aux[i - 1][j] + score["-"], aux[i][j - 1] + score["+"],
                                aux[i - 1][j - 1] + score["~"], aux[i - 2][j - 2] + score["^"])
                if aux[i][j] == aux[i - 1][j - 1] + score["~"]:
                    rule[i][j] = rule[i - 1][j - 1] + "~"
                elif aux[i][j] == aux[i - 1][j] + score["-"]:
                    rule[i][j] = rule[i - 1][j] + "-"
                elif aux[i][j] == aux[i][j - 1] + score["+"]:
                    rule[i][j] = rule[i][j - 1] + "+"
                elif aux[i][j] == aux[i - 2][j - 2] + score["^"]:
                    rule[i][j] = rule[i - 2][j - 2] + "^"
            else:
                aux[i][j] = min(aux[i - 1][j] + score["-"], aux[i][j - 1] + score["+"], aux[i - 1][j - 1] + score["~"])
                if aux[i][j] == aux[i - 1][j - 1] + score["~"]:
                    rule[i][j] = rule[i - 1][j - 1] + "~"
                elif aux[i][j] == aux[i - 1][j] + score["-"]:
                    rule[i][j] = rule[i - 1][j] + "-"
                elif aux[i][j] == aux[i][j - 1] + score["+"]:
                    rule[i][j] = rule[i][j - 1] + "+"
    # print("score", aux[m][n])
    return rule[m][n]


def alignFloatInsDel(score, arrayA, arrayB, threshold):
    # threshold = max(2, second_diffAB)
    # print("阈值", threshold)

    def equal(f1, f2):
        return fabs(f1 - f2) <= threshold

    m = len(arrayA)
    n = len(arrayB)

    sortA = arrayA.copy()
    sortB = arrayB.copy()
    sortA.sort()
    sortB.sort()
    diffA = sys.maxsize
    diffB = sys.maxsize
    second_diffA = sys.maxsize
    second_diffB = sys.maxsize
    diffAB = sys.maxsize
    second_diffAB = sys.maxsize
    for i in range(m - 1):
        diff = abs(sortA[i] - sortA[i + 1])
        if second_diffA > diff:
            if diffA < diff:
                second_diffA = diff
            else:
                second_diffA = diffA
                diffA = diff
    for i in range(n - 1):
        diff = abs(sortB[i] - sortB[i + 1])
        if second_diffB > diff:
            if diffB < diff:
                second_diffB = diff
            else:
                second_diffB = diffB
                diffB = diff
    for i in range(min(m, n)):
        diff = abs(sortA[i] - sortB[i])
        if second_diffAB > diff:
            if diffAB < diff:
                second_diffAB = diff
            else:
                second_diffAB = diffAB
                diffAB = diff
    # print("min diff of A        ", diffA)
    # print("min diff of B        ", diffB)
    # print("min diff of AB       ", diffAB)
    # print("second min diff of A ", second_diffA)
    # print("second min diff of B ", second_diffB)
    # print("second min diff of AB", second_diffAB)

    diffA = 0
    diffB = 0
    second_diffA = 0
    second_diffB = 0
    diffAB = 0
    second_diffAB = 0
    for i in range(m - 1):
        diff = abs(sortA[i] - sortA[i + 1])
        if second_diffA < diff:
            if diffA > diff:
                second_diffA = diff
            else:
                second_diffA = diffA
                diffA = diff
    for i in range(n - 1):
        diff = abs(sortB[i] - sortB[i + 1])
        if second_diffB < diff:
            if diffB > diff:
                second_diffB = diff
            else:
                second_diffB = diffB
                diffB = diff
    for i in range(min(m, n)):
        diff = abs(sortA[i] - sortB[i])
        if second_diffAB < diff:
            if diffAB > diff:
                second_diffAB = diff
            else:
                second_diffAB = diffAB
                diffAB = diff
    # print("max diff of A        ", diffA)
    # print("max diff of B        ", diffB)
    # print("max diff of AB       ", diffAB)
    # print("second max diff of A ", second_diffA)
    # print("second max diff of B ", second_diffB)
    # print("second max diff of AB", second_diffAB)

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
            if equal(arrayA[i - 1], arrayB[j - 1]):
                aux[i][j] = aux[i - 1][j - 1] + score["="]
                rule[i][j] = rule[i - 1][j - 1] + "="
            else:
                aux[i][j] = min(aux[i - 1][j] + score["-"], aux[i][j - 1] + score["+"])
                if aux[i][j] == aux[i - 1][j] + score["-"]:
                    rule[i][j] = rule[i - 1][j] + "-"
                elif aux[i][j] == aux[i][j - 1] + score["+"]:
                    rule[i][j] = rule[i][j - 1] + "+"
    # print("score", aux[m][n])
    return rule[m][n]

def alignFloatIDU(score, arrayA, arrayB, threshold):
    def equal(f1, f2):
        return fabs(f1 - f2) <= threshold

    m = len(arrayA)
    n = len(arrayB)

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
            if equal(arrayA[i - 1], arrayB[j - 1]):
                aux[i][j] = aux[i - 1][j - 1] + score["="]
                rule[i][j] = rule[i - 1][j - 1] + "="
            else:
                aux[i][j] = min(aux[i - 1][j] + score["-"], aux[i][j - 1] + score["+"], aux[i - 1][j - 1] + score["~"])
                if aux[i][j] == aux[i - 1][j - 1] + score["~"]:
                    rule[i][j] = rule[i - 1][j - 1] + "~"
                elif aux[i][j] == aux[i - 1][j] + score["-"]:
                    rule[i][j] = rule[i - 1][j] + "-"
                elif aux[i][j] == aux[i][j - 1] + score["+"]:
                    rule[i][j] = rule[i][j - 1] + "+"
    # print("score", aux[m][n])
    return rule[m][n]



def absolute(f1, f2):
    res = 0
    for i in range(len(f1)):
        res += fabs(f1[i] - f2[i])
    return res / len(f1)


def absolute2(f1, f2):
    res = 0
    for i in range(len(f1)):
        res += f1[i] - f2[i]
    return fabs(res) / len(f1)


# def euclidean(f1, f2):
#     return np.linalg.norm(np.array(f1) - np.array(f2))


def manhattan(f1, f2):
    return np.linalg.norm(np.array(f1) - np.array(f2), ord=1)


# def chebyshev(f1, f2):
#     return np.linalg.norm(np.array(f1) - np.array(f2), ord=np.inf)


def cosine(f1, f2):
    v1 = np.array(f1)
    v2 = np.array(f2)
    m = np.linalg.norm(v1) * (np.linalg.norm(v2))
    if m == 0:
        return 0
    else:
        return np.dot(v1, v2) / m


def correlation(f1, f2):
    n = len(f1)
    v1 = np.array(f1)
    v2 = np.array(f2)
    sum_xy = np.sum(np.sum(v1 * v2))
    sum_x = np.sum(np.sum(v1))
    sum_y = np.sum(np.sum(v2))
    sum_x2 = np.sum(np.sum(v1 * v1))
    sum_y2 = np.sum(np.sum(v2 * v2))
    m = np.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    if m == 0:
        return 1
    else:
        return 1 - (n * sum_xy - sum_x * sum_y) / m


def dtw(a, b):
    dis = np.full((len(a) + 1, len(b) + 1), np.inf)
    dis[0, 0] = 0
    for i in range(0, len(a)):
        for j in range(0, len(b)):
            dis[i + 1, j + 1] = (a[i] - b[j]) ** 2
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dis[i, j] = min(dis[i - 1, j - 1], dis[i, j - 1], dis[i - 1, j]) + dis[i, j]
    result = dis[len(a) - 1, len(b) - 1] / (len(a) + len(b))
    return result

def alignFloatWithMetrics(score, arrayA, arrayB, threshold, metrics):
    m = len(arrayA)
    n = len(arrayB)

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
            if metrics(arrayA[i - 1], arrayB[j - 1]) <= threshold:
                aux[i][j] = aux[i - 1][j - 1] + score["="]
                rule[i][j] = rule[i - 1][j - 1] + "="
            else:
                aux[i][j] = min(aux[i - 1][j] + score["-"], aux[i][j - 1] + score["+"], aux[i - 1][j - 1] + score["~"])
                if aux[i][j] == aux[i - 1][j - 1] + score["~"]:
                    rule[i][j] = rule[i - 1][j - 1] + "~"
                elif aux[i][j] == aux[i - 1][j] + score["-"]:
                    rule[i][j] = rule[i - 1][j] + "-"
                elif aux[i][j] == aux[i][j - 1] + score["+"]:
                    rule[i][j] = rule[i][j - 1] + "+"
    # print("score", aux[m][n])
    return rule[m][n]

def alignFloatInsDelWithMetrics(score, arrayA, arrayB, threshold, metrics):
    # threshold = max(2, second_diffAB)
    # print("阈值", threshold)

    m = len(arrayA)
    n = len(arrayB)

    # sortA = arrayA.copy()
    # sortB = arrayB.copy()
    # sortA.sort()
    # sortB.sort()
    # diffA = sys.maxsize
    # diffB = sys.maxsize
    # second_diffA = sys.maxsize
    # second_diffB = sys.maxsize
    # diffAB = sys.maxsize
    # second_diffAB = sys.maxsize
    # for i in range(m - 1):
    #     diff = abs(sortA[i] - sortA[i + 1])
    #     if second_diffA > diff:
    #         if diffA < diff:
    #             second_diffA = diff
    #         else:
    #             second_diffA = diffA
    #             diffA = diff
    # for i in range(n - 1):
    #     diff = abs(sortB[i] - sortB[i + 1])
    #     if second_diffB > diff:
    #         if diffB < diff:
    #             second_diffB = diff
    #         else:
    #             second_diffB = diffB
    #             diffB = diff
    # for i in range(min(m, n)):
    #     diff = abs(sortA[i] - sortB[i])
    #     if second_diffAB > diff:
    #         if diffAB < diff:
    #             second_diffAB = diff
    #         else:
    #             second_diffAB = diffAB
    #             diffAB = diff
    # print("min diff of A        ", diffA)
    # print("min diff of B        ", diffB)
    # print("min diff of AB       ", diffAB)
    # print("second min diff of A ", second_diffA)
    # print("second min diff of B ", second_diffB)
    # print("second min diff of AB", second_diffAB)

    # diffA = 0
    # diffB = 0
    # second_diffA = 0
    # second_diffB = 0
    # diffAB = 0
    # second_diffAB = 0
    # for i in range(m - 1):
    #     diff = abs(sortA[i] - sortA[i + 1])
    #     if second_diffA < diff:
    #         if diffA > diff:
    #             second_diffA = diff
    #         else:
    #             second_diffA = diffA
    #             diffA = diff
    # for i in range(n - 1):
    #     diff = abs(sortB[i] - sortB[i + 1])
    #     if second_diffB < diff:
    #         if diffB > diff:
    #             second_diffB = diff
    #         else:
    #             second_diffB = diffB
    #             diffB = diff
    # for i in range(min(m, n)):
    #     diff = abs(sortA[i] - sortB[i])
    #     if second_diffAB < diff:
    #         if diffAB > diff:
    #             second_diffAB = diff
    #         else:
    #             second_diffAB = diffAB
    #             diffAB = diff
    # print("max diff of A        ", diffA)
    # print("max diff of B        ", diffB)
    # print("max diff of AB       ", diffAB)
    # print("second max diff of A ", second_diffA)
    # print("second max diff of B ", second_diffB)
    # print("second max diff of AB", second_diffAB)

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
            if metrics(arrayA[i - 1], arrayB[j - 1]) <= threshold:
                aux[i][j] = aux[i - 1][j - 1] + score["="]
                rule[i][j] = rule[i - 1][j - 1] + "="
            else:
                aux[i][j] = min(aux[i - 1][j] + score["-"], aux[i][j - 1] + score["+"])
                if aux[i][j] == aux[i - 1][j] + score["-"]:
                    rule[i][j] = rule[i - 1][j] + "-"
                elif aux[i][j] == aux[i][j - 1] + score["+"]:
                    rule[i][j] = rule[i][j - 1] + "+"
    # print("score", aux[m][n])
    return rule[m][n]


def genLongestContinuous(keys):
    tmp = 1
    longest = 0
    for i in range(1, len(keys)):
        if keys[i] - keys[i - 1] == 1:
            tmp += 1
        else:
            longest = max(longest, tmp)
            tmp = 1
    return longest


def genLongestContinuous2(keys):
    tmp = 1
    longest = 0
    for i in range(1, len(keys)):
        # 密钥中最长的连续操作
        if keys[i][0:1] == keys[i - 1][0:1] and int(keys[i][1:]) - int(keys[i - 1][1:]) == 1:
            tmp += 1
        else:
            longest = max(longest, tmp)
            tmp = 1
    return longest
