from typing import Tuple, List

import Levenshtein
import numpy as np


def weightedEditDistance(del_costs, ins_costs, sub_costs, str_a, str_b):
    matrix_ed = np.zeros((len(str_b) + 1, len(str_a) + 1))
    for i in range(1, len(str_a) + 1):
        matrix_ed[0][i] = matrix_ed[0][i - 1] + del_costs.get(str_a[i - 1])
    for j in range(1, len(str_b) + 1):
        matrix_ed[j][0] = matrix_ed[j - 1][0] + ins_costs.get(str_b[j - 1])
    for i in range(1, len(str_b) + 1):
        for j in range(1, len(str_a) + 1):
            # 表示删除a_i
            dist_1 = matrix_ed[i, j - 1] + del_costs.get(str_a[j - 1])
            # 表示插入b_i
            dist_2 = matrix_ed[i - 1, j] + ins_costs.get(str_b[i - 1])
            # 表示替换b_i
            dist_3 = matrix_ed[i - 1, j - 1] + (
                sub_costs.get(str_a[j - 1]).get(str_b[i - 1]) if str_a[j - 1] != str_b[i - 1] else 0)
            matrix_ed[i, j] = np.min([dist_1, dist_2, dist_3])
    ptr = []
    # for i in range(len(str_b), 0, -1):
    #     for j in range(len(str_a), 0, -1):
    #         diff_1 = matrix_ed[i][j] - matrix_ed[i - 1][j]  # 删除ai
    #         diff_2 = matrix_ed[i][j] - matrix_ed[i][j - 1]  # 插入bj
    #         diff_3 = matrix_ed[i][j] - matrix_ed[i - 1][j - 1]  # 将aj替换为bi
    #         if diff_1 >= max(diff_2, diff_3):
    #             ptr.append({'del': list(del_costs.keys())[list(del_costs.values()).index(diff_1)]})
    #         elif diff_2 >= max(diff_1, diff_3):
    #             ptr.append({'ins': list(ins_costs.keys())[list(ins_costs.values()).index(diff_2)]})
    #         elif diff_3 >= max(diff_1, diff_2):
    #             lists = sub_costs[str_b[i - 1]]
    #             ptr.append({'sub': list(lists.keys())[list(lists.values()).index(diff_3)]})
    #         elif min(abs(diff_1), abs(diff_2), abs(diff_3)) == 0:
    #             ptr.append({'copy': str_b[i - 1]})
    return matrix_ed[-1, -1], ptr


def minDistance(word1, word2) -> Tuple[int, List[List[int]]]:
    if len(word1) == 0:
        return len(word2), []
    elif len(word2) == 0:
        return len(word1), []
    M = len(word1)
    N = len(word2)
    output = [[0] * (N + 1) for _ in range(M + 1)]
    for i in range(M + 1):
        for j in range(N + 1):
            if i == 0 and j == 0:
                output[i][j] = 0
            elif i == 0 and j != 0:
                output[i][j] = j
            elif i != 0 and j == 0:
                output[i][j] = i
            elif word1[i - 1] == word2[j - 1]:
                output[i][j] = output[i - 1][j - 1]
            else:
                output[i][j] = min(output[i - 1][j - 1] + 2, output[i - 1][j] + 1, output[i][j - 1] + 1)
    return output[-1][-1], output


def backtrackingPath(word1, word2):
    _, dp = minDistance(word1, word2)
    m = len(dp) - 1
    n = len(dp[0]) - 1
    operation = []
    spokenstr = []
    writtenstr = []

    while n >= 0 or m >= 0:
        if n and dp[m][n - 1] + 1 == dp[m][n]:
            print("insert %c\n" % (word2[n - 1]))
            spokenstr.append("insert")
            writtenstr.append(word2[n - 1])
            operation.append("NULLREF:" + word2[n - 1])
            n -= 1
            continue
        if m and dp[m - 1][n] + 1 == dp[m][n]:
            print("delete %c\n" % (word1[m - 1]))
            spokenstr.append(word1[m - 1])
            writtenstr.append("delete")
            operation.append(word1[m - 1] + ":NULLHYP")
            m -= 1
            continue
        if dp[m - 1][n - 1] + 1 == dp[m][n]:
            print("replace %c %c\n" % (word1[m - 1], word2[n - 1]))
            spokenstr.append(word1[m - 1])
            writtenstr.append(word2[n - 1])
            operation.append(word1[m - 1] + ":" + word2[n - 1])
            n -= 1
            m -= 1
            continue
        if dp[m - 1][n - 1] == dp[m][n]:
            print("copy %c %c\n" % (word1[m - 1], word2[n - 1]))
            spokenstr.append(' ')
            writtenstr.append(' ')
            operation.append(word1[m - 1])
        n -= 1
        m -= 1
    spokenstr = spokenstr[::-1]
    writtenstr = writtenstr[::-1]
    operation = operation[::-1]
    # print(spokenstr)
    # print(writtenstr)
    print(operation)
    return spokenstr, writtenstr, operation


del_costs = {'l': 1.1, 'o': 2.1, 'v': 3.1, 'e': 4.1, 'g': 5.1}
ins_costs = {'l': 2.5, 'o': 3.5, 'v': 4.5, 'e': 5.5, 'g': 6.5}
sub_costs = {'l': {'l': 0, 'o': 2, 'v': 3, 'e': 4, 'g': 5},
             'o': {'l': 2, '0': 0, 'v': 3, 'e': 4, 'g': 5},
             'v': {'l': 3, 'o': 4, 'v': 0, 'e': 5, 'g': 6},
             'e': {'l': 4, 'o': 5, 'v': 6, 'e': 0, 'g': 7},
             'g': {'l': 5, 'o': 6, 'v': 7, 'e': 8, 'g': 0}}
str1 = "ll"
str2 = "lv"
# backtrackingPath(str1, str2)
# print(editDistance("execution", "intention"))
# print(weightedEditDistance(del_costs, ins_costs, sub_costs, str1, str2))
# print(Levenshtein.distance(str1, str2))
# print(Levenshtein.editops(str1, str2))
# 11113334444333311
#  1113333444333111
print(max(len("11113334444333311"), len("1111555544441111")))
print(Levenshtein.distance("11113334444333311","1111555544441111"))
print(Levenshtein.editops("11113334444333311","1111555544441111"))

# print(Levenshtein.hamming('Hello world!', 'Holly world!'))
# print(Levenshtein.jaro_winkler("yukangrtyu", 'yukangrtyn'))
# fixme = ['Levnhtein', 'Leveshein', 'Leenshten', 'Leveshtei', 'Lenshtein', 'Lvenstein', 'Levenhtin', 'evenshtei']
# print(Levenshtein.opcodes('spam', 'park'))
# print(Levenshtein.ratio('spam', 'spark'))
# print(Levenshtein.jaro_winkler('spam', 'spark'))
# print(Levenshtein.jaro('spam', 'spark'))
# print(Levenshtein.seqratio('spam', 'spark'))
# print(Levenshtein.seqratio(['newspaper', 'litter bin', 'tinny', 'antelope'], ['caribou', 'sausage', 'gorn', 'woody']))
# print(Levenshtein.setratio(['newspaper', 'litter bin', 'tinny', 'antelope'], ['caribou', 'sausage', 'gorn', 'woody']))
e = Levenshtein.editops('man', 'scotsman')
print(e)
# e1 = e[:3]
# print(e1)
# bastard = Levenshtein.apply_edit(e1, 'man', 'scotsman')
# print(bastard)
# print(Levenshtein.subtract_edit(e, e1))
# print(Levenshtein.apply_edit(Levenshtein.subtract_edit(e, e1), bastard, 'scotsman'))
