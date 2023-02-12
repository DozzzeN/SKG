import sys
from math import fabs

# a1P = [1.5319139376546782, 1.0040445442585764, 1.1377085048956799, 1.756917361717805, 1.1309995464745557, 1.8487188404112602, 1.4630473938963906, 1.279332881933639, 0.9809128832096397, 1.0673085433880498, 1.6830179416362794, 1.781251630697289, 1.7091294499206269, 1.558909225408994, 1.2478688568666112]
# a1 =  [0.9642606125846311, 1.6936985591442413, 1.6952198315329703, 0.9869218831607831, 1.6860112235007376, 1.4638441716420556, 1.4630473938963906, 1.6821761124224592, 0.9809128832096397, 1.6816943335021606, 1.6830179416362794, 0.9748195734365094, 1.7091294499206269, 1.558909225408994, 1.58058358980052]
# b1 =  [0.9568085108888017, 1.6901960800285136, 1.6922004289621744, 0.9852767431792937, 1.6824459385139579, 1.4599952560473914, 1.4593924877592308, 1.678943057620203, 0.9761970853273751, 1.6773635823123854, 1.6794278966121188, 0.9731278535996986, 1.707229419327294, 1.5545295388316016, 1.5764181417331617]
a1P = [1.5319139376546782, 1.0040445442585764, 1.1377085048956799, 1.756917361717805, 1.1309995464745557,
       1.8487188404112602, 1.4630473938963906, 1.279332881933639, 0.9809128832096397, 1.0673085433880498,
       1.6830179416362794, 1.781251630697289, 1.7091294499206269]
a1 = [0.9642606125846311, 1.6936985591442413, 1.6952198315329703, 0.9869218831607831, 1.6860112235007376,
      1.4638441716420556, 1.4630473938963906, 1.6821761124224592, 0.9809128832096397, 1.6816943335021606,
      1.6830179416362794, 0.9748195734365094, 1.7091294499206269]
b1 = [0.9568085108888017, 1.6901960800285136, 1.6922004289621744, 0.9852767431792937, 1.6824459385139579,
      1.4599952560473914, 1.4593924877592308, 1.678943057620203, 0.9761970853273751, 1.6773635823123854,
      1.6794278966121188, 0.9731278535996986, 1.707229419327294]


def genAlign(base, strB):
    align = []
    j = 0
    for i in range(len(base)):
        if base[i] == "=":
            align.append(strB[j])
            j += 1
        elif base[i] == "-":
            j += 1
        elif base[i] == "~":
            j += 1
        elif base[i] == "^":
            align.append(strB[j + 1])
            align.append(strB[j])
            j += 2
    return align


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


def alignFloat(score, arrayA, arrayB, threshold):
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
    # print("max diff of AB       ", diffAB)
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
                if aux[i][j] == aux[i - 1][j - 1] + score["~"] and not equal(aux[i][j], aux[i - 1][j - 1]):
                    rule[i][j] = rule[i - 1][j - 1] + "~"
                elif aux[i][j] == aux[i - 1][j] + score["-"]:
                    rule[i][j] = rule[i - 1][j] + "-"
                elif aux[i][j] == aux[i][j - 1] + score["+"]:
                    rule[i][j] = rule[i][j - 1] + "+"
    # print("score", aux[m][n])
    return rule[m][n]


# arrayA = [60.4413907, 37.57888647, 37.95410587, 57.37781942, 41.66437795, 19.26346875, 49.65183698, 47.08151645]
# arrayA = [37.57888647, 60.4413907, 37.95410587, 41.66437795, 57.37781942, 19.26346875, 49.65183698, 47.08151645]
# arrayA = [10.323232, 60.4413907, 37.57888647, 30.233424, 32.3424232, 37.95410587, 57.37781942, 41.66437795, 19.26346875, 49.65183698, 47.08151645]
# arrayA = [60.4413907, 37.95410587, 57.37781942, 19.26346875, 47.08151645]
arrayA = [10.4413907, 27.57888647, 37.95410587, 57.37781942, 31.66437795, 19.26346875, 49.65183698, 27.08151645]
arrayB = [60.4413907, 37.57888647, 37.95410587, 57.37781942, 41.66437795, 19.26346875, 49.65183698, 47.08151645]
# print(re.search('G\w*W\w*', strA))
rule = {"=": 2, "+": -1, "-": -1, "~": 1, "^": -1}  # B复制A的，B添加一位，B删去一位，B替换成A，B交换相邻两位
# ruleArray1 = alignFloat(rule, arrayA, arrayB)
# print(ruleArray1)
# alignStr1 = generate(ruleArray1, arrayA)
# print(alignStr1)

# ruleStr1 = align(rule, strA, strPB)
# print(ruleStr1)
# alignStr1 = generate(ruleStr1, strA)
# print(alignStr1)

# ruleStr2 = align(rule, strB, strPA)
# print(ruleStr2)
# alignStr2 = generate(ruleStr2, strB)
# print(alignStr2)
#
# ruleStrE = align(rule, strPA, strPB)
# print(ruleStrE)
# print(generate(ruleStrE, strPA))
# print(generate(ruleStrE, strPB))
threshold = 0.1
ruleStr1 = alignFloat(rule, a1P, a1, threshold)
alignStr1 = generate(ruleStr1, a1P)
print("ruleStr1", ruleStr1)
ruleStr2 = alignFloat(rule, a1P, b1, threshold)
alignStr2 = generate(ruleStr2, a1P)
print("ruleStr2", ruleStr2)

for i in range(min(len(ruleStr1), len(ruleStr2))):
    if ruleStr1[i] != ruleStr2[i]:
        if i >= len(a1) or i >= len(a1P): continue
        print("\033[0;35;40m", a1[i], b1[i], a1P[i], "\033[0m")
        print(ruleStr1[i], abs(a1[i] - a1P[i]))
        print(ruleStr2[i], abs(a1[i] - a1P[i]))

sum1 = min(len(alignStr1), len(alignStr2))
sum2 = 0
for i in range(0, sum1):
    sum2 += (alignStr1[i] == alignStr2[i])
print(sum2, sum2 / sum1)
