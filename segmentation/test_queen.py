import itertools
import math
import random

import matplotlib.pyplot as plt
import numpy as np

queen = 4


def calculate_distance(x1, y1, x2, y2):
    # return (x1 - x2) ** 2 + (y1 - y2) ** 2
    return abs(x1 - x2) + abs(y1 - y2)


def shortest_distance(solution):
    min_distance = float('inf')  # 初始化最小距离为正无穷
    for i in range(queen):
        for j in range(i + 1, queen):
            distance = calculate_distance(i, solution[i], j, solution[j])
            min_distance = min(min_distance, distance)  # 更新最小距离
    return min_distance


def total_distance(solution):
    total = 0
    for i in range(queen):
        for j in range(i + 1, queen):
            distance = calculate_distance(i, solution[i], j, solution[j])
            total += distance
    return total


def is_safe(board, row, col):
    # 检查列上是否有皇后
    for i in range(row):
        if board[i] == col:
            return False

    # 检查左上方对角线上是否有皇后
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i] == j:
            return False

    # 检查右上方对角线上是否有皇后
    for i, j in zip(range(row, -1, -1), range(col, 8)):
        if board[i] == j:
            return False

    return True


def solve_queens(board, row):
    solutions = []
    if row == queen:  # 所有皇后都放置完毕
        solutions.append(board[:])
        return solutions

    for col in range(queen):
        if is_safe(board, row, col):
            board[row] = col
            solutions += solve_queens(board, row + 1)
            # 如果在当前位置无法放置皇后，则回溯
            board[row] = -1

    return solutions


def print_solution(board):
    for row in range(queen):
        for col in range(queen):
            if board[row] == col:
                print("Q", end=" ")
            else:
                print(".", end=" ")
        print()


def eight_queens():
    # 初始化棋盘，-1表示该位置没有放置皇后
    board = [-1] * queen
    solutions = solve_queens(board, 0)

    if solutions:
        print("Found", len(solutions), "solutions:")
        for idx, solution, in enumerate(solutions, start=1):
            # print_solution(solution)
            # for row, col in enumerate(solution):
            #     print("Queen", row + 1, ":", "Row", row, "Column", col)
            print("Solution", idx, solution, shortest_distance(solution), total_distance(solution))

    else:
        print("No solution exists.")


eight_queens()
max_min_distance = float('-inf')  # 初始化最大最小距离为负无穷
max_min_distance_solutions = []

for combination in itertools.product(range(queen), repeat=queen):
    min_distance = shortest_distance(combination)
    if min_distance > max_min_distance:
        max_min_distance = min_distance
        max_min_distance_solutions = [combination]
    elif min_distance == max_min_distance:
        max_min_distance_solutions.append(combination)

print("所有结果中最大的最小距离为:", max_min_distance)
print("对应的所有解为:")
for idx, solution, in enumerate(max_min_distance_solutions, start=1):
    print("Solution", idx, solution, shortest_distance(solution), total_distance(solution))
print("共有", len(max_min_distance_solutions), "个解")

# 绘图比较
# 所有N皇后的排列种类数
A000170 = [1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512, 95815104, 666090624,
           4968057848, 39029188884, 314666222712, 2691008701644, 24233937684440, 227514171973736, 2207893435808352,
           22317699616364044, 234907967154122528, 2561327494111820312, 28718085567033706624, 330665665962404151344,
           3908888214702474981168, 47352584112327122589552, 608801080906364603555032, 8348329859782379155999728,
           121241622999832161754143280, 1872195402302350332281682160, 30459519939394722597877689936,
           518606028252562362573595786160, 9289360855238799393963914435760, 171759631608961495089393476144384,
           3328402859763924701864163793486368, 67435854536631070779671101303220992, 1421400066662527787309833362449857216]
# 全排列数
# 相当于N皇后中允许对角线有皇后，只限制皇后不能同行同列a
A000142 = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200,
           1307674368000, 20922789888000, 355687428096000, 6402373705728000, 121645100408832000, 2432902008176640000,
           51090942171709440000, 1124000727777607680000, 25852016738884976640000, 620448401733239439360000,
           15511210043330985984000000, 403291461126605650322784000, 10888869450418352160768000000,
           304888344611713860501504000000, 8841761993739701954543616000000, 265252859812191058636308480000000,
           8222838654177922817725562880000000, 263130836933693530167218012160000000,
           8683317618811886495518194401280000000, 295232799039604140847618609643520000000,]
# 2的幂
A000079 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288,
           1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824,
           2147483648, 4294967296, 8589934592, 17179869184, 34359738368, 68719476736, 137438953472, 274877906944,
           549755813888, 1099511627776, 2199023255552, 4398046511104, 8796093022208, 17592186044416, 35184372088832,
           70368744177664, 140737488355328, 281474976710656, 562949953421312, 1125899906842624, 2251799813685248,
           4503599627370496, 9007199254740992, 18014398509481984, 36028797018963968, 72057594037927936]

# 对数坐标绘图
plt.figure()
plt.plot(list(range(len(A000170))), A000170, label='A000170')
plt.plot(list(range(len(A000142))), A000142, label='A000142')
plt.plot(list(range(len(A000079))), A000079, label='A000079')
plt.yscale('log')
plt.legend()
plt.show()
