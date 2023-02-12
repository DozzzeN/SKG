import math
import sys
from collections import deque

import numpy as np

from algorithm import euclidean_metric
from binarytree import Node, NodeNotFoundError


def targetSum(nums, target):
    operators = []
    temp = []
    minimum = [sys.maxsize]
    result = [[], []]
    backtrack(nums, target, 0, temp, operators, minimum, result)
    return minimum[0], result


def backtrack(nums, target, index, temp, operators, minimum, result):
    if index == len(nums):
        if euclidean_metric(temp, target) < minimum[0]:
            minimum[0] = euclidean_metric(temp, target)
            result[0] = temp.copy()
            result[1] = operators.copy()
    else:
        temp.append(temp[-1] + nums[index]) if index > 0 else temp.append(nums[index])
        operators.append("+")
        backtrack(nums, target, index + 1, temp, operators, minimum, result)
        temp.pop()
        operators.pop()

        temp.append(temp[-1] - nums[index]) if index > 0 else temp.append(-nums[index])
        operators.append("-")
        backtrack(nums, target, index + 1, temp, operators, minimum, result)
        temp.pop()
        operators.pop()


# def findTargetMinDiffWays(nums, target):
#     dp = []
#     seq = []
#     n = len(nums)
#     for i in range(n):
#         seq_row = []
#         dp_row = []
#         for j in range(n * 2 + 1):
#             seq_row.append([])
#             dp_row.append([])
#         seq.append(seq_row)
#         dp.append(dp_row)
#
#     dp[0][n + 1].append(euclidean_metric([nums[0]], [target[0]]))
#     dp[0][n - 1].append(euclidean_metric([-nums[0]], [target[0]]))
#     seq[0][n + 1].append([nums[0]])
#     seq[0][n - 1].append([-nums[0]])
#
#     for i in range(1, n):
#         for j in range(n * 2 + 1):
#             left_father_min = []
#             right_father_min = []
#
#             if j - 1 >= 0:
#                 # 收集所有左父亲的序列末尾数字
#                 for k in range(len(seq[i - 1][j - 1])):
#                     number = seq[i - 1][j - 1][k][-1] + nums[i]
#                     tmp = seq[i - 1][j - 1][k].copy()
#                     tmp.append(number)
#                     seq[i][j].append(tmp)
#                     left_father_min.append(euclidean_metric([number], [target[i]]))
#
#             if j + 1 <= n * 2:
#                 # 收集所有右父亲的序列末尾数字
#                 for k in range(len(seq[i - 1][j + 1])):
#                     number = seq[i - 1][j + 1][k][-1] - nums[i]
#                     tmp = seq[i - 1][j + 1][k].copy()
#                     tmp.append(number)
#                     seq[i][j].append(tmp)
#                     right_father_min.append(euclidean_metric([number], [target[i]]))
#
#             if j - 1 < 0:
#                 tmp = dp[i - 1][j + 1].copy()
#                 for k in range(len(right_father_min)):
#                     dp[i][j].append(tmp + right_father_min[k])
#             if j + 1 > n * 2:
#                 tmp = dp[i - 1][j - 1].copy()
#                 for k in range(len(left_father_min)):
#                     dp[i][j].append(tmp + left_father_min[k])
#             else:
#                 tmp1 = dp[i - 1][j + 1].copy()
#                 for k in range(len(right_father_min)):
#                     dp[i][j].append(tmp1 + right_father_min[k])
#                 tmp2 = dp[i - 1][j - 1].copy()
#                 for k in range(len(left_father_min)):
#                     dp[i][j].append(tmp2 + left_father_min[k])
#
#     res_dp = min(dp[n - 1])
#     res_seq = []
#     # for i in range(n * 2 + 1):
#     #     if math.isclose(dp[n - 1][i], res_dp, rel_tol=1e-5):
#     #         for j in range(len(seq[n - 1][i])):
#     #             if math.isclose(euclidean_metric(seq[n - 1][i][j], target), res_dp, rel_tol=1e-5):
#     #                 res_seq.append(seq[n - 1][i][j])
#     return res_dp, res_seq


def findTargetMinDiffWays(nums, target):
    dp = []
    seq = []
    n = len(nums)
    for i in range(n):
        seq_row = []
        dp_row = []
        for j in range(n * 2 + 1):
            seq_row.append([])
            dp_row.append(-1)
        seq.append(seq_row)
        dp.append(dp_row)

    dp[0][n + 1] = euclidean_metric([nums[0]], [target[0]])
    dp[0][n - 1] = euclidean_metric([-nums[0]], [target[0]])
    seq[0][n + 1].append([nums[0]])
    seq[0][n - 1].append([-nums[0]])

    for i in range(1, n):
        for j in range(n * 2 + 1):
            left_father_min = sys.maxsize
            right_father_min = sys.maxsize

            if j - 1 >= 0:
                # 收集所有左父亲的序列末尾数字
                left_father_number = []
                for k in range(len(seq[i - 1][j - 1])):
                    number = seq[i - 1][j - 1][k][-1] + nums[i]
                    left_father_number.append(number)
                    left_father_min = min(left_father_min, euclidean_metric([number], [target[i]]))
                for k in range(len(left_father_number)):
                    number = left_father_number[k]
                    metric = euclidean_metric([number], [target[i]])
                    if math.isclose(left_father_min, metric, rel_tol=1e-5):
                        tmp = seq[i - 1][j - 1][k].copy()
                        tmp.append(number)
                        seq[i][j].append(tmp)

            if j + 1 <= n * 2:
                # 收集所有右父亲的序列末尾数字
                right_father_number = []
                for k in range(len(seq[i - 1][j + 1])):
                    number = seq[i - 1][j + 1][k][-1] - nums[i]
                    right_father_number.append(number)
                    right_father_min = min(right_father_min, euclidean_metric([number], [target[i]]))
                for k in range(len(right_father_number)):
                    number = right_father_number[k]
                    metric = euclidean_metric([number], [target[i]])
                    if math.isclose(right_father_min, metric, rel_tol=1e-5):
                        tmp = seq[i - 1][j + 1][k].copy()
                        tmp.append(number)
                        seq[i][j].append(tmp)

            if j - 1 < 0:
                dp[i][j] = dp[i - 1][j + 1] + right_father_min
            if j + 1 > n * 2:
                dp[i][j] = dp[i - 1][j - 1] + left_father_min
            else:
                dp[i][j] = min(dp[i - 1][j + 1] + right_father_min, dp[i - 1][j - 1] + left_father_min)

    res_dp = min(dp[n - 1])
    res_seq = []
    for i in range(n * 2 + 1):
        if math.isclose(dp[n - 1][i], res_dp, rel_tol=1e-5):
            max_tmp = sys.maxsize
            for j in range(len(seq[n - 1][i])):
                # if math.isclose(euclidean_metric(seq[n - 1][i][j], target), res_dp, rel_tol=1e-5):
                #     res_seq.append(seq[n - 1][i][j])
                if euclidean_metric(seq[n - 1][i][j], target) < max_tmp:
                    res_seq = seq[n - 1][i][j]
                    max_tmp = euclidean_metric(seq[n - 1][i][j], target)
    return res_dp, res_seq


def getLeavePath(root: Node, cur: list, seq: list):
    cur.append(root.value)
    if root.left is None and root.right is None:
        seq.append(cur)
        return
    if root.left is not None:
        # cur.copy()的原因：列表是不可变对象，见http://ask.sov5.cn/q/bZqxOKLvtm
        getLeavePath(root.left, cur.copy(), seq)
    if root.right is not None:
        getLeavePath(root.right, cur.copy(), seq)


def pruning(nums, target):
    root = Node(0)
    levels = len(nums)
    cur = [root]

    # 待完成：最好在层序构建二叉树的时候进行去重，否则复杂度没有变化
    for i in range(levels):
        num = int(math.pow(2, i))
        for j in range(num):
            cur[j].left = Node(cur[j].value + nums[i])
            cur[j].right = Node(cur[j].value - nums[i])
        tmp = []
        for j in range(num):
            tmp.append(cur[j].left)
            tmp.append(cur[j].right)
        cur = tmp

    # 收集出每层需要删除的，统一删除，但注意要从下往上删除，否则从上往下删除后报Node空指针
    # index = 0
    # need_del = []
    # for i in range(levels):
    #     num = int(math.pow(2, i))
    #     metric = []
    #     cur_del = {}
    #     for j in range(num):
    #         metric.append(root[index + j].value)
    #     for j in range(len(metric)):
    #         if metric.count(metric[j]) > 1:
    #             if cur_del.get(metric[j]) is None:
    #                 cur_del[metric[j]] = [j]
    #             else:
    #                 cur_del[metric[j]].append(j)
    #     need_del.append(cur_del)
    #     index += num
    #
    # index = 0
    # for i in range(levels):
    #     num = int(math.pow(2, i))
    #     if len(need_del[i]) != 0:
    #         for j in range(len(need_del[i])):
    #             cur_del = need_del[i]
    #             for v in cur_del.values():
    #                 for k in range(len(v) - 1):
    #                     root[index + k] = None
    #     index += num

    # naive try
    # index = 0
    # for i in range(levels):
    #     num = int(math.pow(2, i))
    #     metric = []
    #     for j in range(num):
    #         try:
    #             metric.append(root[index + j].value)
    #         except NodeNotFoundError:
    #             continue
    #     for j in range(len(metric) - 1, -1, -1):
    #         if metric.count(metric[j]) > 1:
    #             # 如果赋值为None会出错
    #             del metric[j]
    #             try:
    #                 root[index + j] = Node(sys.maxsize)
    #             except NodeNotFoundError:
    #                 continue
    #     index += num

    # leaves = root.leaves

    path = []
    getLeavePath(root, [], path)
    path_sift = []
    for i in range(len(path) - 1, -1, -1):
        if len(path[i]) == len(target) + 1:
            path_sift.append(path[i][1:])

    min_metric = euclidean_metric(path_sift[0], target)
    min_path = []
    for p in path_sift:
        min_metric = min(min_metric, euclidean_metric(p, target))
    for p in path_sift:
        metric = euclidean_metric(p, target)
        if math.isclose(metric, min_metric, rel_tol=1e-5):
            min_path.append(p)
    return min_metric, min_path

# validation with raw data
# b [-0.14347, -0.1384, -0.13286, -0.12683, -0.12029, -0.11326, -0.10573, -0.09773, -0.08928, -0.08042, -0.07118, -0.06157, -0.05162, -0.04131, -0.03076, -0.02022, -0.00986, 0.00045, 0.0116, 0.02532, 0.04238, 0.0618, 0.07979, 0.08815, 0.08324, 0.0772, 0.08896, 0.12565, 0.17308, 0.20216, 0.19776, 0.17725]
# a [-0.09114, -0.08759, -0.08378, -0.07969, -0.07533, -0.0707, -0.06581, -0.06068, -0.05532, -0.04976, -0.04393, -0.03769, -0.03094, -0.02375, -0.01666, -0.01067, -0.00625, -0.00287, 0.00164, 0.01215, 0.0308, 0.05048, 0.05968, 0.0532, 0.03867, 0.03341, 0.04842, 0.07872, 0.11049, 0.12981, 0.12976, 0.11534]
# rec a [-0.09114, -0.08759, -0.08378, -0.07969, -0.07533, -0.0707, -0.06581, -0.06068, -0.05532, -0.04976, -0.04393, -0.03769, -0.03094, -0.02375, -0.01666, -0.01067, -0.00625, -0.00287, 0.00164, 0.01215, 0.0308, 0.05048, 0.05968, 0.0532, 0.03867, 0.03341, 0.04842, 0.07872, 0.11049, 0.12981, 0.12976, 0.11534]
# rec b [-0.09114, -0.08759, -0.08378, -0.07969, -0.07533, -0.0707, -0.06581, -0.06068, -0.05532, -0.04976, -0.04393, -0.03769, -0.03094, -0.02375, -0.01666, -0.01067, -0.00625, -0.00287, 0.00164, 0.01215, 0.0308, 0.05048, 0.05968, 0.06616, 0.05163, 0.04637, 0.06138, 0.09168, 0.12345, 0.14277, 0.14272, 0.1283]
# diff a [0.00355, 0.00381, 0.00409, 0.00436, 0.00463, 0.00489, 0.00513, 0.00536, 0.00556, 0.00583, 0.00624, 0.00675, 0.00719, 0.00709, 0.00599, 0.00442, 0.00338, 0.00451, 0.01051, 0.01865, 0.01968, 0.0092, 0.00648, 0.01453, 0.00526, 0.01501, 0.0303, 0.03177, 0.01932, 5e-05, 0.01442]
# diff b [0.00507, 0.00554, 0.00603, 0.00654, 0.00703, 0.00753, 0.008, 0.00845, 0.00886, 0.00924, 0.00961, 0.00995, 0.01031, 0.01055, 0.01054, 0.01036, 0.01031, 0.01115, 0.01372, 0.01706, 0.01942, 0.01799, 0.00836, 0.00491, 0.00604, 0.01176, 0.03669, 0.04743, 0.02908, 0.0044, 0.02051]
# arr = [-0.09114, 0.00355, 0.00381, 0.00409, 0.00436, 0.00463, 0.00489, 0.00513, 0.00536, 0.00556, 0.00583, 0.00624,
#        0.00675, 0.00719, 0.00709, 0.00599, 0.00442, 0.00338, 0.00451, 0.01051, 0.01865, 0.01968, 0.0092, 0.00648,
#        0.01453, 0.00526, 0.01501, 0.0303, 0.03177, 0.01932, 5e-05, 0.01442]
# target = [-0.14347, -0.1384, -0.13286, -0.12683, -0.12029, -0.11326, -0.10573, -0.09773, -0.08928, -0.08042, -0.07118,
#           -0.06157, -0.05162, -0.04131, -0.03076, -0.02022, -0.00986, 0.00045, 0.0116, 0.02532, 0.04238, 0.0618,
#           0.07979, 0.08815, 0.08324, 0.0772, 0.08896, 0.12565, 0.17308, 0.20216, 0.19776, 0.17725]
# target = target - np.ones(len(target)) * (target[0] - arr[0])
retain = 7
arr = [1, 1, 2, 5, 2, 1, 2, 1, 2]
target = [1, 1, 0, 5, 2, 1, 1, 2, 1]

# for i in range(len(target)):
#     target[i] = round(target[i], 4)
#     arr[i] = round(arr[i], 4)
print(targetSum(arr[0:retain], target[0:retain]))
print(findTargetMinDiffWays(arr[0:retain], target[0:retain]))
print(pruning(arr[0:retain], target[0:retain]))
