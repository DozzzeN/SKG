import math
from typing import List

import numpy as np

# https://takeuforward.org/data-structure/target-sum-dp-21/
# https://leetcode.cn/problems/target-sum/solution/mu-biao-he-by-leetcode-solution-o0cp/
from scipy.stats import pearsonr

from algorithm import euclidean_metric, dtw_metric


def findWays(num, tar):
    mod = int(math.pow(10, 9) + 7)
    n = len(num)
    dp = np.zeros(shape=(n, tar + 1), dtype=int)

    if num[0] == 0:
        dp[0][0] = 2  # 两种结果，num[0]加或减0变为0
    else:
        dp[0][0] = 1  # 一种结果，num[0]减去num[0]变为0

    if num[0] != 0 and num[0] <= tar:
        dp[0][num[0]] = 1  # 一种结果，加上num[0]变为num[0]

    for ind in range(1, n):
        for target in range(tar + 1):
            notTaken = dp[ind - 1][target]  # 不选当前数num[ind]

            taken = 0
            if num[ind] <= target:
                taken = dp[ind - 1][target - num[ind]]

            dp[ind][target] = (notTaken + taken) % mod
    return dp[n - 1][tar]


def targetSum(target, arr):
    toSum = sum(arr)

    if toSum - target < 0 or (toSum - target) % 2 == 1:
        return 0
    return findWays(arr, int((toSum - target) / 2))


arr = [1, 2, 3, 1]
target = 3
print(targetSum(target, arr))


def findTargetSumWays(nums, S):
    d = {}

    def dfs(cur, i, d):
        if i < len(nums) and (cur, i) not in d:  # 搜索周围节点
            d[(cur, i)] = dfs(cur + nums[i], i + 1, d) + dfs(cur - nums[i], i + 1, d)
        return d.get((cur, i), int(math.isclose(cur, S, rel_tol=1e-5)))  # math.isclose(cur, S, rel_tol=1e-5)是浮点判等

    return dfs(0, 0, d)


# a [0.04747, 0.04441, 0.04135, 0.03828, 0.0352, 0.03212, 0.02903, 0.02593, 0.02284, 0.01974, 0.01665, 0.01355, 0.01045, 0.00736, 0.00427, 0.00119, -0.00188, -0.00495, -0.008, -0.01104, -0.01406, -0.01707, -0.02007, -0.02304, -0.02601, -0.02896, -0.03189, -0.03481, -0.03771, -0.04059, -0.04346, -0.04631]
# diff a [0.00306, 0.00306, 0.00307, 0.00308, 0.00308, 0.00309, 0.0031, 0.00309, 0.0031, 0.00309, 0.0031, 0.0031, 0.00309, 0.00309, 0.00308, 0.00307, 0.00307, 0.00305, 0.00304, 0.00302, 0.00301, 0.003, 0.00297, 0.00297, 0.00295, 0.00293, 0.00292, 0.0029, 0.00288, 0.00287, 0.00285]
arr = [0.04747, 0.00306, 0.00306, 0.00307]
target = 0.03828
print(findTargetSumWays(arr, target))

print(pearsonr([1, 2, 3, 5, 1], [1, 1, 2, 5, 1])[0])
print(pearsonr([1, 0, 2, 4, 0], [1, 1, 2, 5, 1])[0])
print(euclidean_metric([1, 2, 3, 5, 1], [1, 1, 2, 5, 1]))
print(euclidean_metric([1, 0, 2, 4, 0], [1, 1, 2, 5, 1]))


def targetSum(nums, target):
    count = [0]
    backtrack(nums, target, 0, 0, count)
    return count[0]

def backtrack(nums, target, index, sum, count):
    if index == len(nums):
        if math.isclose(sum, target, rel_tol=1e-5):
            count[0] += 1
    else:
        backtrack(nums, target, index + 1, sum + nums[index], count)
        backtrack(nums, target, index + 1, sum - nums[index], count)

print(targetSum(arr, target))