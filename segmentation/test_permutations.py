import itertools
import time

import numpy as np

# 是最优分段寻找中，占用时间最多的部分，partition分段其实不多
def unique_permutations(nums):
    def backtrack(path, counter):
        if len(path) == len(nums):
            result.append(path)
            return
        for num in counter:
            if counter[num] > 0:
                counter[num] -= 1
                backtrack(path + [num], counter)
                counter[num] += 1

    result = []
    counter = {}
    for num in nums:
        counter[num] = counter.get(num, 0) + 1
    backtrack([], counter)
    return result


# 迭代的话更慢
def unique_permutations_iter(nums):
    from collections import Counter

    result = []
    counter = Counter(nums)
    stack = [([], counter)]

    while stack:
        path, current_counter = stack.pop()

        if len(path) == len(nums):
            result.append(path)
            continue

        for num in list(current_counter.keys()):
            if current_counter[num] > 0:
                new_counter = current_counter.copy()
                new_counter[num] -= 1
                if new_counter[num] == 0:
                    del new_counter[num]
                stack.append((path + [num], new_counter))

    return result


# # 递归的方式返回不重复的全排列，调库的话慢很多
# nums = np.ones(5, dtype=int).tolist() + np.zeros(5, dtype=int).tolist()
# start_time = time.time()
# result = unique_permutations_iter(nums)
# end_time = time.time()
# print(end_time - start_time)
# print(len(result), result)
#
# start_time = time.time()
# result = unique_permutations(nums)
# end_time = time.time()
# print(end_time - start_time)
# print(len(result), result)
#
# # print(list(itertools.permutations(nums)))
# start_time = time.time()
# perm = list(set(itertools.permutations(nums)))
# end_time = time.time()
# print(end_time - start_time)
# print(len(perm), perm)