import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from itertools import permutations

import numpy as np

from segmentation.test_permutation_of_segment import count_permutations
from segmentation.test_permutations import unique_permutations


def partition(length, min_length=2, max_length=None, num_partitions=None):
    if max_length is None:
        max_length = length

    if length == 0 and (num_partitions is None or num_partitions == 0):
        return [[]]
    if length < min_length or (num_partitions is not None and num_partitions == 0):
        return []

    partitions = []
    for size in range(min_length, min(max_length, length) + 1):
        if num_partitions is None:
            next_partitions = partition(length - size, min_length, size)
        else:
            next_partitions = partition(length - size, min_length, size, num_partitions - 1)
        for sub_partition in next_partitions:
            partitions.append([size] + sub_partition)
    return partitions


# 迭代更快，但是快不了太多
def partition_iter(length, min_length=2, max_length=None, num_partitions=None):
    if max_length is None:
        max_length = length

    stack = [(length, min_length, max_length, num_partitions, [])]
    partitions = []

    while stack:
        current_length, current_min, current_max, current_num_partitions, current_partition = stack.pop()

        if current_length == 0 and (current_num_partitions is None or current_num_partitions == 0):
            partitions.append(current_partition)
            continue
        if current_length < current_min or (current_num_partitions is not None and current_num_partitions == 0):
            continue

        for size in range(current_min, min(current_max, current_length) + 1):
            if current_num_partitions is None:
                stack.append((current_length - size, min_length, size, None, current_partition + [size]))
            else:
                stack.append(
                    (current_length - size, min_length, size, current_num_partitions - 1, current_partition + [size]))

    return partitions


def partition_with_occupy(length, min_length=2, max_length=None, occupied_segments=None, num_partitions=None):
    if max_length is None:
        max_length = length

    if occupied_segments is None:
        occupied_segments = []

    # Create a list representing the availability of each position
    availability = [True] * length
    for start, end in occupied_segments:
        for i in range(start, end):
            availability[i] = False

    def is_available(start, end):
        return all(availability[i] for i in range(start, end))

    def helper(remaining_length, min_length, max_length, num_partitions, current_segments, current_start):
        if remaining_length == 0 and (num_partitions is None or num_partitions == 0):
            return [current_segments]
        if remaining_length < min_length or (num_partitions is not None and num_partitions == 0):
            return []

        partitions = []
        for size in range(min_length, min(max_length, remaining_length) + 1):
            for start in range(current_start, length - size + 1):
                end = start + size
                if is_available(start, end):
                    new_segments = current_segments + [(start, end)]
                    next_partitions = helper(
                        remaining_length - size,
                        min_length,
                        max_length,
                        num_partitions - 1 if num_partitions else None,
                        new_segments,
                        end
                    )
                    partitions.extend(next_partitions)
        return partitions

    intervals = helper(length - sum([index[1] - index[0] for index in occupied_segments]),
                       min_length, max_length, num_partitions, [], 0)
    segment_methods = []
    for interval in intervals:
        segment_methods.append([index[1] - index[0] for index in interval])
    return segment_methods


def partition_with_occupy_dp(length, min_length=2, max_length=None, occupied_segments=None, num_partitions=None):
    if max_length is None:
        max_length = length

    if occupied_segments is None:
        occupied_segments = []

    # Create a list representing the availability of each position
    availability = [True] * length
    for start, end in occupied_segments:
        for i in range(start, end):
            availability[i] = False

    def is_available(start, end):
        return all(availability[i] for i in range(start, end))

    # Dynamic programming table to store the solutions
    dp = {}

    def dp_helper(remaining_length, remaining_partitions, current_start):
        if (remaining_length, remaining_partitions, current_start) in dp:
            return dp[(remaining_length, remaining_partitions, current_start)]

        if remaining_length == 0 and (remaining_partitions is None or remaining_partitions == 0):
            return [[]]
        if remaining_length < min_length or (remaining_partitions is not None and remaining_partitions == 0):
            return []

        partitions = []
        for size in range(min_length, min(max_length, remaining_length) + 1):
            for start in range(current_start, length - size + 1):
                end = start + size
                if is_available(start, end):
                    next_partitions = dp_helper(
                        remaining_length - size,
                        remaining_partitions - 1 if remaining_partitions else None,
                        end
                    )
                    for partition in next_partitions:
                        partitions.append([(start, end)] + partition)

        dp[(remaining_length, remaining_partitions, current_start)] = partitions
        return partitions

    results = dp_helper(length - sum([index[1] - index[0] for index in occupied_segments]), num_partitions, 0)
    segment_methods = []
    for interval in results:
        segment_methods.append([index[1] - index[0] for index in interval])
    return segment_methods


def partition_with_occupy_dp_lru(length, min_length=2, max_length=None, occupied_segments=None, num_partitions=None):
    if max_length is None:
        max_length = length

    if occupied_segments is None:
        occupied_segments = []

    # Create a list representing the availability of each position
    availability = [True] * length
    for start, end in occupied_segments:
        for i in range(start, end):
            availability[i] = False

    def is_available(start, end):
        return all(availability[i] for i in range(start, end))

    @lru_cache(maxsize=None)
    def dp_helper(remaining_length, remaining_partitions, current_start):
        if remaining_length == 0 and (remaining_partitions is None or remaining_partitions == 0):
            return [[]]
        if remaining_length < min_length or (remaining_partitions is not None and remaining_partitions == 0):
            return []

        partitions = []
        for size in range(min_length, min(max_length, remaining_length) + 1):
            for start in range(current_start, length - size + 1):
                end = start + size
                if is_available(start, end):
                    next_partitions = dp_helper(
                        remaining_length - size,
                        remaining_partitions - 1 if remaining_partitions else None,
                        end
                    )
                    for partition in next_partitions:
                        partitions.append([(start, end)] + partition)

        return partitions

    results = dp_helper(length - sum([index[1] - index[0] for index in occupied_segments]), num_partitions, 0)
    segment_methods = []
    for interval in results:
        segment_methods.append([index[1] - index[0] for index in interval])
    return segment_methods


def partition_with_occupy_dp_lru_parallel(length, min_length=2, max_length=None, occupied_segments=None,
                                          num_partitions=None):
    if max_length is None:
        max_length = length

    if occupied_segments is None:
        occupied_segments = []

    # Create a list representing the availability of each position
    availability = [True] * length
    for start, end in occupied_segments:
        for i in range(start, end):
            availability[i] = False

    def is_available(start, end):
        return all(availability[i] for i in range(start, end))

    # @lru_cache(maxsize=None)
    def dp_helper(remaining_length, remaining_partitions, current_start):
        if remaining_length == 0 and (remaining_partitions is None or remaining_partitions == 0):
            return [[]]
        if remaining_length < min_length or (remaining_partitions is not None and remaining_partitions == 0):
            return []

        partitions = []
        for size in range(min_length, min(max_length, remaining_length) + 1):
            for start in range(current_start, length - size + 1):
                end = start + size
                if is_available(start, end):
                    next_partitions = dp_helper(
                        remaining_length - size,
                        remaining_partitions - 1 if remaining_partitions else None,
                        end
                    )
                    for partition in next_partitions:
                        partitions.append([(start, end)] + partition)

        return partitions

    # Using ThreadPoolExecutor for parallel execution
    results = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for start in range(length):
            futures.append(executor.submit(dp_helper, length - start - sum(
                [index[1] - index[0] for index in occupied_segments]), num_partitions, start))

        for future in as_completed(futures):
            result = future.result()
            results.extend(result)

    segment_methods = []
    for interval in results:
        segment_methods.append([index[1] - index[0] for index in interval])
    return segment_methods


def partition_with_occupy_iter(length, min_length=2, max_length=None, occupied_segments=None, num_partitions=None):
    if max_length is None:
        max_length = length

    if occupied_segments is None:
        occupied_segments = []

    # Create a list representing the availability of each position
    availability = [True] * length
    for start, end in occupied_segments:
        for i in range(start, end):
            availability[i] = False

    def is_available(start, end):
        return all(availability[i] for i in range(start, end))

    # Stack to store state during iteration
    stack = [(0, length - sum([index[1] - index[0] for index in occupied_segments]), num_partitions, [])]
    results = []

    while stack:
        current_start, remaining_length, remaining_partitions, current_segments = stack.pop()

        if remaining_length == 0 and (remaining_partitions is None or remaining_partitions == 0):
            results.append(current_segments)
            continue
        if remaining_length < min_length or (remaining_partitions is not None and remaining_partitions == 0):
            continue

        for size in range(min_length, min(max_length, remaining_length) + 1):
            for start in range(current_start, length - size + 1):
                end = start + size
                if is_available(start, end):
                    new_segments = current_segments + [(start, end)]
                    stack.append((end, remaining_length - size,
                                  remaining_partitions - 1 if remaining_partitions else None, new_segments))

    segment_methods = []
    for interval in results:
        segment_methods.append([index[1] - index[0] for index in interval])
    return segment_methods


def partition_with_occupy_dp_iter(length, min_length=2, max_length=None, occupied_segments=None, num_partitions=None):
    if max_length is None:
        max_length = length

    if occupied_segments is None:
        occupied_segments = []

    # Create a list representing the availability of each position
    availability = [True] * length
    for start, end in occupied_segments:
        for i in range(start, end):
            availability[i] = False

    def is_available(start, end):
        return all(availability[i] for i in range(start, end))

    # Memoization table to store the solutions
    dp = {}

    # Stack for iterative processing
    stack = [(length - sum(end - start for start, end in occupied_segments), num_partitions, 0, [])]
    results = []

    while stack:
        remaining_length, remaining_partitions, current_start, partitions = stack.pop()

        if (remaining_length, remaining_partitions, current_start) in dp:
            partitions.extend(dp[(remaining_length, remaining_partitions, current_start)])
            continue

        if remaining_length == 0 and (remaining_partitions is None or remaining_partitions == 0):
            results.append(partitions)
            continue

        if remaining_length < min_length or (remaining_partitions is not None and remaining_partitions == 0):
            continue

        current_partitions = []
        for size in range(min_length, min(max_length, remaining_length) + 1):
            for start in range(current_start, length - size + 1):
                end = start + size
                if is_available(start, end):
                    next_partitions = dp.get((remaining_length - size,
                                              remaining_partitions - 1 if remaining_partitions else None,
                                              end), None)
                    if next_partitions is None:
                        stack.append((remaining_length - size,
                                      remaining_partitions - 1 if remaining_partitions else None,
                                      end,
                                      partitions + [(start, end)]))
                    else:
                        for partition in next_partitions:
                            current_partitions.append([(start, end)] + partition)

        dp[(remaining_length, remaining_partitions, current_start)] = current_partitions

        if current_partitions:
            partitions.extend(current_partitions)

    segment_methods = [[index[1] - index[0] for index in interval] for interval in results]
    return segment_methods



# length = 4 * 8 + 4
# min_length = 3
# max_length = 5
# # 由于传递gaps,导致某些分段已暴露,计算剩余分段时,不再考虑这些分段
# occupied_segments = [(0, 4)]  # Segments are inclusive of start and exclusive of end
# # 速度最快
# start_time = time.time_ns()
# partitions = partition_with_occupy_dp_lru(length, min_length, max_length, occupied_segments)
# print("time", (time.time_ns() - start_time) / 1e6)
# print(len(partitions), partitions)
#
# start_time = time.time_ns()
# partitions = partition_with_occupy_dp(length, min_length, max_length, occupied_segments)
# print("time", (time.time_ns() - start_time) / 1e6)
# print(len(partitions), partitions)
#
# # 计算中有错误
# start_time = time.time_ns()
# partitions = partition_with_occupy_dp_iter(length, min_length, max_length, occupied_segments)
# print("time", (time.time_ns() - start_time) / 1e6)
# print(len(partitions), partitions)


# 计算中有错误
# start_time = time.time_ns()
# partitions = partition_with_occupy_dp_lru_parallel(length, min_length, max_length, occupied_segments)
# print("time", (time.time_ns() - start_time) / 1e6)
# print(len(partitions), partitions)

# start_time = time.time_ns()
# partitions = partition_with_occupy_iter(length, min_length, max_length, occupied_segments)
# print("time", (time.time_ns() - start_time) / 1e6)
# print(len(partitions), partitions)

# start_time = time.time_ns()
# partitions = partition_with_occupy(length, min_length, max_length, occupied_segments)
# print("time", (time.time_ns() - start_time) / 1e6)
# print(len(partitions), partitions)


# 总数据量  最小分段长度  最大分段长度  分段数 分段类型数   总分段类型数
# 4 * 4     1            16           4     34          455[C(15, 3)]
# 4 * 4     3            5            4     3           19
# 4 * 4     3            5            /     4           24
# 4 * 4     3            /            4     5           35
# 4 * 4     3            /            /     21          88

# 4 * 8     1            32           8     919         2629575[C(31, 7)]
# 4 * 8     3            5            8     5           1107
# 4 * 8     3            5            /     12          2121
# 4 * 8     3            /            4     108         1771
# 4 * 8     3            /            /     468         39865

# 4 * 16    3            5            16    9           5196627
# 4 * 32    3            5            32    17
# 4 * 64    3            5            64    33
# 4 * 128   3            5            128   65
# 4 * 256   3            5            256   129
# 4 * 512   3            5            512   257
# 4 * 1024  3            5            1024  513

# all_segments = partition(4 * 16, 3, 5)
# print("分段方法数:", len(all_segments)) if len(all_segments) > 0 else None
# print(all_segments)
# # segment_methods = []
# # for segment in all_segments:
# #     segment_methods += unique_permutations(segment)
# # print("分段方法数:", len(segment_methods)) if len(segment_methods) > 0 else None
#
# segment_method_lens = []
# for segment in all_segments:
#     segment_method_lens.append(len(segment))
#
# segment_method_lens = np.zeros(max(segment_method_lens) + 1)
# for segment in all_segments:
#     different_lengths = list(Counter(segment).values())
#     segment_method_lens[len(segment)] += count_permutations(different_lengths)
# results = {}
# for i, count in enumerate(segment_method_lens):
#     if count != 0:
#         results[i] = int(count)
# print("分段方法数:", results, sum(results.values()))

# segment_lengths = []
# for segment in segment_methods:
#     segment_lengths.append(len(segment))
# print("分段长度类型:", Counter(segment_lengths))


# all_segments = partition(4 * 16, 3, 5, 16)
# print("分段方法数:", len(all_segments)) if len(all_segments) > 0 else None
# print(all_segments)
# segment_methods = []
# for segment in all_segments:
#     segment_methods += unique_permutations(segment)
# print("分段方法数:", len(segment_methods)) if len(segment_methods) > 0 else None

# nums = 8
# all_segments_sum = []
# segment_methods_sum = []
# for i in range(4 * nums):
#     print("分段数", i)
#     all_segments = partition(4 * nums, 3, 4 * nums, i)
#     print("分段方法数:", len(all_segments)) if len(all_segments) > 0 else None
#     # print(all_segments)
#     all_segments_sum.extend(all_segments)
#     segment_methods = []
#     for segment in all_segments:
#         segment_methods += unique_permutations(segment)
#     print("分段方法数:", len(segment_methods)) if len(segment_methods) > 0 else None
#     segment_methods = []
#     for segment in all_segments:
#         all_permutations = list(set(permutations(segment)))
#         for permutation in all_permutations:
#             if permutation not in segment_methods:
#                 segment_methods.append(permutation)
#     segment_methods_sum.extend(segment_methods)
#     print("分段方法数:", len(segment_methods)) if len(segment_methods) > 0 else None
# print(len(all_segments_sum), len(segment_methods_sum))

# min_length = 3
# max_length = 5
# num_segments = 256
#
# res_s = time.time_ns()
# res = partition(4 * num_segments, min_length, max_length, num_segments)
# res_e = time.time_ns()
# res = sorted(res)
# print(len(res))
# res2_s = time.time_ns()
# res2 = partition_iter(4 * num_segments, min_length, max_length, num_segments)
# res2_e = time.time_ns()
# res2 = sorted(res2)
# print(len(res2))
# print(np.array_equal(res, res2))
#
# print("time", (res_e - res_s) / 1e9, (res2_e - res2_s) / 1e9)


def count_partitions(n, k):
    # 初始化动态规划数组
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 1

    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            dp[i][j] = dp[i - 1][j - 1] + dp[i - j][j]

    return dp[n][k]


def math_count_partitions(n, k):
    # 计算组合数
    partitions = math.comb(n - 1, k - 1)
    return partitions


def count_partition_methods(n, k):
    # 初始化动态规划数组
    dp = [0] * (n + 1)
    dp[0] = 1

    # 计算划分方法数
    for i in range(k, n + 1):
        for j in range(i, n + 1):
            dp[j] += dp[j - i]

    return dp[n]


# 示例
# length = 16
# min_length = 2
# num_segment = 4
#
# result1 = partition(length, min_length, None)
# all_segments = []
# for segment in result1:
#     all_permutations = list(permutations(segment))
#     for permutation in all_permutations:
#         if permutation not in all_segments:
#             all_segments.append(permutation)
# print("长为", length, "的序列划分成子序列的方法数:", len(all_segments))
# print(all_segments)
# print()
#
# result = partition(length, min_length, num_partitions=num_segment)
# all_segments = []
# for segment in result:
#     all_permutations = list(permutations(segment))
#     for permutation in all_permutations:
#         if permutation not in all_segments:
#             all_segments.append(permutation)
# print("长为", length, "的序列划分成", num_segment, "个子序列的方法数:", len(all_segments))
# print(all_segments)
# print()
#
# num_segment = 5
# result = partition(length, min_length, num_partitions=num_segment)
# all_segments = []
# for segment in result:
#     all_permutations = list(permutations(segment))
#     for permutation in all_permutations:
#         if permutation not in all_segments:
#             all_segments.append(permutation)
# print("长为", length, "的序列划分成", num_segment, "个子序列的方法数:", len(all_segments))
# print(all_segments)
# print()
#
# # # 等于result1
# # result = count_partition_methods(length, min_length)
# # print("长为", length, "的序列划分成子序列的方法数:", result)
# # print()
# #
# # result = math_count_partitions(length, num_segment)
# # print("长为", length, "的序列分成", num_segment, "个子序列的划分方法数:", result)
# # print()


def generate_subsegments(N, k):
    # 动态规划表，用于记录所有可能的分段方案
    dp = [[[] for _ in range(k + 1)] for _ in range(N + 1)]
    dp[0][0] = [[]]

    # 填充DP表
    for i in range(2, N + 1, 2):
        for j in range(1, k + 1):
            for m in range(2, i + 1, 2):
                if dp[i - m][j - 1]:
                    for sublist in dp[i - m][j - 1]:
                        dp[i][j].append(sublist + [m])

    # 筛选结果，使每种分段长度之和等于N
    result = []
    for segments in dp[N][k]:
        if sum(segments) == N:
            result.append(segments)
    return result

# 示例使用
# N = 16
# k = 4
# result = generate_subsegments(N, k)
# all_segments = []
# for segment in result:
#     all_permutations = list(permutations(segment))
#     for permutation in all_permutations:
#         if permutation not in all_segments:
#             all_segments.append(permutation)
# print("长为", N, "的序列分成", k, "个子序列的划分方法数，以2为基数:", len(result))
# print(all_segments)
