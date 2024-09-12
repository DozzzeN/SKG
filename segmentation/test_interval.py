def find_segments(arr, min_length, max_length):
    from collections import defaultdict

    # 将每个键值对转换为边界
    segments = defaultdict(set)
    for d in arr:
        for key, value in d.items():
            segments[value].add(key)

    # 合并边界形成连续的段，找出长度在[min_length, max_length]之间的段
    all_segments = []
    for key, value in segments.items():
        # 找到起始位置
        start = int(key[:str(key).find('-')])
        if len(value) >= min_length and len(value) <= max_length:
            # 将该段的起止索引加入到结果中
            if [start, start + len(value) - 1] not in all_segments:
                all_segments.append([start, start + len(value)])

    return all_segments


def find_longest_consecutive_subarray(arr):
    if not arr:
        return []

    max_length = 1
    current_length = 1
    start_index = 0
    max_start_index = 0

    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1] + 1:
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
                max_start_index = start_index
            current_length = 1
            start_index = i

    if current_length > max_length:
        max_length = current_length
        max_start_index = start_index

    return arr[max_start_index:max_start_index + max_length]


def find_all_consecutive_subarrays(arr):
    if not arr:
        return []

    subarrays = []
    start_index = 0

    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1] + 1:
            subarrays.append(arr[start_index:i])
            start_index = i

    # 最后一个子段
    subarrays.append(arr[start_index:])

    return subarrays


def find_min_cover_intervals(intervals, target_range):
    # 按起始时间排序，如果起始时间相同则按结束时间排序
    intervals.sort(key=lambda x: (x[0], x[1]))

    start, end = target_range
    current_end = start
    index = 0
    n = len(intervals)
    selected_intervals = []

    while current_end < end:
        best_interval = None
        # 尝试在当前覆盖范围内找到尽可能延长的区间
        while index < n and intervals[index][0] <= current_end:
            if best_interval is None or intervals[index][1] > best_interval[1]:
                best_interval = intervals[index]
            index += 1

        if best_interval is None:
            # 无法覆盖整个区间
            return []

        selected_intervals.append(best_interval)
        current_end = best_interval[1]

        # 如果已经覆盖到或超过目标区间的结束点，停止
        if current_end >= end:
            break

    # 检查最后的 current_end 是否达到了目标 end
    if current_end < end:
        return []

    return selected_intervals


def generate_subintervals(interval, min_length, max_length):
    # 给定单个区间，生成所有可能的子区间，长度在[min_length, max_length]之间
    start, end = interval
    subintervals = []
    for split_point in range(start + 1, end):
        # 生成两个子任务，确保它们的长度在min_length和max_length之间
        subtask1 = [start, split_point]
        subtask2 = [split_point, end]
        if min_length <= subtask1[1] - subtask1[0] <= max_length:
            subintervals.append(subtask1)
        if min_length <= subtask2[1] - subtask2[0] <= max_length:
            subintervals.append(subtask2)
    # 包括原始区间本身，如果它的长度在范围内
    if min_length <= end - start <= max_length:
        subintervals.append(interval)
    return subintervals


def generate_all_subintervals(intervals, min_length, max_length):
    # 给定一些区间，生成所有可能的子区间，长度在[min_length, max_length]之间
    all_subintervals = []
    for interval in intervals:
        subintervals = generate_subintervals(interval, min_length, max_length)
        for subinterval in subintervals:
            if subinterval not in all_subintervals:
                all_subintervals.append(subinterval)
    return all_subintervals


def find_all_cover_intervals(intervals, target_range, min_length, max_length):
    # 给定一些区间，找到所有可能的区间分段方法，使得每个分段的长度在[min_length, max_length]之间
    def backtrack(start, current_cover):
        # 如果已经覆盖到或超过目标区间的结束点
        if start >= target_range[1] and current_cover[:] not in result:
            result.append(current_cover[:])
            return

        for interval in all_subintervals:
            # 下一个区间的起始时间必须在当前区间的结束时间之后，且区间长度在[min_length, max_length]之间
            if interval[0] <= start < interval[1] and min_length <= interval[1] - start <= max_length:
                # 如果当前区间能覆盖start，则选择这个区间并继续寻找
                # 跳过之前分段已经使用的部分
                current_cover.append([start, interval[1]])
                backtrack(interval[1], current_cover)
                current_cover.pop()

    # 生成所有可能的子任务
    # all_subintervals = generate_all_subintervals(intervals, 1, 5 * 32)
    all_subintervals = generate_all_subintervals(intervals, min_length, max_length)
    all_subintervals.sort(key=lambda x: (x[0], x[1]))
    result = []
    backtrack(target_range[0], [])
    return result


def get_segment_lengths(segment_method):
    segment_lengths = []
    for segment in segment_method:
        current_segment_lengths = []
        for seg in segment:
            current_segment_lengths.append(seg[1] - seg[0])
        segment_lengths.append(current_segment_lengths)
    return segment_lengths


# 示例输入
arr = [{0: '0-0', 3: '0-3', 9: '0-9', 12: '0-12', 15: '0-15'}, {1: '0-0', 4: '0-3', 13: '0-12'},
       {2: '0-0', 5: '0-3', 10: '2-10'}, {3: '0-0', 6: '0-3', 11: '2-10', 0: '3-0'},
       {1: '3-0', 4: '0-0', 7: '0-3', 13: '4-13'}, {8: '0-3', 14: '4-13'}, {9: '0-3', 15: '4-13'}, {8: '7-8'},
       {9: '7-8', 6: '8-6'}, {7: '8-6', 10: '7-8', 2: '9-2', 5: '9-5'}, {3: '9-2', 6: '9-5', 11: '7-8'},
       {12: '7-8', 0: '11-0'}, {1: '11-0', 13: '7-8', 4: '12-4'}, {5: '12-4'}, {6: '12-4', 0: '14-0', 9: '14-9'},
       {1: '14-0', 7: '12-4', 10: '14-9'}]

# 计算分段
segments = find_segments(arr, 3, 8)
# print(segments)

# 示例输入
# intervals = [[0, 4], [0, 6], [4, 6], [6, 12], [12, 15]]
# intervals = segments
# intervals = [[0, 5], [5, 10], [10, 15], [15, 20], [20, 25], [25, 30], [29, 35], [35, 40], [40, 45], [45, 50], [50, 55],
#              [55, 60], [60, 65], [65, 70], [70, 75], [73, 77], [75, 81], [79, 85], [85, 90], [90, 95], [95, 100],
#              [100, 112], [110, 115], [113, 120], [120, 125], [125, 135], [135, 140], [140, 150], [150, 155], [155, 160]]
intervals = [[0, 5], [5, 10], [10, 15], [15, 20], [20, 25], [25, 30], [29, 35], [35, 40], [40, 45], [45, 50], [50, 55],
             [55, 60], [60, 65], [65, 70], [70, 75], [73, 77], [75, 81], [79, 85], [85, 90], [90, 95], [95, 100],
             [100, 106], [106, 112], [110, 115], [113, 120], [120, 125], [125, 135], [135, 140], [140, 150], [150, 155],
             [155, 160]]
target_range = (0, 5 * 32)

# 计算覆盖区间
# cover_intervals = find_min_cover_intervals(intervals, target_range)
# print(cover_intervals)

# 计算所有可能的覆盖区间
all_cover_intervals = find_all_cover_intervals(intervals, target_range, 3, 6)
print("所有可能的覆盖区间：")
all_segments = get_segment_lengths(all_cover_intervals)
for segment in all_segments:
    print(segment, len(segment))
# print(get_segment_lengths(all_cover_intervals))
for cover in all_cover_intervals:
    print(cover)
