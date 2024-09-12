import random
import time
import numpy as np
from dtw import accelerated_dtw

def segment_sequence(data, segment_lengths):
    segments = []
    start_index = 0
    for length in segment_lengths:
        end_index = start_index + length
        segments.append(data[start_index:end_index])
        start_index = end_index
    return segments


def dtw_metric(data1, data2):
    distance = lambda x, y: np.abs(x - y)
    data1 = np.array(data1)
    data2 = np.array(data2)
    return accelerated_dtw(data1, data2, dist=distance)


def new_segment_sequence(data, segment_lengths):
    segments = []
    for i in range(1, len(segment_lengths)):
        segments.append(data[segment_lengths[i - 1]:segment_lengths[i]])
    return segments


def compute_min_dtw(data1, data2):
    min_dtw = np.inf
    for i in range(len(data1)):
        for j in range(len(data2)):
            if i == j:
                continue
            min_dtw = min(min_dtw, dtw_metric(data1[i], data2[j])[0])
    return min_dtw


def find_sub_opt_segment_method_sliding(data, min_len, max_len, threshold=np.inf):
    segments = []
    n = len(data)
    i = 0

    # 随机选择第一个分段的长度
    first_segment_len = random.randint(min_len, max_len)
    segments.append(data[:first_segment_len])
    i += first_segment_len

    while i < n:
        segment_found = False
        max_min_distance = 0
        best_segment = None

        # 尝试在规定的区间内找到一个合适的分段长度
        for length in range(min_len, max_len + 1):
            if i + length > n:
                break  # 超出数组范围

            new_segment = data[i:i + length]
            min_distance = float('inf')

            # 计算新分段与之前所有分段的最小DTW距离
            for segment in segments:
                distance = dtw_metric(segment, new_segment)[0]
                if distance < threshold:
                    min_distance = distance

            # 选择最小DTW距离最大的分段
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_segment = new_segment

        if best_segment is not None:
            segments.append(best_segment)
            i += len(best_segment)
            segment_found = True

        # 如果在规定区间内没有找到合适的分段长度，则将剩余部分作为最后一个分段
        if not segment_found:
            segments.append(data[i:])
            break

    # 处理最后一个分段的长度
    if len(segments[-1]) < min_len:
        segments.pop()

    segment_method = []
    for segment in segments:
        segment_method.append(len(segment))
    return segment_method


def find_sub_opt_segment_method_sliding_max(data, min_len, max_len, threshold=np.inf):
    n = len(data)
    min_dtw = 0
    min_segment_method = []

    for first_segment_len in range(min_len, max_len + 1):
        segments = []
        i = 0
        segments.append(data[:first_segment_len])
        i += first_segment_len

        while i < n:
            segment_found = False
            max_min_distance = 0
            best_segment = None

            # 尝试在规定的区间内找到一个合适的分段长度
            for length in range(min_len, max_len + 1):
                if i + length > n:
                    break  # 超出数组范围

                new_segment = data[i:i + length]
                min_distance = float('inf')

                # 计算新分段与之前所有分段的最小DTW距离
                for segment in segments:
                    distance = dtw_metric(segment, new_segment)[0]
                    if distance < threshold:
                        min_distance = distance

                # 选择最小DTW距离最大的分段
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_segment = new_segment

            if best_segment is not None:
                segments.append(best_segment)
                i += len(best_segment)
                segment_found = True

            # 如果在规定区间内没有找到合适的分段长度，则将剩余部分作为最后一个分段
            if not segment_found:
                segments.append(data[i:])
                break

        # 处理最后一个分段的长度
        if len(segments[-1]) < min_len:
            segments.pop()

        segment_method = []
        for segment in segments:
            segment_method.append(len(segment))

        this_min_dtw = compute_min_dtw(segment_sequence(data, segment_method), segment_sequence(data, segment_method))
        if this_min_dtw > min_dtw:
            min_dtw = this_min_dtw
            min_segment_method = segment_method
    return min_segment_method


def find_sub_opt_segment_method_sliding_threshold_old(data, min_len, max_len, threshold):
    segments = []
    n = len(data)
    i = 0

    # 随机选择第一个分段的长度
    first_segment_len = random.randint(min_len, max_len)
    segments.append(data[:first_segment_len])
    i += first_segment_len

    while i < n:
        # 尝试在规定的区间内找到一个合适的分段长度
        segment_found = False
        for length in range(min_len, max_len + 1):
            if i + length > n:
                break  # 超出数组范围

            new_segment = data[i:i + length]
            valid = True

            # 检查新分段与之前所有分段的DTW距离
            for segment in segments:
                distance = dtw_metric(segment, new_segment)[0]
                if distance <= threshold:
                    valid = False
                    break

            if valid:
                segments.append(new_segment)
                i += length
                segment_found = True
                break

        # 如果在规定区间内没有找到合适的分段长度，则将剩余部分作为最后一个分段
        if not segment_found:
            segments.append(data[i:])
            break

    # 处理最后一个分段的长度
    if len(segments[-1]) < min_len:
        segments.pop()

    segment_method = []
    for segment in segments:
        segment_method.append(len(segment))
    return segment_method


def find_sub_opt_segment_method_sliding_threshold(data, min_len, max_len, threshold):
    segments = []
    n = len(data)
    i = 0

    # 随机选择第一个分段的长度
    first_segment_len = random.randint(min_len, max_len)
    segments.append(data[:first_segment_len])
    i += first_segment_len

    while i < n:
        # 尝试在规定的区间内找到一个合适的分段长度
        segment_found = False
        for length in range(min_len, max_len + 1):
            if i + length > n:
                break  # 超出数组范围

            new_segment = data[i:i + length]
            valid = True

            # 检查新分段与之前所有分段的DTW距离
            for segment in segments:
                distance = dtw_metric(segment, new_segment)[0]
                if distance <= threshold:
                    valid = False
                    break

            if valid:
                segments.append(new_segment)
                i += length
                segment_found = True
                break

        # 如果在规定区间内没有找到合适的分段长度，则将剩余部分作为最后一个分段
        if not segment_found:
            # segments.append(data[i:])
            while i < n:
                remaining_length = n - i
                if remaining_length <= max_len:
                    if remaining_length >= min_len:
                        segments.append(data[i:])
                    else:
                        if segments:
                            previous_segment = segments.pop()
                            combined_segment = list(previous_segment)
                            combined_segment.extend(data[i:])
                            if len(combined_segment) <= max_len:
                                segments.append(combined_segment)
                            else:
                                segments.append(previous_segment)
                        else:
                            segments.append(data[i:])
                    break
                else:
                    segment_length = np.random.randint(min_len, max_len + 1)
                    segments.append(data[i:i + segment_length])
                    i += segment_length
            break

    # 处理最后一个分段的长度
    if len(segments[-1]) < min_len:
        segments.pop()

    segment_method = []
    for segment in segments:
        segment_method.append(len(segment))
    return segment_method



def find_sub_opt_segment_method_sliding_threshold_max(data, min_len, max_len, threshold):
    n = len(data)
    min_dtw = 0
    min_segment_method = []

    # 随机选择第一个分段的长度
    for first_segment_len in range(min_len, max_len + 1):
        segments = []
        i = 0
        segments.append(data[:first_segment_len])
        i += first_segment_len

        while i < n:
            # 尝试在规定的区间内找到一个合适的分段长度
            segment_found = False
            for length in range(min_len, max_len + 1):
                if i + length > n:
                    break  # 超出数组范围

                new_segment = data[i:i + length]
                valid = True

                # 检查新分段与之前所有分段的DTW距离
                for segment in segments:
                    distance = dtw_metric(segment, new_segment)[0]
                    if distance <= threshold:
                        valid = False
                        break

                if valid:
                    segments.append(new_segment)
                    i += length
                    segment_found = True
                    break

            # 如果在规定区间内没有找到合适的分段长度，则将剩余部分作为最后一个分段
            if not segment_found:
                # segments.append(data[i:])
                while i < n:
                    remaining_length = n - i
                    if remaining_length <= max_len:
                        if remaining_length >= min_len:
                            segments.append(data[i:])
                        else:
                            if segments:
                                previous_segment = segments.pop()
                                combined_segment = list(previous_segment)
                                combined_segment.extend(data[i:])
                                if len(combined_segment) <= max_len:
                                    segments.append(combined_segment)
                                else:
                                    segments.append(previous_segment)
                            else:
                                segments.append(data[i:])
                        break
                    else:
                        segment_length = np.random.randint(min_len, max_len + 1)
                        segments.append(data[i:i + segment_length])
                        i += segment_length
                break

        # 处理最后一个分段的长度
        if len(segments[-1]) < min_len:
            segments.pop()

        segment_method = []
        for segment in segments:
            segment_method.append(len(segment))

        this_min_dtw = compute_min_dtw(segment_sequence(data, segment_method), segment_sequence(data, segment_method))
        if this_min_dtw > min_dtw:
            min_dtw = this_min_dtw
            min_segment_method = segment_method
    return min_segment_method

# 滑动窗口分段，找出分段方法中的最优解：分段之间最小的DTW距离最大
# np.random.seed(0)
data = np.random.permutation(4 * 128)
min_length = 3
max_length = 5
num_segments = 128
dtw_threshold = 40

cal_dtw_times = 0

algorithm_start = time.time_ns()
segments = find_sub_opt_segment_method_sliding(data, min_length, max_length, dtw_threshold)
algorithm_end = time.time_ns()
print("滑动窗口耗时:", (algorithm_end - algorithm_start) / 1e6)
print("Segment method:", segments, sum(segments))
print(compute_min_dtw(segment_sequence(data, segments), segment_sequence(data, segments)))

algorithm_start = time.time_ns()
segments = find_sub_opt_segment_method_sliding(data, min_length, max_length)
algorithm_end = time.time_ns()
print("滑动窗口，不限定阈值，耗时:", (algorithm_end - algorithm_start) / 1e6)
print("Segment method:", segments, sum(segments))
print(compute_min_dtw(segment_sequence(data, segments), segment_sequence(data, segments)))

algorithm_start = time.time_ns()
segments = find_sub_opt_segment_method_sliding_max(data, min_length, max_length, dtw_threshold)
algorithm_end = time.time_ns()
print("滑动窗口，选取最大dtw的分段，耗时:", (algorithm_end - algorithm_start) / 1e6)
print("Segment method:", segments, sum(segments))
print(compute_min_dtw(segment_sequence(data, segments), segment_sequence(data, segments)))

algorithm_start = time.time_ns()
segments = find_sub_opt_segment_method_sliding_threshold_old(data, min_length, max_length, dtw_threshold)
algorithm_end = time.time_ns()
print("滑动窗口限定阈值耗时:", (algorithm_end - algorithm_start) / 1e6)
print("Segment method:", segments, sum(segments))
print(compute_min_dtw(segment_sequence(data, segments), segment_sequence(data, segments)))

# 从速度上最优，从min_dtw上看还能接受
algorithm_start = time.time_ns()
segments = find_sub_opt_segment_method_sliding_threshold(data, min_length, max_length, dtw_threshold)
algorithm_end = time.time_ns()
print("滑动窗口限定阈值，限定最后一个分段长度，耗时:", (algorithm_end - algorithm_start) / 1e6)
print("Segment method:", segments, sum(segments))
print(compute_min_dtw(segment_sequence(data, segments), segment_sequence(data, segments)))

algorithm_start = time.time_ns()
segments = find_sub_opt_segment_method_sliding_threshold_max(data, min_length, max_length, dtw_threshold)
algorithm_end = time.time_ns()
print("滑动窗口限定阈值，选取最大dtw的分段，耗时:", (algorithm_end - algorithm_start) / 1e6)
print("Segment method:", segments, sum(segments))
print(compute_min_dtw(segment_sequence(data, segments), segment_sequence(data, segments)))