import random


def segment_data(data, segments, segment_lengths):
    segmented_data = []
    start_index = 0
    for segment_length in segment_lengths:
        end_index = start_index + segment_length
        if end_index > len(data):
            end_index = len(data)
        segment = data[start_index:end_index]
        segmented_data.append(segment)
        start_index = end_index
    # 如果分段数超过指定数量，将剩余数据作为最后一个分段
    if len(segmented_data) < segments:
        segmented_data.append(data[start_index:])
    return segmented_data


def generate_segment_lengths(segments, total_length):
    segment_lengths = []
    remaining_length = total_length
    for _ in range(segments - 1):
        if remaining_length <= 1:
            break
        # 生成一个随机分段长度，范围在1到剩余长度之间
        random.seed(1)
        segment_length = random.randint(1, remaining_length - segments + len(segment_lengths))
        segment_lengths.append(segment_length)
        remaining_length -= segment_length
    # 最后一个分段的长度为剩余长度
    last_segment_length = max(1, remaining_length)
    segment_lengths.append(last_segment_length)
    return segment_lengths


def generate_condition_segment_lengths(segments, min_length, max_length, total_length):
    segment_lengths = []
    remaining_length = total_length
    for _ in range(segments - 1):
        if remaining_length <= 1:
            break
        if remaining_length - segments + len(segment_lengths) < min_length:
            break
        # 生成一个随机分段长度，范围在min_length到max_length和剩余长度的较小值之间
        random.seed(1)
        print(min_length, min(max_length, remaining_length - segments + len(segment_lengths)))
        segment_length = random.randint(min_length, min(max_length, remaining_length - segments + len(segment_lengths)))
        # segment_length = random.randint(1, remaining_length - segments + len(segment_lengths))
        segment_lengths.append(segment_length)
        remaining_length -= segment_length
    # 最后一个分段的长度为剩余长度
    last_segment_length = max(1, remaining_length)
    segment_lengths.append(last_segment_length)
    return segment_lengths


# 示例用法
# segments = 3
# total_length = 8
# segment_lengths = generate_segment_lengths(segments, total_length)
# print(segment_lengths)
# data = list(range(total_length))
# segmented_data = segment_data(data, segments, segment_lengths)
# print(segmented_data)
#
# segments = 3
# total_length = 8
# segment_lengths = generate_condition_segment_lengths(segments, 3, 6, total_length)
# print(segment_lengths)
# data = list(range(total_length))
# segmented_data = segment_data(data, segments, segment_lengths)
# print(segmented_data)
