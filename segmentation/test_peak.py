import numpy as np
from scipy.signal import find_peaks


def my_find_peaks(data, height=None, distance=None):
    peaks = []
    peak_height = []
    last_peak_index = 0

    for i in range(1, len(data) - 1):
        if data[i] >= data[i - 1] and data[i] >= data[i + 1]:
            if (height is None or data[i] > height) and (distance is None or i - last_peak_index >= distance):
                peaks.append(i)
                peak_height.append(data[i])
                last_peak_index = i

    return peaks


def find_peaks_in_overlapping_segments(data, segment_length):
    # 拷贝一个数组用于删除重叠部分
    data_copy = data.copy()
    peaks = []

    last_peak_index = 0
    i = 0
    while i < len(data):
        segment = data_copy[i:i + segment_length]
        max_value = max(segment)
        # 如果最大值等于最小值，直接找到了峰值
        if max_value == min(segment):
            peaks.append(len(segment) - 1 + i)
            i = len(segment) + i
            continue
        # 如果最大值在最后一个元素，说明发现了重叠，否则找到了峰值
        if segment[-1] != max_value:
            peaks.append(np.argmax(segment) + i)
            i = len(segment) + i
            continue
        # 找出第二大的值，忽略重复的最大值
        second_max_value = -np.inf
        peak_pos = 0
        for j in range(len(segment)):
            if segment[j] < max_value and segment[j] >= second_max_value:
                peak_pos = j
                second_max_value = segment[j]
        peaks.append(peak_pos + i)
        for j in range(len(segment) - 1, -1, -1):
            if segment[j] == max_value:
                data_copy[i + j] -= second_max_value
                last_peak_index = j + i
        i = last_peak_index

    return peaks, np.array(data_copy)[peaks]


def find_peaks_in_segments(data, height, segment_length):
    peaks = []
    peak_values = []

    i = 0
    while i < len(data):
        if data[i] <= height:
            i += 1
            continue
        segment = data[i:i + segment_length]
        segment_peaks = my_find_peaks(segment, height, segment_length // 2)
        # 前四个值不是最值，但是真实值，且后四个值递增，这种情况my_find_peaks找不到峰值，且两个分段重复
        # if segment_peaks == [] and max(segment) > height:
        #     segment_peaks = [np.argmax(segment)]
        if segment_peaks:
            segment_peaks = [peak + i for peak in segment_peaks]
            # 如果分段最后一个元素大于峰值，即找到了重叠分段
            last_value = segment[-1]
            # 有时候峰值就是最大值，但还是重复了，就寻找分段后一个值
            if last_value > data[segment_peaks[-1]] or (
                    last_value == data[segment_peaks[-1]] and i + segment_length < len(data)
                    and data[i + segment_length] > height):
                j = i + segment_length
                while j < len(data):
                    if data[j] <= height:
                        break
                    j += 1
                segment = data[i:j]
                peaks_overlapping = find_peaks_in_overlapping_segments(segment, segment_length)
                peaks.extend(np.array(peaks_overlapping[0]) + i)
                peak_values.extend([int(peak_value) for peak_value in peaks_overlapping[1]])
                i = j
            else:
                peaks.extend(segment_peaks)
                peak_values.extend([int(data[peak]) for peak in segment_peaks])
                i += segment_length
        else:
            i += 1

    return peaks, peak_values


# [ 4 10 10 10 10 10 10]
# Peaks: [4, 30, 30, 10, 10]
# Update: [4, 10, 20, 10, 10, 10]
# data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 9.0, 10.0, 30.0, 30.0, 30.0, 20.0, 20.0, 20.0, 30.0, 10.0, 10.0, 10.0,
#         10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0,
#         10.0, 10.0, 10.0, 9.0, 8.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# [ 6  6  6 22  6  6  6  6]
# Peaks: [28, 12, 6, 6, 6]
# Update: [28, 12, 6, 6, 6, 6]
# data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.0, 26.0, 27.0, 28.0, 28.0, 28.0, 28.0,
#         0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 10.0, 11.0, 12.0, 12.0, 12.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0,
#         6.0, 6.0, 5.0, 4.0, 9.0, 6.0, 6.0, 6.0, 5.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# [3 9 9 9 7 9 9 9]
# Update: [9, 3, 17, 10, 7, 9, 9]
# data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 0.0, 1.0,
#         2.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 15.0, 16.0, 17.0, 27.0, 27.0, 27.0, 27.0, 9.0, 9.0, 9.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0, 8.0, 7.0,
#         15.0, 9.0, 9.0, 9.0, 8.0, 7.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

data1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 2.0, 3.0, 4.0, 5.0,
         4.0, 4.0, 9.0, 10.0, 9.0, 8.0, 8.0, 8.0, 6.0, 4.0, 5.0, 5.0, 5.0, 5.0, 4.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
         4.0, 5.0, 5.0, 5.0, 5.0, 4.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]

data2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 0.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0,
         5.0, 4.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 6.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 5.0, 5.0, 5.0, 5.0, 4.0, 3.0, 2.0, 4.0, 4.0, 4.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 找出连续出现峰值的长数
segment_length = 1
height = 2
max_segment_length = 0
for i in range(1, len(data1)):
    if data1[i - 1] > height:
        if data1[i] == data1[i - 1]:
            segment_length += 1
        else:
            max_segment_length = max(segment_length, max_segment_length)
            segment_length = 1
print("Max Segment Length:", max_segment_length)
peaks = find_peaks_in_segments(data1, 2, 5)
print("Peaks:", peaks[0], peaks[1])
print(np.array(data1)[peaks[0]], np.array(data2)[peaks[0]])
peaks = find_peaks_in_segments(data2, 2, 5)
print("Peaks:", peaks[1])
