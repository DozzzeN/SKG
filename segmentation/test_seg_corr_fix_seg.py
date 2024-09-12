import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.io import loadmat
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
        if segment_peaks:
            segment_peaks = [peak + i for peak in segment_peaks]
            # 如果分段最后一个元素大于峰值，即找到了重叠分段
            last_value = segment[-1]
            # 有时候峰值就是最大值，但还是重复了，就寻找分段后一个值
            if last_value > data[segment_peaks[-1]] or (last_value == data[segment_peaks[-1]] and i + segment_length < len(data)
                and data[i + segment_length] > height):
                j = i + segment_length
                while j < len(data):
                    if data[j] <= height:
                        break
                    j += 1
                segment = data[i:j]
                peaks_overlapping = find_peaks_in_overlapping_segments(segment, segment_length)
                peaks.extend(peaks_overlapping[0])
                peak_values.extend([int(peak_value) for peak_value in peaks_overlapping[1]])
                i = j
            else:
                peaks.extend(segment_peaks)
                peak_values.extend([int(data[peak]) for peak in segment_peaks])
                i += segment_length
        else:
            i += 1

    return peaks, peak_values


def step_corr(data1, data2, step, threshold):
    # 根据step和阈值计算两个序列的相似性
    n, m = len(data1), len(data2)

    corr = np.zeros(n)
    for i in range(n):
        distance = np.inf
        for j in range(max(0, i - step), min(m, i + step + 1)):
            distance = min(distance, np.abs(data1[i] - data2[j]))
        if distance > threshold:
            corr[i] = 0
        # 如果一个窗内有匹配到的数据，则corr置为1
        else:
            corr[i] = 1
    return np.sum(corr)


def find_segments(cross_correlation, step):
    # 寻找局部极值，来找到分段位置
    local_maxima = []
    for j in range(step, len(cross_correlation) - step):
        if cross_correlation[j] >= cross_correlation[j - step:j + step + 1].max():
            if cross_correlation[j] != 0:
                local_maxima.append(int(cross_correlation[j]))
    return local_maxima


def step_corr_metric(data1, data2, step, threshold):
    return min(step_corr(data1, data2, step, threshold), step_corr(data2, data1, step, threshold))


def isSuccessive(key):
    key = list(key)
    for i in range(1, len(key)):
        if key[i] - key[i - 1] == 1:
            return True
    return False


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


def generate_segment_lengths(segments, total_length, seed):
    segment_lengths = []
    remaining_length = total_length
    for _ in range(segments - 1):
        if remaining_length <= 1:
            break
        # 生成一个随机分段长度，范围在1到剩余长度之间
        random.seed(seed)
        segment_length = random.randint(1, remaining_length - segments + len(segment_lengths))
        segment_lengths.append(segment_length)
        remaining_length -= segment_length
    # 最后一个分段的长度为剩余长度
    last_segment_length = max(1, remaining_length)
    segment_lengths.append(last_segment_length)
    return segment_lengths


def discrete_corr(data1, data2, threshold):
    # 计算互相关
    length = len(data1) + len(data2) - 1
    corr = np.zeros(length)
    data2 = np.flip(data2)
    # 卷积
    for i in range(length):
        for j in range(len(data1)):
            if i - j >= 0 and i - j < len(data2):
                # print(i, j, i - j, data1[j], data2[i - j])
                # 只有在对应位置的数据差值小于阈值时才计算
                if abs(data1[j] - data2[i - j]) <= threshold:
                    corr[i] += 1
                # corr[i] += data1[j] * data2[i - j]
    return corr


def step_discrete_corr(data1, data2, step, threshold):
    # 按照一个固定窗口计算互相关
    length = len(data1) + len(data2) - 1
    corr = np.zeros(length)
    data2 = np.flip(data2)
    # 卷积
    for i in range(length):
        tmp1 = []
        tmp2 = []
        for j in range(len(data1)):
            # 根据step选定窗口大小来进行互相关的计算
            if i - j >= 0 and i - j < len(data2):
                tmp1.append(data1[j])
                tmp2.append(data2[i - j])
        corr[i] = step_corr_metric(tmp1, tmp2, step, threshold)
    return corr


def compute_threshold(data1, data2):
    data1_sort = np.sort(data1)
    data2_sort = np.sort(data2)
    threshold = 0
    for i in range(len(data1)):
        temp = abs(data1_sort[i] - data2_sort[i])
        if temp > threshold:
            threshold = temp
    return threshold


def smooth(x, window_len=11, window='hanning'):
    # ndim返回数组的维度
    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser'")

    # np.r_拼接多个数组，要求待拼接的多个数组的列数必须相同
    # 切片[开始索引:结束索引:步进长度]
    # 使用算术平均矩阵来平滑数据
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        # 元素为float，返回window_len个1.的数组
        w = np.ones(window_len, 'd')
    elif window == 'kaiser':
        beta = 5
        w = eval('np.' + window + '(window_len, beta)')
    else:
        w = eval('np.' + window + '(window_len)')

    # 进行卷积操作
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


# 用互相关衡量两个序列分段之间的相似性
withNoise = True
cross_correlations = []
cross_correlations_of_A = []
cross_correlations_of_B = []
M = 8
N = 8
# 是否固定随机数种子
fixSeed = True

data_mean = []
fileName = ["../data/data_mobile_indoor_1.mat"]
# fileName = ["../csi/csi_mobile_indoor_1_r"]
rawData = loadmat(fileName[0])

data1 = rawData['A'][:, 0]
data2 = rawData['A'][:, 1]

if fixSeed:
    np.random.seed(10)
startInd = np.random.randint(0, len(data1) - M * N)
print("startInd: ", startInd)
data1 = smooth(np.array(data1), window_len=30, window='flat')[startInd:startInd + M * N]
data2 = smooth(np.array(data2), window_len=30, window='flat')[startInd:startInd + M * N]

data1 = data1 - np.mean(data1)
data2 = data2 - np.mean(data2)

# data1 = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1))
# data2 = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2))

if withNoise:
    if fixSeed:
        np.random.seed(1)
    noise = np.random.normal(0, 1, (M * N, M * N))
    data1 = data1 @ noise
    data2 = data2 @ noise

data1 = np.argsort(data1)
data2 = np.argsort(data2)
data2_back = data2.copy()

# data1 = data1 - np.mean(data1)
# data2 = data2 - np.mean(data2)

# segment_length_A = generate_segment_lengths(M, M * N, 0)
# print("segment_length_A: ", segment_length_A)
# data1 = segment_data(data1, M, segment_length_A)
# segment_length_B = generate_segment_lengths(M, M * N, segment)
# data2 = segment_data(data2, M, segment_length_B)
# print("segment_length_B: ", segment_length_B)

data1 = np.reshape(data1, (M, N))
data2 = np.reshape(data2, (M, N))
print("data1: ", data1.tolist())
print("data2: ", data2.tolist())

# np.random.seed(1)
# noise = np.random.normal(0, 1, (N, N))
# data1 = data1 @ noise
# data2 = data2 @ noise

permutation = list(range(M))
combineMetric = list(zip(data2, permutation))
# while True:
#     np.random.shuffle(combineMetric)
#     data2, permutation = zip(*combineMetric)
#     if isSuccessive(permutation) is False:
#         break
if fixSeed:
    np.random.seed(100)
np.random.shuffle(combineMetric)
data2, permutation = zip(*combineMetric)
print("permutation: ", permutation)
print("data1 segment: ", [list(data) for data in data1])
print("data2 segment: ", [list(data) for data in data2])

data1_flat = np.hstack((data1))
data2_flat = np.hstack((data2))
data_mean.append(np.mean(data1_flat) * np.mean(data2_flat))

for segment in range(2):
    # 初始化一个列表用于存储互相关结果
    data1_seg = data1_flat[segment * int(M * N / 2):(segment + 1) * int(M * N / 2)]
    data2_seg = data2_flat[segment * int(M * N / 2):(segment + 1) * int(M * N / 2)]
    data2_bck = data2_back[segment * int(M * N / 2):(segment + 1) * int(M * N / 2)]
    print()
    print("data1_seg: ", data1_seg.tolist())
    print("data2_seg: ", data2_seg.tolist())
    # threshold = compute_threshold(data1_seg, data2_seg)
    threshold = 1
    step = int(N / 2) - 1
    print("threshold: ", threshold, "step: ", step)
    # 测自相关来避免连续密钥
    # cross_correlation = discrete_corr(data1_flat, data2_flat, threshold)
    # cross_correlation = discrete_corr(data2_back, data2_flat, threshold)
    cross_correlation_of_A = step_discrete_corr(data2_bck, data2_seg, step, threshold)
    cross_correlation_of_B = step_discrete_corr(data1_seg, data2_seg, step, threshold)
    segment_of_A = find_segments(cross_correlation_of_A, 2 * step + 1)
    segment_of_B = find_segments(cross_correlation_of_B, 2 * step + 1)
    segmentsA = cross_correlation_of_A[find_peaks(cross_correlation_of_A, height=0, distance=2 * step + 1)[0]]
    segmentsB = cross_correlation_of_B[find_peaks(cross_correlation_of_B, height=0, distance=2 * step + 1)[0]]
    segment_peaks_A = find_peaks_in_segments(cross_correlation_of_A, 2, 2 * step + 1)[1]
    segment_peaks_B = find_peaks_in_segments(cross_correlation_of_B, 2, 2 * step + 1)[1]

    print("segment_of_A: ", segment_of_A)
    print("segment_of_B: ", segment_of_B)
    print("segmentsA: ", segmentsA)
    print("segmentsB: ", segmentsB)
    print("cross_correlation_of_A: ", np.sum(cross_correlation_of_A), cross_correlation_of_A.tolist())
    print("cross_correlation_of_B: ", np.sum(cross_correlation_of_B), cross_correlation_of_B.tolist())
    print("segment_peaks_A: ", segment_peaks_A)
    print("segment_peaks_B: ", segment_peaks_B)
    cross_correlation = cross_correlation_of_B

    # 滤波
    # cross_correlation = smooth(cross_correlation, window_len=N, window='hanning')
    # print("cross_correlation: ", np.sum(cross_correlation), cross_correlation.tolist())
    print()

    # cross_correlation = np.correlate(data1_flat, data2_flat, 'full')
    # cross_correlation = np.array(cross_correlation, dtype=float)
    # max_len = len(data1_flat)
    # j = 1.0
    # intervals = np.arange(1, max_len + 1)
    # intervals = np.append(intervals, np.arange(max_len - 1, 0, -1))
    # cross_correlation /= intervals

    # 对互相关进行滤波
    # cross_correlation = np.convolve(cross_correlation, np.ones(N - 1) / N - 1, mode='same')
    # cross_correlation = smooth(cross_correlation, window_len=N, window='hanning')

    cross_correlations.append(cross_correlation)
    cross_correlations_of_A.append(cross_correlation_of_A)
    cross_correlations_of_B.append(cross_correlation_of_B)

    # cross_correlations = []
    # for i in range(M):
    #     cross_correlations.append(np.correlate(data2[i], data1_flat, 'valid'))
    # plt.figure()
    # plt.plot(cross_correlations)
    # plt.show()

# 绘图，不同的分段数对应不同的子图
plt.figure()
for i in range(2):
    plt.subplot(2, 2, i + 1)
    plt.plot(cross_correlations_of_A[i])
    # 均值作为参考线
    # plt.axhline(y=data_mean[i], color='r', linestyle='--')
    if withNoise:
        plt.title("M = " + str(M) + ", N = " + str(N) + ", w. noise")
    else:
        plt.title("M = " + str(M) + ", N = " + str(N) + ", w.o. noise")
    # 图之间加上间隔
    # 标出局部极值
    local_maxima = []
    step = N
    for j in range(step, len(cross_correlations_of_A[i]) - step):
        if cross_correlations_of_A[i][j] >= cross_correlations_of_A[i][j - step:j + step + 1].max():
            local_maxima.append(j)
    # print("i", i, "local_maxima: ", local_maxima, cross_correlations_of_A[i][local_maxima].tolist())
    plt.tight_layout()
# plt.savefig("corr_of_A.png")
plt.show()

plt.figure()
for i in range(2):
    plt.subplot(2, 2, i + 1)
    plt.plot(cross_correlations_of_B[i])
    # 均值作为参考线
    # plt.axhline(y=data_mean[i], color='r', linestyle='--')
    if withNoise:
        plt.title("M = " + str(M) + ", N = " + str(N) + ", w. noise")
    else:
        plt.title("M = " + str(M) + ", N = " + str(N) + ", w.o. noise")
    # 图之间加上间隔
    # 标出局部极值
    local_maxima = []
    step = N
    for j in range(step, len(cross_correlations_of_B[i]) - step):
        if cross_correlations_of_B[i][j] >= cross_correlations_of_B[i][j - step:j + step + 1].max():
            local_maxima.append(j)
    # print("i", i, "local_maxima: ", local_maxima, cross_correlations_of_B[i][local_maxima].tolist())
    plt.tight_layout()
# plt.savefig("corr_of_B.png")
plt.show()