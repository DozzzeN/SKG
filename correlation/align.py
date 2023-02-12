import json
import math
import multiprocessing
import sys
import threading
import time
from concurrent.futures._base import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from functools import wraps
from multiprocessing import Process, freeze_support

import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr


def list_abs(list1, list2):
    res = 0
    for i in range(len(list1)):
        res += abs(list1[i] - list2[i])
    return res


# 以短的CSI数据为基准从长的CSI数据中筛选
# 不重复选取，且递增顺序选取
def uniqueIncrementSelect(long, short):
    long_len = len(long)
    short_len = len(short)

    left = 0
    step = 100

    long_res = []
    short_res = short.copy()
    for i in range(short_len):
        min_diff = sys.maxsize
        for j in range(left, min(step + left, long_len)):
            diff = abs(short[i] - long[j])
            min_diff = min(diff, min_diff)
            if diff == min_diff:
                left = j
        long_res.append(long[left])
    return long_res, short_res


# 以短的CSI数据为基准从长的CSI数据中筛选
# 重复选取
def repeatedSelect(long, short):
    long_len = len(long)
    short_len = len(short)

    left = 0

    long_res = []
    short_res = short.copy()
    for i in range(short_len):
        min_diff = sys.maxsize
        for j in range(long_len):
            diff = abs(short[i] - long[j])
            min_diff = min(diff, min_diff)
            if diff == min_diff:
                left = j
        long_res.append(long[left])
    return long_res, short_res


# 以短的CSI数据为基准从长的CSI数据中筛选
# 不重复选取
def uniqueSelect(long, short):
    long_len = len(long)
    short_len = len(short)

    left = 0

    long_res = []
    short_res = short.copy()

    occupied = []
    for i in range(short_len):
        min_diff = sys.maxsize
        for j in range(long_len):
            if j not in occupied:
                diff = abs(short[i] - long[j])
                min_diff = min(diff, min_diff)
                if diff == min_diff:
                    left = j
        long_res.append(long[left])
        occupied.append(left)
    return long_res, short_res


# 以两个CSI数据的分布来选取，都假设为高斯分布
# 可重复选取
def threeSigmaSelected(long, short):
    long_left = np.mean(long) - 3 * abs(np.std(long))
    long_right = np.mean(long) + 3 * abs(np.std(long))
    short_left = np.mean(short) - 3 * abs(np.std(short))
    short_right = np.mean(short) + 3 * abs(np.std(short))
    # 根据两个CSI数据的3σ分布找到一个公共区间
    common_left = max(long_left, short_left)
    common_right = min(long_right, short_right)

    long_len = len(long)
    short_len = len(short)

    short_res = []
    long_res = []

    for i in range(short_len):
        if short[i] < common_right and short[i] > common_left:
            short_res.append(short[i])

    print("short_res", len(short_res))

    cur = 0
    for i in range(len(short_res)):
        min_diff = sys.maxsize
        for j in range(long_len):
            diff = abs(short_res[i] - long[j])
            min_diff = min(diff, min_diff)
            if diff == min_diff:
                cur = j
        long_res.append(long[cur])

    return long_res, short_res


class searchThread(threading.Thread):

    def __init__(self, func, args=()):
        super(searchThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


# class searchProcess(Process):
#
#     def __init__(self, func, args=()):
#         super(searchProcess, self).__init__()
#         self.func = func
#         self.args = args
#
#     def run(self):
#         self.result = self.func(*self.args)
#
#     def get_result(self):
#         try:
#             return self.result
#         except Exception:
#             return None

def searchMultiprocess(index, return_queue, short_res, long):
    cur = 0
    step = math.ceil(len(short_res) / multiprocessing.cpu_count())
    left = step * index
    right = min(left + step, len(short_res))
    local_long_res = np.zeros(len(short_res))
    # print("range:" + str(left) + "-" + str(right))
    for i in range(left, right):
        min_diff = sys.maxsize
        for j in range(len(long)):
            diff = abs(short_res[i] - long[j])
            min_diff = min(diff, min_diff)
            if diff == min_diff:
                cur = j
        local_long_res[i] = long[cur]
        # print("thread" + str(index) + "-" + str(i))
    return_queue.put(local_long_res)

# 以两个CSI数据的分布来选取，都假设为高斯分布
# 可重复选取
def threeSigmaSelectedMultiProcess(long, short):
    long_left = np.mean(long) - 3 * abs(np.std(long))
    long_right = np.mean(long) + 3 * abs(np.std(long))
    short_left = np.mean(short) - 3 * abs(np.std(short))
    short_right = np.mean(short) + 3 * abs(np.std(short))
    # 根据两个CSI数据的3σ分布找到一个公共区间
    common_left = max(long_left, short_left)
    common_right = min(long_right, short_right)

    short_len = len(short)

    short_res = []

    for i in range(short_len):
        if short[i] < common_right and short[i] > common_left:
            short_res.append(short[i])

    print("short_res", len(short_res))

    jobs = []
    return_queue = multiprocessing.Queue()
    for i in range(multiprocessing.cpu_count()):
        p = multiprocessing.Process(target=searchMultiprocess, args=(i, return_queue, short_res, long))
        jobs.append(p)
        p.start()

    # for p in jobs:
    #     p.join()
    all_local_long_res = [return_queue.get() for j in jobs]

    long_res = np.zeros(len(short_res))
    for i in range(len(long_res)):
        for j in range(len(all_local_long_res)):
            if all_local_long_res[j][i] != 0:
                long_res[i] = all_local_long_res[j][i]
    return long_res, short_res


# 以两个CSI数据的分布来选取，都假设为高斯分布
# 可重复选取
def threeSigmaSelectedParallel(long, short):
    long_left = np.mean(long) - 3 * abs(np.std(long))
    long_right = np.mean(long) + 3 * abs(np.std(long))
    short_left = np.mean(short) - 3 * abs(np.std(short))
    short_right = np.mean(short) + 3 * abs(np.std(short))
    # 根据两个CSI数据的3σ分布找到一个公共区间
    common_left = max(long_left, short_left)
    common_right = min(long_right, short_right)

    long_len = len(long)
    short_len = len(short)

    short_res = []

    for i in range(short_len):
        if short[i] < common_right and short[i] > common_left:
            short_res.append(short[i])

    print("short_res", len(short_res))

    def search(index):
        cur = 0
        step = math.ceil(len(short_res) / multiprocessing.cpu_count())
        left = step * index
        right = min(left + step, len(short_res))
        local_long_res = np.zeros(len(short_res))
        # print("range:" + str(left) + "-" + str(right))
        for i in range(left, right):
            min_diff = sys.maxsize
            for j in range(long_len):
                diff = abs(short_res[i] - long[j])
                min_diff = min(diff, min_diff)
                if diff == min_diff:
                    cur = j
            local_long_res[i] = long[cur]
            # print("thread" + str(index) + "-" + str(i))
        return local_long_res

    threads = []
    for i in range(multiprocessing.cpu_count()):
        threads.append(searchThread(search, args=(i, )))
        threads[i].start()

    all_local_long_res = []
    for i in range(len(threads)):
        threads[i].join()
        all_local_long_res.append(threads[i].get_result())

    long_res = np.zeros(len(short_res))
    for i in range(len(long_res)):
        for j in range(len(all_local_long_res)):
            if all_local_long_res[j][i] != 0:
                long_res[i] = all_local_long_res[j][i]
    return long_res, short_res


# 以两个CSI数据的分布来选取，都假设为高斯分布
# 可重复选取
def threeSigmaSelectedQueue(long, short):
    long_left = np.mean(long) - 3 * abs(np.std(long))
    long_right = np.mean(long) + 3 * abs(np.std(long))
    short_left = np.mean(short) - 3 * abs(np.std(short))
    short_right = np.mean(short) + 3 * abs(np.std(short))
    # 根据两个CSI数据的3σ分布找到一个公共区间
    common_left = max(long_left, short_left)
    common_right = min(long_right, short_right)

    long_len = len(long)
    short_len = len(short)

    short_res = []

    for i in range(short_len):
        if short[i] < common_right and short[i] > common_left:
            short_res.append(short[i])

    print("short_res", len(short_res))

    def search(index):
        cur = 0
        step = math.ceil(len(short_res) / multiprocessing.cpu_count())
        left = step * index
        right = min(left + step, len(short_res))
        local_long_res = np.zeros(len(short_res))
        # print("range:" + str(left) + "-" + str(right))
        for i in range(left, right):
            min_diff = sys.maxsize
            for j in range(long_len):
                diff = abs(short_res[i] - long[j])
                min_diff = min(diff, min_diff)
                if diff == min_diff:
                    cur = j
            local_long_res[i] = long[cur]
            # print("thread" + str(index) + "-" + str(i))
        return local_long_res

    executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
    tasks = []
    all_local_long_res = []
    for i in range(multiprocessing.cpu_count()):
        task = executor.submit(search, i)
        tasks.append(task)
    for future in as_completed(tasks):
        all_local_long_res.append(future.result())

    long_res = np.zeros(len(short_res))
    for i in range(len(long_res)):
        for j in range(len(all_local_long_res)):
            if all_local_long_res[j][i] != 0:
                long_res[i] = all_local_long_res[j][i]
    return long_res, short_res


global global_long_res


# 以两个CSI数据的分布来选取，都假设为高斯分布
# 可重复选取
def threeSigmaSelectedLock(long, short):
    long_left = np.mean(long) - 3 * abs(np.std(long))
    long_right = np.mean(long) + 3 * abs(np.std(long))
    short_left = np.mean(short) - 3 * abs(np.std(short))
    short_right = np.mean(short) + 3 * abs(np.std(short))
    # 根据两个CSI数据的3σ分布找到一个公共区间
    common_left = max(long_left, short_left)
    common_right = min(long_right, short_right)

    long_len = len(long)
    short_len = len(short)

    short_res = []

    for i in range(short_len):
        if short[i] < common_right and short[i] > common_left:
            short_res.append(short[i])

    print("short_res", len(short_res))

    lock = threading.Lock()
    global_long_res = np.zeros(len(short_res))

    def search(index):
        cur = 0
        step = math.ceil(len(short_res) / multiprocessing.cpu_count())
        left = step * index
        right = min(left + step, len(short_res))
        # print("range:" + str(left) + "-" + str(right))
        for i in range(left, right):
            min_diff = sys.maxsize
            for j in range(long_len):
                diff = abs(short_res[i] - long[j])
                min_diff = min(diff, min_diff)
                if diff == min_diff:
                    cur = j
            lock.acquire()
            try:
                global_long_res[i] = long[cur]
            finally:
                lock.release()
            # print("thread" + str(index) + "-" + str(i))

    threads = []
    for i in range(multiprocessing.cpu_count()):
        threads.append(threading.Thread(target=search, args=(i,)))
    for i in range(len(threads)):
        threads[i].start()
    for i in range(len(threads)):
        threads[i].join()

    return global_long_res, short_res


# 以两个CSI数据的分布来选取，都假设为高斯分布
# 不重复递增选取
def threeSigmaIncrementSelected(long, short):
    long_left = np.mean(long) - 3 * abs(np.std(long))
    long_right = np.mean(long) + 3 * abs(np.std(long))
    short_left = np.mean(short) - 3 * abs(np.std(short))
    short_right = np.mean(short) + 3 * abs(np.std(short))
    # 根据两个CSI数据的3σ分布找到一个公共区间
    common_left = max(long_left, short_left)
    common_right = min(long_right, short_right)

    long_len = len(long)
    short_len = len(short)

    left = 0
    step = 100

    long_res = []
    short_res = []

    for i in range(short_len):
        if short[i] < common_right and short[i] > common_left:
            short_res.append(short[i])

    occupied = []
    for i in range(len(short_res)):
        min_diff = sys.maxsize
        for j in range(left, min(step + left, long_len)):
            if j not in occupied:
                diff = abs(short_res[i] - long[j])
                min_diff = min(diff, min_diff)
                if diff == min_diff:
                    left = j
        occupied.append(left)
        long_res.append(long[left])
    return long_res, short_res


fileName = "1-10"
# long
csi1 = loadmat("../data/data/" + fileName + "_1.mat")['csi1'][0]
# short
csi2 = loadmat("../data/data/" + fileName + "_2.mat")['csi2'][0]

# 比较不同处理方法的相关性，相关性越大越好
# long, short = uniqueIncrementSelect(csi1, csi2)
# print(pearsonr(long, short)[0])
# print(list_abs(long, short))
# long, short = repeatedSelect(csi1, csi2)
# print(pearsonr(long, short)[0])
# print(list_abs(long, short))
# long, short = uniqueSelect(csi1, csi2)
# print(pearsonr(long, short)[0])
# print(list_abs(long, short))

if __name__ == '__main__':
    # 公共数组加锁与一个线程返回一个数组的性能比较
    time1 = time.time()
    long, short = threeSigmaSelectedLock(csi1, csi2)
    print(pearsonr(long, short)[0])
    print(list_abs(long, short))
    time2 = time.time()
    print("time1", time2 - time1)
    print()

    # long, short = threeSigmaSelectedQueue(csi1, csi2)
    # print(pearsonr(long, short)[0])
    # print(list_abs(long, short))
    # time3 = time.time()
    # print("time2", time3 - time2)
    # print()

    # long, short = threeSigmaSelectedParallel(csi1, csi2)
    # print(pearsonr(long, short)[0])
    # print(list_abs(long, short))
    # print("time3", time.time() - time3)
    # print()

    # 与单线程的结果比较，可以看出多线程的计算是对的
    # single_s = time.time()
    # long, short = threeSigmaSelected(csi1, csi2)
    # print(pearsonr(long, short)[0])
    # print(list_abs(long, short))
    # print("time4", time.time() - single_s)

    # 与多进程的结果比较
    # multi_s = time.time()
    # long, short = threeSigmaSelectedMultiProcess(csi1, csi2)
    # print(pearsonr(long, short)[0])
    # print(list_abs(long, short))
    # print("time5", time.time() - multi_s)
    csv1 = open("../correlation/csi1.csv", "a+")
    csv2 = open("../correlation/csi2.csv", "a+")
    for r1 in long:
        csv1.write(str(r1) + "\n")
    for r2 in short:
        csv2.write(str(r2) + "\n")
    csv1.close()
    csv2.close()

# for i in range(len2):
#     time = time2[i] + align
#     min_diff = 999999999999
#     cur_ind = -1
#     for j in range(i, len1):
#         diff = time1[j] - time
#         min_diff = min(min_diff, diff)
#         if diff == min_diff:
#             cur_ind = j
#     res1.append(csi1[cur_ind])
# print(pearsonr(res1, res2)[0])

# for i in range(len2):
#     time = time2[i] + align
#     min_diff = 999999999999
#     cur_ind = -1
#     for j in range(i, len1):
#         diff = time1[j] - time
#         min_diff = min(min_diff, diff)
#         if diff == min_diff:
#             cur_ind = j
#     res1.append(csi1[cur_ind])
# print(pearsonr(res1, res2)[0])

# corr = -1
# idx = 0
# for i in range(0, len1 - len2 - 1):
#     tmp = csi1[i:i + len2]
#     cur_corr = pearsonr(tmp, csi2)[0]
#     corr = max(cur_corr, corr)
#     if corr == cur_corr:
#         idx = i
# print(corr, idx)
# tmp = csi1[idx:idx + len1]
