import math
from collections import deque

import numpy as np


# list 待排序的数组
# length 比较中所需的最小比较单元的长度
def binaryTreeSort(list, length):
    extend_list = list.copy()
    for i in range(2 ** math.ceil(np.log2(len(list))) - len(list)):
        extend_list.append(0)

    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])
            # 为什么不写成q.append([math.floor((l[0] + l[1]) / 2) + 1, l[1]])
            # 因为要保证生成的二叉树是完全二叉树，同时除了根节点外，中间节点的range不能为1（如下面的[7 7]）
            # 例如使用ceil时，长度为7的数组产生的二叉树为
            #                      [1 7] --从第1个到第7个
            #                   [1 4] [4 7]
            #             [1 2] [3 4] [4 5] [6 7]
            # [1 1] [2 2] [3 3] [4 4] [4 4] [5 5] [6 6] [7 7]
            # 这里最底层出现的两次[4 4]与中间节点[1 4][4 7]中重复使用的4并不会影响密钥的随机性
            # 否则使用floor+1时的二叉树为
            #                [1 7]
            #             [1 4] [5 7]
            #       [1 2] [3 4] [5 6] [7 7]
            # [1 1] [2 2] [3 3] [4 4] [5 5] [6 6]
            # 不仅[7 7]过早出现在中间节点，而且叶子节点中没有[7 7]

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("1")
    code = [["1"]]
    cur = 1
    for i in range(1, max_level):
        cur_code = []
        for j in range(cur, cur + 2 ** i, 2):
            tmp1 = linear_code[int(j / 2)]
            tmp2 = linear_code[int(j / 2)]
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l2l = index[j + 1][0] - 1
            l2r = index[j + 1][1]
            sum1 = np.sum(extend_list[l1l: l1r])
            sum2 = np.sum(extend_list[l2l: l2r])
            while abs(sum1 - sum2) <= 2:
                np.random.shuffle(extend_list)
                sum1 = np.sum(extend_list[l1l: l1r])
                sum2 = np.sum(extend_list[l2l: l2r])
            if sum1 >= sum2:
                tmp1 += "1"
                tmp2 += "2"
            else:
                tmp1 += "2"
                tmp2 += "1"
            # 记录当前code，用于子节点使用
            linear_code.append(tmp1)
            linear_code.append(tmp2)
            # 记录当前code，用于输出
            cur_code.append(tmp1)
            cur_code.append(tmp2)
        cur = cur + 2 ** i
        code.append(cur_code)

    return_code = []
    for i in range(1, len(intervals)):
        if intervals[i] < length:
            return_code = code[i - 1]
            break

    for i in range(len(return_code)):
        return_code[i] = return_code[i][1:]

    return return_code, extend_list


# list 待排序的数组
# length 比较中所需的最小比较单元的长度
def binaryLevelSort(list, length):
    extend_list = list.copy()
    for i in range(2 ** math.ceil(np.log2(len(list))) - len(list)):
        extend_list.append(0)

    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为2
    max_level = int(np.log2(len(index) + 1))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 2 ** i
    intervals.append(0)
    linear_code.append("0")
    code = [["0"]]
    cur = 1
    for i in range(1, max_level):
        cur_code = []
        cur_sum = []
        for j in range(cur, cur + 2 ** i):
            ll = index[j][0] - 1
            lr = index[j][1]
            cur_sum.append(sum(extend_list[ll:lr]))
        sort_sum = np.argsort(cur_sum)
        # 记录当前code，用于子节点使用
        for j in range(len(sort_sum)):
            father = int((len(linear_code) - 1) / 2)
            linear_code.append(linear_code[father]+str(sort_sum[j]))
            cur_code.append(linear_code[father]+str(sort_sum[j]))
        cur = cur + 2 ** i
        code.append(cur_code)

    return_code = []
    for i in range(1, len(intervals)):
        if intervals[i] < length:
            return_code = code[i - 1]
            break

    for i in range(len(return_code)):
        return_code[i] = return_code[i][1:]

    return return_code, extend_list


# list 待排序的数组
# length 比较中所需的最小比较单元的长度
def ternaryTreeSort(list, length):
    q = deque()
    q.append([1, len(list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] != l[0]:
            q.append([l[0], math.floor((l[0] + l[1]) / 2)])
            q.append([math.ceil((l[0] + l[1]) / 2), l[1]])

    max_level = math.ceil(np.log2(len(list))) + 1
    linear_code.append("1")
    code = [["1"]]
    cur = 1
    for i in range(1, max_level):
        cur_code = []
        for j in range(cur, cur + 2 ** i, 2):
            tmp1 = linear_code[int(j / 2)]
            tmp2 = linear_code[int(j / 2)]
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l1r = l1l + 1 if l1l == l1r else l1r
            l2l = l1r
            l2r = index[j + 1][1]
            l2r = l2l + 1 if l2l == l2r else l2r
            sum1 = np.sum(list[l1l: l1r])
            sum2 = np.sum(list[l2l: l2r])
            if sum1 >= sum2:
                tmp1 += "0"
                tmp2 += "1"
            else:
                tmp1 += "1"
                tmp2 += "0"
            linear_code.append(tmp1)
            linear_code.append(tmp2)
            cur_code.append(tmp1)
            cur_code.append(tmp2)
        cur = cur + 2 ** i
        code.append(cur_code)
    intvl = int(np.log2(length))
    return code[-intvl - 1]


# list 待排序的数组
# length 比较中所需的最小比较单元的长度
# pyramid shape
def quadTreeSort(list, length):
    extend_list = list.copy()
    for i in range(2 ** math.ceil(np.log2(len(list))) - len(list)):
        extend_list.append(0)

    q = deque()
    q.append([1, len(extend_list)])
    index = []
    linear_code = []
    while q:
        l = q.popleft()
        index.append(l)
        if l[1] - l[0] >= 3:
            q.append([l[0], l[0] + math.floor((l[1] - l[0]) / 4)])
            q.append([l[0] + math.ceil((l[1] - l[0]) / 4), l[0] + math.floor((l[1] - l[0]) / 2)])
            q.append([l[0] + math.ceil((l[1] - l[0]) / 2), l[0] + math.floor((l[1] - l[0]) * 3 / 4)])
            q.append([l[0] + math.ceil((l[1] - l[0]) * 3 / 4), l[1]])

    # 用等比数列求和公式算出index总共有几层，注意公比为4
    max_level = math.ceil(np.log10(len(index) * 3 + 1) / np.log10(4))
    intervals = []
    cur_interval = 0
    for i in range(max_level):
        intervals.append(index[cur_interval][1] - index[cur_interval][0] + 1)
        cur_interval = cur_interval + 4 ** i
    intervals.append(0)
    linear_code.append("1")
    code = [["1"]]
    cur = 1
    for i in range(1, max_level):
        cur_code = []
        for j in range(cur, cur + 4 ** i, 4):
            tmp1 = linear_code[int(j / 4)]
            tmp2 = linear_code[int(j / 4)]
            tmp3 = linear_code[int(j / 4)]
            tmp4 = linear_code[int(j / 4)]
            l1l = index[j][0] - 1
            l1r = index[j][1]
            l2l = index[j + 1][0] - 1
            l2r = index[j + 1][1]
            l3l = index[j + 2][0] - 1
            l3r = index[j + 2][1]
            l4l = index[j + 3][0] - 1
            l4r = index[j + 3][1]
            sum1 = np.sum(list[l1l: l1r])
            sum2 = np.sum(list[l2l: l2r])
            sum3 = np.sum(list[l3l: l3r])
            sum4 = np.sum(list[l4l: l4r])
            # 四组求和值分别比大小，按照大小顺序分别编码为1 2 3 4
            sort_sum = [[sum1, 1], [sum2, 2], [sum3, 3], [sum4, 4]]
            sort_sum.sort(key=lambda x: (x[0]))
            for k in range(len(sort_sum)):
                if sort_sum[k][1] == 1:
                    tmp1 += str(k + 1)
                elif sort_sum[k][1] == 2:
                    tmp2 += str(k + 1)
                elif sort_sum[k][1] == 3:
                    tmp3 += str(k + 1)
                elif sort_sum[k][1] == 4:
                    tmp4 += str(k + 1)
            # 记录当前code，用于子节点使用
            linear_code.append(tmp1)
            linear_code.append(tmp2)
            linear_code.append(tmp3)
            linear_code.append(tmp4)
            # 记录当前code，用于输出
            cur_code.append(tmp1)
            cur_code.append(tmp2)
            cur_code.append(tmp3)
            cur_code.append(tmp4)
        cur = cur + 4 ** i
        code.append(cur_code)

    for i in range(0, len(intervals)):
        if intervals[i] < length:
            return code[i - 1]


# print(quadTreeSort([3, 2, 1, 6, 10, 22, 1], 4))
# print(quadTreeSort([3, 2, 1, 6, 10, 22, 1, 2], 4))
# print(quadTreeSort([3, 2, 1, 6, 10, 22, 3, 2, 1, 6, 10, 10, 22, 3, 2, 1, 6, 10, 2, 22], 4))
print(binaryLevelSort([3, 2, 1, 6, 10, 22, 3, 2, 1, 6, 10, 10, 22, 3, 2, 1, 6, 10, 2, 22], 4))
