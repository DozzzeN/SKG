import sys
import time

import numpy as np
from typing import List, Union

from scipy.spatial import distance


class EuclideanLSH:

    def __init__(self, tables_num: int, a: int, depth: int):
        """

        :param tables_num: hash_table的个数
        :param a: a越大，被纳入同个位置的向量就越多，即可以提高原来相似的向量映射到同个位置的概率，
                反之，则可以降低原来不相似的向量映射到同个位置的概率。
        :param depth: 向量的维度数
        """
        self.tables_num = tables_num
        self.a = a
        # 为了方便矩阵运算，调整了shape，每一列代表一个hash_table的随机向量
        self.R = np.random.random([depth, tables_num])
        self.b = np.random.uniform(0, a, [1, tables_num])
        # 初始化空的hash_table
        self.hash_tables = [dict() for _ in range(tables_num)]

    def _hash(self, inputs: Union[List[List], np.ndarray]):
        """
        将向量映射到对应的hash_table的索引
        :param inputs: 输入的单个或多个向量
        :return: 每一行代表一个向量输出的所有索引，每一列代表位于一个hash_table中的索引
        """
        # H(V) = |V·R + b| / a，R是一个随机向量，a是桶宽，b是一个在[0,a]之间均匀分布的随机变量
        hash_val = np.floor(np.abs(np.matmul(inputs, self.R) + self.b) / self.a)

        return hash_val

    def insert(self, inputs):
        """
        将向量映射到对应的hash_table的索引，并插入到所有hash_table中
        :param inputs:
        :return:
        """
        # 将inputs转化为二维向量
        inputs = np.array(inputs)
        if len(inputs.shape) == 1:
            inputs = inputs.reshape([1, -1])

        hash_index = self._hash(inputs)
        for inputs_one, indexes in zip(inputs, hash_index):
            for i, key in enumerate(indexes):
                # i代表第i个hash_table，key则为当前hash_table的索引位置
                # inputs_one代表当前向量
                self.hash_tables[i].setdefault(key, []).append(tuple(inputs_one))

    def query(self, inputs, nums=20):
        """
        查询与inputs相似的向量，并输出相似度最高的nums个
        :param inputs: 输入向量
        :param nums:
        :return:
        """
        # ravel将多维数组转为一维数组
        hash_val = self._hash(inputs).ravel()
        candidates = set()

        # 将相同索引位置的向量添加到候选集中
        for i, key in enumerate(hash_val):
            candidates.update(self.hash_tables[i][key])

        # 根据向量距离进行排序
        candidates = sorted(candidates, key=lambda x: self.euclidean_dis(x, inputs))

        return candidates[:nums]

    @staticmethod
    def euclidean_dis(x, y):
        """
        计算欧式距离
        :param x:
        :param y:
        :return:
        """
        x = np.array(x)
        y = np.array(y)

        return np.sqrt(np.sum(np.power(x - y, 2)))


if __name__ == '__main__':
    max_dim = 7
    max_len = 256
    data = np.random.random([max_len, max_dim])

    repeat = 100
    query = np.random.random([max_dim])

    start1 = time.time()
    # 速度太慢，比顺序搜索还要慢
    lsh = EuclideanLSH(10, 1, max_dim)
    lsh.insert(data)
    for i in range(repeat):
        res = lsh.query(query, 1)
        res = np.array(res)
        top = -1
        for j in range(len(data)):
            if np.sum(np.power(data[j] - query, 2), axis=-1) == np.sum(np.power(res[0] - query, 2), axis=-1):
                top = j
        print(top)
        print(np.sum(np.power(res - query, 2), axis=-1))
    end1 = time.time()

    print("------------------------------------------------")
    start2 = time.time()
    for i in range(repeat):
        min_dist = sys.maxsize
        min_index = -1
        for j in range(max_len):
            dist = np.sum(np.power(query - data[j], 2), axis=-1)
            if dist < min_dist:
                min_dist = dist
                min_index = i
        sort = np.argsort(np.sum(np.power(data - query, 2), axis=-1))
        top = np.argmin(np.sum(np.power(data - query, 2), axis=-1))
        print(top)
        print(np.sum(np.power(data[sort[:1]] - query, 2), axis=-1))
    end2 = time.time()

    print(end1 - start1)
    print(end2 - start2)
