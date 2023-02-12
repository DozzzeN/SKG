import sys

import dijkstra as dj
import numpy as np


def Dijkstra(name_list, W, s):
    """
    Dijkstra 算法：计算从源点到其他点的最短距离
    W：权重矩阵，存放图中所有边的非负权值，对称矩阵
    name_list：按照顺序存放W中点的名称，从0开始
    s：源点（点名称）
    return：存放源点到其他点的最短距离
    """

    # 将点名称s改为点索引s
    for i, name in enumerate(name_list):
        if name == s:
            s = i
            break
    # 初始化
    MAX = sys.maxsize  # 权值的上限
    dist = [MAX for _ in range(len(name_list))]  # s到所有点的距离，初始化为最大值
    dist[s] = 0  # s到s的距离为0
    min_point = s  # 距离s最短的点为s
    T = set()  # 存放已经算出最短距离的点，初始化为空

    # 开始循环
    while len(T) < len(name_list):  # 一直循环直至T包含了所有点
        T.add(min_point)  # 将距离最短的点加入到T中
        for i, w in enumerate(W[min_point]):  # 遍历min_point的所有直接相连的点
            if i not in T and w > 0:  # 只需要更新不属于T的、权值大于0的点
                dist[i] = min(dist[i], dist[min_point] + W[min_point][i])  # 取最小值
        # 选出不属于T的距离的最小值
        min_dist = MAX
        for i, d in enumerate(dist):
            if i not in T and 0 < d < min_dist:
                min_dist = d
                min_point = i
    return dict((name_list[i], d) for i, d in enumerate(dist))  # 将结果集中的点索引换为点名称，放入到词典中


def getWeightedMatrix(CSITmp):
    CSILen = len(CSITmp)

    weightCSIa1 = [[0 for i in range(CSILen)] for j in range(CSILen)]
    for i in range(0, CSILen):
        for j in range(0, CSILen):
            weightCSIa1[i][j] = round(np.abs(CSITmp[i] - CSITmp[j]), 6)
    return weightCSIa1


CSIa1 = [1, 2, 1000, 3]

if __name__ == '__main__':
    A, B, C, D = nodes = list("ABCD")

    graph = dj.Graph()
    graph.add_edge(A, B, 1)
    graph.add_edge(B, A, 1)
    graph.add_edge(A, C, 3)
    graph.add_edge(C, A, 3)
    graph.add_edge(A, D, 10)
    graph.add_edge(D, A, 10)
    graph.add_edge(D, B, 9)
    graph.add_edge(B, C, 1)
    graph.add_edge(C, B, 1)
    graph.add_edge(C, D, 1)
    graph.add_edge(D, C, 1)

    dijkstra = dj.DijkstraSPF(graph, A)

    print("%-5s %-5s" % ("label", "distance"))
    for u in nodes:
        print("%-5s %8d" % (u, dijkstra.get_distance(u)))

    print(" -> ".join(dijkstra.get_path(C)))
    # print(getWeightedMatrix(CSIa1))
    # name_list = list(range(1, len(CSIa1) + 1))
    # print(Dijkstra(name_list, getWeightedMatrix(CSIa1), 3))
