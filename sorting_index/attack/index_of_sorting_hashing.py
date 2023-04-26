import sys

import numpy as np
from scipy.stats import pearsonr

# 输入向量长度
N = 2

while N < 64:
    N = N * 2
    # 输出向量长度
    L = N

    print(N)

    corrEst = 0
    corrGue = 0
    attempts = 100  # 对实际结果影响不大
    dotsEst = 0
    dotsGue = 0
    for attempt in range(attempts):
        np.random.seed((10 ** attempt + 1) % (2 ** 32 - 1))  # 保证不同的K1下，随机数相同
        x = np.random.uniform(0, 1, N)
        np.random.seed((10 ** attempt + 2) % (2 ** 32 - 1))
        xr = np.random.uniform(0, 1, N)
        np.random.seed((10 ** attempt + 3) % (2 ** 32 - 1))
        r = np.random.uniform(0, 1, (1, L, N))
        # x = np.array([1, 2, 3])
        # r = np.array([[[1, 0, 1], [-1, 0, 0], [0, 0, 1]],
        #               [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        #               [[1, 0, 0], [0, 1, 1], [0, 1, 0]]])
        y = np.matmul(r, x)
        ra = np.argsort(y, axis=1)
        x = x / np.linalg.norm(x, ord=2)

        d = []
        for i in range(len(r)):
            dk = []
            for j in range(len(r[i]) - 1, 0, -1):
                # 按照y中每个矩阵从大到小的索引找出r中对应的向量
                dk.append((r[i][ra[i][j]] - r[i][ra[i][j - 1]]) /
                          np.linalg.norm(r[i][ra[i][j]] - r[i][ra[i][j - 1]], ord=2))
            d.append(dk)

        xe = np.mean(d, axis=(1, 0)).T

        for e in range(10):
            y = np.matmul(r, xe)
            yMin = np.min(y, axis=(1, 0))
            yMinIndex = []
            for i in range(len(y)):
                for j in range(len(y[i])):
                    if np.isclose(y[i][j], yMin):
                        yMinIndex.append([i, j])
            kStar = r[yMinIndex[0][0]][yMinIndex[0][1]]
            xe = np.array(xe.T - np.dot(kStar, xe) * np.array(kStar).T / np.linalg.norm(kStar, ord=2)).T
            xe = xe / np.linalg.norm(xe, ord=2)
        # print(x.T)
        # print(xe.T)
        # print(xr.T)

        # print(pearsonr(x.flatten(), xe.flatten())[0])
        # print(pearsonr(x.flatten(), xr.flatten())[0])

        # print("normalized projection")
        # print(np.dot(x.T, xe))
        dotsEst += abs(np.dot(x.T, xe) / (np.linalg.norm(x, ord=2) * np.linalg.norm(xe, ord=2)))
        dotsGue += abs(np.dot(x.T, xr) / (np.linalg.norm(x, ord=2) * np.linalg.norm(xr, ord=2)))
        corrEst += abs(pearsonr(x.flatten(), xe.flatten())[0])
        corrGue += abs(pearsonr(x.flatten(), xr.flatten())[0])
    print(corrEst / attempts, corrGue / attempts)
    print(dotsEst / attempts, dotsGue / attempts)
