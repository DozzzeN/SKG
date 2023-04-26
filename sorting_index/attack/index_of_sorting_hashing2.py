import sys

import numpy as np
from scipy.stats import pearsonr

# 输入向量长度
N = 2
singular = True

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
        x = np.random.normal(0, 1, N)
        np.random.seed((10 ** attempt + 2) % (2 ** 32 - 1))
        xr = np.random.normal(0, 1, N)

        np.random.seed((10 ** attempt + 3) % (2 ** 32 - 1))
        if singular:
            r = np.random.uniform(0, 1, size=(1, L, N - 1))
            linearElement = []
            np.random.seed((10 ** attempt + 3) % (2 ** 32 - 1))
            randomCoff = np.random.uniform(0, 1, size=N - 1)
            for i in range(N):
                linearElement.append(np.sum(np.multiply(randomCoff, r[0][i])))
            # 随机选一列插入
            np.random.seed((10 ** attempt + 3) % (2 ** 32 - 1))
            randomIndex = np.random.randint(0, N)
            r = np.array([np.insert(r[0], randomIndex, linearElement, axis=1)])
        else:
            r = np.random.uniform(0, 1, size=(1, L, N))

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
            for j in range(len(r[i]) - 1):
                for k in range(j + 1, len(r[i])):
                    if np.dot(r[i][j], x) > np.dot(r[i][k], x):
                        dk.append((r[i][j] - r[i][k]) / np.linalg.norm(r[i][j] - r[i][k], ord=2))
                    else:
                        dk.append((r[i][k] - r[i][j]) / np.linalg.norm(r[i][k] - r[i][j], ord=2))
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

# non-singular
# 4
# 0.48812563021942823 0.47341061422030056
# 0.42338704127879423 0.43251120262193665
# 8
# 0.2865299775022982 0.30412381844101716
# 0.2736866194564115 0.29347214494076423
# 16
# 0.2137795066266619 0.20406125704248826
# 0.20719296276229332 0.2032034798912373
# 32
# 0.1261801666542439 0.14526641088711542
# 0.12155889999800565 0.14645118125219836
# 64
# 0.1152973059449252 0.08972488609502634
# 0.11440410324362255 0.09252291375783049

# singular
# 4
# 0.6075516589911141 0.47341061422030056
# 0.5315512298005308 0.43251120262193665
# 8
# 0.4428249794875543 0.30412381844101716
# 0.4434130706246545 0.29347214494076423
# 16
# 0.20847648910023175 0.20406125704248826
# 0.20826897363836377 0.2032034798912373
# 32
# 0.15807267457985766 0.14526641088711542
# 0.16250744025242167 0.14645118125219836
# 64
# 0.10157551634864166 0.08972488609502634
# 0.10115760162202765 0.09252291375783049
