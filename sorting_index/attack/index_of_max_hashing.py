import sys

import numpy as np
from scipy.stats import pearsonr

# 输入向量长度
N = 8
# 随机矩阵个数
K1 = 16
singular = True

while N < 16:
    N = N * 2
    # 输出向量长度
    L = N

    print(N, K1)
    corrEst = 0
    corrGue = 0
    attempts = 100  # 对实际结果影响不大
    dotsEst = 0
    dotsGue = 0
    corrEsts = []
    corrGues = []

    for attempt in range(attempts):
        np.random.seed((10 ** attempt + 1) % (2**32 - 1))  # 保证不同的K1下，随机数相同
        x = np.random.normal(0, 1, N)
        np.random.seed((10 ** attempt + 2) % (2**32 - 1))
        xr = np.random.normal(0, 1, N)

        np.random.seed((10 ** attempt + 3) % (2**32 - 1))
        if singular:
            r = np.random.uniform(0, 1, size=(1, L, N - 1))
            linearElement = []
            np.random.seed((10 ** attempt + 3) % (2 ** 32 - 1))
            randomCoff = np.random.uniform(0, 1, size=N - 1)
            for i in range(L):
                linearElement.append(np.sum(np.multiply(randomCoff, r[0][i])))
            # 随机选一列插入
            np.random.seed((10 ** attempt + 3) % (2 ** 32 - 1))
            randomIndex = np.random.randint(0, N)
            r = np.array([np.insert(r[0], randomIndex, linearElement, axis=1)])
        else:
            r = np.random.normal(0, 1, size=(K1, L, N))

        # x = np.array([1, 2, -1])
        # r = np.array([[[1, 0, 1], [-1, 0, 0], [0, 0, 1]],
        #               [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        #               [[1, 0, 0], [0, 1, 1], [0, 1, 0]]])
        y = np.matmul(r, x)
        ra = np.argmax(y, axis=1)
        x = x / np.linalg.norm(x, ord=2)

        d = []
        for i in range(len(r)):
            dk = []
            for j in range(len(r[i])):
                if j == ra[i]:
                    continue
                # 按照y中每个矩阵最大值的索引找出r中对应的向量
                dk.append(r[i][ra[i]] - r[i][j])
            d.append(dk)

        xe = np.mean(d, axis=(1, 0)).T

        # 如果不要以下refinement部分，相关性会更高
        # for e in range(10):
        #     y = np.matmul(r, xe)
        #     yMin = np.min(y, axis=(1, 0))
        #     yMinIndex = []
        #     for i in range(len(y)):
        #         for j in range(len(y[i])):
        #             if np.isclose(y[i][j], yMin):
        #                 yMinIndex.append([i, j])
        #     kStar = r[yMinIndex[0][0]][yMinIndex[0][1]]
        #     xe = np.array(xe.T - np.dot(kStar, xe) * np.array(kStar).T).T
        #     # xe = np.array(xe.T - np.dot(kStar, xe) * np.array(kStar).T / np.linalg.norm(kStar, ord=2)).T
        #     xe = xe / np.linalg.norm(xe, ord=2)
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
        corrEsts.append(pearsonr(x.flatten(), xe.flatten())[0])
        corrGues.append(pearsonr(x.flatten(), xr.flatten())[0])
    print(corrEst / attempts, corrGue / attempts)
    print(dotsEst / attempts, dotsGue / attempts)
    print("mean", np.mean(corrEsts), np.mean(corrGues))
    print("var", np.var(corrEsts), np.var(corrGues))
    print("max", np.max(corrEsts), np.max(corrGues))

# K1 = 16
# 8
# 0.4003923030659625 0.30412381844101716
# 0.3828291922654924 0.29347214494076423
# 16
# 0.21960651634916445 0.20406125704248826
# 0.2133985734527402 0.2032034798912373
# 32
# 0.15758923016158127 0.14526641088711542
# 0.16011916627210032 0.14645118125219836
# 64
# 0.1019687068209055 0.08972488609502634
# 0.10257133494225874 0.09252291375783049


# K1 = 32
# 16
# 0.20934032853466833 0.20406125704248826
# 0.19941316162942813 0.2032034798912373
# mean -0.03904204796440229 -0.013475312108002884
# var 0.06679582008699521 0.06907457486503497
# max 0.457664231284945 0.6359525302807687
# 32
# 0.13797411653065256 0.14526641088711542
# 0.14021063928128524 0.14645118125219836
# mean -0.029135630409238028 -0.001761684404602329
# var 0.03084526620518822 0.032687435257678865
# max 0.5127736371088385 0.4348078030376682
# 64
# 0.10444738667360387 0.08972488609502634
# 0.10487809520640934 0.09252291375783049
# mean -0.00577158399486786 -0.004842375277338504
# var 0.017079456224179976 0.013242852523785289
# max 0.38826426417135607 0.28092631887803937
# 128
# 0.07301031113196074 0.07077179758272807
# 0.07304711910928612 0.07183708687369089
# mean 0.0069070622779129264 -0.008520896208887607
# var 0.008069485855937098 0.00782414024319485
# max 0.21477162148235 0.28403619731492424
