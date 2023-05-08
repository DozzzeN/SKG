import sys

import numpy as np
from scipy.stats import pearsonr
import gurobipy as gp


# 输入向量长度
N = 4
# 随机矩阵个数
K1 = 16

while N < 32:
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

    notFoundSolution = 0
    for attempt in range(attempts):
        np.random.seed((10 ** attempt + 1) % (2**32 - 1))  # 保证不同的K1下，随机数相同
        x = np.random.normal(0, 1, N)
        np.random.seed((10 ** attempt + 2) % (2**32 - 1))
        xr = np.random.normal(0, 1, N)
        np.random.seed((10 ** attempt + 3) % (2**32 - 1))
        r = np.random.normal(0, 1, (K1, L, N))
        # x = np.array([1, 2, -1])
        # r = np.array([[[1, 0, 1], [-1, 0, 0], [0, 0, 1]],
        #               [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        #               [[1, 0, 0], [0, 1, 1], [0, 1, 0]]])
        y = np.matmul(r, x)
        rir = np.matmul(r, np.linalg.pinv(r))
        M = rir - np.array(np.eye(N))
        ra = np.argmax(y, axis=1)
        x = x / np.linalg.norm(x, ord=2)

        d = []

        model = gp.Model()
        inputs = []
        for i in range(K1):
            inputTmp = []
            for j in range(N):
                inputTmp.append(model.addVar(lb=-100, ub=100, name=f'x{i}{j}'))
            inputs.append(inputTmp)
            obj = sum(np.dot(M[i], inputs[i]) ** 2)
            model.setObjective(obj, sense=gp.GRB.MINIMIZE)

        for i in range(len(r)):
            dk = []
            for j in range(len(r[i])):
                if j == ra[i]:
                    continue
                # 按照y中每个矩阵最大值的索引找出r中对应的向量
                dk.append(r[i][ra[i]] - r[i][j])
            d.append(dk)

        for i in range(len(d)):
            for j in range(len(d[i])):
                model.addConstr(np.dot(d[i][j], np.array(inputs[i])) >= 0, name=f'c{i}{j}')

        params = {"NonConvex": 2, "OutputFlag": 0, 'LogToConsole': 0}
        # set OutputFlag to 0 to suppress solver output
        for key, value in params.items():
            model.setParam(key, value)
        model.optimize()

        if model.status == gp.GRB.OPTIMAL:
            xe = []
            for v in model.getVars():
                xe.append(v.x)
            xe = np.array(xe).reshape((K1, N))
            xe = np.mean(xe, axis=0)
            xe = xe / np.linalg.norm(xe, ord=2)
        else:
            notFoundSolution += 1
            xe = xr

        # xe = xe @ np.linalg.pinv(r[0].T @ r[0]) @ r[0].T  # 恢复x

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
    print("notFoundSolution", notFoundSolution)

# K1 = 16
# 16
# 0.6868910957441863 0.20406125704248826
# 0.2507893567825516 0.2032034798912373
# mean 0.6868910957441859 -0.013475312108002884
# var 0.012504923809263326 0.06907457486503497
# max 0.9001883889497908 0.6359525302807687
# 32
# 0.5695636944786082 0.14526641088711542
# 0.1777042971870631 0.14645118125219836
# mean 0.569563694478608 -0.001761684404602329
# var 0.01228348675750636 0.032687435257678865
# max 0.8322925551778788 0.4348078030376682
# notFoundSolution 0
# 64
# 0.41445605925214485 0.08972488609502634
# 0.13057764223738846 0.09252291375783049
# mean 0.41445605925214485 -0.004842375277338504
# var 0.007776262116659204 0.013242852523785289
# max 0.5862304932126245 0.28092631887803937
# notFoundSolution 0
# 128
# 0.26535823244688084 0.07077179758272807
# 0.08605535298089513 0.07183708687369089
# mean 0.26535823244688084 -0.008520896208887607
# var 0.006185996631865357 0.00782414024319485
# max 0.4587134837264681 0.28403619731492424
# notFoundSolution 0

# K1 = 32
# 16
# 0.7988862187321432 0.20406125704248826
# 0.25037319303375233 0.2032034798912373
# mean 0.7988862187321434 -0.013475312108002884
# var 0.006172370559459373 0.06907457486503497
# max 0.9339722448535535 0.6359525302807687
# notFoundSolution 0
# 32
# 0.6874679518414805 0.14526641088711542
# 0.17695377858126385 0.14645118125219836
# mean 0.6874679518414801 -0.001761684404602329
# var 0.006873755442983717 0.032687435257678865
# max 0.856372489183813 0.4348078030376682
# notFoundSolution 0
# 64
# 0.5223557613181478 0.08972488609502634
# 0.13204388921374807 0.09252291375783049
# mean 0.5223557613181478 -0.004842375277338504
# var 0.004881644729267099 0.013242852523785289
# max 0.6750470532426811 0.28092631887803937
# notFoundSolution 0
# 128
# 0.3708368819146833 0.07077179758272807
# 0.08376159169969086 0.07183708687369089
# mean 0.37083688191468317 -0.008520896208887607
# var 0.0032595848644716187 0.00782414024319485
# max 0.505888525874916 0.28403619731492424
# notFoundSolution 0
