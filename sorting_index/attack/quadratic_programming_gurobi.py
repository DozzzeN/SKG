import numpy as np

from scipy.stats import pearsonr
import gurobipy as gp


# 输入向量长度
N = 128
singular = True

while N < 256:
    N = N * 2
    # 输出向量长度
    L = N

    print(N)

    corrEst = 0
    corrGue = 0
    attempts = 100  # 对实际结果影响不大
    dotsEst = 0
    dotsGue = 0
    corrEsts = []
    corrGues = []

    notFoundSolution = 0
    for attempt in range(attempts):
        np.random.seed((10 ** attempt + 1) % (2 ** 32 - 1))  # 保证不同的K1下，随机数相同
        x = np.random.normal(0, 1, N)
        np.random.seed((10 ** attempt + 2) % (2 ** 32 - 1))
        xr = np.random.normal(0, 1, N)

        np.random.seed((10 ** attempt + 3) % (2 ** 32 - 1))
        if singular:
            r = np.random.normal(0, 1, size=(L, N - 1))
            linearElement = []
            np.random.seed((10 ** attempt + 3) % (2 ** 32 - 1))
            randomCoff = np.random.normal(0, 1, size=N - 1)
            for i in range(N):
                linearElement.append(np.sum(np.multiply(randomCoff, r[i])))
            # 随机选一列插入
            np.random.seed((10 ** attempt + 3) % (2 ** 32 - 1))
            randomIndex = np.random.randint(0, N)
            r = np.insert(r, randomIndex, linearElement, axis=1)
        else:
            r = np.random.normal(0, 1, size=(L, N))

        y = np.matmul(r, x)
        rir = np.matmul(r, np.linalg.pinv(r))
        M = rir - np.array(np.eye(N))

        model = gp.Model()
        inputs = []
        for i in range(N):
            inputs.append(model.addVar(lb=-100, ub=100, name=f'x{i}'))
        obj = sum(np.dot(M, inputs) ** 2)
        model.setObjective(obj, sense=gp.GRB.MINIMIZE)

        ra = np.argsort(y)
        dk = []
        for j in range(len(r) - 1, 0, -1):
            # 将y的不等式转为向量
            dktmp = np.zeros(N)
            dktmp[ra[j]] = 1
            dktmp[ra[j - 1]] = -1
            dk.append(dktmp)

        for i in range(len(dk)):
            model.addConstr(np.dot(dk[i], np.array(inputs)) >= 0, name=f'c{i}')

        params = {"NonConvex": 2, "OutputFlag": 0, "LogToConsole": 0}
        # set OutputFlag to 0 to suppress solver output
        for key, value in params.items():
            model.setParam(key, value)
        model.optimize()

        if model.status == gp.GRB.OPTIMAL:
            xe = []
            for v in model.getVars():
                xe.append(v.x)
            xe = np.array(xe)
        else:
            notFoundSolution += 1
            xe = xr

        xe = xe @ np.linalg.pinv(r.T @ r) @ r.T  # 恢复x
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

# 没有恢复x non-singular
# normal
# 4
# 0.49663845279925956 0.4734106142203007
# 0.42401595852808655 0.43251120262193665
# 8
# 0.3093940117829977 0.30412381844101716
# 0.2894194947412508 0.2934721449407642
# 16
# 0.1965349821887396 0.20406125704248823
# 0.18991604492567724 0.2032034798912373
# 32
# 0.1313358815507023 0.14526641088711542
# 0.13076005673513635 0.1464511812521984
# 64
# 0.09419562299759011 0.08972488609502634
# 0.09471724085517122 0.0925229137578305
# 128
# 0.071561126266906 0.07077179758272809
# 0.07121297753850463 0.07183708687369088
# 256
# 0.05009944453002439 0.04756697099122104
# 0.05003958997613279 0.047808218980279794

# 没有恢复x singular
# normal
# 4
# 0.49663845279925956 0.4734106142203007
# 0.42401595852808655 0.43251120262193665
# 8
# 0.3239211739984451 0.30412381844101716
# 0.3188223208523657 0.2934721449407642
# 16
# 0.22884528468408236 0.20406125704248823
# 0.21563274104468114 0.2032034798912373
# 32
# 0.1313358815507023 0.14526641088711542
# 0.13076005673513635 0.1464511812521984
# 64
# 0.09419562299759011 0.08972488609502634
# 0.09471724085517122 0.0925229137578305
# 128
# 0.06511280431644367 0.07077179758272809
# 0.06508194482126499 0.07183708687369088
# 256
# 0.044373448997020226 0.04756697099122104
# 0.04369631464032248 0.047808218980279794

# 恢复x non-singular
# 4
# 0.5311862321886537 0.4734106142203007
# 0.44199390397337396 0.43251120262193665
# 8
# 0.29867535334057405 0.30412381844101716
# 0.2906685340403372 0.2934721449407642
# 16
# 0.20598584627473315 0.20406125704248823
# 0.19962057506216252 0.2032034798912373
# 32
# 0.13785186545041025 0.14526641088711542
# 0.13419384341840265 0.1464511812521984
# 64
# 0.08575815966567277 0.08972488609502634
# 0.08709282155092855 0.0925229137578305
# 128
# 0.0588940998849472 0.07077179758272809
# 0.05882981090135704 0.07183708687369088

# 恢复x singular
# normal
# 4
# 0.5490192830734251 0.4734106142203007
# 0.4749995242151396 0.43251120262193665
# 8
# 0.3193037250531577 0.30412381844101716
# 0.2959498200737098 0.2934721449407642
# 16
# 0.20598584627473315 0.20406125704248823
# 0.19962057506216252 0.2032034798912373
# 32
# 0.13785186545041025 0.14526641088711542
# 0.13419384341840265 0.1464511812521984
# 64
# 0.10537236019343671 0.08972488609502634
# 0.10589734796347536 0.0925229137578305
# 128
# 0.055098488837267735 0.07077179758272809
# 0.05552284146329124 0.07183708687369088