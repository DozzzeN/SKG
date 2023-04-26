import numpy as np

from scipy.stats import pearsonr
import gurobipy as gp


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

    notFoundSolution = 0
    for attempt in range(attempts):
        np.random.seed((10 ** attempt + 1) % (2 ** 32 - 1))  # 保证不同的K1下，随机数相同
        x = np.random.uniform(0, 1, N)
        np.random.seed((10 ** attempt + 2) % (2 ** 32 - 1))
        xr = np.random.uniform(0, 1, N)

        np.random.seed((10 ** attempt + 3) % (2 ** 32 - 1))
        if singular:
            r = np.random.uniform(0, 1, size=(L, N - 1))
            linearElement = []
            np.random.seed((10 ** attempt + 3) % (2 ** 32 - 1))
            randomCoff = np.random.uniform(0, 1, size=N - 1)
            for i in range(N):
                linearElement.append(np.sum(np.multiply(randomCoff, r[i])))
            # 随机选一列插入
            np.random.seed((10 ** attempt + 3) % (2 ** 32 - 1))
            randomIndex = np.random.randint(0, N)
            r = np.insert(r, randomIndex, linearElement, axis=1)
        else:
            r = np.random.uniform(0, 1, size=(L, N))

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

        params = {"NonConvex": 2, "OutputFlag": 0, 'LogToConsole': 0}
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

        dotsEst += abs(np.dot(x.T, xe) / (np.linalg.norm(x, ord=2) * np.linalg.norm(xe, ord=2)))
        dotsGue += abs(np.dot(x.T, xr) / (np.linalg.norm(x, ord=2) * np.linalg.norm(xr, ord=2)))
        corrEst += abs(pearsonr(x.flatten(), xe.flatten())[0])
        corrGue += abs(pearsonr(x.flatten(), xr.flatten())[0])
    print(corrEst / attempts, corrGue / attempts)
    print(dotsEst / attempts, dotsGue / attempts)
    print("notFoundSolution", notFoundSolution)

# 4
# 0.5692086725198601 0.48774881480077253
# 0.531548670314852 0.7937770167377599
# notFoundSolution 0
# 8
# 0.3166160806324376 0.2944898737295601
# 0.3805353477097866 0.7596331667852055
# notFoundSolution 0
# 16
# 0.20149163426193234 0.18853370905441263
# 0.2788700586229442 0.7484796357213649
# notFoundSolution 0
# 32
# 0.13396230922793137 0.14217614542690932
# 0.2076169553313932 0.7491604932095164
# notFoundSolution 0
# 64
# 0.1214557729990168 0.10066655376774966
# 0.18419276640610519 0.7498649056676492
# notFoundSolution 0