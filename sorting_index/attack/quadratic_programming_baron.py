import numpy as np

from scipy.stats import pearsonr

import pyomo.environ as pyo


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
        x = np.random.uniform(0, 1, N)
        xr = np.random.uniform(0, 1, N)
        if singular:
            r = np.random.uniform(0, 1, size=(L, N - 1))
            linearElement = []
            randomCoff = np.random.uniform(0, 1, size=N - 1)
            for i in range(N):
                linearElement.append(np.sum(np.multiply(randomCoff, r[i])))
            # 随机选一列插入
            randomIndex = np.random.randint(0, N)
            r = np.insert(r, randomIndex, linearElement, axis=1)
        else:
            r = np.random.uniform(0, 1, size=(L, N))

        y = np.matmul(r, x)
        rir = np.matmul(r, np.linalg.pinv(r))
        M = rir - np.array(np.eye(N))

        exec("model = pyo.ConcreteModel()")
        exec("inputs = []")
        for i in range(N):
            exec("model.x" + str(i) + " = pyo.Var(within=pyo.Reals)")
            exec("inputs.append(" + "model.x" + str(i) + ")")

        exec("model.obj = pyo.Objective(expr=(np.dot(M, np.array(inputs))[0][0]) ** 2, sense=pyo.minimize)")

        ra = np.argsort(y)
        dk = []
        for j in range(len(r) - 1, 0, -1):
            # 将y的不等式转为向量
            dktmp = np.zeros(N)
            dktmp[ra[j]] = 1
            dktmp[ra[j - 1]] = -1
            dk.append(dktmp)

        exec("cons = []")
        for i in range(len(dk)):
            exec("cons.append(np.dot(dk[i], np.array(inputs))[0])")

        for i in range(len(dk)):
            exec("addModel = str(cons[i]).replace(\"x\", \"model.x\")")
            addMd = locals()['addModel']
            exec("model.con" + str(i) + " = pyo.Constraint(expr=" + addMd + ">=0)")

        # for i in range(N):
        #     exec("model.x" + str(i) + ".setlb(0)")
        #     exec("model.x" + str(i) + ".setub(10000)")

        solver = pyo.SolverFactory('baron', executable="D:\\baron\\baron.exe")

        # more than 10 constraints and should buy a license
        # https://stackoverflow.com/questions/61231085/valueerror-cannot-load-a-solverresults-object-with-bad-status-aborted
        exec("results = solver.solve(model)")

        xe = []

        for i in range(N):
            exec("xe.append(pyo.value(model.x" + str(i) + "))")
        xe = np.array(xe)

        dotsEst += abs(np.dot(x.T, xe) / (np.linalg.norm(x, ord=2) * np.linalg.norm(xe, ord=2)))
        dotsGue += abs(np.dot(x.T, xr) / (np.linalg.norm(x, ord=2) * np.linalg.norm(xr, ord=2)))
        corrEst += abs(pearsonr(x.flatten(), xe.flatten())[0])
        corrGue += abs(pearsonr(x.flatten(), xr.flatten())[0])
    print(corrEst / attempts, corrGue / attempts)
    print(dotsEst / attempts, dotsGue / attempts)
# 4
# 0.5598586909573073 0.48774881480077253
# 0.5522372928709478 0.7937770167377599
# 8
# 0.30368356146709585 0.2944898737295601
# 0.3924791963366537 0.7596331667852055