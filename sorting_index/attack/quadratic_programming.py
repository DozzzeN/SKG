import numpy as np

from scipy.stats import pearsonr
from qpsolvers import solve_qp

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
        P = np.dot(M.T, M)  # 见https://scaron.info/blog/quadratic-programming-in-python.html
        isConvex = "positive semi-definite, convex" if np.all(np.linalg.eigvals(P) >= 0) \
            else "negative semi-definite, non-convex"
        print(isConvex)
        q = np.dot(M.T, np.zeros(N))

        # model = pyo.ConcreteModel()
        # inputs = []
        # for i in range(N):
        #     exec("model.x" + str(i) + " = pyo.Var(within=pyo.Reals)")
        #     exec("inputs.append(" + "model.x" + str(i) + ")")
        #
        # model.obj = pyo.Objective(expr=(np.dot(M, np.array(inputs))[0][0]) ** 2, sense=pyo.minimize)

        ra = np.argsort(y)
        dk = []
        for j in range(len(r) - 1, 0, -1):
            # 将y的不等式转为向量
            dktmp = np.zeros(N)
            dktmp[ra[j]] = 1
            dktmp[ra[j - 1]] = -1
            dk.append(dktmp)

        # cons = []
        # for i in range(len(dk)):
        #     cons.append(np.dot(dk[i], np.array(inputs))[0])
        #
        # for i in range(len(cons)):
        #     addModel = str(cons[i]).replace("x", "model.x")
        #     exec("model.con" + str(i) + " = pyo.Constraint(expr=" + addModel + ">=0)")
        #
        # solver = pyo.SolverFactory('baron', executable="C:\\baron\\baron.exe")

        # solver.options['logfile'] = 'baron.log'
        # results = solver.solve(model, tee=True, keepfiles=True, load_solutions=True)

        # xe = []
        # for i in range(N):
        #     exec("xe.append(pyo.value(model.x" + str(i) + "))")
        # xe = np.array(xe)

        G = -np.array(dk)
        G = np.append(G, np.zeros(N))
        G = G.reshape(N, N)
        h = np.zeros(N)
        ye = solve_qp(P, q, G, h, solver="cvxopt")
        # 除了cvxopt，其他都无法解决负定矩阵的非凸优化
        # ['cvxopt', 'daqp', 'ecos', 'osqp', 'scs']
        if ye is None:
            xe = xr
            notFoundSolution += 1
        else:
            xe = np.matmul(np.linalg.pinv(r), ye)

        dotsEst += abs(np.dot(x.T, xe) / (np.linalg.norm(x, ord=2) * np.linalg.norm(xe, ord=2)))
        dotsGue += abs(np.dot(x.T, xr) / (np.linalg.norm(x, ord=2) * np.linalg.norm(xr, ord=2)))
        corrEst += abs(pearsonr(x.flatten(), xe.flatten())[0])
        corrGue += abs(pearsonr(x.flatten(), xr.flatten())[0])
    print(corrEst / attempts, corrGue / attempts)
    print(dotsEst / attempts, dotsGue / attempts)
    # print("notFoundSolution", notFoundSolution)
# 4
# 0.4966204004996226 0.48774881480077253
# 0.6526569466969123 0.7937770167377599
# notFoundSolution 0
# 8
# 0.3934444653417774 0.2944898737295601
# 0.46230199241809544 0.7596331667852055
# notFoundSolution 0
# 16
# 0.3295201329989418 0.18853370905441263
# 0.34415827261428583 0.7484796357213649
# notFoundSolution 0
# 32
# 0.2499927779051785 0.14217614542690932
# 0.25299573770823675 0.7491604932095164
# notFoundSolution 0
# 64
# 0.21364756023630974 0.10066655376774966
# 0.18745337348622024 0.7498649056676492
# notFoundSolution 1