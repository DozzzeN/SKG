import numpy as np
from scipy.io import loadmat
from scipy.linalg import circulant
from scipy.stats import pearsonr, spearmanr
import cvxpy as cp
from scipy.signal import convolve, medfilt

from chirpKey.RandomWayPoint import RandomWayPoint

N = 256
p_before = []
p_after = []
mismatch = []


def addNoiseFuc(origin, SNR):
    dataLen = len(origin)
    noise = np.random.normal(0, 1, size=dataLen)
    signal_power = np.sum(origin ** 2) / dataLen
    noise_power = np.sum(noise ** 2) / dataLen
    noise_variance = signal_power / (10 ** (SNR / 10))
    noise = noise * np.sqrt(noise_variance / noise_power)
    return origin + noise, noise


fileName = ["../../csi/csi_mobile_indoor_1_r",
            "../../csi/csi_mobile_outdoor_r",
            "../../csi/csi_static_indoor_1_r",
            "../../csi/csi_static_outdoor_r"]

csi_data = np.append(loadmat(fileName[0])['testdata'][:, 0], loadmat(fileName[1])['testdata'][:, 0])
csi_data = np.append(csi_data, loadmat(fileName[2])['testdata'][:, 0])
csi_data = np.append(csi_data, loadmat(fileName[3])['testdata'][:, 0])

times = 1000
for SNR in range(-20, 20, 4):
    print(SNR)
    for i in range(times):
        # model = RandomWayPoint(steps=N, x_range=np.array([0, 11]), y_range=np.array([0, 11]))
        # trace_data = model.generate_trace(start_coor=[1, 1])
        # a = trace_data[:, 0]
        # a = np.random.normal(0, 1, N)
        a = csi_data[i * N % len(csi_data):(i + 1) * N % len(csi_data)]
        if len(a) < N:
            a = np.append(a, csi_data[0:N - len(a)])
        a = a - np.mean(a)
        b = addNoiseFuc(a, SNR)[0]
        # b = a + np.random.normal(0, 2, N)
        b = b - np.mean(b)

        n = np.random.normal(0, 1, (N, N))
        a = a @ n
        b = b @ n

        p_before.append(pearsonr(a, b)[0])

        key = np.random.binomial(1, 0.5, N)
        basic = [0, 1]

        a_con = np.array(circulant(a))
        b_con = np.array(circulant(b))

        U, S, Vt = np.linalg.svd(a_con)
        S = medfilt(S, kernel_size=7)
        D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
        a_con = U @ D @ Vt

        U, S, Vt = np.linalg.svd(b_con)
        S = medfilt(S, kernel_size=7)
        D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
        b_con = U @ D @ Vt

        mulA = a_con @ key

        lambda_ = 0
        solvers = [cp.SCS, cp.GUROBI, cp.OSQP, cp.CVXOPT, cp.ECOS, cp.SCIP, cp.PROXQP, cp.MOSEK,
                   cp.CLARABEL, cp.NAG, cp.XPRESS]
        solver = solvers[1]  # OSQP会失败
        x = cp.Variable(N)
        obj = cp.Minimize(cp.sum_squares(b_con @ x - mulA) + lambda_ * cp.sum_squares(x))
        prob = cp.Problem(obj, [x >= min(basic), x <= max(basic)])
        # prob = cp.Problem(obj)
        prob.solve(solver=solver, max_iter=1000)
        inferred = [i.value for i in x]

        if inferred[0] is None:
            inferred = np.random.binomial(1, 0.5, N)

        inferred = np.array([round(abs(i)) % 2 for i in inferred])

        # 防止协方差出现nan
        if np.var(inferred) < 0.001:
            inferred = np.random.binomial(1, 0.5, N)

        p_after.append(pearsonr(key, inferred)[0])
        mis = 0
        for i in range(N):
            if inferred[i] != key[i]:
                mis += 1
        mismatch.append(mis / N)
        # print(p_before[-1], p_after[-1], mismatch[-1])

# print(np.mean(p_before), np.mean(p_after), np.mean(mismatch))
# 按照[0:0.05, 0.05:0.1, ..., 0.95:1]进行分租
counterMismatch0 = []
counterMismatch1 = []
counterMismatch2 = []
counterMismatch3 = []
counterMismatch4 = []
counterMismatch5 = []
counterMismatch6 = []
counterMismatch7 = []
counterMismatch8 = []
counterMismatch9 = []
for i in range(times * len(range(-20, 20, 4))):
    if p_before[i] < 0.1:
        counterMismatch0.append(mismatch[i])
    elif p_before[i] < 0.2:
        counterMismatch1.append(mismatch[i])
    elif p_before[i] < 0.3:
        counterMismatch2.append(mismatch[i])
    elif p_before[i] < 0.4:
        counterMismatch3.append(mismatch[i])
    elif p_before[i] < 0.5:
        counterMismatch4.append(mismatch[i])
    elif p_before[i] < 0.6:
        counterMismatch5.append(mismatch[i])
    elif p_before[i] < 0.7:
        counterMismatch6.append(mismatch[i])
    elif p_before[i] < 0.8:
        counterMismatch7.append(mismatch[i])
    elif p_before[i] < 0.9:
        counterMismatch8.append(mismatch[i])
    else:
        counterMismatch9.append(mismatch[i])
# 保存成mat文件
import scipy.io as scio

for i in range(10):
    # scio.savemat('test_corr' + str(i) + '.mat', {'counterMismatch' + str(i): eval('counterMismatch' + str(i))})
    scio.savemat('test_corr_csi' + str(i) + '.mat', {'counterMismatch' + str(i): eval('counterMismatch' + str(i))})
