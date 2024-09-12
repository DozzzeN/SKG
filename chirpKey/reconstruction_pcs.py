import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr


def wthresh(data, threshold):
    for i in range(len(data)):
        if data[i] > threshold:
            data[i] = data[i] - threshold
        elif data[i] < -threshold:
            data[i] = data[i] + threshold
        else:
            data[i] = 0
    return data


def l2norm2(data):
    return np.sqrt(np.sum(np.square(data))) ** 2


# 给定A=A0+e1, b=A0h+e2, 求解x: Ax=b, 其中A0是已知的, e1, e2是噪声, h是未知的
def ass_pg_stls_f(A, b, N, K, lam, h, ni):
    # adaptive - step - size proximal - gradient
    AA = np.matmul(np.array(A).T, np.array(A))
    Ab = np.matmul(np.array(A).T, np.array(b))

    er2 = np.zeros(ni)  # error
    er0a = np.zeros(ni)  # missed detections
    er0b = np.zeros(ni)  # wrong detections
    xo = np.zeros(N)  # initialization of solution
    g = -2 * Ab
    mu0 = .2
    x = wthresh(-mu0 * g, mu0 * lam)
    y = 1 / (np.matmul(np.array(x).T, np.array(x)) + 1)
    c = y * l2norm2(np.matmul(A, x) - b)
    muo = mu0

    for nn in range(ni):
        # iterations loop
        # calculate gradient
        go = g  # g0
        co = c  # f1
        g = 2 * y * (np.matmul(AA, x) - Ab - co * x)  # gn

        #  calculate step - size
        if np.matmul(np.array(x - xo).T, (g - go)) == 0:
            mu = muo
        else:
            mus = np.matmul(np.array(x - xo).T, (x - xo)) / np.matmul(np.array(x - xo).T, (g - go))
            mum = np.matmul(np.array(x - xo).T, (g - go)) / np.matmul(np.array(g - go).T, (g - go))
            if mum / mus > .5:
                mu = mum
            else:
                mu = mus - mum / 2
            if mu <= 0:
                mu = muo

        # backtracking line-search
        while True:
            # proximal - gradient
            z = wthresh(x - mu * g, mu * lam)  # xn + 1
            y = 1 / (np.matmul(np.array(z).T, np.array(z)) + 1)
            c = y * l2norm2(np.matmul(A, z) - b)  # fn + 1
            if c <= co + np.matmul(np.array(z - x).T, g) + l2norm2(z - x) / (2 * mu):
                break
            mu = mu / 2
        muo = mu
        xo = x
        x = z

        # calculate errors
        er2[nn] = l2norm2(x - h)
        # ll = length(intersect(find(h), find(x)))
        # er0a(nn) = K - ll
        # er0b(nn) = length(find(x)) - ll
    return er2, er0a, er0b, x


def addNoise(origin, SNR):
    dataLen = len(origin)
    # np.random.seed(seed)
    noise = np.random.normal(0, 1, size=dataLen)
    signal_power = np.sum(origin ** 2) / dataLen
    noise_power = np.sum(noise ** 2) / dataLen
    noise_variance = signal_power / (10 ** (SNR / 10))
    noise = noise * np.sqrt(noise_variance / noise_power)
    return origin + noise


def perturbedMatrix(data, M):
    perturbed = []
    N = len(data)
    for i in range(N):
        row = []
        for j in range(M):
            row.append(data[(N - i + j) % N])
        perturbed.append(row)
    return np.array(perturbed).T


# N = 40  # length of x
# M = 20  # rows of A
# K = 5  # support size
# sA = .01 / M  # A noise variance
# sb = sA  # b noise variance
# # lam = .02  # regularization parameter
# lam = 1  # regularization parameter
# ni = 150  # no. of iterations

# A = loadmat('A.mat')['A'][:, :]
# b = loadmat('b.mat')['b'][:, 0]
A0 = loadmat('A0h-gau.mat')['A0'][:, :]
# h = loadmat('A0h-gau.mat')['h'][:, 0]
# m22 = np.zeros(ni)  # mean-square error
#
# A0 = np.random.normal(0, 1, size=(M, N))  # A0 matrix
# h = []
# for i in range(int(N / 5)):
#     h.append(np.random.normal(0, 1))
#     h.extend([0, 0, 0, 0])
# A = A0 + np.sqrt(sA) * np.random.normal(0, 1, (M, N))  # noisy A matrix
# b = np.matmul(A0, h) + np.sqrt(sb) * np.random.normal(0, 1, M)  # noisy b vector
# [e23, _, _, x] = ass_pg_stls_f(A, b, N, K, lam, h, ni)
#
# m22 = m22 + e23
# # 查看方差的收敛
# print(m22)

kmr_AB = []
kmr_AE = []
epsilon_AB = []
epsilon_AE = []

originSum = 0
correctSum = 0
randomSum = 0

originWholeSum = 0
correctWholeSum = 0
randomWholeSum = 0

# 错误的实现，以reconstruction_demo为准
for times in range(10):
    N = 128  # length of x
    M = N  # rows of A
    K = 5  # support size
    sA = .01 / M  # A noise variance
    sb = sA  # b noise variance
    lam = .02  # regularization parameter
    ni = 30  # no. of iterations
    m22 = np.zeros(ni)  # mean-square error

    A0 = np.random.normal(0, 1, size=(M, M))  # A0 matrix

    dataLen = N
    SNR = 20
    SA = np.random.normal(0, 1, size=dataLen)
    SB = addNoise(SA, SNR)
    # SB = SA
    SE = np.random.normal(np.mean(SA), np.std(SA, ddof=1), size=dataLen)

    eta = 35
    SA /= eta
    SB /= eta
    SE /= eta

    Ea = perturbedMatrix(SA, M)
    Eb = perturbedMatrix(SB, M)
    Ee = perturbedMatrix(SE, M)

    Aa = np.matmul(A0, (Ea + np.identity(N)))
    Ab = np.matmul(A0, (Eb + np.identity(N)))
    Ae = np.matmul(A0, (Ee + np.identity(N)))

    epsilon_AB.append(np.sqrt(l2norm2(Ab - Aa)) / np.sqrt(l2norm2(Aa)))
    epsilon_AE.append(np.sqrt(l2norm2(Ae - Aa)) / np.sqrt(l2norm2(Aa)))

    np.random.seed(times)
    KA = np.random.randint(2, size=N)
    KAs = []

    for i in range(len(KA)):
        KAs.append(KA[i])
        KAs.extend([0, 0, 0, 0])

    Aas = []
    for i in range(M):
        row = []
        for t in range(5):
            for j in range(N):
                row.append(Aa[i][j])
        Aas.append(row)

    Abs = []
    for i in range(M):
        row = []
        for t in range(5):
            for j in range(N):
                row.append(Ab[i][j])
        Abs.append(row)

    Aes = []
    for i in range(M):
        row = []
        for t in range(5):
            for j in range(N):
                row.append(Ae[i][j])
        Aes.append(row)

    syn1 = np.matmul(Aas, KAs)
    syn2 = np.matmul(Aa, KA)

    [e23, _, _, KBs] = ass_pg_stls_f(Abs, syn1, 5 * N, K, lam, KAs, ni)
    # [_, _, _, KB] = ass_pg_stls_f(Ab, syn2, N, K, lam, KA, ni)
    [_, _, _, KEs] = ass_pg_stls_f(Aes, syn1, 5 * N, K, lam, KAs, ni)
    m22 = m22 + e23
    # print(m22)

    KA_de_sparse = []
    KB_de_sparse = []
    KE_de_sparse = []
    for i in range(0, len(KAs), 5):
        if KAs[i] > 0.5:
            KA_de_sparse.append(1)
        elif KAs[i] < -0.5:
            KA_de_sparse.append(1)
        else:
            KA_de_sparse.append(0)

        if KBs[i] > 0.5:
            KB_de_sparse.append(1)
        elif KBs[i] < -0.5:
            KB_de_sparse.append(1)
        else:
            KB_de_sparse.append(0)

        if KEs[i] > 0.5:
            KE_de_sparse.append(1)
        elif KEs[i] < -0.5:
            KE_de_sparse.append(1)
        else:
            KE_de_sparse.append(0)

    delta_AB = np.matmul(Ab, KB_de_sparse) - syn2
    delta_AE = np.matmul(Ae, KE_de_sparse) - syn2

    hAB = KBs.copy()
    hAE = KEs.copy()
    [_, _, _, delta_AB_solve] = ass_pg_stls_f(Abs, delta_AB, 5 * N, K, lam, hAB, ni)
    [_, _, _, delta_AE_solve] = ass_pg_stls_f(Abs, delta_AE, 5 * N, K, lam, hAE, ni)

    for i in range(len(delta_AB_solve)):
        if delta_AB_solve[i] > 0.5:
            delta_AB_solve[i] = 1
        elif delta_AB_solve[i] < -0.5:
            delta_AB_solve[i] = 1
        else:
            delta_AB_solve[i] = 0

        if delta_AE_solve[i] > 0.5:
            delta_AE_solve[i] = 1
        elif delta_AE_solve[i] < -0.5:
            delta_AE_solve[i] = 1
        else:
            delta_AE_solve[i] = 0

    KB = []
    KE = []

    for i in range(len(KB_de_sparse)):
        KB.append(KB_de_sparse[i] ^ int(delta_AB_solve[i]))
        KE.append(KE_de_sparse[i] ^ int(delta_AE_solve[i]))

    sum1 = min(len(KA_de_sparse), len(KB))
    sum2 = 0
    sum3 = 0
    for i in range(0, sum1):
        sum2 += (KA_de_sparse[i] == KB[i])
    for i in range(0, sum1):
        sum3 += (KA_de_sparse[i] == KE[i])

    originSum += sum1
    correctSum += sum2
    randomSum += sum3

    originWholeSum += 1
    correctWholeSum = correctWholeSum + 1 if sum2 == sum1 else correctWholeSum
    randomWholeSum = randomWholeSum + 1 if sum3 == sum1 else randomWholeSum

print("\033[0;34;40ma-b all", correctSum, "/", originSum, "=", round(correctSum / originSum, 10), "\033[0m")
print("\033[0;34;40ma-e all", randomSum, "/", originSum, "=", round(randomSum / originSum, 10), "\033[0m")
print("\033[0;34;40ma-b whole match", correctWholeSum, "/", originWholeSum, "=",
      round(correctWholeSum / originWholeSum, 10), "\033[0m")
print("\033[0;34;40ma-e whole match", randomWholeSum, "/", originWholeSum, "=",
      round(randomWholeSum / originWholeSum, 10), "\033[0m")
print(epsilon_AB)
print(epsilon_AE)
