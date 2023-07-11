import numpy as np
from scipy.io import loadmat
from scipy.linalg import circulant
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
from scipy.stats import zscore


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

# 仿真数据
# 虽然设置了dataLen，但后续还是用依据A0的长度（40）进行操作，每次只能生成8位密钥
# dataLen = 2400
# SNR = 20
# SA = np.random.normal(0, 1, size=dataLen)
# SB = addNoise(SA, SNR)
# SE = np.random.normal(np.mean(SA), np.std(SA, ddof=1), size=dataLen)

# 实验数据
dataLen = 240
SA = loadmat('rss.mat')['rss'][:, 0]
SB = loadmat('rss.mat')['rss'][:, 1]
# SE = loadmat('rss.mat')['rss'][:, 2]
SE = np.random.normal(np.mean(SB), np.std(SB, ddof=1), size=len(SB))

print(pearsonr(SA, SB)[0])
print(pearsonr(SA, SE)[0])

# SA = np.array(SA).argsort().argsort()
# SB = np.array(SB).argsort().argsort()
# SE = np.array(SE).argsort().argsort()
# print(pearsonr(SA, SB)[0])
# print(pearsonr(SA, SE)[0])

SA = savgol_filter(SA, 11, 5, axis=0)
SB = savgol_filter(SB, 11, 5, axis=0)
SE = savgol_filter(SE, 11, 5, axis=0)

# SA = SA - np.mean(SA)
# SB = SB - np.mean(SB)
# SE = SE - np.mean(SE)
SA = zscore(SA, ddof=1)
SB = zscore(SB, ddof=1)
SE = zscore(SE, ddof=1)

eta = 35
SA /= eta
SB /= eta
SE /= eta

originSum1 = 0
correctSum1 = 0
randomSum1 = 0

originWholeSum1 = 0
correctWholeSum1 = 0
randomWholeSum1 = 0

originSum2 = 0
correctSum2 = 0
randomSum2 = 0

originWholeSum2 = 0
correctWholeSum2 = 0
randomWholeSum2 = 0

N = 40  # length of x
M = N  # rows of A
K = 5  # support size
sA = .1 / M  # A noise variance
sb = sA  # b noise variance
lam = .02  # regularization parameter
# lam = 1  # regularization parameter
ni = 30  # no. of iterations
m22 = np.zeros(ni)  # mean-square error
m22_eve = np.zeros(ni)  # mean-square error
# 必须固定测量矩阵，或相似分布的测量矩阵
A0 = loadmat('A0h-gau.mat')['A0'][:, :]
# A0 = np.random.normal(np.mean(A0), np.std(A0, ddof=1), size=(int(N / 2), N))
# A0 = np.random.normal(0, 0.2, size=(int(N / 2), N))
# 不同分布的测量矩阵效果很差
# A0 = np.random.normal(0, 2, size=(int(N / 2), N))

for j in range(int(dataLen / N)):
    Sa = SA[j * N:(j + 1) * N]
    Sb = SB[j * N:(j + 1) * N]
    Se = SE[j * N:(j + 1) * N]

    # Ea = perturbedMatrix(Sa, N)
    # Eb = perturbedMatrix(Sb, N)
    # Ee = perturbedMatrix(Se, N)

    # 压缩矩阵复用
    for times in range(10):
        np.random.seed(times)
        perm = np.random.permutation(len(Sa))
        Sa = Sa[perm]
        Sb = Sb[perm]
        Se = Se[perm]
        np.random.seed(times)
        perm = np.random.permutation(len(A0))
        A0 = A0[perm]

        # Aa = np.matmul(A0, (Ea + np.identity(N)))
        # Ab = np.matmul(A0, (Eb + np.identity(N)))
        # Ea = np.matmul(A0, Sa)
        # Eb = np.matmul(A0, Sb)
        # Ee = np.matmul(A0, Se)
        Ea = np.matmul(A0, (Sa + np.identity(N)))
        Eb = np.matmul(A0, (Sb + np.identity(N)))
        Ee = np.matmul(A0, (Se + np.identity(N)))

        Aa = np.array(perturbedMatrix(Ea[:, 0], N)).T
        Ab = np.array(perturbedMatrix(Eb[:, 0], N)).T
        Ae = np.array(perturbedMatrix(Ee[:, 0], N)).T

        # Aa = circulant(Ea[::-1])
        # Ab = circulant(Eb[::-1])
        # Ae = circulant(Ee[::-1])
        # Aa = np.append(Aa, Aa, axis=1)
        # Ab = np.append(Ab, Ab, axis=1)
        # Ae = np.append(Ae, Ae, axis=1)

        # KAs = np.random.randint(2, size=N)
        np.random.seed(times)
        KA = np.random.randint(2, size=int(N / 5))
        KAs = []

        for i in range(len(KA)):
            KAs.append(KA[i])
            KAs.extend([0, 0, 0, 0])

        b = np.matmul(Aa, KAs)

        [e23, _, _, KBs] = ass_pg_stls_f(Ab, b, N, K, lam, KAs, ni)
        [e23_eve, _, _, KEs] = ass_pg_stls_f(Ae, b, N, K, lam, KAs, ni)
        m22 = m22 + e23
        m22_eve = m22_eve + e23_eve

        # print(m22)
        # print(m22_eve)

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

        sum1 = min(len(KA_de_sparse), len(KB_de_sparse))
        sum2 = 0
        sum3 = 0
        for i in range(0, sum1):
            sum2 += (KA_de_sparse[i] == KB_de_sparse[i])
        for i in range(0, sum1):
            sum3 += (KA_de_sparse[i] == KE_de_sparse[i])

        originSum1 += sum1
        correctSum1 += sum2
        randomSum1 += sum3

        originWholeSum1 += 1
        correctWholeSum1 = correctWholeSum1 + 1 if sum2 == sum1 else correctWholeSum1
        randomWholeSum1 = randomWholeSum1 + 1 if sum3 == sum1 else randomWholeSum1

        # error correction
        KAs_de_sparse = []
        KBs_de_sparse = []
        KEs_de_sparse = []
        for i in range(len(KA_de_sparse)):
            KAs_de_sparse.append(KA_de_sparse[i])
            KAs_de_sparse.extend([0, 0, 0, 0])

            KBs_de_sparse.append(KB_de_sparse[i])
            KBs_de_sparse.extend([0, 0, 0, 0])

            KEs_de_sparse.append(KE_de_sparse[i])
            KEs_de_sparse.extend([0, 0, 0, 0])

        mismatch_AB = np.bitwise_xor(KAs_de_sparse, KBs_de_sparse)
        mismatch_AE = np.bitwise_xor(KAs_de_sparse, KEs_de_sparse)

        np.random.seed(times)
        perm = np.random.permutation(len(Sa))
        Sa = Sa[perm]
        Sb = Sb[perm]
        Se = Se[perm]
        np.random.seed(times)
        perm = np.random.permutation(len(A0))
        A0 = A0[perm]

        # Aa = np.matmul(A0, (Ea + np.identity(N)))
        # Ab = np.matmul(A0, (Eb + np.identity(N)))
        # Ea = np.matmul(A0, Sa)
        # Eb = np.matmul(A0, Sb)
        # Ee = np.matmul(A0, Se)
        Ea = np.matmul(A0, (Sa + np.identity(N)))
        Eb = np.matmul(A0, (Sb + np.identity(N)))
        Ee = np.matmul(A0, (Se + np.identity(N)))

        Aa = np.array(perturbedMatrix(Ea[:, 0], N)).T
        Ab = np.array(perturbedMatrix(Eb[:, 0], N)).T
        Ae = np.array(perturbedMatrix(Ee[:, 0], N)).T

        # Aa = circulant(Ea[::-1])
        # Ab = circulant(Eb[::-1])
        # Ae = circulant(Ee[::-1])
        # Aa = np.append(Aa, Aa, axis=1)
        # Ab = np.append(Ab, Ab, axis=1)
        # Ae = np.append(Ae, Ae, axis=1)

        b = np.matmul(Aa, mismatch_AB)

        [e23, _, _, KBs] = ass_pg_stls_f(Ab, b, N, K, lam, mismatch_AB, ni)
        [e23_eve, _, _, KEs] = ass_pg_stls_f(Ae, b, N, K, lam, mismatch_AB, ni)
        m22 = m22 + e23
        m22_eve = m22_eve + e23_eve

        # print(m22)
        # print(m22_eve)

        KB_de_sparse = []
        KE_de_sparse = []
        for i in range(0, len(KAs), 5):
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

        sum1 = min(len(mismatch_AB), len(KB_de_sparse))
        sum2 = 0
        sum3 = 0
        for i in range(0, sum1):
            sum2 += (mismatch_AB[i] == KB_de_sparse[i])
        for i in range(0, sum1):
            sum3 += (mismatch_AE[i] == KE_de_sparse[i])

        originSum2 += sum1
        correctSum2 += sum2
        randomSum2 += sum3

        originWholeSum2 += 1
        correctWholeSum2 = correctWholeSum2 + 1 if sum2 == sum1 else correctWholeSum2
        randomWholeSum2 = randomWholeSum2 + 1 if sum3 == sum1 else randomWholeSum2

print("\033[0;34;40ma-b all", correctSum1, "/", originSum1, "=", round(correctSum1 / originSum1, 10), "\033[0m")
print("\033[0;34;40ma-e all", randomSum1, "/", originSum1, "=", round(randomSum1 / originSum1, 10), "\033[0m")
print("\033[0;34;40ma-b whole match", correctWholeSum1, "/", originWholeSum1, "=",
      round(correctWholeSum1 / originWholeSum1, 10), "\033[0m")
print("\033[0;34;40ma-e whole match", randomWholeSum1, "/", originWholeSum1, "=",
      round(randomWholeSum1 / originWholeSum1, 10), "\033[0m")
print("bit generation rate", round(correctSum2 / dataLen, 10))

print("\033[0;34;40ma-b all", correctSum2, "/", originSum2, "=", round(correctSum2 / originSum2, 10), "\033[0m")
print("\033[0;34;40ma-e all", randomSum2, "/", originSum2, "=", round(randomSum2 / originSum2, 10), "\033[0m")
print("\033[0;34;40ma-b whole match", correctWholeSum2, "/", originWholeSum2, "=",
      round(correctWholeSum2 / originWholeSum2, 10), "\033[0m")
print("\033[0;34;40ma-e whole match", randomWholeSum2, "/", originWholeSum2, "=",
      round(randomWholeSum2 / originWholeSum2, 10), "\033[0m")
print("bit generation rate", round(correctSum2 / dataLen, 10))
