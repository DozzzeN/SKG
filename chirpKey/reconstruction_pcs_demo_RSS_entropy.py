import numpy as np
from scipy.io import loadmat
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

# 根据样本数据估计概率分布
def frequency(samples):
    samples = np.array(samples)
    total_samples = len(samples)

    # 使用字典来记录每个数值出现的次数
    frequency_count = {}
    for sample in samples:
        if sample in frequency_count:
            frequency_count[sample] += 1
        else:
            frequency_count[sample] = 1

    # 计算每个数值的频率，即概率分布
    frequency = []
    for sample in frequency_count:
        frequency.append(frequency_count[sample] / total_samples)

    return frequency


def minEntropy(probabilities):
    return -np.log2(np.max(probabilities) + 1e-12)

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


fileName = ["../../data/data_mobile_indoor_1.mat",
            "../../data/data_mobile_outdoor_1.mat",
            "../../data/data_static_indoor_1.mat",
            "../../data/data_static_outdoor_1.mat"
            ]

# fileName = ["../csi/csi_mobile_indoor_1_r",
#             "../csi/csi_static_indoor_1_r",
#             "../csi/csi_mobile_outdoor_r",
#             "../csi/csi_static_outdoor_r"]

# RSS security strength
# mi1   0.20295046388843316
# si1   0.1763170133800085
# mo1   0.17740941699425483
# so1   0.23398947595122438

# CSI security strength
# mi1   0.21726164252088986
# si1   0.20294633611570395
# mo1   0.1892813973272623
# so1   0.20151937337875153

for f in fileName:
    rawData = loadmat(f)

    if f.find("csi") != -1:
        CSIa1Orig = rawData['testdata'][:, 0]
        CSIb1Orig = rawData['testdata'][:, 1]

        CSIa1Orig = np.tile(CSIa1Orig, 10)
        CSIb1Orig = np.tile(CSIb1Orig, 10)
    else:
        CSIa1Orig = rawData['A'][:, 0]
        CSIb1Orig = rawData['A'][:, 1]

    # 分段进行滤波的影响不大
    dataLen = 40

    # 稀疏比例是4，故除以5
    bit_len = int(dataLen / 5)

    keys = []

    lam = .02  # regularization parameter
    ni = 30  # no. of iterations
    # 必须固定测量矩阵，或相似分布的测量矩阵
    A0 = loadmat('A0h-gau.mat')['A0'][:, :]
    M = 40
    N = 40
    K = 5  # support size
    A0 = np.random.normal(np.mean(A0), np.std(A0, ddof=1), size=(M, N))

    # 原始方法有漏洞，密钥主要由A0决定
    isOriginMethod = False

    print()
    print("fileName:", f)

    times = 0

    for staInd in range(0, 2 ** bit_len * 100):
        endInd = staInd + dataLen
        if endInd >= len(CSIa1Orig):
            print("too long")
            break

        times += 1

        SA = CSIa1Orig[staInd:endInd]
        SB = CSIb1Orig[staInd:endInd]

        SA = savgol_filter(SA, 11, 5, axis=0)
        SB = savgol_filter(SB, 11, 5, axis=0)

        SA = SA - np.mean(SA)
        SB = SB - np.mean(SB)
        SA = zscore(SA, ddof=1)
        SB = zscore(SB, ddof=1)

        # eta = 25
        # SA /= eta
        # SB /= eta

        Sa = SA
        Sb = SB

        Ea = perturbedMatrix(Sa, N)
        Eb = perturbedMatrix(Sb, N)

        # 压缩矩阵复用
        perm = np.random.permutation(len(Ea))
        Sa = Sa[perm]
        Sb = Sb[perm]
        perm = np.random.permutation(len(A0))
        A0 = A0[perm]

        Ea = perturbedMatrix(Sa, N)

        if isOriginMethod:
            Aa = np.matmul(A0, (Ea + np.identity(N)))
            Ab = np.matmul(A0, (Eb + np.identity(N)))
        else:
            Aa = np.matmul(A0, Ea)
            Ab = np.matmul(A0, Eb)

        KA = np.random.randint(2, size=int(N / 5))
        KAs = []

        for i in range(len(KA)):
            KAs.append(KA[i])
            KAs.extend([0, 0, 0, 0])

        b = np.matmul(Aa, KAs)

        [e23, _, _, KBs] = ass_pg_stls_f(Ab, b, N, K, lam, KAs, ni)

        KA_de_sparse = []
        KB_de_sparse = []
        for i in range(0, len(KAs), 5):
            KA_de_sparse.append(KAs[i])

            if KBs[i] > 0.5:
                KB_de_sparse.append(1)
            elif KBs[i] < -0.5:
                KB_de_sparse.append(1)
            else:
                KB_de_sparse.append(0)

        # error correction
        KAs_de_sparse = []
        KBs_de_sparse = []
        for i in range(len(KA_de_sparse)):
            KAs_de_sparse.append(KA_de_sparse[i])
            KAs_de_sparse.extend([0, 0, 0, 0])

            KBs_de_sparse.append(KB_de_sparse[i])
            KBs_de_sparse.extend([0, 0, 0, 0])

        mismatch_AB = np.bitwise_xor(KAs_de_sparse, KBs_de_sparse)

        perm = np.random.permutation(len(Ea))
        Sa = Sa[perm]
        Sb = Sb[perm]
        perm = np.random.permutation(len(A0))
        A0 = A0[perm]

        if isOriginMethod:
            Aa = np.matmul(A0, (Ea + np.identity(N)))
            Ab = np.matmul(A0, (Eb + np.identity(N)))
        else:
            Aa = np.matmul(A0, Ea)
            Ab = np.matmul(A0, Eb)

        b = np.matmul(Aa, mismatch_AB)

        [e23, _, _, KBs] = ass_pg_stls_f(Ab, b, N, K, lam, mismatch_AB, ni)

        delta_AB = []
        KB_de_sparse = []
        for i in range(0, len(KAs), 5):
            delta_AB.append(mismatch_AB[i])

            if KBs[i] > 0.5:
                KB_de_sparse.append(1)
            elif KBs[i] < -0.5:
                KB_de_sparse.append(1)
            else:
                KB_de_sparse.append(0)

        keys.append("".join(map(str, KB_de_sparse)))
        # print(delta_AB)

    distribution = frequency(keys)
    print("minEntropy", minEntropy(distribution) / bit_len, "bit_len", bit_len, "keyLen", len(keys))