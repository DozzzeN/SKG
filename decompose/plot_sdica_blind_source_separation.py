import sys

import numpy as np
import pywt
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA, SparsePCA


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def performance_index(G):
    n = len(G)
    m = len(G[0])
    tmp1 = 0
    for i in range(n):
        tmp2 = 0
        for j in range(m):
            max_j = np.max(np.array(G).T[j])
            tmp2 += G[i][j] / max_j - 1
        tmp1 += tmp2
    return tmp1 / (n - 1) / m


def MMSE(real, observed):
    real /= real.std(axis=0)  # Standardize data
    observed /= observed.std(axis=0)  # Standardize data
    minMMSE = []
    for i in range(len(real.T)):
        MMSE = sys.maxsize
        for j in range(len(real.T)):
            if MMSE > np.mean((real.T[i] - observed.T[j]) ** 2):
                MMSE = np.mean((real.T[i] - observed.T[j]) ** 2)
        minMMSE.append(MMSE)
    return minMMSE


# np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = 2 * np.sin(2 * time) + 3 * np.cos(3 * time)  # Signal 1 : sinusoidal signal
# s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s2 = 3 * np.sin(2 * time) + 2 * np.cos(3 * time)  # Signal 2 : sinusoidal signal
# s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal
s3 = np.random.normal(size=n_samples)  # Signal 3: noise signal

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data

# Mix data
# A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
A = np.random.normal(size=(S.shape[1], S.shape[1]))
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=S.shape[1], whiten="arbitrary-variance")
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# band-pass filter
fs = n_samples / 8

# plt.figure()
# plt.clf()
# z = np.fft.fft(X.T[0])
# plt.plot(time, np.abs(z), label='FFT')
# plt.show()

# 小波分解
wave = 'sym2'
wavelet = pywt.Wavelet(wave)
maxLevel = pywt.dwt_max_level(len(X), wavelet.dec_len)
print("maxLevel", maxLevel)
# 小波分解
X_f = pywt.wavedec(X, wave, level=S.shape[1])

# X_f = []
# L = 100
# # print(np.mean(X.T[0]))
# for k in range(L - 1):
#     tmp = []
#     for i in range(3):
#         tmp.append(butter_bandpass_filter(X.T[i], k * fs / 2 / L + 1, (k + 1) * fs / 2 / L + 1, fs, order=10))
#     tmp = np.nan_to_num(tmp)
#     # print(np.mean(tmp), k, k * fs / 2 / L + 1, (k + 1) * fs / 2 / L + 1)
#     X_f.append(tmp)


S_f = []
W_f = []

for k in range(len(X_f)):
    ica = FastICA(n_components=S.shape[1], whiten="arbitrary-variance", max_iter=10000)
    S_f.append(ica.fit_transform(np.array(X_f[k])))
    W_f.append(np.linalg.pinv(ica.mixing_))

G = []
for i in range(len(W_f)):
    for j in range(i + 1, len(W_f)):
        Gij = W_f[i] @ np.linalg.pinv(W_f[j])
        G.append([Gij, i, j])

PI = []
for i in range(len(G)):
    PI.append(performance_index(G[i][0]))

G_for_minPI = G[np.argmin(PI)][0]
Wi_for_minPI = W_f[G[np.argmin(PI)][1]]
Wj_for_minPI = W_f[G[np.argmin(PI)][2]]
Si_for_minPI = S_f[G[np.argmin(PI)][1]]
Sj_for_minPI = S_f[G[np.argmin(PI)][2]]

# For comparison, compute PCA
pca = PCA(n_components=S.shape[1])
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

minMMSE_mixed = MMSE(S, X)
minMMSE_ica = MMSE(S, S_)
minMMSE_sd1 = MMSE(S, Si_for_minPI)
minMMSE_sd2 = MMSE(S, Sj_for_minPI)
minMMSE_pca = MMSE(S, H)

print("mixed", "ica", "sd1", "sd2", "pca")
print(np.mean(minMMSE_mixed), np.mean(minMMSE_ica), np.mean(minMMSE_sd1), np.mean(minMMSE_sd2), np.mean(minMMSE_pca))

# Plot results
import matplotlib.pyplot as plt

plt.figure()

models = [X, S, S_, Si_for_minPI, Sj_for_minPI, H]
names = [
    "Observations (mixed signal)",
    "True Sources",
    "ICA recovered signals",
    "SD1 ICA recovered signals",
    "SD2 ICA recovered signals",
    "PCA recovered signals",
]
colors = ["red", "steelblue", "orange"]

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(len(models), 1, ii)
    plt.title(name, fontsize=10)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(hspace=0.5)  # 调整水平方向的间距
# 隐藏上面子图的横坐标轴刻度
plt.setp(plt.gcf().axes[:-1], xticks=[])
# plt.tight_layout()
plt.show()
