import numpy as np
import pywt
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr, spearmanr

# 使用小波去噪将ata_static_indoor_1_r_m变为data_static_indoor_1_r_m_d
from sympy.physics.quantum.identitysearch import scipy

name = "data_static_indoor_1"
fileName = "../data/" + name + ".mat"
rawData = loadmat(fileName)

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]
CSIe1Orig = np.random.normal(loc=np.mean(CSIa1Orig), scale=np.std(CSIa1Orig, ddof=1), size=len(CSIa1Orig))

dataLen = len(CSIa1Orig)

print(pearsonr(CSIa1Orig, CSIb1Orig)[0])
print(spearmanr(CSIa1Orig, CSIb1Orig)[0])

plt.figure()
plt.subplot(2, 3, 1)
plt.plot(CSIa1Orig[0:100])
plt.subplot(2, 3, 2)
plt.plot(CSIb1Orig[0:100])
plt.subplot(2, 3, 3)
plt.plot(CSIe1Orig[0:100])

# https://blog.csdn.net/qq_28156907/article/details/103171284
# candidates wave
# 'bior1.1', 'bior1.3', 'bior1.5',
# 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8',
# 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9',
# 'bior4.4', 'bior5.5', 'bior6.8',
# 'coif1' - 'coif17',
# 'db1' - 'db38',
# 'dmey',
# 'haar',
# 'rbio1.1', 'rbio1.3', 'rbio1.5',
# 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8',
# 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9',
# 'rbio4.4', 'rbio5.5',
# 'rbio6.8',
# 'sym2' - 'sym20'
# continuous wavelet
# 'cgau1', - 'cgau8', 'cmor', 'fbsp', 'gaus1' - 'gaus8', 'mexh', 'morl', 'shan'
wave = 'sym2'
wavelet = pywt.Wavelet(wave)
maxLevel = pywt.dwt_max_level(len(CSIa1Orig), wavelet.dec_len)
print("maxLevel", maxLevel)
threshold = 0.5
# 小波分解
coefficient_a = pywt.wavedec(CSIa1Orig, wave, level=maxLevel)
coefficient_b = pywt.wavedec(CSIb1Orig, wave, level=maxLevel)
coefficient_e = pywt.wavedec(CSIe1Orig, wave, level=maxLevel)
# 噪声滤波
for i in range(1, len(coefficient_a)):
    coefficient_a[i] = pywt.threshold(coefficient_a[i], threshold * max(coefficient_a[i]))
    coefficient_b[i] = pywt.threshold(coefficient_b[i], threshold * max(coefficient_b[i]))
    coefficient_e[i] = pywt.threshold(coefficient_e[i], threshold * max(coefficient_e[i]))
rec_a = pywt.waverec(coefficient_a, wave)
rec_b = pywt.waverec(coefficient_b, wave)
rec_e = pywt.waverec(coefficient_e, wave)

# CSIa1Orig = CSIa1Orig - np.mean(CSIa1Orig)
# CSIb1Orig = CSIb1Orig - np.mean(CSIb1Orig)
# CSIe1Orig = CSIe1Orig - np.mean(CSIe1Orig)
# rec_a = scipy.fft.dct(CSIa1Orig)
# rec_b = scipy.fft.dct(CSIb1Orig)
# rec_e = scipy.fft.dct(CSIe1Orig)
# rec_a = np.abs(np.fft.fft(CSIa1Orig))
# rec_b = np.abs(np.fft.fft(CSIb1Orig))
# rec_e = np.abs(np.fft.fft(CSIe1Orig))
plt.subplot(2, 3, 4)
plt.plot(rec_a[0:100])
plt.subplot(2, 3, 5)
plt.plot(rec_b[0:100])
plt.subplot(2, 3, 6)
plt.plot(rec_e[0:100])
# plt.show()

print(pearsonr(rec_a, rec_b)[0])
print(spearmanr(rec_a, rec_b)[0])
savemat("../data/" + name + "_r.mat", {'A': np.array([rec_a, rec_b]).T})
savemat("../data/" + name + "_e_r.mat", {'A': np.array([rec_e]).T})
