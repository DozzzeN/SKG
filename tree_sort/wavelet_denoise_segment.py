import numpy as np
import pywt
from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr, spearmanr

# 使用小波去噪将ata_static_indoor_1_r_m变为data_static_indoor_1_r_m_d
fileName = "../data/data_static_indoor_1.mat"
rawData = loadmat(fileName)

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

dataLen = len(CSIa1Orig)

print(pearsonr(CSIa1Orig, CSIb1Orig)[0])
print(spearmanr(CSIa1Orig, CSIb1Orig)[0])

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(CSIa1Orig)
plt.subplot(2, 2, 2)
plt.plot(CSIb1Orig)

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

length = int(len(CSIa1Orig) / 4)
CSIa1Orig = CSIa1Orig[0:int(len(CSIa1Orig) / length) * length]
CSIb1Orig = CSIb1Orig[0:int(len(CSIb1Orig) / length) * length]
CSIa1OrigReshape = np.array(CSIa1Orig).reshape(int(len(CSIa1Orig) / length), length)
CSIb1OrigReshape = np.array(CSIb1Orig).reshape(int(len(CSIb1Orig) / length), length)

rec_a = []
rec_b = []
for i in range(len(CSIb1OrigReshape)):
    wave = 'sym2'
    wavelet = pywt.Wavelet(wave)
    maxLevel = pywt.dwt_max_level(len(CSIb1OrigReshape[i]), wavelet.dec_len)
    threshold = 0.5
    # 小波分解
    coefficient_a = pywt.wavedec(CSIa1OrigReshape[i], wave, level=maxLevel)
    coefficient_b = pywt.wavedec(CSIb1OrigReshape[i], wave, level=maxLevel)
    # 噪声滤波
    for i in range(1, len(coefficient_a)):
        coefficient_a[i] = pywt.threshold(coefficient_a[i], threshold * max(coefficient_a[i]))
        coefficient_b[i] = pywt.threshold(coefficient_b[i], threshold * max(coefficient_b[i]))
    rec_a += list(pywt.waverec(coefficient_a, wave))
    rec_b += list(pywt.waverec(coefficient_b, wave))

plt.subplot(2, 2, 3)
plt.plot(rec_a)
plt.subplot(2, 2, 4)
plt.plot(rec_b)
plt.show()

print(pearsonr(rec_a, rec_b)[0])
print(spearmanr(rec_a, rec_b)[0])
savemat('../data/data_static_indoor_1_r.mat', {'A': np.array([rec_a, rec_b]).T})
