import numpy as np
from scipy.io import loadmat
from scipy.linalg import circulant
from scipy.optimize import leastsq


def normalize(data):
    return (np.array(data) - np.min(data)) / (np.max(data) - np.min(data))


def zero_mean(data):
    return data - np.mean(data)


fileName = ["../data/data_mobile_indoor_1.mat",
            "../data/data_mobile_outdoor_1.mat",
            "../data/data_static_outdoor_1.mat",
            "../data/data_static_indoor_1.mat"
            ]

rawData = loadmat(fileName[3])

start = np.random.randint(70000, 100000)
keyLen = 4

CSIa1Orig = rawData['A'][:, 0][start:start + keyLen]
CSIb1Orig = rawData['A'][:, 1][start:start + keyLen]

np.random.seed(10000)
noise = np.random.normal(0, 1, (keyLen, keyLen))
CSIa1Orig = zero_mean(noise @ CSIa1Orig)
CSIb1Orig = zero_mean(noise @ CSIb1Orig)

CSIa1Orig = [24.08866736, 21.6753997, 26.34084049, 27.10490755]
CSIb1Orig = [26.00030832, 22.43956518, 27.74571149, 28.185585]

keyBin = np.random.binomial(1, 0.5, keyLen)
tmpCSIa1IndPerm = np.round(circulant(CSIa1Orig),1)
tmpCSIb1IndPerm = np.round(circulant(CSIb1Orig),1)
tmpMulA1 = np.dot(tmpCSIa1IndPerm, keyBin)

print(tmpCSIa1IndPerm, np.mean(tmpCSIa1IndPerm))
print(tmpCSIb1IndPerm, np.mean(tmpCSIb1IndPerm))
print(keyBin)
print(tmpMulA1)


def residuals(x, tmpMulA1, tmpCSIx1IndPerm):
    return tmpMulA1 - np.dot(tmpCSIx1IndPerm, x)


a_list_number = leastsq(residuals, np.random.binomial(1, 0.5, keyLen),
                        args=(tmpMulA1, tmpCSIa1IndPerm))[0]
b_list_number = leastsq(residuals, np.random.binomial(1, 0.5, keyLen),
                        args=(tmpMulA1, tmpCSIb1IndPerm))[0]
print(a_list_number, np.round(a_list_number, 1))
print(b_list_number, np.round(b_list_number, 1))
print(np.sum(np.square(np.dot(tmpCSIa1IndPerm, a_list_number) - tmpMulA1)))
print(np.sum(np.square(np.dot(tmpCSIb1IndPerm, b_list_number) - tmpMulA1)))