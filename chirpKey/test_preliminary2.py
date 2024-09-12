import numpy as np
from scipy.io import loadmat
from scipy.linalg import circulant
from scipy.optimize import leastsq
from scipy.stats import pearsonr


def normalize(data):
    return (np.array(data) - np.min(data)) / (np.max(data) - np.min(data))


def zero_mean(data):
    return data - np.mean(data)


fileName = ["./data_alignment/csi_ear.mat"]

rawData = loadmat(fileName[0])

start = np.random.randint(0, 1000)
keyLen = 4

# CSIa1Orig = rawData['csi'][:, 0][start:start + keyLen]
# CSIb1Orig = rawData['csi'][:, 1][start:start + keyLen]

# np.random.seed(1000)
# noise = np.random.normal(0, 1, (keyLen, keyLen))
# CSIa1Orig = zero_mean(noise @ CSIa1Orig)
# CSIb1Orig = zero_mean(noise @ CSIb1Orig)

# CSIa1Orig = [26.08866736, 27.6753997, 26.34084049, 27.10490755]
# CSIb1Orig = [24.00030832, 26.43956518, 27.74571149, 28.185585]
# CSIb1Orig1 = [20.00030832, 26.43956518, 27.74571149, 28.185585]

CSIa1Orig = [1.5, 1.3, 1.6, 1.7]
# CSIa1Orig = [1.5, 1.35, 1.6, 1.7]
CSIb1Orig = [1.6, 1.3, 1.8, 1.6]
# CSIb1Orig = [1.6, 1.32, 1.73, 1.6]

print(pearsonr(CSIa1Orig, CSIb1Orig)[0], np.var(CSIa1Orig), np.var(CSIb1Orig))
# keyBin = np.random.binomial(1, 0.5, keyLen)
keyBin = np.array([1, 0, 0, 1])
tmpCSIa1IndPerm = np.round(circulant(CSIa1Orig), 1)
tmpCSIb1IndPerm = np.round(circulant(CSIb1Orig), 1)
tmpMulA1 = np.dot(tmpCSIa1IndPerm, keyBin)

print(tmpCSIa1IndPerm, np.mean(tmpCSIa1IndPerm))
print(tmpCSIb1IndPerm, np.mean(tmpCSIb1IndPerm))
print(keyBin)
print("tmpMulA1", tmpMulA1)


def residuals(x, tmpMulA1, tmpCSIx1IndPerm):
    return tmpMulA1 - np.dot(tmpCSIx1IndPerm, x)


def gaussian_elimination(A, b):
    # 将增广矩阵[A|b]构建为numpy数组
    augmented_matrix = np.column_stack((A, b))

    # 获取矩阵的行数和列数
    rows, cols = augmented_matrix.shape

    # 遍历每一列
    for pivot_col in range(cols - 1):
        # 找到主元素的行号，从对角线开始
        pivot_row = pivot_col + np.argmax(np.abs(augmented_matrix[pivot_col:, pivot_col]))

        # 交换当前行与主元素所在行
        augmented_matrix[[pivot_col, pivot_row]] = augmented_matrix[[pivot_row, pivot_col]]

        # 将主元素所在列的其他元素消为0
        for row in range(pivot_col + 1, rows):
            factor = augmented_matrix[row, pivot_col] / augmented_matrix[pivot_col, pivot_col]
            augmented_matrix[row, pivot_col:] -= factor * augmented_matrix[pivot_col, pivot_col:]

    # 回代，解出方程组
    x = np.zeros(cols - 1)
    for i in range(rows - 1, -1, -1):
        x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i:cols - 1], x[i:])) / augmented_matrix[i, i]

    return x

a_list_number = leastsq(residuals, np.random.binomial(1, 0.5, keyLen),
                        args=(tmpMulA1, tmpCSIa1IndPerm))[0]
b_list_number = leastsq(residuals, np.random.binomial(1, 0.5, keyLen),
                        args=(tmpMulA1, tmpCSIb1IndPerm))[0]
a_list_number_1 = gaussian_elimination(tmpCSIa1IndPerm, tmpMulA1)
b_list_number_1 = gaussian_elimination(tmpCSIb1IndPerm, tmpMulA1)
print("a key", a_list_number, np.round(a_list_number, 1))
print("b key", b_list_number, np.round(b_list_number, 1))
print("a key", a_list_number_1, np.round(a_list_number_1, 1))
print("b key", b_list_number_1, np.round(b_list_number_1, 1))

a_mean = np.mean(CSIa1Orig)
b_mean = np.mean(CSIb1Orig)
a_std = np.std(CSIa1Orig)
b_std = np.std(CSIb1Orig)
alpha = 0.1
a_quan = []
b_quan = []

print("a_mean", a_mean, "b_mean", b_mean)
print("a_std", a_std, "b_std", b_std)

for i in range(len(CSIa1Orig)):
    if CSIa1Orig[i] < a_mean - alpha * a_std:
        a_quan.append(0)
    elif CSIa1Orig[i] > a_mean + alpha * a_std:
        a_quan.append(1)

    if CSIb1Orig[i] < b_mean - alpha * b_std:
        b_quan.append(0)
    elif CSIb1Orig[i] > b_mean + alpha * b_std:
        b_quan.append(1)

print(a_mean - alpha * a_std)
print(a_mean + alpha * a_std)
print(b_mean - alpha * b_std)
print(b_mean + alpha * b_std)
print("a_quan", a_quan)
print("b_quan", b_quan)
# print(np.sum(np.square(np.dot(tmpCSIa1IndPerm, a_list_number) - tmpMulA1)))
# print(np.sum(np.square(np.dot(tmpCSIb1IndPerm, b_list_number) - tmpMulA1)))
