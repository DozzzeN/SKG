import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy.stats import boxcox


def normal2uniform(data):
    data1 = data[:int(len(data) / 2)]
    data2 = data[int(len(data) / 2):]
    data_reshape = np.array(data[0: 2 * int(len(data) / 2)])
    data_reshape = data_reshape.reshape(int(len(data_reshape) / 2), 2)
    x_list = []
    for i in range(len(data_reshape)):
        # r = np.sum(np.square(data_reshape[i]))
        # r = np.sum(data1[i] * data1[i] + data2[i] * data2[i])
        # x_list.append(np.exp(-0.5 * r))
        r = data2[i] / data1[i]
        x_list.append(math.atan(r) / math.pi + 0.5)

    return x_list


fileName = "data_static_indoor_1"
rawData = loadmat("../data/" + fileName + ".mat")

CSIa1Orig = rawData['A'][:, 0][10000:10320]
CSIb1Orig = rawData['A'][:, 1][10000:10320]
# CSIa1Orig = rawData['A'][:, 0]
# CSIb1Orig = rawData['A'][:, 1]
dataLen = len(CSIa1Orig)

# plt.figure()
# plt.hist(CSIa1Orig - np.mean(CSIa1Orig), bins=30)
# plt.show()

plt.figure()
plt.plot(CSIa1Orig)
plt.show()

mulNoise = np.random.uniform(0, np.std(CSIa1Orig) * 4, size=(dataLen, dataLen))
perturb = np.matmul(CSIa1Orig - np.mean(CSIa1Orig), mulNoise)

# perturb = boxcox(np.abs(CSIa1Orig))[0]
# perturb = normal2uniform(perturb)

# plt.figure()
# plt.hist(perturb, bins=30)
# plt.show()

plt.figure()
plt.plot(perturb)
plt.show()

sortRSS = np.sort(CSIa1Orig - np.mean(CSIa1Orig))
# plt.figure()
# plt.plot(sortRSS)
# plt.show()

perturb = np.sort(perturb - np.mean(perturb))
sortMulRSS = np.sort(perturb)
# plt.figure()
# plt.plot(sortMulRSS)
# plt.show()


