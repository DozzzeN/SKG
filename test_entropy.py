import numpy as np
from scipy.io import loadmat

import entropy_estimators

CSIa1Orig1 = np.array(loadmat("./data/data_mobile_indoor_1.mat")['A'][:, 0])
CSIa1Orig2 = np.array(loadmat("./data/data_mobile_outdoor_1.mat")['A'][:, 0])
CSIa1Orig3 = np.array(loadmat("./data/data_static_indoor_1.mat")['A'][:, 0])
CSIa1Orig4 = np.array(loadmat("./data/data_static_outdoor_1.mat")['A'][:, 0])

adv_range = [np.std(CSIa1Orig1), np.std(CSIa1Orig2), np.std(CSIa1Orig3), np.std(CSIa1Orig4)]
print(adv_range)
adv_range = np.mean(adv_range)
print(adv_range)

keyLen = 5 * 1024
# keyLen = 7 * 256
# keyLen = 5 * 128
randomMatrix1 = np.random.uniform(0, adv_range * 4, size=(keyLen, keyLen))
randomMatrix2 = np.random.uniform(0, adv_range * 4, size=(keyLen, 1))

randomVector = np.arange(0, keyLen)
np.random.shuffle(randomVector)
randomVector = randomVector.reshape(len(randomVector), 1)

print(entropy_estimators.entropy(randomMatrix1))
print(entropy_estimators.entropy(randomMatrix2))

# log(7*256)=10.8，与结果比较接近
print(entropy_estimators.entropy(randomVector))
print(keyLen * np.log(adv_range * 4))

CSIa1Orig = np.array(loadmat("./data/data_mobile_indoor_1.mat")['A'][:, 0])
CSIa1Orig = CSIa1Orig.reshape(len(CSIa1Orig), 1)
CSIb1Orig = np.array(loadmat("./data/data_mobile_indoor_1.mat")['A'][:, 1])
CSIb1Orig = CSIb1Orig.reshape(len(CSIb1Orig), 1)
print(entropy_estimators.entropy(CSIa1Orig))
print(entropy_estimators.entropy(CSIb1Orig))

# [6.364928963976272, 12.841546077791309, 2.683944723394222, 4.927364730845812]
# 6.704446124001905
# 29137.542612354548
# 4.735562873457393
# 12.990719319012795
# 16840.014172306404