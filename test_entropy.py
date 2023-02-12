import numpy as np
from scipy.io import loadmat

import entropy_estimators

keyLen = 7 * 256
# keyLen = 5 * 128
randomMatrix1 = np.random.uniform(0, 4, size=(keyLen, keyLen))
randomMatrix2 = np.random.uniform(0, 4, size=(keyLen, 1))

randomVector = np.arange(0, keyLen)
np.random.shuffle(randomVector)
randomVector = randomVector.reshape(len(randomVector), 1)

print(entropy_estimators.entropy(randomMatrix1))
print(entropy_estimators.entropy(randomMatrix2))

# log(7*256)=10.8，与结果比较接近
print(entropy_estimators.entropy(randomVector))


CSIa1Orig = np.array(loadmat("testdata.mat")['testdata'][:, 0])
CSIa1Orig = CSIa1Orig.reshape(len(CSIa1Orig), 1)
CSIb1Orig = np.array(loadmat("testdata.mat")['testdata'][:, 1])
CSIb1Orig = CSIb1Orig.reshape(len(CSIb1Orig), 1)
print(entropy_estimators.entropy(CSIa1Orig))
print(entropy_estimators.entropy(CSIb1Orig))