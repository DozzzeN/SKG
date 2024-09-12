import numpy as np

# Example data
keyLen = 64

# Ensure rawCSIa1 and randomMatrix are zero-mean
# rawCSIa1 = np.random.randn(keyLen, keyLen)
rawCSIa1 = np.random.normal(0, 1, size=(keyLen, keyLen))
randomMatrix = np.random.normal(0, 1, size=(keyLen, keyLen))
# randomMatrix = np.random.uniform(0, 1, size=(keyLen, keyLen))

# Zero-mean adjustments
rawCSIa1 = rawCSIa1 - np.mean(rawCSIa1)
randomMatrix = randomMatrix - np.mean(randomMatrix)

# Matrix multiplication
result = np.matmul(rawCSIa1, randomMatrix)

# Mean of the resulting matrix
result_mean = np.mean(result)

print("Raw CSIa1 Matrix (zero-mean):")
print(rawCSIa1)
print("Random Matrix (zero-mean):")
print(randomMatrix)
print("Resulting Matrix:")
print(result)
# 零均值的向量与矩阵相乘，乘积的均值不一定为零
print("Mean of the resulting matrix:", result_mean)

