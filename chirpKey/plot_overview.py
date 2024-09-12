from PIL import Image
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import numpy as np
from scipy.linalg import circulant
from scipy.signal import medfilt


def normalize(data):
    return (np.array(data) - np.min(data)) / (np.max(data) - np.min(data)) * 50


fileName = "../data/data_mobile_indoor_1.mat"
rawData = loadmat(fileName)

CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

keyLen = 16
staInd = 220
lw = 7
ms = 14
endInd = staInd + keyLen
x = range(staInd, endInd, 1)
tmpCSIa1 = np.array(normalize(CSIa1Orig[range(staInd, endInd, 1)])).astype(np.int)
tmpCSIb1 = np.array(normalize(CSIb1Orig[range(staInd, endInd, 1)])).astype(np.int)
tmpCSIb1[0] += 6
tmpCSIb1[1] += 6
tmpCSIb1[14] -= 6


# bins_interval = 3
# margin = 1
# bins = range(min(tmpCSIa1), max(tmpCSIa1) + bins_interval - 1, bins_interval)
# plt.xlim(min(tmpCSIa1) - margin, max(tmpCSIa1) + margin)
# plt.axis('off')
# plt.hist(x=tmpCSIa1, bins=bins, histtype='bar', color=['r'])
# plt.savefig('a1.svg')
# plt.show()

plt.plot(x, tmpCSIa1, color='r', linewidth=lw, marker='o', markersize=ms, linestyle='-')
plt.axhline(np.mean(tmpCSIa1), color='black', linewidth=lw, linestyle='--')
plt.axis('off')
plt.tight_layout()
plt.savefig('a1.svg', dpi=1200, bbox_inches='tight')
# plt.show()
plt.close()

plt.plot(x, tmpCSIb1, color='k', linewidth=lw, marker='s', markersize=ms, linestyle='-')
plt.axhline(np.mean(tmpCSIb1), color='black', linewidth=lw, linestyle='--')
plt.axis('off')
plt.tight_layout()
plt.savefig('b1.svg', dpi=120, bbox_inches='tight')
plt.close()

# plt.show()
# noise perturbation

randomMatrix = np.random.uniform(0, 2, size=(keyLen, keyLen))

# tmpCSIa1 = np.array(normalize(np.matmul(tmpCSIa1, randomMatrix)).astype(np.int))
# tmpCSIb1 = np.array(normalize(np.matmul(tmpCSIb1, randomMatrix)).astype(np.int))

# plt.plot(x, tmpCSIa1, color='r', linewidth=lw, marker='o', markersize=ms, linestyle='-')
# plt.axhline(np.mean(tmpCSIa1), color='black', linewidth=lw, linestyle='--')
# plt.axis('off')
# plt.tight_layout()
# plt.savefig('a2.svg')
# plt.show()
#
# plt.plot(x, tmpCSIb1, color='k', linewidth=lw, marker='s', markersize=ms, linestyle='-')
# plt.axhline(np.mean(tmpCSIb1), color='black', linewidth=lw, linestyle='--')
# plt.axis('off')
# plt.tight_layout()
# plt.savefig('b2.svg')
# plt.show()

# matrix generation and equilibration
tmpCSIa1 = circulant(tmpCSIa1[::-1])
tmpCSIb1 = circulant(tmpCSIb1[::-1])

# U, S, Vt = np.linalg.svd(tmpCSIa1)
# S = medfilt(S, kernel_size=3)
# D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
# tmpCSIa1 = U @ D @ Vt

# U, S, Vt = np.linalg.svd(tmpCSIb1)
# S = medfilt(S, kernel_size=3)
# D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
# tmpCSIb1 = U @ D @ Vt

plt.imshow(tmpCSIa1, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig('a3.svg', dpi=1200, bbox_inches='tight')
# plt.show()
plt.close()

plt.imshow(tmpCSIb1, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig('b3.svg', dpi=1200, bbox_inches='tight')
# plt.show()
plt.close()

x = np.random.randint(0, 2, keyLen)
bA = np.array(tmpCSIa1 @ x)
bA = bA.reshape((keyLen, 1))

plt.imshow(bA, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig('a4.svg', dpi=1200, bbox_inches='tight')
# plt.show()
plt.close()
