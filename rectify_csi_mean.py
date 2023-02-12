import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr

import entropy_estimators

CSIa1Orig = list(loadmat("csi_static_indoor.mat")['testdata'][:, 0])
CSIb1Orig = list(loadmat("csi_static_indoor.mat")['testdata'][:, 1])

CSIa = []
CSIb = []
print(len(CSIa1Orig), len(CSIb1Orig))

# mobile选的step是10
# static选的step是5000
step = 5000
for i in range(0, len(CSIa1Orig)):
    CSIa.append(CSIa1Orig[i])
    CSIb.append(CSIb1Orig[i] - (np.mean(CSIb1Orig[i:i+step]-np.mean(CSIa1Orig[i:i+step]))))
print(len(CSIa), len(CSIb))

print(len(CSIa), len(CSIb))

print(pearsonr(CSIa, CSIb)[0])
print(pearsonr(CSIa1Orig, CSIb1Orig)[0])

savemat('csi_mobile_indoor_r.mat', {"testdata": np.array([CSIa, CSIb]).T})

# plt.figure()
# plt.plot(CSIa)
# plt.plot(CSIb)
# plt.show()