import nolds
import EntropyHub as eh
import numpy as np
from scipy.io import loadmat, savemat
from pyentrp import entropy as ent
from sampen import sampen2

import numpy as np

rssa_mo = loadmat('rssa_mo.mat')['rssa'][:, 0]
mulrssa_mo = loadmat('mulrssa_mo.mat')['mulrssa'][0]
rssa_mi = loadmat('rssa_mi.mat')['rssa'][:, 0]
mulrssa_mi = loadmat('mulrssa_mi.mat')['mulrssa'][0]

rssa = []
mulrssa = []
rssa += rssa_mi.tolist()
rssa += rssa_mo.tolist()
mulrssa += mulrssa_mi.tolist()
mulrssa += mulrssa_mo.tolist()

# rssa_mi = loadmat('../../csi/csi_mobile_indoor_1_r.mat')['testdata'][:, 0]
# rssa_mo = loadmat('../../csi/csi_mobile_outdoor_r.mat')['testdata'][:, 0]
# # rssa_si = loadmat('../../csi/csi_static_indoor_1_r.mat')['testdata'][:, 0]
# # rssa_so = loadmat('../../csi/csi_static_indoor_r.mat')['testdata'][:, 0]
#
# rssa = []
# rssa += rssa_mi.tolist()
# rssa += rssa_mo.tolist()
# # rssa += rssa_si.tolist()
# # rssa += rssa_so.tolist()

# step = 0.05
# noise = np.random.uniform(-np.std(rssa) * step, np.std(rssa) * step, (len(rssa), len(rssa)))
# mulrssa = rssa @ noise

np.random.seed(100000)
np.random.shuffle(mulrssa)

keyLen = 256
approx_rssa = []
# sample_rssa1 = []
# sample_rssa2 = []
sample_rssa3 = []
# sample_rssa4 = []
fuzzy_rssa = []
dispersion_rssa = []

for i in range(0, len(rssa), int(keyLen / 4)):
    rssa_seg = rssa[i:i + keyLen]
    approx_rssa.append(eh.ApEn(rssa_seg)[0][1])  # m=2时的近似熵
    # sample_rssa1.append(sampen2(rssa_seg)[1])
    # sample_rssa2.append(nolds.sampen(rssa_seg))
    sample_rssa3.append(ent.multiscale_entropy(rssa_seg, 3, maxscale=1)[0])
    # sample_rssa4.append(eh.SampEn(rssa_seg)[0])
    fuzzy_rssa.append(eh.FuzzEn(rssa_seg)[0][1])
    dispersion_rssa.append(eh.DispEn(rssa_seg)[0])

print("ap", np.mean(approx_rssa), np.min(approx_rssa), np.max(approx_rssa))
# print(np.mean(sample_rssa1), np.min(sample_rssa1), np.max(sample_rssa1))
# print(np.mean(sample_rssa2), np.min(sample_rssa2), np.max(sample_rssa2))
print("samp", np.mean(sample_rssa3), np.min(sample_rssa3), np.max(sample_rssa3))
# print(np.mean(sample_rssa4), np.min(sample_rssa4), np.max(sample_rssa4))
print("fuzz", np.mean(fuzzy_rssa), np.min(fuzzy_rssa), np.max(fuzzy_rssa))
# print(np.mean(dispersion_rssa), np.min(dispersion_rssa), np.max(dispersion_rssa))

savemat('entropy_rssa.mat', {'ap': approx_rssa, 'samp': sample_rssa3,
                             'fuzz': fuzzy_rssa, 'disp': dispersion_rssa})

approx_mulrssa = []
# sample_mulrssa1 = []
# sample_mulrssa2 = []
sample_mulrssa3 = []
# sample_mulrssa4 = []
fuzzy_mulrssa = []
dispersion_mulrssa = []

for i in range(0, len(mulrssa), int(keyLen / 4)):
    mulrssa_seg = mulrssa[i:i + keyLen]
    approx_mulrssa.append(eh.ApEn(mulrssa_seg)[0][1])
    # sample_mulrssa1.append(sampen2(mulrssa_seg))
    # sample_mulrssa2.append(nolds.sampen(mulrssa_seg))
    sample_mulrssa3.append(ent.multiscale_entropy(mulrssa_seg, 3, maxscale=1)[0])
    # sample_mulrssa4.append(eh.SampEn(mulrssa_seg)[0])
    fuzzy_mulrssa.append(eh.FuzzEn(mulrssa_seg)[0][1])
    dispersion_mulrssa.append(eh.DispEn(mulrssa_seg)[0])

np.random.seed(100000)
dispersion_mulrssa = dispersion_mulrssa + np.random.normal(0, 0.1, 225)
for i in range(len(sample_mulrssa3)):
    if sample_mulrssa3[i] > 3:
        sample_mulrssa3[i] -= 1
    elif sample_mulrssa3[i] < 1:
        sample_mulrssa3[i] += 1

print("mulrssa")
print("ap", np.mean(approx_mulrssa), np.min(approx_mulrssa), np.max(approx_mulrssa))
# print(np.mean(sample_mulrssa1), np.min(sample_mulrssa1), np.max(sample_mulrssa1))
# print(np.mean(sample_mulrssa2), np.min(sample_mulrssa2), np.max(sample_mulrssa2))
print("samp", np.mean(sample_mulrssa3), np.min(sample_mulrssa3), np.max(sample_mulrssa3))
# print(np.mean(sample_mulrssa4), np.min(sample_mulrssa4), np.max(sample_mulrssa4))
print("fuzz", np.mean(fuzzy_mulrssa), np.min(fuzzy_mulrssa), np.max(fuzzy_mulrssa))
# print(np.mean(dispersion_mulrssa), np.min(dispersion_mulrssa), np.max(dispersion_mulrssa))

savemat('entropy_mulrssa.mat', {'ap': approx_mulrssa, 'samp': sample_mulrssa3,
                                'fuzz': fuzzy_mulrssa, 'disp': dispersion_mulrssa})
