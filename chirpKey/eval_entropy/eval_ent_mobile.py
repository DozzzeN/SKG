import nolds
import EntropyHub as eh
import numpy as np
from scipy.io import loadmat, savemat
from pyentrp import entropy as ent
from sampen import sampen2

import numpy as np

rssa_o = loadmat('rssa_mi.mat')['rssa'][:, 0]

rssa = []
rssa += rssa_o.tolist()
rssa = np.round(rssa, 2)
rssa += np.random.normal(0, 0.001, len(rssa))

keyLen = 256
approx_rssa = []
# sample_rssa1 = []
# sample_rssa2 = []
sample_rssa3 = []
# sample_rssa4 = []
fuzzy_rssa = []
dispersion_rssa = []

print(np.var(rssa))
print(eh.ApEn(rssa)[0][1])
print(ent.multiscale_entropy(rssa, 3, maxscale=1)[0])
print(eh.FuzzEn(rssa)[0][1])
print(eh.DispEn(rssa)[0])

exit()

for i in range(0, len(rssa), int(len(rssa) / 4)):
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

savemat('entropy_rssa_mi.mat', {'ap': approx_rssa, 'samp': sample_rssa3,
                             'fuzz': fuzzy_rssa, 'disp': dispersion_rssa})
