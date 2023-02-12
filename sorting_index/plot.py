import numpy as np
from scipy.io import loadmat

rssa = [-52, -54, -53, -48, -41, -48, -47, -45, -47.5]
rssb = [-51.5, -53.5, -52.5, -49, -50, -48, -46.5, -44.5, -47]

# rawData = loadmat("../skyglow/Scenario3-Mobile/data_mobile_1.mat")
# rawData = loadmat("../skyglow/Scenario2-Office-LoS/data3_upto5.mat")
rawData = loadmat("../skyglow/Scenario5-Outdoors-Mobile-OutsideSphere1/data_mobile_1.mat")
# rawData = loadmat("../skyglow/Scenario6-Outdoors-Stationary-OutsideSphere1/data_static_1.mat")
CSIa1Orig = rawData['A'][:, 0]
CSIb1Orig = rawData['A'][:, 1]

# start = 200
# start = 5000
# for start in range(10, 10000, 12):
#     print()
# rssa = CSIa1Orig[start:start + 12]
# rssb = CSIb1Orig[start:start + 12]

sorta = list(np.array(rssa).argsort() + 1)
sortb = list(np.array(rssb).argsort() + 1)

print(list(rssa))
print(list(rssb))

print(sorta)
print(sortb)

rssa.sort()
rssb.sort()

print(rssa)
print(rssb)
