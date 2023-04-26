import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as ss
import pandas as pd
from scipy.stats import pearsonr
import entropy_estimators as ee

n = 100000

# p1=s1*sigma1,p2=s2*sigma2
data = []
theory = []
for i in range(1, n):
    a = np.random.normal(0, 1)
    b = np.random.normal(0, 1)
    data.append(a * b)
    # bessel函数不包含0
    theory.append(ss.kn(0, i / n) / math.pi)
plt.figure()
plt.hist(data, bins=50)
# plt.show()
plt.figure()
theoryf = pd.DataFrame(theory)
plt.hist(theoryf[np.isfinite(theoryf)].values, bins=50)
plt.hist(-theoryf[np.isfinite(theoryf)].values, bins=50)
# plt.show()

data2 = []
for i in range(1, n):
    a = np.random.normal(0, 1)
    b = np.random.normal(0, 1)
    data2.append(a * b)
print("corr", pearsonr(data, data2)[0])
print(np.sum(np.multiply(data, data2)) / len(data))

# p1=s*sigma1, p2=s*sigma2
s = []
sigma1 = []
sigma2 = []
for i in range(n):
    s.append(np.random.normal(0, 1))
    sigma1.append(np.random.normal(0, 1))
    sigma2.append(np.random.normal(0, 1))
p1 = np.multiply(np.array(s), np.array(sigma1))
p2 = np.multiply(np.array(s), np.array(sigma2))
print("mean", np.mean(p1), np.mean(p2))
print("var", np.var(p1), np.var(p2))
print("corr", pearsonr(p1, p2)[0])
print(np.sum(np.multiply(p1, p2)) / len(p1))

sl = []
p1l = []
p2l = []
for i in range(len(s)):
    sl.append(list([s[i]]))
    p1l.append(list([p1[i]]))
    p2l.append(list([p2[i]]))
print("entropy", ee.entropy(sl), np.log2(2 * np.pi * np.e) / 2)
print("entropy", ee.entropy(p1l), np.log2(2 * np.pi * np.e))
print("entropy", ee.entropy(p2l), np.log2(2 * np.pi * np.e))
