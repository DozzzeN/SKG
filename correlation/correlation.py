from random import shuffle

import numpy as np
import scipy.spatial.distance
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
import pandas as pd

# https://www.zhihu.com/question/19734616
# 0.8-1.0 极强相关
# 0.6-0.8 强相关
# 0.4-0.6 中等程度相关
# 0.2-0.4 弱相关
# 0.0-0.2 极弱相关或无相关
n = 8
x1 = [0.5, 0.4, 0.6, 0.3, 0.6, 0.2, 0.7, 0.5]
x2 = [0.5, 0.6, 0.4, 0.7, 0.4, 0.8, 0.3, 0.5]  # 1 - x1
x3 = [1.5, 1.4, 1.6, 1.3, 1.6, 1.2, 1.7, 1.5]  # 1 + x1
shuffle(x3)  # x3 shuffle
gauss = np.random.normal(loc=np.mean(x1), scale=np.std(x1), size=n)
x4 = np.array(x1) * gauss
x5 = np.array(x1) + gauss

# p12 = 1 - pearsonr(x1, x2)[0]
# p13 = 1 - pearsonr(x1, x3)[0]
# p23 = 1 - pearsonr(x2, x3)[0]

p12 = pearsonr(x1, x2)[0]
p13 = pearsonr(x1, x3)[0]
p23 = pearsonr(x2, x3)[0]

print(pd.Series(x1).corr(pd.Series(x2)))
print(pd.Series(x1).corr(pd.Series(x3)))
print(pd.Series(x2).corr(pd.Series(x3)))

p144 = pearsonr(x4, gauss)[0]
p14 = pearsonr(np.log(np.abs(x4)), np.log(np.abs(gauss)))[0]
p15 = pearsonr(x1, x5)[0]

d12 = (euclidean(x1, x2) ** 2) / (2 * n)
d13 = (euclidean(x1, x3) ** 2) / (2 * n)
d23 = (euclidean(x2, x3) ** 2) / (2 * n)

c12 = cosine(x1, x2)
c13 = cosine(x1, x3)
c23 = cosine(x2, x3)

print("p144:", p144, "p14:", p14, "p15:", p15)
print('pearson:', np.round(p12, decimals=4), np.round(p13, decimals=4), np.round(p23, decimals=4))
print('cos:', np.round(c12, decimals=4), np.round(c13, decimals=4), np.round(c23, decimals=4))
print('euclidean sq:', np.round(d12, decimals=4), np.round(d13, decimals=4), np.round(d23, decimals=4))
