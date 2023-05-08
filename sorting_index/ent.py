import random

import numpy as np
import scipy.stats as st

# smoke-grenade中H(x3kappa3+x4kappa4)的仿真验证
n = 10000
k1 = 10
k2 = 10

# a = []
# for i in range(n):
#     t = st.norm.pdf((i - int(n / 2) / 10), loc=0, scale=1)
#     if round(t, 10) != 0:
#         a.append(t)
#
# print(np.sum(-np.log(a) * a))
# print(np.log(2 * np.e * np.pi) / 2)

a = []
step = 0.01
t1 = st.norm.pdf(np.arange(0, 100, step), loc=0, scale=1)
t2 = st.norm.pdf(np.arange(0, 100, step), loc=0, scale=k1)
t3 = st.norm.pdf(np.arange(0, 100, step), loc=0, scale=1)
t4 = st.norm.pdf(np.arange(0, 100, step), loc=0, scale=k2)
t = t1 * t2 + t3 * t4
for i in range(len(t)):
    if round(t[i], 5) != 0:
        a.append(t[i])

print(np.sum(-np.log(a) * a) * step) # 乘step表示积分，因为求的是微分熵
print(np.log(2 * np.e * np.sqrt(np.square(k1) + 1) / 2))
print(np.log(4 * np.pi * np.e * np.square(k1)) / 2 + (1 / np.e - 1) / 2)
