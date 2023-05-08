import random

import numpy as np
from matplotlib import pyplot as plt
from sympy.stats import Erlang

n = 1000000

isShow = False

expo1 = []
expo2 = []
erlang = []
for i in range(n):
    expo1.append(np.random.exponential(1))
    expo2.append(np.random.exponential(1))
    erlang.append(random.gammavariate(2, 1))
p1 = np.array(expo1) + np.array(expo2)
print("mean", np.mean(p1), np.mean(erlang))
print("var", np.var(p1), np.var(erlang))

if isShow:
    plt.figure()
    plt.hist(p1, bins=50)
    plt.show()

    plt.figure()
    plt.hist(erlang, bins=50)
    plt.show()