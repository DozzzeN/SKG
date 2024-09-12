import random

import numpy as np
from matplotlib import pyplot as plt

n = 100000
k = 10

isShow = False

a = []
d = []
e = []
for i in range(n):
    a.append(np.sum(np.random.laplace(0, 1, k)))
    b = np.random.normal(0, 1, k)
    c = np.random.normal(0, 1, k)
    d.append(np.sum(np.multiply(b, c)))
    e.append(random.gammavariate(k, 1) - random.gammavariate(k, 1))
print("mean", np.mean(a))
print("var", np.var(a))
print("mean", np.mean(d))
print("var", np.var(d))
print("mean", np.mean(e))
print("var", np.var(e))

if isShow:
    plt.figure()
    plt.hist(a, bins=50)
    plt.show()

    plt.figure()
    plt.hist(e, bins=50)
    plt.show()

