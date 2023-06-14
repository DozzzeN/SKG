import numpy as np
from scipy.io import loadmat

import poly

shuffles = np.random.permutation(1024)
x = np.random.randint(0, 12289, 1024)[shuffles]
x_prime = x.copy() + np.random.randint(0, 1000, 1024)[shuffles]
e = np.random.randint(0, 12289, 1024)[shuffles]

r = poly.helprec(x)
k = poly.rec(x, r)
k_prime = poly.rec(x_prime, r)

e_prime = poly.rec(e, r)
print(x)
print(x_prime)

print("max distance", (1 - 1 / (2 ** 2)) * 12289 - 2)
print(sum(abs(x - x_prime)))

print(k)
print(k_prime)
print(e_prime)
