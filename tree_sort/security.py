import numpy as np


def asymmetricKL(P, Q):
    l = np.log(P / Q)
    return sum(P * l)


def symmetricalKL(P, Q):
    return (asymmetricKL(P, Q) + asymmetricKL(Q, P)) / 2.00


a = [1, 3, 2, 3, 20]
b = [2, 3, 2, 3, 20]
c = [20, 2, 3, 2, 3]
d = [1, 3, 20, 30, 20]
a, b, c, d = np.array(a), np.array(b), np.array(c), np.array(d)
# a = a - np.min(a) + 0.1
# b = b - np.min(b) + 0.1

print(symmetricalKL(a, b))
print(symmetricalKL(a, c))
print(symmetricalKL(a, d))
print()
print(symmetricalKL(a, b))
np.random.shuffle(b)
print(symmetricalKL(a, b))
np.random.shuffle(b)
print(symmetricalKL(a, b))
