import numpy as np

q = 33
g = 3


def signal(sigma):
    res = int(sigma * 2 ** (g + 1) / q) - int(2 / q * sigma) * 2 ** (g + 1)
    return bin(res)[2:].zfill(g)


def coordinate(v, b):
    tmp0 = (b ^ int(v[0])) * int(q / 2)
    for i in range(len(v)):
        tmp0 += int(v[i]) * int(q / 2 ** (i + 2))
    tmp1 = tmp0 + int(q / 2 ** (g + 1))
    return np.arange(tmp0, tmp1)


D = np.arange(int(q / 4), int(3 * q / 4))
E = np.arange(int(-q / 4 * (1 - 0.5 ** g)), int(q / 4 * (1 - 0.5 ** g))).astype(int)

def neg(sigma):
    v0 = 0 if sigma in D else 1
    kl_prime = int(sigma + v0 * (q - 1) / 2) % q % 2
    kh_prime = int(2 * sigma / q + 1 / 2) % 2
    return kh_prime, kl_prime


def com(sigma, v):
    C0 = coordinate(v, 0)
    C1 = coordinate(v, 1)
    assert len(C0) == len(C1) == int(q / 2 ** (g + 1))
    E0 = np.arange(min(E) + min(C0), max(E) + max(C0) + 1)
    E1 = np.arange(min(E) + min(C1), max(E) + max(C1) + 1)

    kh = None
    if sigma in E0:
        kh = 0
    elif sigma in E1:
        kh = 1
    tmp = (kh ^ int(v[0])) * int(q / 2)
    for i in range(len(v)):
        tmp += int(v[i]) * int(q / 2 ** (i + 2))
    v0prime = 0 if tmp in D else 1
    kl = int(sigma + v0prime * (q - 1) / 2) % q % 2
    return kh, kl

delta = np.random.randint(0, q)
sigma1 = q + 1
sigma2 = q + 2 * delta + 1
# sigma1 = (q + 1) % q
# sigma2 = (q + 2 * delta + 1) % q
print(sigma1, sigma2)
d = int(q / 4 * (1 - 0.5 ** g))
v = signal(sigma2)
print(v)
k_prime = neg(sigma2)
print(k_prime)
k = com(sigma1, v)
print(k)