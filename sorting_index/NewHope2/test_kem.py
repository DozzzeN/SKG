import numpy as np

import poly, newhope


def decode(k):
    miu = np.zeros(32, dtype=np.int64)
    halfQ = int((12289 - 1) / 2)
    for i in range(256):
        t = abs(k[i] - halfQ)
        t += abs(k[i + 256] - halfQ)
        t += abs(k[i + 512] - halfQ)
        t += abs(k[i + 768] - halfQ)
        t = t - 12289
        if t >= 0:
            t = t >> 63
        else:
            # t = t >> 63
            t = int(bin(t)[bin(t).find('b') + 1:bin(t).find('b') + 2])
        miu[i >> 3] |= (t << (i & 7))
    return miu


def NHSEncode(v):
    k = np.zeros(1024, dtype=np.int64)
    halfQ = int((12289 - 1) / 2)
    for i in range(256):
        k[i] = v[i] * halfQ
        k[i + 256] = v[i] * halfQ
        k[i + 512] = v[i] * halfQ
        k[i + 768] = v[i] * halfQ
    return k


def NHSDecode(k):
    v = np.zeros(256, dtype=np.int64)
    halfQ = int((12289 - 1) / 2)
    for i in range(256):
        t = abs(k[i] - halfQ) + abs(k[i + 256] - halfQ) + abs(k[i + 512] - halfQ) + abs(k[i + 768] - halfQ)
        if t < 12289:
            v[i] = 1
        else:
            v[i] = 0
    return v


def NHSCompress(c):
    c_prime = []
    for i in range(1024):
        c_prime.append(int(c[i] * 8 / 12289) % 8)
    return c_prime

def NHSDecompress(c_prime):
    c = []
    for i in range(1024):
        c.append(int(c_prime[i] * 12289 / 8))
    return c


v = np.random.randint(0, 2, 256)
v_prime = NHSDecode(NHSEncode(v))
print(v)
print(v_prime)
print(v == v_prime)

v = np.random.randint(0, 12289, 1024)
v_com = NHSCompress(v)
print(v)
print(np.array(NHSDecompress(v_com)))
print(v == NHSDecompress(v_com))

# reconciliation-based key encapsulation mechanism

a = np.array(np.random.randint(0, 12289, 1024), dtype=np.int64)
s1 = newhope.get_noise()
e = newhope.get_noise()
b = poly.add(e, poly.pointwise(s1, a))

a_prime = np.array(a + np.random.randint(0, 100, 1024), dtype=np.int64)
s2 = newhope.get_noise()
e_prime = newhope.get_noise()
e_prime_prime = poly.get_noise()
u = poly.add(poly.pointwise(a_prime, s2), e_prime)
v = poly.add(poly.pointwise(b, s2), e_prime_prime)
k = np.array(np.random.randint(0, 2, 256), dtype=np.int64)
k_encode = NHSEncode(k)

c = poly.add(v, k_encode)
# c = poly.add(v, k)
c_prime = NHSCompress(c)
# r = poly.helprec(k)

keyb = k
print(keyb)

c_prime_prime = NHSDecompress(c_prime)
v_prime = poly.pointwise(s1, u)
# k_prime = poly.sub(c, v_prime)
k_prime = poly.sub(c_prime_prime, v_prime)
keya = NHSDecode(k_prime)
print(keya)

print(keya == keyb)
keye = NHSDecode(poly.sub(c_prime, np.random.randint(0, 12289, 1024)))
print(keya == keye)