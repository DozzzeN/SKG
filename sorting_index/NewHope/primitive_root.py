import math

import numpy as np

import precomp

print(len(precomp.omegas_montgomery), np.sort(precomp.omegas_montgomery))
for i in range(0, len(precomp.omegas_montgomery)):
    if pow(precomp.omegas_montgomery[i], 1024, 12289) != 1:
        print(i, precomp.omegas_montgomery[i], pow(precomp.omegas_montgomery[i], 1024, 12289))

roots = []
for i in range(0, 12289):
    if pow(i, 1024, 12289) == 1:
        isPrime = True
        for j in range(1, 1024):
            if pow(i, j, 12289) == 1:
                isPrime = False
                break
        if isPrime:
            roots.append(i)
print(len(roots), roots)
matchs = []
for r in roots:
    for om in precomp.omegas_montgomery:
        if r * 2 ** 18 % 12289 == om:
            matchs.append([r, om])
print(np.sort(matchs, axis=0))
print(len(matchs))

exit()
roots_mul = []
for i in range(0, len(roots)):
    res = roots[i] ** i * 2**18 % 12289
    if res not in roots_mul:
        roots_mul.append(res)
# roots_mul.sort()
# print(len(roots_mul), roots_mul)

r = 2
roots_mul = []
for i in range(0, 1024):
    res = (r ** i * 2**18) % 12289
    if res not in roots_mul:
        roots_mul.append(res)
roots_mul.sort()
print(len(roots_mul), roots_mul)
