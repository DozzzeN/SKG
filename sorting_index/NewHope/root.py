import math

import numpy as np


def primRoots(modulo):
    coprime_set = {num for num in range(1, modulo) if math.gcd(num, modulo) == 1}
    return [g for g in range(1, modulo) if coprime_set == {pow(g, powers, modulo)
                                                           for powers in range(1, modulo)}]


# print(primRoots(12289))
def inv(a, b):
    if b == 0:
        return 1, 0, a
    else:
        x, y, q = inv(b, a % b)
        x, y = y, (x - (a // b) * y)
        return x, y, q


def ModReverse(a, p):
    x, y, q = inv(a, p)
    if q != 1:
        raise Exception("No solution.")
    else:
        return (x + p) % p  # 防止负数


# print(primRoots(1024))
print(primRoots(12289))

# print(2 ** 18 % 12289)
# print(ModReverse(2 ** 18, 12289))
# print(576 * 6974 % 12289)
# print((2 ** 18 * 10810) % 12289)
#
# print()
# for j in range(12289):
#     muls = []
#     for i in range(12289):
#         muls.append(pow(j, i, 12289))
#     muls.sort()
#     if muls == list(range(12289)):
#         print(j)
