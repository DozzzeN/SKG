import sys
import numpy as np
import random
import math

nvals = 20

B = []
e = []
s = 20
M1 = 6
M2 = 4
M12 = M1 + M2

q = 97


def get_uv(A, B, M, q):
    u = 0
    v = 0
    sample = random.sample(range(nvals - 1), nvals // 4)
    r = np.random.randint(1, 3)
    e1 = list(np.random.randint(1, 3, len(A)))
    e2 = list(np.random.randint(1, 3, len(B)))
    uu = list(A) * r + e1
    vv = list(B) * r + e2
    print("sample", sample)
    for x in range(0, len(sample)):
        u = u + (uu[sample[x]])

        v = v + vv[sample[x]]

    v = v + math.floor(q / 2) * M
    return u % q, v % q


def get_result(u1, v1, u2, v2, q):
    res = ((v1 - s * u1) + (v2 - s * u2)) % q

    if (res > q // 2):
        return 1

    return 0

def dec(u1, v1, q):
    res = (v1 - s * u1) % q

    if (res > q // 2):
        return 1

    return 0

def tobits(val):
    l = [0] * (8)

    l[0] = val & 0x1
    l[1] = (val & 0x2) >> 1
    l[2] = (val & 0x4) >> 2
    l[3] = (val & 0x8) >> 3
    return l


if (len(sys.argv) > 1):
    M1 = int(sys.argv[1])

if (len(sys.argv) > 2):
    M2 = int(sys.argv[2])

if (len(sys.argv) > 3):
    s = int(sys.argv[3])

if (len(sys.argv) > 4):
    q = int(sys.argv[4])

A = random.sample(range(q), nvals)

for x in range(0, len(A)):
    e.append(random.randint(1, 3))
    B.append((A[x] * s + e[x]) % q)

print("\n------Parameters and keys-------")
print("Value to cipher:\t", M1, M2)
print("Public Key (A):\t", A)
print("Public Key (B):\t", B)
print("Errors (e):\t\t", e)
print("Secret key:\t\t", s)
print("Prime number:\t\t", q)

print("\n------Sampling Process from public key-------")

bits1 = tobits(M1)
bits2 = tobits(M2)
bitsM12 = tobits(M12)

print("Bits to be ciphered:", bits1, bits2, bitsM12)

u1_1, v1_1 = get_uv(A, B, bits1[0], q)
u2_1, v2_1 = get_uv(A, B, bits1[1], q)
u3_1, v3_1 = get_uv(A, B, bits1[2], q)
u4_1, v4_1 = get_uv(A, B, bits1[3], q)

u1_2, v1_2 = get_uv(A, B, bits2[0], q)
u2_2, v2_2 = get_uv(A, B, bits2[1], q)
u3_2, v3_2 = get_uv(A, B, bits2[2], q)
u4_2, v4_2 = get_uv(A, B, bits2[3], q)

u1_12, v1_12 = get_uv(A, B, bitsM12[0], q)
u2_12, v2_12 = get_uv(A, B, bitsM12[1], q)
u3_12, v3_12 = get_uv(A, B, bitsM12[2], q)
u4_12, v4_12 = get_uv(A, B, bitsM12[3], q)

print("\n------Results                -----------------")

print("Result bit0 is", get_result(u1_1, v1_1, u1_2, v1_2, q), dec(u1_1, v1_1, q), dec(u1_2, v1_2, q), dec(u1_12, v1_12, q))
print("Result bit1 is", get_result(u2_1, v2_1, u2_2, v2_2, q), dec(u2_1, v2_1, q), dec(u2_2, v2_2, q), dec(u2_12, v2_12, q))
print("Result bit2 is", get_result(u3_1, v3_1, u3_2, v3_2, q), dec(u3_1, v3_1, q), dec(u3_2, v3_2, q), dec(u3_12, v3_12, q))
print("Result bit3 is", get_result(u4_1, v4_1, u4_2, v4_2, q), dec(u4_1, v4_1, q), dec(u4_2, v4_2, q), dec(u4_12, v4_12, q))

# print("\n------Key Exchange           -----------------")
# e1 = []
# e2 = []
# A1 = []
# A2 = []
#
# s1 = 20
# s2 = 13
#
# k1 = []
# k2 = []
#
# for x in range(0, len(A)):
#     e1.append(random.randint(1, 3))
#     e2.append(random.randint(1, 3))
#     A1.append((A[x] * s1 + e1[x]) % q)
#     A2.append((A[x] * s2 + e2[x]) % q)
#     k1.append((A2[x] * s1) % q)
#     k2.append((A1[x] * s2) % q)
# print(k1)
# print(k2)
