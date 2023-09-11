import numpy as np

q = 32
B = 2
assert B < int(np.log2(q) - 1)


# modular rounding []
def modular_rounding(x):
    return int((x * 2 / q + 0.5) % 2)


# cross-rounding <>
def cross_rounding(x):
    return int((x * 4 / q) % 2)

for i in range(q):
    print(i, bin(i), f"[{i}]", modular_rounding(i), f"<{i}>", cross_rounding(i))

print(((np.arange(0, q) * 2 / q + 0.5) % 2).astype(int))
print(((np.arange(0, q) * 4 / q) % 2).astype(int))
# q = 16
# I0 = np.arange(0, 4)
# I1_q2 = np.arange(4, 8) % q
# I0_q2 = np.arange(8, 12)
# I1 = np.arange(12, q)
# q = 32
I0 = np.arange(0, int(q / 4))
I1_q2 = np.arange(int(q / 4), int(q / 2))
I0_q2 = np.arange(int(q / 2), int(3 * q / 4))
I1 = np.arange(int(3 * q / 4), q)
# q = 18
# I0 = np.arange(0, 5)
# I1 = np.arange(14, q)
# I0_q2 = np.arange(0 + 9, 5 + 9)
# I1_q2 = np.arange(14 + 9, q + 9) % q
# <x>=0: 0, 1, 2, 3, 4, 9, 10, 11, 12, 13: I0, I0_q2
# <x>=1: 5, 6, 7, 8, 14, 15, 16, 17: I1, I1_q2
# [x]=0: 0, 1, 2, 3, 4, 14, 15, 16, 17: I0, I1
# [x]=1: 5, 6, 7, 8, 9, 10, 11, 12, 13: I0_q2, I1_q2
print(I0, I1_q2, I0_q2, I1)
E = np.arange(int(-q / 8), (q / 8)).astype(int)
print(E)

# for <x>=0: I0, I0_q2
w0 = np.arange(min(I0) + min(E), max(I0) + max(E) + 1)
w1 = np.arange(min(I0_q2) + min(E), max(I0_q2) + max(E) + 1)
print(w0 % q)
print(w1 % q)

# for <x>=1: I1, I1_q2
w0 = np.arange(min(I1) + min(E), max(I1) + max(E) + 1)
w1 = np.arange(min(I1_q2) + min(E), max(I1_q2) + max(E) + 1)
print(w0 % q)
print(w1 % q)


def rec(x, b):
    if b == 0:
        if x in I0 or x in I0_q2:
            return 0
        else:
            return 1
    else:
        if x in I1 or x in I1_q2:
            return 0
        else:
            return 1


def rec2(x, b):
    if b == 0:
        if x in np.arange(min(I0) - int(q / 8), max(I0) + int(q / 8) + 1) % q:
            return 0
        else:
            return 1
    else:
        if x in np.arange(min(I1) - int(q / 8), max(I1) + int(q / 8) + 1) % q:
            return 0
        else:
            return 1


e = int(q / 8)
for i in range(q):
    v = i
    w = np.arange(v - e, v + e + 1) % q
    for j in range(len(w)):
        print(v, w[j], rec2(w[j], cross_rounding(v)), modular_rounding(v))
        assert rec2(w[j], cross_rounding(v)) == modular_rounding(v)

# print(cross_rounding(w))
# print(cross_rounding(v))
# print(modular_rounding(v))
# print(modular_rounding(w))
