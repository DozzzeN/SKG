import numpy as np

q = 15
# ri = np.random.choice([-1, 0, 0, 1], q)
ri = [0, -1, 1, 0, 0, 0, 0, 0, 1, -1, -1, -1, 1, 0, 0]
print(ri)


# modular rounding []
def modular_rounding(x):
    k = int(x * 2 / (2 * q) + 0.5) % 2
    return k


# cross-rounding <>
def cross_rounding(x):
    c = int(x * 4 / (2 * q)) % 2
    return c


for i in range(q):
    dbl_v = 2 * i - ri[i]
    print(i, bin(i), dbl_v, f"k{i}", modular_rounding(dbl_v), f"c{i}", cross_rounding(dbl_v))

I0 = np.arange(0, int(q / 2 + 1 / 2))
I1 = np.arange(-int(q / 2), 0) % q
print(I0, I1)
E = np.arange(int(-q / 4), int(q / 4)).astype(int)
print(E)


def rec(x, b):
    if b == 0:
        if 2 * x in (np.arange(min(I0) - min(E), max(I0) + max(E)) % (2 * q)):
            return 0
        else:
            return 1
    else:
        if 2 * x in (np.arange(min(I1) - min(E), max(I1) + max(E)) % (2 * q)):
            return 0
        else:
            return 1


e = int(q / 8)
for i in range(q):
    w = i
    v = np.arange(w - e, w + e + 1) % q
    for j in range(len(v)):
        dbl_v = 2 * v[j] - ri[v[j]]
        print(w, v[j], rec(w, cross_rounding(dbl_v)), modular_rounding(dbl_v))
