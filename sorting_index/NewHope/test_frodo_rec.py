import numpy as np

q = 64
print(int(np.log2(q) - 1))
B = 4
assert B < int(np.log2(q) - 1)


# modular rounding []
def modular_rounding(x):
    return int((x * 2 ** B / q + 0.5)) % (2 ** B)


# cross-rounding <>
def cross_rounding(x):
    return int((x * 2 ** (B + 1)) / q) % 2


for i in range(q):
    print(i, bin(i), f"[{i}]", modular_rounding(i), f"<{i}>", cross_rounding(i))


def dist(x, y, q):
    # 如果x=0, y=17, q=18, 则距离为1; 但如果用abs来计算，距离为17
    return min(abs(x - y), q - abs(x - y))


def recB(x, b):
    equal_modular = []
    for i in range(q):
        if b == cross_rounding(i):
            equal_modular.append(i)
    res = -1
    diff = q
    for i in range(len(equal_modular)):
        if dist(equal_modular[i], x, q) < diff:
            res = equal_modular[i]
            diff = dist(equal_modular[i], x, q)
    return modular_rounding(res)

print()
e = int(q / 2 ** (B + 2))
for i in range(q):
    v = i
    w = np.arange(v - e + 1, v + e) % q
    for j in range(len(w)):
        print(v, w[j], recB(w[j], cross_rounding(v)), modular_rounding(v), cross_rounding(v))
        assert recB(w[j], cross_rounding(v)) == modular_rounding(v)
