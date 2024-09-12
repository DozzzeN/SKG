import numpy as np

q = 32
print(int(np.log2(q) - 1))
B = 3
assert B < int(np.log2(q) - 1)


# modular rounding []
def modular_rounding(x):
    return int((x * 2 ** B / q + 0.5)) % (2 ** B)


# cross-rounding <>
def cross_rounding(x):
    return int((x * 2 ** (B + 1)) / q) % 2


for i in range(q):
    print(i, bin(i), f"[{i}]", modular_rounding(i), f"<{i}>", cross_rounding(i))



def recB(x, b, cross_roundings):
    equal_modular = []
    for i in range(q):
        if b == cross_roundings[i]:
            equal_modular.append(i)
    # 如果x=0, y=17, q=18, 则距离为1; 但如果用abs来计算，距离为17
    dists = []
    for i in range(len(equal_modular)):
        dists.append(min(abs(x - equal_modular[i]), q - abs(x - equal_modular[i])))
    return modular_rounding(equal_modular[np.argmin(dists)])


print()
e = int(q / 2 ** (B + 2))
cross_roundings = []
for i in range(q):
    cross_roundings.append(cross_rounding(i))
for i in range(q):
    v = i
    w = np.arange(v - e + 1, v + e) % q
    for j in range(len(w)):
        print(v, bin(v), w[j], bin(w[j]), recB(w[j], cross_rounding(v), cross_roundings), modular_rounding(v))
        assert recB(w[j], cross_rounding(v), cross_roundings) == modular_rounding(v)

