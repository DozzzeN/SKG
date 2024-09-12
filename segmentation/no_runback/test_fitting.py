import math

from sympy import symbols, expand

t = symbols('t')


def f(fk, fk_1, t, m, k):
    return (t - (m - 1) / 2) * fk - (k ** 2 * m ** 2 - k ** 4) / (16 * k ** 2 - 4) * fk_1


def a(fk, qs, m, k):
    fs_mod = (math.factorial(k) ** 4 / (math.factorial(2 * k) * math.factorial(2 * k + 1)))
    for i in range(-k, k + 1):
        fs_mod *= (m + i)
    right = 0
    for t in range(m):
        right += qs[t] * fk.subs('t', t)
    ak = right / fs_mod
    return ak


f0_1 = 0 + 0 * t
f0 = 1 + 0 * t
m = 5
f1 = f(f0, f0_1, t, m, 0)
f2 = f(f1, f0, t, m, 1)
f3 = f(f2, f1, t, m, 2)
f4 = f(f3, f2, t, m, 3)
print(f1)
print(expand(f2))
print(expand(f3))
print(expand(f4))

Q = [10, 10, 6, 4, 10]
a0 = a(f0, Q, m, 0)
a1 = a(f1, Q, m, 1)
a2 = a(f2, Q, m, 2)
a3 = a(f3, Q, m, 3)
a4 = a(f4, Q, m, 4)
print(a0)
print(a1)
print(a2)
print(a3)
print(a4)

F1 = [a0 * f0, a1 * f1, a2 * f2, a3 * f3, a4 * f4]
F10 = F1[0]
F11 = F1[0] + F1[1]
F12 = F1[0] + F1[1] + F1[2]
F13 = F1[0] + F1[1] + F1[2] + F1[3]
F14 = F1[0] + F1[1] + F1[2] + F1[3] + F1[4]
F10_fitting = [F10.subs('t', i) for i in range(m)]
F11_fitting = [F11.subs('t', i) for i in range(m)]
F12_fitting = [F12.subs('t', i) for i in range(m)]
F13_fitting = [F13.subs('t', i) for i in range(m)]
F14_fitting = [F14.subs('t', i) for i in range(m)]
print()
print(F10_fitting)
print(F11_fitting)
print(F12_fitting)
print(F13_fitting)
print(F14_fitting)
