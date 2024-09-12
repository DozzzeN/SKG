import numpy as np

p = np.random.randint(0, 2, 10)
a = abs(p + np.random.normal(0, 0.2, 10))
b = abs(p + np.random.normal(0, 0.2, 10))

def rec(a, bits):
    alpha = 0.1
    if bits == 1:
        a[0] = round(a[0])
        for i in range(1, len(a)):
            if a[i] >= 1 / 2 - alpha and a[i] <= 1 / 2 + alpha:
                if a[i - 1] == 0:
                    a[i] = round(a[i] + 1 / 2)
                elif a[i - 1] == 1:
                    a[i] = round(a[i] - 1 / 2)
            else:
                a[i] = round(a[i])
    elif bits == 2:
        a[0] = round(a[0])
        for i in range(1, len(a)):
            if a[i] >= 1 / 2 - alpha and a[i] <= 1 / 2 + alpha:
                if a[i - 1] == 0:
                    a[i] = round(a[i] + 1 / 2)
                elif a[i - 1] == 1:
                    a[i] = round(a[i] - 1 / 2)
            elif a[i] >= 3 / 2 - alpha and a[i] <= 3 / 2 + alpha:
                if a[i - 1] == 1:
                    a[i] = round(a[i] + 1 / 2)
                elif a[i - 1] == 2:
                    a[i] = round(a[i] - 1 / 2)
            elif a[i] >= 5 / 2 - alpha and a[i] <= 5/ 2 + alpha:
                if a[i - 1] == 2:
                    a[i] = round(a[i] + 1 / 2)
                elif a[i - 1] == 3:
                    a[i] = round(a[i] - 1 / 2)
            else:
                a[i] = round(a[i])
    return a


print(a)
print(b)
print(rec(a, 1))
print(rec(b, 1))
print(np.allclose(rec(a, 1), rec(b, 1)))
