import numpy as np
import poly

x = np.random.randint(0, 12289, 1024)
r = poly.helprec(x)
k = poly.rec(x, r)
# x_prime = np.random.randint(0, 12289, 1024)
x_prime = x.copy() + np.random.randint(0, 1000, 1024)
k_prime = poly.rec(x_prime, r)

print("max distance", (1 - 1 / (2 ** 2)) * 12289 - 2)
print(sum(abs(x - x_prime)))

print(k)
print(k_prime)
