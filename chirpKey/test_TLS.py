import numpy as np
from scipy.linalg import circulant

from tls import tls, stls

A0 = np.array([[1.529, 0.584], [0.989, 0.839], [1.094, -0.091]])
A1 = np.array([[1.038, 0.935], [0.177, -0.140], [0.681, -0.148]])
A2 = np.array([[1.074, 1.132], [1.287, 0.224], [0.092, 1.195]])
# A0 = np.array([[1, 1], [1, 1], [1, 0]])
# A1 = np.array([[1, 1], [0, 0], [1, 0]])
# A2 = np.array([[1, 1], [1, 0], [0, 1]])
b0 = np.array([5.934, 2.925, 2.941])
b1 = np.array([5.656, 2.989, 3.043])
b2 = np.array([6.434, 3.114, 3.163])
b = np.hstack((b0, b1, b2))
# b = np.array([5.292, 3.376, 2.823, 5.292, 3.376, 2.823, 5.292, 3.376, 2.823])
A = np.vstack((np.hstack((A0, A1, A2)), np.hstack((A2, A0, A1)), np.hstack((A1, A2, A0))))
print(A @ np.ones(len(A[0])))

# x_tls = np.array([0.6832, 1.0906, 0.8109, 1.3365, 0.9744, 1.1405])
# print(A @ np.ones(len(x_tls)))
# print(abs(A @ x_tls - b).sum())
# print(abs(A @ np.ones(len(x_tls)) - b).sum())


print(tls(A, b))
print(abs(A @ tls(A, b) - b).sum())

N = 3
F0A = A0 + A1 + A2
print("FA")
print(F0A)
omega = np.exp(2 * np.pi * 1j / N)
F1A = A0 + omega * A1 + omega ** 2 * A2
print(F1A)
F2A = A0 + omega ** 2 * A1 + omega * A2
print(F2A)

f0b = b0 + b1 + b2
print("fb")
print(f0b)
omega = np.exp(-2 * np.pi * 1j / N)
f1b = b0 + omega * b1 + omega ** 2 * b2
print(f1b)
f2b = b0 + omega ** 2 * b1 + omega * b2
print(f2b)

f0x = tls(F0A, f0b)
f1x = tls(F1A, f1b)
f2x = tls(F2A, f2b)
print("fx")
print(f0x)
print(f1x)
print(f2x)
x_stls = 1 / 3 * np.vstack((np.array([f0x + f1x + f2x]).T, np.array([f0x + omega * f1x + omega ** 2 * f2x]).T,
                            np.array([f0x + omega ** 2 * f1x + omega * f2x]).T))
print("x_stls")
print(x_stls)
x_stls = x_stls.reshape(1, -1)[0]
print(x_stls)

ls = list(np.linalg.lstsq(A, b, rcond=None)[0])
print("ls")
print(ls)

x_tls = tls(A, b)
print("x_tls")
print(x_tls)

x_stls = stls(np.array([A0, A1, A2]), np.array([b0, b1, b2]))
print("x_stls")
print(x_stls)

print("diff")
print(abs(x_tls - np.ones(len(x_tls))).sum())
print(abs(x_stls - np.ones(len(x_stls))).sum())

print(abs(x_tls[0] - 1) + abs(x_tls[1] - 1))
print(abs(x_stls[0] - 1) + abs(x_stls[1] - 1))
print(abs(x_tls[2] - 1) + abs(x_tls[3] - 1))
print(abs(x_stls[2] - 1) + abs(x_stls[3] - 1))
print(abs(x_tls[4] - 1) + abs(x_tls[5] - 1))
print(abs(x_stls[4] - 1) + abs(x_stls[5] - 1))

print(stls(np.array([A0, A1, A2]), np.array([b0, b1, b2])))

print()
print(omega ** 4, omega)
print(omega ** 5, omega ** 2)
print(omega + omega ** 2 + omega ** 3)
print(omega ** 2 + omega ** 4 + omega ** 6)

print()
print(tls(A, b))
print(stls(np.array([A0, A1, A2]), np.array([b0, b1, b2])))
print(np.linalg.pinv(A) @ b)