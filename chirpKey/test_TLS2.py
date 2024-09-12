import numpy as np
from scipy.linalg import circulant

from tls import tls, stls, stls_qp
import cvxpy as cp

n = 2
m = 3
x = np.random.randint(0, 4, n * m)
A0 = np.random.normal(0, 1, (m, n))
A1 = np.random.normal(0, 1, (m, n))
A2 = np.random.normal(0, 1, (m, n))
A = np.vstack((np.hstack((A0, A1, A2)), np.hstack((A2, A0, A1)), np.hstack((A1, A2, A0))))
b = A @ x
perturb = 0.1
A0p = A0 + np.random.normal(0, perturb, (m, n))
A1p = A1 + np.random.normal(0, perturb, (m, n))
A2p = A2 + np.random.normal(0, perturb, (m, n))
Ap = np.vstack((np.hstack((A0p, A1p, A2p)), np.hstack((A2p, A0p, A1p)), np.hstack((A1p, A2p, A0p))))
b0p = b[:3] + np.random.normal(0, perturb, m)
b1p = b[3:6] + np.random.normal(0, perturb, m)
b2p = b[6:] + np.random.normal(0, perturb, m)
bp = np.hstack((b0p, b1p, b2p))

x_tls = tls(Ap, bp)
x_stls = stls(np.array([A0p, A1p, A2p]), np.array([b0p, b1p, b2p]))
x_inv = np.linalg.pinv(Ap) @ bp
x_stls_qp = stls_qp(np.array([A0p, A1p, A2p]), np.array([b0p, b1p, b2p]))

c = cp.Variable(len(Ap[0]))
obj = cp.Minimize(cp.sum_squares(Ap @ c - bp))
prob = cp.Problem(obj)
# prob = cp.Problem(obj, [x >= 0, x <= 3])
prob.solve()
x_qp = [i.value for i in c]

print(x)
print(x_inv)
print(x_stls)
print(x_tls)
print(x_stls_qp)
print(x_qp)

print(abs(x - x_tls).sum())
print(abs(x - x_stls).sum())
print(abs(x - x_inv).sum())
print(abs(x - x_stls_qp).sum())
print(abs(x - x_qp).sum())
