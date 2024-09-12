import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
n = 500
A1 = np.random.normal(0, 1, (n, n))
x1 = np.random.randint(0, 4, n)
b1 = np.matmul(A1, x1)
A1 += np.random.normal(0, 0.1, (n, n))

solvers = [cp.SCS, cp.GUROBI, cp.OSQP, cp.CVXOPT, cp.ECOS, cp.SCIP, cp.PROXQP, cp.MOSEK,
                           cp.CLARABEL, cp.NAG, cp.XPRESS]
solver = solvers[2]
x_prime1 = cp.Variable(n)
obj = cp.Minimize(cp.sum_squares(A1 @ x_prime1 - b1))
prob = cp.Problem(obj, [x_prime1 >= 0, x_prime1 <= 3])
prob.solve(solver=solver)
x_prime1 = [i.value for i in x_prime1]

x_round1 = [round(i) for i in x_prime1]

A2 = np.random.normal(0, 1, (n, n))
x2 = np.random.randint(0, 4, n)
b2 = np.matmul(A2, x2)
A2 += np.random.normal(0, 0.1, (n, n))

solvers = [cp.SCS, cp.GUROBI, cp.OSQP, cp.CVXOPT, cp.ECOS, cp.SCIP, cp.PROXQP, cp.MOSEK,
                           cp.CLARABEL, cp.NAG, cp.XPRESS]
solver = solvers[2]
x_prime2 = cp.Variable(n)
obj = cp.Minimize(cp.sum_squares(A2 @ x_prime2 - b2))
prob = cp.Problem(obj, [x_prime2 >= 0, x_prime2 <= 3])
prob.solve(solver=solver)
x_prime2 = [i.value for i in x_prime2]

x_round2 = [round(i) for i in x_prime2]

# plt.figure()
# plt.hist(x_round1)
# plt.show()
#
# plt.figure()
# plt.hist(x_round2)
# plt.show()
#
# plt.figure()
# plt.hist([round(i / 2) for i in np.array(x_round1) + np.array(x_round2)])
# plt.show()
#
# x_sum = [round(i / 2) for i in np.array(x_prime1) + np.array(x_prime2)]
# plt.figure()
# plt.hist(x_sum)
# plt.show()
#
# x_mod = [round(i) % 4 for i in np.array(x_round1) - np.array(x_round2)]
# plt.figure()
# plt.hist(x_mod)
# plt.show()

n = 1000
a = []
b = []
for i in range(n):
    a.append(np.random.rand() * 3)
    b.append(np.random.rand() * 3)

a_round = np.round(a)
b_round = np.round(b)

plt.figure()
plt.hist(a)
plt.show()

plt.figure()
plt.hist(a_round)
plt.show()

plt.figure()
plt.hist(b_round)
plt.show()

a_mod = [round(i) % 4 for i in np.array(a_round) - np.array(b_round)]
plt.figure()
plt.hist(a_mod)
plt.show()