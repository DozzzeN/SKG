import numpy as np
import cvxpy as cp


def maximize_distance_sum_cvxpy(A):
    solvers = [cp.SCS, cp.GUROBI, cp.OSQP, cp.CVXOPT, cp.ECOS, cp.SCIP, cp.PROXQP, cp.MOSEK,
               cp.CLARABEL, cp.NAG, cp.XPRESS]
    n = len(A)
    A = np.reshape(A, (-1, 1))  # Ensure A is a column vector

    K = cp.Variable((n, n))

    B = A.T @ K

    obj = cp.Maximize(cp.sum_squares(B))

    prob = cp.Problem(obj, [B >= 0, B <= 1])

    prob.solve(solver=solvers[4])

    K_opt = K.value

    return K_opt


# Problem does not follow DCP rules
# 无法进行优化
A = np.random.normal(0, 1, (16, 1))
K_opt = maximize_distance_sum_cvxpy(A)
print("Optimal K:")
print(K_opt)
