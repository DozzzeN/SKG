# perturbed least squares problems
import numpy as np
import pulp

m = 15
n = 10
np.random.seed(0)
A = np.random.normal(0, 1, (m, n))
np.random.seed(1)
delta_A = np.random.normal(0, 0.25, (m, n))
np.random.seed(2)
x = np.random.normal(0, 1, n)
b = A @ x
delta_b = delta_A @ x
print(np.allclose(b + delta_b, (A + delta_A) @ x))

print(1 / max(np.linalg.svd(np.linalg.pinv(A))[1]))
print(max(np.linalg.svd(delta_A)[1]))

for alpha in range(0, 10):
    # Tichonov regularization
    print()
    print(alpha)
    alpha /= 10
    ATA = A.T @ A
    A_wave = np.vstack((A, np.sqrt(alpha) * np.eye(n)))
    print(np.allclose(np.linalg.svd(ATA)[1], np.linalg.svd(A)[1] ** 2))
    print(np.allclose(np.linalg.svd(A_wave.T @ A_wave)[1], np.linalg.svd(A)[1] ** 2 + alpha))
    U, S, V, = np.linalg.svd(A)
    V = V.T
    S = np.diag(S)
    beta = (U.T @ b)[:n]
    sol = V @ np.linalg.inv(S.T @ S + alpha * np.eye(n)) @ S.T @ beta
    print(np.sum(np.abs(x - sol)))
    print(np.sum(np.abs(b - (A + delta_A) @ sol)))

    # pulp
    problem = pulp.LpProblem("Perturbed_Least_Squares", pulp.LpMinimize)
    sol2 = [pulp.LpVariable(f"x_{i}") for i in range(n)]
    problem += pulp.lpSum(x)
    for i in range(len(A)):
        constraint = pulp.lpSum(A[i][j] * x[j] for j in range(n)) == b[i]
        problem += constraint
    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    sol2 = np.array([round(pulp.value(x[i])) for i in range(n)])
    print("pulp")
    print(np.sum(np.abs(x - sol2)))
    print(np.sum(np.abs(b - (A + delta_A) @ sol2)))

    # regularization by truncation
    print("trunc")
    r = 2
    U, S, V, = np.linalg.svd(A)
    V = V.T
    for i in range(len(S)):
        if S[i] < r:
            S[i] = 0
    S = np.diag(S)
    # Sr = np.vstack((S, np.zeros((m - n, n))))
    print(alpha)
    beta = (U.T @ b)[:n]
    sol3 = V @ np.linalg.pinv(S.T @ S + alpha * np.eye(n)) @ S.T @ beta
    # Ar = U @ Sr @ V.T
    # sol3 = np.linalg.pinv(Ar) @ b
    print(np.sum(np.abs(x - sol3)))
    print(np.sum(np.abs(b - (A + delta_A) @ sol3)))

