import numpy as np


def bnewt(A):
    # X0: initial guess. TOL: error tolerance.
    # delta/Delta: how close/far balancing vectors can get
    # to/from the edge of the positive cone.
    # We use a relative measure on the size of elements.
    # FL: intermediate convergence statistics on/off.
    # RES: residual error, measured by norm(diag(x)*A*x - e).
    n = A.shape[0]
    e = np.ones((n, 1))
    tol = 1e-6
    x0 = e
    delta = 0.1
    Delta = 3
    fl = 0

    g = 0.9
    etamax = 0.1
    eta = etamax
    stop_tol = tol * .5
    x = x0
    rt = tol ** 2
    v = A @ x
    rk = 1 - v
    rho_km1 = (rk.T @ rk)[0][0]
    rout = rho_km1
    rold = rout
    MVP = 0
    i = 0

    if fl == 1:
        print('it in. it res\n')

    while rout > rt:
        i = i + 1
        k = 0
        y = e
        innertol = max([eta ** 2 * rout, rt])

        while rho_km1 > innertol:
            k = k + 1

            if k == 1:
                Z = rk / v
                p = Z
                rho_km1 = (rk.T @ Z)[0][0]
            else:
                beta = rho_km1 / rho_km2
                p = Z + beta * p
            w = x * (A @ (x * p)) + v * p
            alpha = rho_km1 / (p.T @ w)[0][0]
            ap = alpha * p

            ynew = y + ap
            if min(ynew) <= delta:
                if delta == 0:
                    break
                ind = np.where(ap < 0)[0]
                gamma = min((delta - y[ind]) / ap[ind])
                y = y + gamma * ap
                break

            if max(ynew) >= Delta:
                ind = np.where(ynew > Delta)[0]
                gamma = min((Delta - y[ind]) / ap[ind])
                y = y + gamma * ap
                break

            y = ynew
            rk = rk - alpha * w
            rho_km2 = rho_km1
            Z = rk / v
            rho_km1 = (rk.T @ Z)[0][0]

        x = x * y
        v = x * (A @ x)
        rk = 1 - v
        rho_km1 = (rk.T @ rk)[0][0]
        rout = rho_km1
        MVP = MVP + k + 1

        rat = rout / rold
        rold = rout
        res_norm = np.sqrt(rout)
        eta_o = eta
        eta = g * rat
        if (g * eta_o) ** 2 > 0.1:
            eta = max([eta, (g * eta_o) ** 2])

        eta = max([min([eta, etamax]), stop_tol / res_norm])

    return x


# A = np.array([[3.79026173675223, 1.52356681656773, 2.97018832455993],
#               [1.52356681656773, 1.69370753046105, 1.28751401741427],
#               [2.97018832455993, 1.28751401741427, 2.61403299955598]])
A = np.random.normal(0, 1, (8, 8))
A = abs(A @ A)
# 需要输入是对称非负矩阵
# print(A)
x = bnewt(A)
B = np.diag(x.reshape(-1)) @ A @ np.diag(x.reshape(-1))

# print(B)
# 只是矩阵平衡，新矩阵B的行列范数近似1，但是条件数很大
print(np.linalg.cond(A))
print(np.linalg.cond(B))
print(max(np.linalg.svd(B)[1]) / min(np.linalg.svd(B)[1]))
print(np.linalg.svd(B)[1])

# 标准化的影响
B = (B - np.min(B)) / (np.max(B) - np.min(B))
print(np.linalg.cond(B))
print(max(np.linalg.svd(B)[1]) / min(np.linalg.svd(B)[1]))
print(np.linalg.svd(B)[1])
