import numpy as np
from scipy.linalg import circulant, toeplitz

import cvxpy as cp


# 输入n个b1*b2的block，输出每个block按照toeplitz循环矩阵组成的大矩阵，维度为(n*b1)*(n*b2)
def block_toeplitz(A):
    n = len(A)
    res = []

    for i in range(n):
        row = A[i]
        for j in range(1, n):
            if j > i:
                row = np.hstack((row, A[(-i + j) % n]))
            else:
                row = np.hstack((row, A[(i - j) % n]))
        res = np.vstack((res, row)) if len(res) else row

    return res


def block_circulant(A):
    n = len(A)
    res = []

    for i in range(n):
        row = A[i]
        for j in range(1, n):
            row = np.hstack((row, A[(i - j) % n]))
        res = np.vstack((res, row)) if len(res) else row

    return res


# print(np.linalg.matrix_rank(block_circulant([block_circulant([1, 2, 3]), block_circulant([2, 3, 1])])))
# print(np.linalg.matrix_rank(block_circulant([1, 2, 3, 2, 3, 1])))


# Algorithm TLS in paper "STLS WITH BLOCK CIRCULANT MATRICES"
# refer to "AN ANALYSIS OF THE TOTAL LEAST SQUARES PROBLEM"
def tls(A, b):
    alpha = 1 / len(A)
    Ab = np.hstack((A, np.sqrt(alpha) * b.reshape(-1, 1)))
    U, S, V = np.linalg.svd(Ab)
    Ua, Sa, Va = np.linalg.svd(A)
    n = len(Sa)
    sigma_n = Sa[n - 1]
    sigma_n_plus_1 = S[n] if n < len(S) else 0
    if sigma_n > sigma_n_plus_1:
        # delta_a_and_b = (sigma_n_plus_1 * U.T[len(Sa)].reshape(-1, 1) @
        #                  np.array(np.array(V.T[len(Sa)]).T @
        #                           np.diag(np.hstack((np.ones(n), np.sqrt(alpha))))).reshape(1, -1))
        # delta_a = delta_a_and_b[:, :n]
        # delta_b = delta_a_and_b[:, n:]
        x = -1 / (np.sqrt(alpha) * V[n][n]) * V[n][:n]
        return x
    else:
        return None


def tls2(A, b):
    alpha = 1 / len(A)
    Ab = np.hstack((A, np.sqrt(alpha) * b.reshape(-1, 1)))
    U, S, V = np.linalg.svd(Ab)
    n = len(S) - 1

    x = -1 / (np.sqrt(alpha) * V[n][n]) * V[n][:n]
    return x


def dft_matrix(A):
    omega = np.exp(2 * np.pi * 1j / len(A))
    FA = []
    for j in range(len(A)):
        FAj = np.zeros(A[0].shape).astype(complex)
        for i in range(len(A)):
            FAj += omega ** (i * j) * A[i]
        FA.append(FAj)
    return np.array(FA)


def dft_vector(b):
    omega = np.exp(-2 * np.pi * 1j / len(b))
    fb = []
    for j in range(len(b)):
        fbj = np.zeros(b[0].shape).astype(complex)
        for i in range(len(b)):
            fbj += omega ** (i * j) * b[i]
        fb.append(fbj)
    return np.array(fb)


def stls(A, b):
    FA = dft_matrix(A)
    fb = dft_vector(b)
    fx = []
    for i in range(len(FA)):
        fx.append(tls(FA[i], fb[i]))
    fx = np.array(fx)
    x_stls = 1 / len(FA) * dft_vector(fx)
    return np.real(x_stls.reshape(1, -1)[0])


def stls_qp(A, b):
    FA = dft_matrix(A)
    fb = dft_vector(b)
    fx = []
    for i in range(len(FA)):
        x = cp.Variable(len(FA[i][0]))
        obj = cp.Minimize(cp.sum_squares(FA[i] @ x - fb[i]))
        prob = cp.Problem(obj)
        # prob = cp.Problem(obj, [x >= 0, x <= 3])
        prob.solve()
        fx.append([i.value for i in x])
    fx = np.array(fx)
    x_stls = 1 / len(FA) * dft_vector(fx)
    return np.real(x_stls.reshape(1, -1)[0])

# A0 = np.random.randn(3, 2)
# A1 = np.random.randn(3, 2)
# A2 = np.random.randn(3, 2)
# A3 = np.random.randn(3, 2)
# b0 = np.random.randn(3)
# b1 = np.random.randn(3)
# b2 = np.random.randn(3)
# b3 = np.random.randn(3)
# b = np.hstack((b0, b1, b2, b3))
# A = np.vstack((np.hstack((A0, A1, A2, A3)), np.hstack((A3, A0, A1, A2)), np.hstack((A2, A3, A0, A1)),
#                np.hstack((A1, A2, A3, A0))))
# print(np.allclose(A, block_circulant([A0, A1, A2, A3])))
# print(stls(np.array([A0, A1, A2, A3]), np.array([b0, b1, b2, b3])))

# keyLen = 16
# A = np.random.randn(keyLen)
# block_shape = 1
# block_number = int(keyLen / block_shape / block_shape)
# A1 = np.array(A).reshape(block_number, block_shape, block_shape)
# AC = block_circulant(A1)
# keyBin = np.random.randint(0, 4, block_number * block_shape)
#
# tmpMulA = np.dot(AC, keyBin)
# tmpMulA = tmpMulA.reshape(block_number, block_shape)
# a_list_number = tls2(AC, tmpMulA)
#
# AC1 = circulant(A)
# tmpMulA1 = np.dot(AC1, keyBin)
# a_list_number1 = tls2(AC1, tmpMulA1)
# print(a_list_number)
# print(a_list_number1)
