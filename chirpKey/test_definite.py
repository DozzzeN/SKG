import numpy as np
from scipy.linalg import circulant, toeplitz, solve_toeplitz
from scipy.signal import medfilt

def normalize(data):
    return (np.array(data) - np.min(data)) / (np.max(data) - np.min(data))

# a = toeplitz(np.random.normal(0, 1, 5))
# print(np.linalg.svd(a)[1])
# print(np.linalg.eigvals(a))
# # a1 = np.linalg.eig(a)[1]
# # a2 = np.linalg.eig(a)[1].T
# # print(a1)
# # print(np.linalg.cond(a), np.linalg.cond(np.diag(np.linalg.eigvals(a))))
# U, S, Vt = np.linalg.svd(a)
# S = medfilt(S, kernel_size=3)
# D = np.diag(np.convolve(S, np.ones((len(S),)) / len(S), mode="same"))
# D = np.sqrt(D)
# a = U @ D @ Vt
# print(a)
# print(U @ Vt)
for i in range(100):
    a = circulant(np.random.uniform(1, 10, 10))
    print(np.linalg.matrix_rank(a) == len(a))
    # a = a @ a.T
    # a = np.linalg.eig(a)[1] @ np.diag(np.abs(np.linalg.eig(a)[0])) @ np.linalg.eig(a)[1].T
    # a = np.linalg.eig(a)[1] @ np.diag(normalize(np.linalg.eig(a)[0])) @ np.linalg.eig(a)[1].T
    # print(np.linalg.eig(a)[0])
    # print(np.abs(np.linalg.eig(a)[0]))

    print(np.linalg.cond(a), np.min(np.linalg.eigvals(a)))
    U, S, Vt = np.linalg.svd(a)
    S = medfilt(S, kernel_size=7)
    S = np.convolve(S, np.ones((len(S),)) / len(S), mode="same")
    S = np.sqrt(S)
    D = np.diag(S)
    a = U @ D @ Vt
    # a = a + a.T
    a = a + 10 * np.eye(len(a))
    print(np.linalg.cond(a), np.min(np.linalg.eigvals(a)))

    # a = np.linalg.eig(a)[1] @ np.diag(np.abs(np.linalg.eig(a)[0])) @ np.linalg.eig(a)[1].T
    # print(np.linalg.cond(a), np.min(np.linalg.eigvals(a)))
    print()
    # print(np.linalg.eig(a)[0])
    np.linalg.cholesky(a)
