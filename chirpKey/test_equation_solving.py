import numpy as np
import scipy
import sympy as sp


def block_circulant(A):
    n = len(A)
    res = []

    for i in range(n):
        row = A[i]
        for j in range(1, n):
            row = np.hstack((row, A[(i - j) % n]))
        res = np.vstack((res, row)) if len(res) else row

    return res


# c = block_circulant([block_circulant([1, 2, 3]), block_circulant([0, 0, 0])])
# # c = block_circulant([1, 2, 1, 2, 0])
# print(c)
# print(np.linalg.matrix_rank(c))
# exit()

# 即使有8个方程，8个未知数，还是解不出来
# 定义符号
x1, x2, x3, x4, a1, a2, b1, b2 = sp.symbols('x1, x2, x3, x4, a1, a2, b1, b2')

# 方程组
equations = [
    sp.Eq(a1 * x1 + a2 * x2, 2.5),
    sp.Eq(a2 * x1 + a1 * x2, 1.2),
    sp.Eq(a1 * x3 + a2 * x4, 3.7),
    sp.Eq(a2 * x3 + a1 * x4, 3.7),
    sp.Eq(b1 * x1 + b2 * x2, 2.1),
    sp.Eq(b2 * x1 + b1 * x2, 1.8),
    sp.Eq(b1 * x3 + b2 * x4, 3.9),
    sp.Eq(b2 * x3 + b1 * x4, 3.9)
]

# 解方程组
solution = sp.solve(equations, (x1, x2, x3, x4, a1, a2, b1, b2))

# 打印解
print(solution)
