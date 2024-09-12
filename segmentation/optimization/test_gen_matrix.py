import numpy as np
from matplotlib import pyplot as plt


def generate_matrix_and_solve_last(vector, target):
    # 根据向量vector生成矩阵A_full，使得vector @ A_full = target
    # 同时A_full的前n-1行是随机生成的，最后一行是根据vector和target计算得到的
    n = len(vector)

    np.random.seed(int(np.mean(vector)) + 10000)
    A = np.random.normal(0, 10, (n - 1, n))

    A_extended = np.vstack([A, np.zeros(n)])

    result = vector @ A_extended

    offset = target - result

    vector_last = offset / vector[-1]

    A_full = np.vstack([A, vector_last])

    return A_full


def generate_matrix_and_solve_diag(vector, target):
    # 根据向量vector生成矩阵A_full，使得vector @ A_full = target
    # 同时A_full的前n-1行是随机生成的，最后一行是根据vector和target计算得到的
    n = len(vector)

    # Generate a random n x n matrix
    np.random.seed(int(np.mean(vector)) + 10000)
    A = np.random.normal(0, 1, (n, n))

    # Ensure the target vector has length n
    if len(target) < n:
        target = np.append(target, [1] * (n - len(target)))

    # Copy the matrix to keep original random values
    A_full = np.copy(A)

    # Compute the diagonal elements
    for i in range(n):
        A_full[i, i] = (target[i] - np.dot(vector, A_full[:, i]) + vector[i] * A_full[i, i]) / vector[i]

    return A_full

# Example usage
# n = 10
# vector = np.random.rand(n)  # Example 1*10 vector
# target = np.array([0] * (n // 2) + [1] * (n // 2))
# target = np.random.permutation(target)
# matrix = generate_matrix_and_solve_last(vector, target)
# # plt.figure()
# # plt.imshow(matrix, cmap='hot', interpolation='nearest')
# # plt.colorbar()
# # plt.show()
# print(vector @ matrix)
#
# matrix = generate_matrix_and_solve_diag(vector, target)
# # plt.figure()
# # plt.imshow(matrix, cmap='hot', interpolation='nearest')
# # plt.colorbar()
# # plt.show()
# print(vector @ matrix)
