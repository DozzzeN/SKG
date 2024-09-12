import numpy as np


def characteristic_polynomial(matrix):
    # 计算矩阵的特征值
    eigenvalues, _ = np.linalg.eig(matrix)

    # 构建特征多项式的系数
    coefficients = [1]
    for eigenvalue in eigenvalues:
        coefficients = np.convolve(coefficients, [1, -eigenvalue])

    # 返回特征多项式的系数
    return coefficients


def evaluate_characteristic_polynomial(coefficients, eigenvalue):
    # 使用np.polyval计算特征多项式在给定特征值上的值
    result = np.polyval(coefficients, eigenvalue)
    return result


# 例子
matrix = [[1, 2, 1, 3],
          [3, 1, 2, 1],
          [1, 3, 1, 2],
          [2, 1, 3, 1]]

poly_coefficients = characteristic_polynomial(matrix)
print("特征多项式的系数：", poly_coefficients)

# 例子
eigenvalue = -3  # 你想要评估的特征值

result = evaluate_characteristic_polynomial(poly_coefficients, eigenvalue)
print(f"特征多项式在 λ={eigenvalue} 上的值：{result}")

def eigen_analysis(matrix):
    # 计算矩阵的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return np.round(eigenvalues), np.round(eigenvectors)

# 例子
eigenvalues, eigenvectors = eigen_analysis(matrix)

print("特征值：", eigenvalues)
print("特征向量：", eigenvectors)