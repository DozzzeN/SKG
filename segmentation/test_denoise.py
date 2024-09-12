import numpy as np
from scipy.stats import pearsonr


def generate_random_matrix(size):
    """生成随机矩阵和其逆矩阵。"""
    R = np.random.randn(size, size)
    Q, _ = np.linalg.qr(R)
    return Q


def process_data_with_random_matrix(data1, data2):
    """用随机矩阵处理两组数据，并消除噪音。"""
    # 数据的维度
    size = data1.shape[1]

    # 生成随机矩阵
    R = generate_random_matrix(size)

    # 变换数据
    transformed_data1 = np.dot(data1, R)
    transformed_data2 = np.dot(data2, R)

    # QR分解反向变换
    Q1, _ = np.linalg.qr(transformed_data1)
    Q2, _ = np.linalg.qr(transformed_data2)

    # 消除噪音后重新变换回原空间
    processed_data1 = np.dot(Q1, R.T)
    processed_data2 = np.dot(Q2, R.T)

    return processed_data1, processed_data2


# 示例数据
np.random.seed(0)
data1 = np.random.randn(100, 5)
data2 = data1 + np.random.normal(0, 0.1, (100, 5))

# 处理数据
processed_data1, processed_data2 = process_data_with_random_matrix(data1, data2)
# 计算相关性
original_correlation = pearsonr(data1.reshape(1, -1)[0], data2.reshape(1, -1)[0])[0]
processed_correlation = pearsonr(processed_data1.reshape(1, -1)[0], processed_data2.reshape(1, -1)[0])[0]

# 输出相关性结果
print(f"Original Data Correlation: {original_correlation:.4f}")
print(f"Processed Data Correlation: {processed_correlation:.4f}")

# 比较处理前后数据
print("Original Data1:\n", data1[:5])
print("Processed Data1:\n", processed_data1[:5])
print("Original Data2:\n", data2[:5])
print("Processed Data2:\n", processed_data2[:5])
