import numpy as np
from scipy.linalg import circulant
from scipy.stats import pearsonr

N = 100
circulant_matrix1 = circulant((np.random.normal(0, 1, N)))
circulant_matrix2 = circulant_matrix1 + np.random.normal(0, 0.2, (N, N))
noise = circulant((np.random.normal(0, 1, N))) + np.random.normal(0, 0.2, (N, N))

left_decomposition = np.eye(N - 1)
decomposition = np.zeros((N, N))
decomposition[0, N - 1] = 1
decomposition[1:, 0:N - 1] = left_decomposition

decomposition += np.random.normal(0, 1, (N, N))
# decomposition = np.random.normal(0, 1, (N, N))
# print("decomposition", decomposition)

# print("circulant_matrix1", circulant_matrix1)
composition1 = circulant_matrix1[0][0] * np.eye(N)
for i in range(1, N):
    composition1 += circulant_matrix1[i][0] * np.linalg.matrix_power(decomposition, i)
    # print(np.linalg.matrix_power(decomposition, i))
# print("composition1", composition1)

# print("circulant_matrix2", circulant_matrix2)
composition2 = circulant_matrix2[0][0] * np.eye(N)
for i in range(1, N):
    composition2 += circulant_matrix2[i][0] * np.linalg.matrix_power(decomposition, i)
    # print(np.linalg.matrix_power(decomposition, i))
# print("composition2", composition2)

noise_comp = noise[0][0] * np.eye(N)
for i in range(1, N):
    noise_comp += noise[i][0] * np.linalg.matrix_power(decomposition, i)
    # print(np.linalg.matrix_power(decomposition, i))

print(pearsonr(circulant_matrix1[0], circulant_matrix2[0])[0])
print(pearsonr(composition1[0], composition2[0])[0])
print(pearsonr(noise_comp[0], composition1[0])[0])
