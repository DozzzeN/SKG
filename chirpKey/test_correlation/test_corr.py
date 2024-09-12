import numpy as np
from scipy.linalg import circulant
from scipy.stats import pearsonr, spearmanr

a = [1.5, 1.3, 1.6, 1.7]
b = [1.4, 1.3, 1.8, 1.8]

a_con = np.array(circulant(a)).reshape(-1)
b_con = np.array(circulant(b)).reshape(-1)

print(a_con)

print(pearsonr(a, b)[0])
print(pearsonr(a_con, b_con)[0])

print(spearmanr(a, b)[0])
print(spearmanr(a_con, b_con)[0])