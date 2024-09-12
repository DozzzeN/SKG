import numpy as np
from scipy.linalg import circulant
from scipy.stats import pearsonr, spearmanr

N = 10
p_before = []
p_after = []

for i in range(10000):
    a = np.random.normal(0, 1, N)
    a = (a - np.mean(a)) / np.std(a)
    b = a + np.random.normal(0, 0.6, N)
    b = (b - np.mean(b)) / np.std(b)

    key = np.random.normal(0, 1, N)
    key = (key - np.mean(key)) / np.std(key)

    a_con = np.array(circulant(a))
    b_con = np.array(circulant(b))

    an = key @ a_con
    bn = key @ b_con

    # print(a_con)

    p_before.append(pearsonr(a, b)[0])
    p_after.append(pearsonr(an, bn)[0])
    # print(pearsonr(a, b)[0], pearsonr(an, bn)[0])
    # print(np.corrcoef(a, b) / (np.corrcoef(a, a) * np.corrcoef(b, b)) ** 0.5)
    # print(np.corrcoef(an, bn) / (np.corrcoef(an, an) * np.corrcoef(bn, bn)) ** 0.5)

print(np.mean(p_before), np.mean(p_after))