import numpy as np
from scipy.linalg import circulant
from scipy.stats import pearsonr, spearmanr

N = 10
p_before = []
p_after = []

var_before = []
var_after = []

for i in range(10000):
    a = np.random.normal(0, 1, N)
    var_before.append(np.var(a))
    a = (a - np.mean(a)) / np.std(a)
    b = a + np.random.normal(0, 0.6, N)
    b = (b - np.mean(b)) / np.std(b)

    n = np.random.normal(0, 1, (N, N))
    n = (n - np.mean(n)) / np.std(n)

    an = a @ n
    bn = b @ n
    
    var_after.append(np.var(an))

    # an = (an - np.mean(an)) / np.std(an)
    # bn = (bn - np.mean(bn)) / np.std(bn)
    # a_con = np.array(circulant(a)).reshape(-1)
    # b_con = np.array(circulant(b)).reshape(-1)

    # print(a_con)

    p_before.append(pearsonr(a, b)[0])
    p_after.append(pearsonr(an, bn)[0])
    # print(pearsonr(a, b)[0], pearsonr(an, bn)[0])
    # print(np.corrcoef(a, b) / (np.corrcoef(a, a) * np.corrcoef(b, b)) ** 0.5)
    # print(np.corrcoef(an, bn) / (np.corrcoef(an, an) * np.corrcoef(bn, bn)) ** 0.5)

print(np.mean(p_before), np.mean(p_after))
print(np.mean(var_before), np.mean(var_after))