import numpy as np
from scipy.linalg import circulant

import EntropyHub as eh
from scipy.stats import pearsonr

cond_a = []
cond_b = []
cond_c = []

ent_a = []
ent_b = []
ent_c = []

random_a = []
random_b = []
pearson_a = []
pearson_b = []
for i in range(100):
    a = np.random.normal(0, 1, 100)
    a_matrix = circulant(a)
    cond_a.append(np.linalg.cond(a_matrix))
    ent_a.append(eh.ApEn(a)[0][1])
    # slow varying example
    b = np.random.randint(0, 3, 100) + np.random.normal(0, 0.001, 100)
    b_matrix = circulant(b)
    cond_b.append(np.linalg.cond(b_matrix))
    ent_b.append(eh.ApEn(b)[0][1])
    # random permutation example
    b = b - np.mean(b)
    c = np.random.normal(0, 1, (100, 100)) @ b
    c_matrix = circulant(c)
    cond_c.append(np.linalg.cond(c_matrix))
    ent_c.append(eh.ApEn(c)[0][1])

    slow = np.random.randint(0, 3, 100)
    fast = np.random.randint(0, 10, 100)
    random_a.append(eh.ApEn(slow)[0][1])
    random_b.append(eh.ApEn(fast)[0][1])

    pearson_a.append(pearsonr(circulant(slow)[0], circulant(slow)[1])[0])
    pearson_b.append(pearsonr(circulant(fast)[0], circulant(fast)[1])[0])

print(np.mean(cond_a), np.mean(ent_a))
print(np.mean(cond_b), np.mean(ent_b))
print(np.mean(cond_c), np.mean(ent_c))

print(np.mean(random_a), np.mean(random_b))
print(np.mean(pearson_a), np.mean(pearson_b))