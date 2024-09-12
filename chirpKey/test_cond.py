import numpy as np

def normalize(data):
    if np.max(data) == np.min(data):
        return (data - np.min(data)) / np.max(data)
    else:
        return (np.array(data) - np.min(data)) / (np.max(data) - np.min(data))

t = 10000
conds = []
for i in range(t):
    A = np.random.normal(0, 1, (100, 100))
    A = normalize(A)
    conds.append(np.linalg.cond(A))
print(np.mean(conds))

t = 10000
conds = []
for i in range(t):
    A = np.random.normal(1, 1, (100, 100))
    A = normalize(A)
    conds.append(np.linalg.cond(A))
print(np.mean(conds))

t = 10000
conds = []
for i in range(t):
    A = np.random.normal(1, 2, (100, 100))
    A = normalize(A)
    conds.append(np.linalg.cond(A))
print(np.mean(conds))