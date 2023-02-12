import numpy as np

from algorithm import euclidean_metric

a = [0, 10, 2]
b = [0.8, 13, 0]

a_res = []
b_res = []

for i in range(12):
    tmp = []
    for j in range(len(a)):
        tmp.append(euclidean_metric(a[j], i))
    a_res.append(list(np.argsort(tmp)))
    tmp = []
    for j in range(len(b)):
        tmp.append(euclidean_metric(b[j], i))
    b_res.append(list(np.argsort(tmp)))

a_mean = np.zeros(len(a))
b_mean = np.zeros(len(b))

for i in range(len(a_res)):
    for j in range(len(a)):
        a_mean[j] += a_res[i][j]
        b_mean[j] += b_res[i][j]

print(a_res)
print(b_res)
print(a_mean)
print(b_mean)