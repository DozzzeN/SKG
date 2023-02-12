import matplotlib.pyplot as plt

from algorithm import pearson_metric, dtw_metric

a = [1, 0, 2, 3]
br = [2, 2, 3, 4]
b = [4, 2, 1, 3, 0, 6, 7, 5]
ar = [1, 1, 0, 0, 2, 2, 3, 3]
plt.figure()
plt.plot(a, "b")
plt.plot(br, "k")
plt.plot(ar, 'r')
plt.plot(b, "y")
# plt.show()

print(pearson_metric(a, br))
print(pearson_metric(b, ar))
print(dtw_metric(a, b) / ((len(a) + len(b)) / 2))
print(dtw_metric(a, br) / ((len(a) + len(br)) / 2))
print(dtw_metric(b, ar) / ((len(ar) + len(b)) / 2))
