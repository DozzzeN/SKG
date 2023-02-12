# 只改变固定游程的0和1，自相关不变
# x1 = [0, 1, 1, 0, 0, 0, 1]
# x2 = [0, 1, 0, 0, 0, 1, 1]
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import rel_entr
from scipy.stats import pearsonr, entropy

x1 = [1, 2, 3, 4]
x2 = [2, 3, 4, 1]
x3 = [3, 4, 1, 2]
x4 = [4, 1, 2, 3]
x0 = [2, 3, 4, 5]

plt.figure()
plt.plot(x1, x2)
plt.plot(x1, x0)
plt.show()

# print(sum(rel_entr(x1, x2)))
# print(sum(rel_entr(x1, x3)))
# print(sum(rel_entr(x2, x3)))

# print(pearsonr(x1, x1)[0])
print(pearsonr(x1, x2)[0])
print(pearsonr(x1, x3)[0])
# print(pearsonr(x1, x4)[0])
print(pearsonr(x1, x0)[0])

#
# y1 = [0, 1, 1, 0, 0, 0, 1]
# y2 = [0, 1, 0, 0, 0, 1, 1]
# y3 = [0, 1, 1, 1, 0, 0, 0]
#
# print(pearsonr(y1, y2)[0])
# print(pearsonr(y1, y3)[0])

# print(np.corrcoef(y1[0:len(y1) - 1], y1[1:])[0][1])
# print(np.corrcoef(y2[0:len(y2) - 1], y2[1:])[0][1])
# print(np.corrcoef(y3[0:len(y3) - 1], y3[1:])[0][1])

