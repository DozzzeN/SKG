import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb, perm

from alignment import genAlignInsDel, alignFloatInsDel

# for np in range(104, 153):
#     guess = 0
#     for ni in range(22, 44):
#         guess += comb(np, ni) * comb(np, 64 - ni)
#     print(guess, np)

# print("pow", pow(2.0, 128))
# print("pow", pow(2.0, 104))
# print("pow", pow(2.0, 152))

# for np in range(104, 153):
#     guess = 0
#     for ni in range(86, 114):
#         guess += comb(np, ni)
#     print(guess, np)

guess = 0
for le in range(0, 128 + 1):
    guess += comb(128, le)
print(guess)
print(pow(2.0, 128))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
a = [1, 2, 3, 4, 5]
ap = [1, 2, 7, 8, 5, 6]
rule = alignFloatInsDel({'=': 0, '+': 1, '-': 1}, ap, a, 0.1)
key = genAlignInsDel(rule)
print(rule)
print(key)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

# 测试li的范围和加密强度的关系
res = []
maxGuess = 0
maxLi = 0
halfKeyLen = 64
for li in range(0, halfKeyLen + 1):
    ld = halfKeyLen - li
    guess = comb(halfKeyLen * 2 + li - ld, li) * comb(halfKeyLen * 2 + li - ld, ld)
    maxGuess = max(maxGuess, guess)
    if maxGuess == guess:
        maxLi = li
    # print(guess, li)
    res.append(guess)
print("最大的攻击强度为 ", maxGuess, maxLi)
# plt.figure()
# plt.scatter(range(0, halfKeyLen + 1), res, color="red")
# plt.figure()
# plt.scatter(range(0, halfKeyLen + 1), numpy.log(res), color="blue")
# plt.show()
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

# 测试li的范围和加密强度的关系
res = []
maxGuess = 0
maxLi = 0
halfKeyLen = 64
for li in range(0, halfKeyLen + 1):
    ld = halfKeyLen - li
    guess = comb(halfKeyLen * 2 + li, li) * comb(halfKeyLen * 2, ld)
    maxGuess = max(maxGuess, guess)
    if maxGuess == guess:
        maxLi = li
    # print(guess, li)
    res.append(guess)
print("最大的攻击强度为 ", maxGuess, maxLi)
# plt.figure()
# plt.scatter(range(0, halfKeyLen + 1), res, color="red")
# plt.figure()
# plt.scatter(range(0, halfKeyLen + 1), numpy.log(res), color="blue")
# plt.show()
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
guess = 0
for li in range(0, halfKeyLen + 1):
    guess += comb(halfKeyLen * 2, li) * comb(halfKeyLen * 2, halfKeyLen - li)
print("从N中选择的次数", guess)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
guess = 0
for li in range(0, halfKeyLen + 1):
    guess += comb(halfKeyLen + 2 * li, li) * comb(halfKeyLen + 2 * li, halfKeyLen - li)
print("从Np中选择的次数", guess)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# 从rule中找出li的插入的，再从剩下的找出ld个删除的
guess = 0
for li in range(0, halfKeyLen + 1):
    guess += comb(halfKeyLen * 2 + li, li) * comb(halfKeyLen * 2, halfKeyLen - li)
print("理论上的次数", guess)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# 从rule中找出le个相等的
guess = 0
for li in range(0, halfKeyLen + 1):
    guess += comb(halfKeyLen * 2 + li, halfKeyLen * 2 - (halfKeyLen - li))
print("从Np中找相等的次数", guess)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# 从rule中找出li个插入的，再找出ld个删除的
guess = 0
maxGuess = 0
maxLi = 0
maxLd = 0
points = np.zeros([halfKeyLen * 2 + 1, halfKeyLen * 2 + 1])
for li in range(0, halfKeyLen * 2 + 1):
    for ld in range(0, halfKeyLen * 2 + 1):
        tmp = comb(halfKeyLen * 2 + li, li) * comb(halfKeyLen * 2, ld)
        guess += tmp
        points[li][ld] = math.log10(tmp)
        maxGuess = max(maxGuess, tmp)
        if maxGuess == tmp:
            maxLi = li
            maxLd = ld
print("分别计算插入和删除的次数", guess)
print("最大的攻击强度下的插入和删除次数", maxLi, maxLd)
x = np.arange(0, halfKeyLen * 2 + 1, 1)
y = np.arange(0, halfKeyLen * 2 + 1, 1)
plt.close()
plt.figure()
X, Y = np.meshgrid(x, y)
plt.axes(projection='3d').plot_surface(X, Y, points, cmap='rainbow')
plt.show()

# 从rule中找出li个插入的，再找出le个相等的
guess = 0
maxGuess = 0
maxLi = 0
maxLd = 0
# for li in range(0, halfKeyLen * 2 + 1):
#     for ld in range(0, halfKeyLen * 2 + 1):
#         tmp = comb(halfKeyLen * 2 + li, li) * comb(halfKeyLen * 2, halfKeyLen * 2 - ld)
#         guess += tmp
#         maxGuess = max(maxGuess, tmp)
#         if maxGuess == tmp:
#             maxLi = li
#             maxLd = ld
for li in range(halfKeyLen - int(halfKeyLen / 2), halfKeyLen + int(halfKeyLen / 2) + 1):
    for ld in range(halfKeyLen - int(halfKeyLen / 2), halfKeyLen + int(halfKeyLen / 2) + 1):
        tmp = comb(halfKeyLen * 2 + li, li) * comb(halfKeyLen * 2, halfKeyLen * 2 - ld)
        guess += tmp
        maxGuess = max(maxGuess, tmp)
        if maxGuess == tmp:
            maxLi = li
            maxLd = ld
print("分别计算插入和删除的次数", guess)
print("最大的攻击强度下的插入和删除次数", maxLi, maxLd)

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
guess = 0
maxGuess = 0
maxNa = 0
for na in range(0, halfKeyLen * 2 + 1):  # 假设np=3n
    tmp = comb(halfKeyLen * 2, na) * comb(halfKeyLen * 3, na)
    guess += tmp
    maxGuess = max(maxGuess, tmp)
    if maxGuess == tmp:
        maxNa = na
print("猜测次数", guess)
print("最大的攻击强度下的插入和删除次数", maxNa)
# x = np.arange(0, halfKeyLen * 2 + 1, 1)
# y = np.arange(0, halfKeyLen * 2 + 1, 1)
# plt.close()
# plt.figure()
# X, Y = np.meshgrid(x, y)
# plt.axes(projection='3d').plot_surface(X, Y, points, cmap='rainbow')
# plt.show()
print("2^128", pow(2.0, halfKeyLen * 2))
print("128!", perm(halfKeyLen * 2, halfKeyLen * 2))

points = np.zeros([halfKeyLen * 2 + 1, halfKeyLen * 4 + 1])
for ld in range(0, halfKeyLen * 2 + 1):
    for np in range(0, halfKeyLen * 4 + 1):
        points[ld][np] = comb(halfKeyLen * 2, ld) * comb(np, halfKeyLen * 2 - ld)
plt.close()
plt.figure()
X, Y = np.meshgrid(np.arange(0, halfKeyLen * 2 + 1, 1), np.arange(0, halfKeyLen * 4 + 1, 1))
plt.axes(projection='3d').plot_surface(X, Y, points, cmap='rainbow')
plt.show()