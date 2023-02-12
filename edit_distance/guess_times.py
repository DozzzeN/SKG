import math

from scipy.special import comb, perm

# 计算排列数
# A = perm(3, 2)
# 计算组合数
# C = comb(45, 2)
# print(A, C)

i = 50
# 2个一组交换
print("swap", pow(2, i / 2))
# 插入
print("insert", comb(i * 2, i))
# 更新
print("update", comb(i, int(i / 2)))
# 删除
print("delete", comb(i, int(i / 2)))
# 全排列
print("permute", perm(i, i))
# 幂
print("key", pow(2.0, i))

keyLen = 128
print(comb(keyLen * 4, keyLen) * comb(keyLen * 3, keyLen) * comb(keyLen * 2, keyLen) * comb(keyLen * 1, keyLen))
print(comb(keyLen * 2, keyLen))
print(perm(keyLen, keyLen))
print(pow(2.0, keyLen))

segLen = 2
keyLen = 128
L = int(keyLen / segLen)
# 猜出两种编辑操作各自的位置
guess = 0
guessTmp = 0
for i in range(1, L + 1):
    for j in range(1, i + 1):
        for k in range(1, i - j + 1):
            guessTmp += comb(i, j) * comb(i - j, k)
    guess += guessTmp
print("guess", guess)

# 猜出两种编辑操作各自的位置，让编辑操作个数约等于密钥的一半
guess = 0
guessTmp = 0
for i in range(1, L + 1):
    for j in range(1, int(L / 2) + 1):
        for k in range(1, i - j + 1):
            guessTmp += comb(i, j) * comb(i - j, k)
    guess += guessTmp
print("guess", guess)

# 只猜相等的，让编辑操作个数约等于密钥的一半
guess = 0
guessTmp = 0
for j in range(int(L / 2) - 8, int(L / 2) + 8):
    guessTmp += comb(128, j)
guess += guessTmp
print("guess", guess)

guess = 0
guessTmp = 0
for i in range(L - 32, L + 32):
    for j in range(1, int(L / 2) + 1):
        guessTmp += comb(i, j)
    guess += guessTmp
print("guess", guess)

guess = 0
guessTmp = 0
for i in range(L - 32, L + 32):
    for j in range(1, i + 1):
        guessTmp += comb(i, j)
    guess += guessTmp
print("guess", guess)

print(pow(2.0, L))
print(perm(L, L))

#
# for i in range(1, 1000):
#     if math.log10(math.factorial(i)) < math.log10(pow(2.0, i) * pow(comb(2 * i, i), 3)):
#         print(i)
#         print(math.log10(math.factorial(i)))
#         print(math.log10(pow(2.0, i) * pow(comb(2 * i, i), 3)))
#     else:
#         print(math.factorial(i))
#         print(math.log10(pow(2.0, i) * pow(comb(2 * i, i), 3)))
#         break

def C2n_n(n):
    return math.factorial(2 * n) / pow(math.factorial(n), 2)

# print(pow(C2n_n(2), 128 / 4))
# print(pow(C2n_n(4), 128 / 8))
# print(pow(C2n_n(8), 128 / 16))
# print(pow(C2n_n(16), 128 / 32))
# print(pow(C2n_n(32), 128 / 64))
# print(C2n_n(64))
