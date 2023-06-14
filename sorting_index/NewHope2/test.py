import matplotlib.pyplot as plt
import numpy as np
import newhope
import csv
import poly

csv1 = open("a.csv", "r")
reader1 = csv.reader(csv1)
a = []
for item in reader1:
    a.append(int(item[0]))

csv2 = open("a_prime.csv", "r")
reader2 = csv.reader(csv2)
a_prime = []
for item in reader2:
    a_prime.append(int(item[0]))

a = np.random.randint(0, 1024, 1024)
a_prime = a + np.random.randint(0, 2, 1024)

a = np.array(a, dtype=np.int64) * 117
a_prime = np.array(a_prime, dtype=np.int64) * 117

# np.random.seed(0)
# ro = np.random.permutation(1024)
# a = a[ro]
# a_prime = a_prime[ro]

s1 = newhope.get_noise()
e = newhope.get_noise()
b = poly.add(e, poly.pointwise(s1, a))

s2 = newhope.get_noise()
e_prime = newhope.get_noise()
e_prime_prime = poly.get_noise()
u = poly.add(poly.pointwise(a_prime, s2), e_prime)
v = poly.add(poly.pointwise(b, s2), e_prime_prime)

# np.random.seed(0)
# ro = np.random.permutation(len(v))
# v = np.array(v)[ro]
r = poly.helprec(v)
b_list_number = poly.rec(v, r)

v_prime = poly.pointwise(s1, u)
# v_prime = np.array(v_prime)[ro]
# print(r)
# r = np.random.randint(0, 4, 1024)
a_list_number = poly.rec(v_prime, r)

# r可以分段发送，key依次分段产生
r_short = np.ones(1024, dtype=np.int64)
v_prime_short = np.ones(1024, dtype=np.int64)
step = 8
mul = 256
off = 0
# r_short[0:8] = r[0:8]
# r_short[256:264] = r[256:264]
# r_short[512:520] = r[512:520]
# r_short[768:776] = r[768:776]
r_short[0 * mul + off * step:0 * mul + step + off * step] = r[0 * mul + off * step:0 * mul + step + off * step]
r_short[1 * mul + off * step:1 * mul + step + off * step] = r[1 * mul + off * step:1 * mul + step + off * step]
r_short[2 * mul + off * step:2 * mul + step + off * step] = r[2 * mul + off * step:2 * mul + step + off * step]
r_short[3 * mul + off * step:3 * mul + step + off * step] = r[3 * mul + off * step:3 * mul + step + off * step]
v_prime_short[0 * mul + off * step:0 * mul + step + off * step] = v_prime[0 * mul + off * step:0 * mul + step + off * step]
v_prime_short[1 * mul + off * step:1 * mul + step + off * step] = v_prime[1 * mul + off * step:1 * mul + step + off * step]
v_prime_short[2 * mul + off * step:2 * mul + step + off * step] = v_prime[2 * mul + off * step:2 * mul + step + off * step]
v_prime_short[3 * mul + off * step:3 * mul + step + off * step] = v_prime[3 * mul + off * step:3 * mul + step + off * step]
a_list_number1 = poly.recKeyBit(v_prime_short, r_short, off * step)

perm = np.random.permutation(1024)
# vp = np.array(v)[perm]
# e1_list_number = poly.rec(vp, poly.get_noise())

e1_v = poly.add(poly.pointwise(a[perm], newhope.get_noise()), newhope.get_noise())
e1_list_number = poly.rec(e1_v, r)
# e1_list_number = poly.rec(np.random.randint(0, 1024, 1024, dtype=np.int64), poly.get_noise())
print(a_list_number[0], a_list_number1)

# plt.figure()
# # plt.plot(a)
# # plt.plot(a_prime)
# plt.plot(a_list_number)
# plt.show()

print(a == a_prime)
print(a_list_number == b_list_number)

a_list = []
b_list = []
e1_list = []
# 转成二进制，0填充成0000
for i in range(len(a_list_number)):
    number = bin(a_list_number[i])[2:].zfill(8)
    a_list += number
for i in range(len(b_list_number)):
    number = bin(b_list_number[i])[2:].zfill(8)
    b_list += number
for i in range(len(e1_list_number)):
    number = bin(e1_list_number[i])[2:].zfill(8)
    e1_list += number

sum1 = min(len(a_list), len(b_list))
sum2 = 0
sum31 = 0
for i in range(0, sum1):
    sum2 += (a_list[i] == b_list[i])
for i in range(min(len(a_list), len(e1_list))):
    sum31 += (a_list[i] == e1_list[i])

print(sum2 / sum1)
print(sum31 / sum1)