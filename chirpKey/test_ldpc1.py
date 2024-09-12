import time

import numpy as np
# 需要numpy的版本为1.20
from pyldpc import make_ldpc, encode, decode, get_message

n = 135   # 码长
d_v = 3  # 列重，即每列的1个数
d_c = 45  # 行重，即每行的1个数
snr = 20
seed = np.random.RandomState(42)
H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)
k = G.shape[1]  # 原始消息长度
print(k)
v1 = np.random.randint(2, size=k)  # A的原始消息
r1 = np.random.randint(2, size=k)
r2 = np.random.randint(2, size=k)
# print((v1 + r1) % 2)
# print((v1 + r2) % 2)
# print(abs((v1 + r1) % 2 - (r2 + v1) % 2).sum())

# 将噪音进行编码的方式，虽然解码时间短，但是解码错误率高，改成test_ldpc2.py的方式
b = np.round((v1 + np.random.normal(0, 0.2, k))) % 2  # B的带噪音的原始消息
print(abs(b - v1).sum())

y1 = encode(G, (v1 + r1) % 2, seed=seed)
y2 = encode(G, (b + r2) % 2, seed=seed)
d1 = decode(H, y2, snr)

decode_start1 = time.time_ns() / 10 ** 6
d2 = decode(H, y1, snr)
decode_end1 = time.time_ns() / 10 ** 6

x1 = get_message(G, d1)

decode_start2 = time.time_ns() / 10 ** 6
x2 = get_message(G, d2)
decode_end2 = time.time_ns() / 10 ** 6
# print((x1 + r1) % 2)
# print((x2 + r2) % 2)
print(abs((x1 + r1) % 2 - (r2 + x2) % 2).sum())
print(abs(x1 - x2).sum())
print("解码时间 (ms)", decode_end1 - decode_start1, decode_end2 - decode_start2)
