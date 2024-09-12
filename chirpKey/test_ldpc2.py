import time

import numpy as np
# 需要numpy的版本为1.20
from pyldpc import make_ldpc, encode, decode, get_message

n = 135  # 码长
d_v = 3  # 列重，即每列的1个数
d_c = 45  # 行重，即每行的1个数
snr = 20
seed = np.random.RandomState(42)
H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)
k = G.shape[1]
print(k)
v1 = np.random.randint(2, size=k)

b = np.round((v1 + np.random.normal(0, 1, k))) % 2
print(abs(b - v1).sum())

r1 = np.random.randint(2, size=n)
r2 = np.random.randint(2, size=n)

y1 = (encode(G, v1, seed=seed) + r1) % 2
y2 = (encode(G, b, seed=seed) + r2) % 2

decode_start1 = time.time_ns() / 10 ** 6
d2 = (decode(H, y1, snr) + r2) % 2
decode_end1 = time.time_ns() / 10 ** 6

d1 = (decode(H, y2, snr) + r1) % 2
x1 = get_message(G, d1)

decode_start2 = time.time_ns() / 10 ** 6
x2 = get_message(G, d2)
decode_end2 = time.time_ns() / 10 ** 6
print(abs(x1 - x2).sum())
print("解码时间 (ms)", decode_end1 - decode_start1, decode_end2 - decode_start2)
