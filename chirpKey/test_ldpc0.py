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
k = G.shape[1]  # 原始消息长度
print("消息长度", k)
v1 = np.random.randint(2, size=k)  # 原始消息

b = np.round((v1 + np.random.normal(0, 1, k))) % 2  # 带噪音的原始消息
print("纠错前错误", abs(b - v1).sum())

y1 = encode(G, v1, seed=seed)
decode_start1 = time.time_ns() / 10 ** 6
d2 = decode(H, y1, snr)
decode_end1 = time.time_ns() / 10 ** 6

decode_start2 = time.time_ns() / 10 ** 6
x2 = get_message(G, d2)
decode_end2 = time.time_ns() / 10 ** 6

# e2 = get_message(G, np.random.normal(0, 1, len(d2)) % 2)
e2 = get_message(G, d2)
print("纠错后错误", abs(v1 - x2).sum())
# 因此不能公开发送校验信息，否则泄露比特率为1，需要双方都加入噪音，然后相互发送
print("泄露比特率", 1 - abs(v1 - e2).sum() / len(v1))
print("解码时间 (ms)", decode_end1 - decode_start1, decode_end2 - decode_start2)