import numpy as np
# 需要numpy的版本为1.20
from pyldpc import make_ldpc, encode, decode, get_message


n = 320
d_v = 9
d_c = 40
snr = 20
seed = np.random.RandomState(42)
H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)
k = G.shape[1]
print(k)
v = np.random.randint(2, size=k)

y = encode(G, v, seed=seed)
print(y.shape)
d = decode(H, y, snr)
print(d.shape)
x = get_message(G, d)
print(v)
print(y)
print(d)
print(x)
G1 = np.round(G + np.random.normal(0, 0.1, G.shape)) % 2
d1 = np.round(d + np.random.normal(0, 0.1, d.shape)) % 2
print(abs(G - G1).sum())
print(abs(d - d1).sum())
x1 = get_message(G1, d1)
print(abs(x - x1).sum())
x2 = get_message(G, d1)
print(abs(x - x2).sum())
y1 = y + np.round(np.random.normal(0, 0.2, y.shape))
print(abs(y - y1).sum())
d2 = decode(H, y1, snr)
x3 = get_message(G, d2)
print(abs(x - x3).sum())