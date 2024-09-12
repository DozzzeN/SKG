import time

import numpy as np
# 需要numpy的版本为1.20
from pyldpc import make_ldpc, encode, decode, get_message


for i in range(2, 50):
    for j in range(1, 100):
        d_v = i
        d_c = d_v + j
        # 设置n = d_c到n = 8 * d_c
        n = 10 * d_c
        snr = 20
        seed = np.random.RandomState(42)
        H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)
        k = G.shape[1]
        # if k == 64 or k == 128 or k == 256 or k == 512:
        #     print(n, d_v, d_c, k)
        if k == 1024:
            print(n, d_v, d_c, k)

# n_v n_c越小代表校验矩阵越稀疏，从而解码速度更快
# 77 2 11 64
# 68 3 34 64
# 147 3 21 128
# 135 3 45 128
# 266 2 53 256
# 540 3 54 512
# 1340 35 134 1024