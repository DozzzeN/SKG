import os
import params, precomp
import numpy as np

QINV = 12287  # -inverse_mod(p,2^18)
RLOG = 18


def LDDecode(xi0, xi1, xi2, xi3):
    t = g(xi0)
    t += g(xi1)
    t += g(xi2)
    t += g(xi3)
    t -= 8 * params.Q
    t >>= 31
    return t & 1  # round(t / 2^32)


def nh_abs(x):
    mask = x >> 31
    return (x ^ mask) - mask


def f(x):
    # Next 6 lines compute t = x / params.Q
    b = x * 2730
    t = b >> 25
    b = x - t * 12289
    b = 12288 - b
    b >>= 31
    t -= b

    r = t & 1
    xit = t >> 1
    v0 = xit + r  # v0 = round(x / (2 * params.Q))

    t -= 1
    r = t & 1
    v1 = (t >> 1) + r
    return (v0, v1, nh_abs(x - v0 * 2 * params.Q))


def g(x):
    # Next 6 lines compute t = x / (4 * params.Q)
    b = x * 2730
    t = b >> 27
    b = x - t * 49156
    b = 49155 - b
    b >>= 31
    t -= b
    # t_prime = int(2730 * x / 2 ** 27) - int((49155 - x + int(2730 * x / 2 ** 27) * 49156) / 2 ** 31)
    # tt = 2730 * x / 2 ** 27 - (49155 - x + 2730 * x / 2 ** 27 * 49156) / 2 ** 31
    # tt_prime = x / (4 * params.Q)

    c = t & 1
    t = (t >> 1) + c  # t = round(x / (8 * params.Q))
    t *= 8 * params.Q
    return nh_abs(t - x)


def helprec(coefficients):
    rand = []
    output = []
    for i in range(0, 1024):
        output.append(0)
    v0 = [0, 0, 0, 0]
    v1 = [0, 0, 0, 0]
    v_tmp = [0, 0, 0, 0]
    for i in range(0, 32):
        rand.append(int.from_bytes(os.urandom(4), byteorder='little'))
        # np.random.seed(0)
        # rand.append(np.random.randint(0, 2 ** 16))
    for i in range(0, 256):
        rbit = rand[i >> 3] >> (i & 7) & 1
        (v0[0], v1[0], k) = f(8 * coefficients[0 + i] + 4 * rbit)
        (v0[1], v1[1], x) = f(8 * coefficients[256 + i] + 4 * rbit)
        k += x
        (v0[2], v1[2], x) = f(8 * coefficients[512 + i] + 4 * rbit)
        k += x
        (v0[3], v1[3], x) = f(8 * coefficients[768 + i] + 4 * rbit)
        k += x
        k = 2 * params.Q - 1 - k >> 31
        v_tmp[0] = ((~k) & v0[0]) ^ (k & v1[0])
        v_tmp[1] = ((~k) & v0[1]) ^ (k & v1[1])
        v_tmp[2] = ((~k) & v0[2]) ^ (k & v1[2])
        v_tmp[3] = ((~k) & v0[3]) ^ (k & v1[3])
        output[0 + i] = (v_tmp[0] - v_tmp[3]) & 3
        output[256 + i] = (v_tmp[1] - v_tmp[3]) & 3
        output[512 + i] = (v_tmp[2] - v_tmp[3]) & 3
        output[768 + i] = (-k + 2 * v_tmp[3]) & 3
    return output


def rec(v_coeffs, c_coeffs):
    key = []
    tmp = [0, 0, 0, 0]
    for i in range(0, 32):
        key.append(0)
    for i in range(0, 256):
        tmp[0] = (
                16 * params.Q
                + 8 * v_coeffs[0 + i]
                - params.Q * (2 * c_coeffs[0 + i] + c_coeffs[768 + i]))
        tmp[1] = (
                16 * params.Q
                + 8 * v_coeffs[256 + i]
                - params.Q * (2 * c_coeffs[256 + i] + c_coeffs[768 + i]))
        tmp[2] = (
                16 * params.Q
                + 8 * v_coeffs[512 + i]
                - params.Q * (2 * c_coeffs[512 + i] + c_coeffs[768 + i]))
        tmp[3] = (
                16 * params.Q
                + 8 * v_coeffs[768 + i]
                - params.Q * (c_coeffs[768 + i]))
        key[i >> 3] |= LDDecode(tmp[0], tmp[1], tmp[2], tmp[3]) << (i & 7)
    return key


def recKeyBit(v_coeffs, c_coeffs, index):
    key = 0

    # keytmp0 = 16 * params.Q + 8 * v_coeffs[0] - params.Q * (2 * c_coeffs[0] + c_coeffs[768])
    # keytmp1 = 16 * params.Q + 8 * v_coeffs[256] - params.Q * (2 * c_coeffs[256] + c_coeffs[768])
    # keytmp2 = 16 * params.Q + 8 * v_coeffs[512] - params.Q * (2 * c_coeffs[512] + c_coeffs[768])
    # keytmp3 = 16 * params.Q + 8 * v_coeffs[768] - params.Q * (c_coeffs[768])
    # key |= LDDecode(keytmp0, keytmp1, keytmp2, keytmp3) << (0 & 7)
    #
    # keytmp0 = 16 * params.Q + 8 * v_coeffs[1] - params.Q * (2 * c_coeffs[1] + c_coeffs[769])
    # keytmp1 = 16 * params.Q + 8 * v_coeffs[257] - params.Q * (2 * c_coeffs[257] + c_coeffs[769])
    # keytmp2 = 16 * params.Q + 8 * v_coeffs[513] - params.Q * (2 * c_coeffs[513] + c_coeffs[769])
    # keytmp3 = 16 * params.Q + 8 * v_coeffs[769] - params.Q * (c_coeffs[769])
    # key |= LDDecode(keytmp0, keytmp1, keytmp2, keytmp3) << (1 & 7)
    #
    # keytmp0 = 16 * params.Q + 8 * v_coeffs[2] - params.Q * (2 * c_coeffs[2] + c_coeffs[770])
    # keytmp1 = 16 * params.Q + 8 * v_coeffs[258] - params.Q * (2 * c_coeffs[258] + c_coeffs[770])
    # keytmp2 = 16 * params.Q + 8 * v_coeffs[514] - params.Q * (2 * c_coeffs[514] + c_coeffs[770])
    # keytmp3 = 16 * params.Q + 8 * v_coeffs[770] - params.Q * (c_coeffs[770])
    # key |= LDDecode(keytmp0, keytmp1, keytmp2, keytmp3) << (2 & 7)
    #
    # keytmp0 = 16 * params.Q + 8 * v_coeffs[3] - params.Q * (2 * c_coeffs[3] + c_coeffs[771])
    # keytmp1 = 16 * params.Q + 8 * v_coeffs[259] - params.Q * (2 * c_coeffs[259] + c_coeffs[771])
    # keytmp2 = 16 * params.Q + 8 * v_coeffs[515] - params.Q * (2 * c_coeffs[515] + c_coeffs[771])
    # keytmp3 = 16 * params.Q + 8 * v_coeffs[771] - params.Q * (c_coeffs[771])
    # key |= LDDecode(keytmp0, keytmp1, keytmp2, keytmp3) << (3 & 7)
    #
    # keytmp0 = 16 * params.Q + 8 * v_coeffs[4] - params.Q * (2 * c_coeffs[4] + c_coeffs[772])
    # keytmp1 = 16 * params.Q + 8 * v_coeffs[260] - params.Q * (2 * c_coeffs[260] + c_coeffs[772])
    # keytmp2 = 16 * params.Q + 8 * v_coeffs[516] - params.Q * (2 * c_coeffs[516] + c_coeffs[772])
    # keytmp3 = 16 * params.Q + 8 * v_coeffs[772] - params.Q * (c_coeffs[772])
    # key |= LDDecode(keytmp0, keytmp1, keytmp2, keytmp3) << (4 & 7)
    #
    # keytmp0 = 16 * params.Q + 8 * v_coeffs[5] - params.Q * (2 * c_coeffs[5] + c_coeffs[773])
    # keytmp1 = 16 * params.Q + 8 * v_coeffs[261] - params.Q * (2 * c_coeffs[261] + c_coeffs[773])
    # keytmp2 = 16 * params.Q + 8 * v_coeffs[517] - params.Q * (2 * c_coeffs[517] + c_coeffs[773])
    # keytmp3 = 16 * params.Q + 8 * v_coeffs[773] - params.Q * (c_coeffs[773])
    # key |= LDDecode(keytmp0, keytmp1, keytmp2, keytmp3) << (5 & 7)
    #
    # keytmp0 = 16 * params.Q + 8 * v_coeffs[6] - params.Q * (2 * c_coeffs[6] + c_coeffs[774])
    # keytmp1 = 16 * params.Q + 8 * v_coeffs[262] - params.Q * (2 * c_coeffs[262] + c_coeffs[774])
    # keytmp2 = 16 * params.Q + 8 * v_coeffs[518] - params.Q * (2 * c_coeffs[518] + c_coeffs[774])
    # keytmp3 = 16 * params.Q + 8 * v_coeffs[774] - params.Q * (c_coeffs[774])
    # key |= LDDecode(keytmp0, keytmp1, keytmp2, keytmp3) << (6 & 7)
    #
    # keytmp0 = 16 * params.Q + 8 * v_coeffs[7] - params.Q * (2 * c_coeffs[7] + c_coeffs[775])
    # keytmp1 = 16 * params.Q + 8 * v_coeffs[263] - params.Q * (2 * c_coeffs[263] + c_coeffs[775])
    # keytmp2 = 16 * params.Q + 8 * v_coeffs[519] - params.Q * (2 * c_coeffs[519] + c_coeffs[775])
    # keytmp3 = 16 * params.Q + 8 * v_coeffs[775] - params.Q * (c_coeffs[775])
    # key |= LDDecode(keytmp0, keytmp1, keytmp2, keytmp3) << (7 & 7)

    for i in range(0, 8):
        keytmp0 = 16 * params.Q + 8 * v_coeffs[0 + i + index] - params.Q * (
                    2 * c_coeffs[0 + i + index] + c_coeffs[768 + i + index])
        keytmp1 = 16 * params.Q + 8 * v_coeffs[256 + i + index] - params.Q * (
                    2 * c_coeffs[256 + i + index] + c_coeffs[768 + i + index])
        keytmp2 = 16 * params.Q + 8 * v_coeffs[512 + i + index] - params.Q * (
                    2 * c_coeffs[512 + i + index] + c_coeffs[768 + i + index])
        keytmp3 = 16 * params.Q + 8 * v_coeffs[768 + i + index] - params.Q * (c_coeffs[768 + i + index])
        key |= LDDecode(keytmp0, keytmp1, keytmp2, keytmp3) << i

    return key


def bitrev_vector(coefficients):
    for i in range(0, params.N):
        r = precomp.bitrev_table[i]
        if i < r:
            tmp = coefficients[i]
            coefficients[i] = coefficients[r]
            coefficients[r] = tmp
    return coefficients


# def invntt(coefficients):
#     coefficients = bitrev_vector(coefficients)
#     coefficients = ntt(coefficients, precomp.omegas_inv_montgomery)
#     coefficients = mul_coefficients(coefficients, precomp.psis_inv_montgomery)
#     return coefficients


# Get a random sampling of integers from a normal distribution around parameter
# Q.
def get_noise():
    coeffs = []
    for i in range(0, params.N):
        t = int.from_bytes(os.urandom(4), byteorder='little')
        # np.random.seed(0)
        # t = np.random.randint(0, 2 ** 16)
        d = 0
        for j in range(0, 8):
            d += (t >> j) & 0x01010101
        a = ((d >> 8) & 0xff) + (d & 0xff)
        b = (d >> 24) + ((d >> 16) & 0xff)
        coeffs.append(a + params.Q - b)
    return coeffs


def get_noise_from_channel(s_coeffs):
    coeffs = []
    for i in range(0, params.N):
        t = s_coeffs[i]
        d = 0
        for j in range(0, 8):
            d += (t >> j) & 0x01010101
        a = ((d >> 8) & 0xff) + (d & 0xff)
        b = (d >> 24) + ((d >> 16) & 0xff)
        coeffs.append(a + params.Q - b)
    return coeffs


def ntt(coefficients, omega):
    for i in range(0, 10, 2):
        distance = 1 << i
        for start in range(0, distance):
            jTwiddle = 0
            for j in range(start, params.N - 1, 2 * distance):
                W = omega[jTwiddle]
                jTwiddle += 1
                temp = coefficients[j]
                coefficients[j] = temp + coefficients[j + distance]
                coefficients[j + distance] = montgomery_reduce(
                    W * (temp + 3 * params.Q - coefficients[j + distance]))
        distance <<= 1
        for start in range(0, distance):
            jTwiddle = 0
            for j in range(start, params.N - 1, 2 * distance):
                W = omega[jTwiddle]
                jTwiddle += 1
                temp = coefficients[j]
                coefficients[j] = barrett_reduce(temp + coefficients[j + distance])
                coefficients[j + distance] = montgomery_reduce(
                    W * (temp + 3 * params.Q - coefficients[j + distance]))
    return coefficients


# def poly_ntt(coefficients):
#     coefficients = mul_coefficients(coefficients, precomp.psis_bitrev_montgomery)
#     coefficients = ntt(coefficients, precomp.omegas_montgomery)
#     return coefficients


# a and b are the coefficients of these polys as lists.
def pointwise(a, b):
    coefficients = []
    for i in range(0, params.N):
        t = montgomery_reduce(3186 * b[i])
        coefficients.append(montgomery_reduce(a[i] * t))
    return coefficients


# a and b are the coefficients of these polys as lists.
def add(a, b):
    coefficients = []
    for i in range(0, params.N):
        coefficients.append(barrett_reduce(a[i] + b[i]))
    return coefficients


def sub(a, b):
    coefficients = []
    for i in range(0, params.N):
        coefficients.append(barrett_reduce(a[i] - b[i]))
    return coefficients


def mul_coefficients(coefficients, factors):
    for i in range(0, params.N):
        coefficients[i] = montgomery_reduce(coefficients[i] * factors[i])
    return coefficients


def montgomery_reduce(a):
    u = a * QINV
    u &= (1 << RLOG) - 1
    u *= params.Q
    a += u
    return a >> 18
    # 捕获RuntimeWarning
    # import warnings
    # warnings.filterwarnings("error")
    # try:
    #     u = a * QINV
    #     u &= (1 << RLOG) - 1
    #     u *= params.Q
    #     a += u
    # except Warning as e:
    #     return a >> 18


def barrett_reduce(a):
    u = (a * 5) >> 16
    u *= params.Q
    a -= u
    return a
