import os

import numpy as np
from reedsolo import RSCodec


def binary_to_bytes(binary_array):
    byte_array = bytearray()
    for i in range(0, len(binary_array), 8):
        byte = binary_array[i:i + 8]
        byte_value = int(''.join(map(str, byte)), 2)
        byte_array.append(byte_value)
    return byte_array


def bytes_to_binary(byte_array):
    binary_array = []
    for byte_value in byte_array:
        byte_binary = bin(byte_value)[2:].zfill(8)  # 将字节值转换为二进制字符串并补零
        binary_array.extend(map(int, byte_binary))
    return binary_array


def bytearray_diff(a, b):
    if len(a) != len(b):
        raise ValueError("array length not equal")
    diff = 0
    for i in range(len(a)):
        diff += 1 if a[i] != b[i] else 0
    return diff


# a = [1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
# b = [1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1]
a = [1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
b = [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]

print(bytearray_diff(a, b))
# 最多错误个数为字节数，而不是比特数
print(bytearray_diff(binary_to_bytes(a), binary_to_bytes(b)))
print()

ecc = RSCodec(4)  # 纠错码字长度，可纠正n/2个字节错误
print("max_error_number:", ecc.maxerrata()[0])
data = np.random.randint(0, 2, 32)
data = binary_to_bytes(data)
print(data)
print(len(data))
encoded = ecc.encode(data)
encoded_tampered = np.int8(
    np.round(bytes_to_binary(encoded) + np.random.normal(0, 0.2, len(bytes_to_binary(encoded)))) % 2)
encoded_tampered = binary_to_bytes(encoded_tampered)
print("error", abs(np.array(bytes_to_binary(encoded)) - np.array(bytes_to_binary(encoded_tampered))).sum())
print("error", bytearray_diff(encoded, encoded_tampered))
# 512 bits
print(len(encoded))
# decoded = ecc.decode(encoded)[0]
decoded = ecc.decode(encoded_tampered)[0]
print(decoded)
print(len(decoded))
decoded = bytes_to_binary(decoded)
