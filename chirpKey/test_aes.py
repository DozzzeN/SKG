import time

import numpy as np
from Crypto.Cipher import AES

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

password = np.random.randint(0, 2, 128).tolist()
password_bytes = binary_to_bytes(password)
# text = b'abcdefghijklmnhi'  # 需要加密的内容，bytes类型
enc_time = []
for i in range(100000):
    text = np.random.randint(0, 2, 1024 * 10).tolist()
    text_bytes = binary_to_bytes(text)
    enc_start = time.time_ns() / 10 ** 6
    aes = AES.new(password_bytes, AES.MODE_ECB)  # 创建一个aes对象
    # AES.MODE_ECB 表示模式是ECB模式
    en_text = aes.encrypt(text_bytes)  # 加密明文
    enc_time.append(time.time_ns() / 10 ** 6 - enc_start)
    # print("密文：", en_text)  # 加密明文，bytes类型
    den_text = aes.decrypt(en_text)  # 解密密文
    # print("明文：", den_text)
    exit()
print("加密平均时间：", sum(enc_time) / len(enc_time))