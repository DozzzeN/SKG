from hashlib import sha3_256

import numpy as np


def sha3_256_hash(data):
    # 将整数数组转换为字节串
    byte_array = bytes(data)

    # 创建SHA3-256哈希对象
    sha3_256_hash = sha3_256()

    # 更新哈希对象的数据
    sha3_256_hash.update(byte_array)

    # 获取哈希值的十六进制表示
    hash_value = sha3_256_hash.hexdigest()

    return hash_value

def hash_to_binary(hash_value):
    # 将十六进制哈希值转换为整数
    decimal_value = int(hash_value, 16)

    # 将整数转换为二进制字符串，去掉开头的 '0b'
    binary_string = bin(decimal_value)[2:]

    # 补齐长度至256位
    binary_string = binary_string.zfill(256)

    return binary_string

data = np.random.randint(0, 255, 32)

hash_value = sha3_256_hash(data)
binary_string = hash_to_binary(hash_value)
print("SHA3-256 哈希值（二进制）：", len(binary_string), binary_string)
