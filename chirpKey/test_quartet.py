# 将四进制的数组（0-3）转换为字节数组
import numpy as np


def quartet_to_bytes(quartet_array):
    byte_array = bytearray()
    current_byte = 0
    bits_written = 0

    for quartet in quartet_array:
        current_byte = (current_byte << 2) | quartet
        bits_written += 2

        if bits_written == 8:
            byte_array.append(current_byte)
            current_byte = 0
            bits_written = 0

    if bits_written > 0:
        # 在最后一个字节上添加零位，以确保字节数组的长度是整数倍
        current_byte <<= 8 - bits_written
        byte_array.append(current_byte)

    return byte_array

# 将字节数组转换为四进制的数组
def bytes_to_quartet(byte_array):
    quartet_array = []
    current_byte = 0
    bits_read = 0

    for byte in byte_array:
        current_byte = (current_byte << 8) | byte
        bits_read += 8

        while bits_read >= 2:
            bits_read -= 2
            quartet = (current_byte >> bits_read) & 0b11
            quartet_array.append(quartet)

    return quartet_array

# 测试
input_quartet_array = np.random.randint(0, 4, 16)
byte_array = quartet_to_bytes(input_quartet_array)
output_quartet_array = bytes_to_quartet(byte_array)

print("输入四进制数组:", input_quartet_array)
print("字节数组:", byte_array)
print("输出四进制数组:", output_quartet_array)

# 示例的二进制数组
binary_array = [0, 1, 1, 0, 1, 0, 0, 1]

# 将二进制数组转换为字节数组
def binary_to_bytes(binary_array):
    byte_array = bytearray()
    for i in range(0, len(binary_array), 8):
        byte = binary_array[i:i+8]
        byte_value = int(''.join(map(str, byte)), 2)
        byte_array.append(byte_value)
    return byte_array

# 打印字节数组
print(binary_to_bytes(binary_array))

# 将字节数组转换为二进制数组
def bytes_to_binary(byte_array):
    binary_array = []
    for byte_value in byte_array:
        byte_binary = bin(byte_value)[2:].zfill(8)  # 将字节值转换为二进制字符串并补零
        binary_array.extend(map(int, byte_binary))
    return binary_array

# 打印二进制数组
print(bytes_to_binary(binary_to_bytes(binary_array)))

