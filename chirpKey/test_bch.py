#!/usr/bin/env python

import binascii
import hashlib
import os
import random
import unittest

import bchlib
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

class BCHTestCase(unittest.TestCase):
    def exercise(self, *args, **kwargs):
        # create a bch object
        bch = bchlib.BCH(*args, **kwargs)
        max_data_len = bch.n // 8 - (bch.ecc_bits + 14) // 8

        print('max_data_len: %d' % (max_data_len,))
        print('ecc_bits: %d (ecc_bytes: %d)' % (bch.ecc_bits, bch.ecc_bytes))
        print('m: %d' % (bch.m,))
        print('n: %d (%d bytes)' % (bch.n, bch.n // 8))
        print('k: %d (%d bytes)' % (bch.n - bch.ecc_bits, (bch.n - bch.ecc_bits) // 8))
        print('prim_poly: 0x%x' % (bch.prim_poly,))
        print('t: %d' % (bch.t,))
        print('capability', bch.t / bch.n)

        # random data
        # data = bytearray(os.urandom(max_data_len))
        # ecc = bch.encode(data)
        # packet = data + ecc

        data = np.random.randint(0, 4, max_data_len * 4, dtype=np.int8)
        data = quartet_to_bytes(data)

        # data = np.random.randint(0, 4, max_data_len * 8, dtype=np.int8)
        # data = (data.reshape((-1, 8))[:, ::-1] << np.arange(7, -1, -1)).sum(axis=1).astype(np.uint8)
        # data = bytearray(data.tobytes())

        # encode and make a "packet"
        ecc = bch.encode(data)
        packet = data + ecc
        print("code len:", len(data), len(ecc), len(packet) * 8)
        print('packet:', binascii.hexlify(packet).decode('utf-8'))

        data_quartet_array = bytes_to_quartet(data)
        print("before", len(data_quartet_array), data_quartet_array)
        print("data", data)

        packet_quartet_array = bytes_to_quartet(packet)

        # de-packetize
        a_data_quartet_array = packet_quartet_array[:max_data_len * 4]
        # a_data_quartet_array[0] = (a_data_quartet_array[0] + 1) % 4
        # a_data_quartet_array[3] = (a_data_quartet_array[3] + 3) % 4
        # print("corrupted", a_data_quartet_array)
        a_data = quartet_to_bytes(a_data_quartet_array)
        print("a_data", a_data)

        a_ecc_quartet_array = packet_quartet_array[max_data_len * 4:]
        # a_ecc_quartet_array[0] = (a_ecc_quartet_array[0] + 1) % 4
        # a_ecc_quartet_array[3] = (a_ecc_quartet_array[3] + 2) % 4
        # a_ecc_quartet_array[4] = (a_ecc_quartet_array[4] + 2) % 4
        a_ecc = quartet_to_bytes(a_ecc_quartet_array)
        # a_packet = a_data + a_ecc
        # print('packet:', binascii.hexlify(a_packet).decode('utf-8'))
        bch.decode(a_data, a_ecc)
        bch.correct(a_data, a_ecc)

        # packetize
        a_packet = a_data + a_ecc
        print('packet:', binascii.hexlify(a_packet).decode('utf-8'))
        a_data_quartet_array = bytes_to_quartet(a_data)
        print('after:', a_data_quartet_array)

        assert data_quartet_array == a_data_quartet_array
    def test_t_eq_6(self):
        # 2 total len
        # self.exercise(3, m=1)
        # 16 total len 不知道为什么这个参数无法选择，对应于BCH(15,11)
        # self.exercise(1, m=5)
        # 32 total len
        # self.exercise(5, m=5)
        # 64 total len
        # self.exercise(1, m=6)
        # self.exercise(6, m=6)
        self.exercise(10, m=6)
        # 128 total len
        # self.exercise(11, m=7)
        # self.exercise(10, m=7)
        # 256 total len
        # self.exercise(19, m=8)
        # self.exercise(17, m=8)
        # self.exercise(21, m=8)
        # 512 total len
        # self.exercise(19, m=9)
        # self.exercise(31, m=9)
        # self.exercise(36, m=9)
        # 1024 total len
        # self.exercise(22, m=10)
        # self.exercise(58, m=10)
        # self.exercise(60, m=10)


    # def test_t_eq_12(self):
    #     self.exercise(12, prim_poly=17475, swap_bits=True)
    #
    # def test_t_eq_16(self):
    #     self.exercise(16, m=13)
    #
    # def test_t_eq_32(self):
    #     self.exercise(32, m=14)
    #
    # def test_t_eq_64(self):
    #     self.exercise(64, m=15)


if __name__ == '__main__':
    unittest.main()
