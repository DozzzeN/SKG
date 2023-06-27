#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名：client.py

import socket  # 导入 socket 模块

s = socket.socket()  # 创建 socket 对象
host = socket.gethostname()  # 获取本地主机名
port = 1231  # 设置端口号
s.connect((host, port))

while True:
    send_data = input("请输入要发送的数据：")
    s.send(send_data.encode("utf-8"))
    print(s.recv(1024))
