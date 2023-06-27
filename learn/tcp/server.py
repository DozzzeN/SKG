#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 文件名：server.py
import socket  # 导入 socket 模块

s = socket.socket()  # 创建 socket 对象
host = socket.gethostname()  # 获取本地主机名
port = 1231  # 设置端口
s.bind((host, port))  # 绑定端口

s.listen(5)  # 等待客户端连接

while True:
    c, addr = s.accept()  # 建立客户端连接
    while True:
        recv_data = c.recv(1024)
        if recv_data:
            print('接收到的数据为:', recv_data.decode('utf-8'))
            print('连接地址：', addr)
            c.send("连接成功".encode('utf-8'))
        else:
            break
