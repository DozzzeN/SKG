# coding=utf-8
import socket

# 创建udp套接字,
# AF_INET表示ip地址的类型是ipv4，
# SOCK_DGRAM表示传输的协议类型是udp
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 要发送的ip地址和端口（元组的形式）
send_addr = ('127.0.0.1', 8080)
print('send_addr = ', send_addr)

while True:
    # 要发送的信息
    test_data = input('请输入要发送的消息：')
    print('send_data: ', test_data)

    # 发送消息
    udp_socket.sendto(test_data.encode("utf-8"), send_addr)

    # 等待接收数据
    recv = udp_socket.recvfrom(1024)  # 1024表示本次接收的最大字节数

    # 打印接收到的数据
    print("recv_from: {0} , recv_data: {1}".format(recv[1], recv[0]))

# 关闭套接字
# udp_socket.close()
