import hashlib

# def shuffle_str(s):
#     # 将字符串转换成列表
#     str_list = list(s)
#     # 调用random模块的shuffle函数打乱列表
#     random.shuffle(str_list)
#     # 将列表转字符串
#     return ''.join(str_list)

with open('./evaluations/key_data_static_outdoor_1.txt', 'r+') as f:
    read_data = f.read()

codings = ""
with open('./evaluations/so.txt', 'w+') as f:
    for i in range(0, len(read_data), 25):
        raw = read_data[i:i + 25] + bin(i)
        codings += bin(int(hashlib.sha256(raw.encode('utf-8')).hexdigest()[0:30], 16))[2:] + "\n"
    f.write(codings)
