import precomp

bitrev_table = []
for v in range(0, 1024):
    res = 0
    for i in range(0, 10):
        res += ((v >> i) & 1) << (10 - 1 - i)
    res = res % 1024
    bitrev_table.append(res)
print(bitrev_table == precomp.bitrev_table)
