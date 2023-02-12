def genDiff(data):
    diff = []
    for i in range(len(data) - 1):
        diff.append(abs(data[i + 1] - data[i]))
    return diff


def recoverData(start, diff, ref):
    rec = []
    rec.append(start)
    for i in range(len(diff)):
        if ref[i + 1] - ref[i] > 0:
            rec.append(rec[i] + diff[i])
        else:
            rec.append(rec[i] - diff[i])
    return rec

a = [1, 2, 4, 5, -1]
b = [2, 3, 5, 6, -1]

diff = genDiff(a)
rec = recoverData(a[0], diff, b)
print(rec)