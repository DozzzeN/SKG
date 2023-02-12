import itertools
import sys

import numpy as np

times = 10000000

matches1 = 0
matches2 = 0
for t in range(times):
    n1 = np.random.rand()
    n2 = np.random.rand()
    x1 = np.random.rand()
    x2 = np.random.rand()
    y1 = np.random.rand()
    y2 = np.random.rand()

    # if 2 * y2 - x1 < 1:
    #     matches1 += 1

    test = [y1, x1, n1, n2, y2, x2]
    m1 = True
    m2 = True
    for i in range(1, len(test)):
        if test[i - 1] > test[i] or x1 + x2 + n1 > 2 * y2 + n2 or x1 + x2 > 2 * y2 or 2 * y2 - x1 < 1:
            m1 = False
        if test[i - 1] > test[i] or x1 + x2 + n1 > 2 * y2 + n2 or x1 + x2 > 2 * y2 or 2 * y2 - x1 > 1:
            m2 = False
    if m1:
        matches1 += 1
    if m2:
        matches2 += 1
print("%.8f" % (matches1 / times))
print("%.8f" % (matches2 / times))
exit()

matches = 0
all_orders = {}
orders = {}

for t in range(times):
    n1 = np.random.rand()
    n2 = np.random.rand()
    x1 = np.random.rand()
    x2 = np.random.rand()
    y1 = np.random.rand()
    y2 = np.random.rand()

    tmp = np.array([n1, n2, x1, x2, y1, y2])
    tmp = list(tmp.argsort().argsort())
    s = "".join(str(j) for j in tmp)
    if all_orders.get(s) is None:
        all_orders[s] = 1
    else:
        all_orders[s] += 1

    # if x1 > y1 and x2 < y2 and x2 + n2 > y1 + n1 and x1 + n1 < x2 + n2:
    if x1 > y1 and x2 > y2 and x2 + n2 > y1 + n1 and x1 + n1 < x2 + n2 and x1 + x2 + n1 + n2 < 2 * y2 + 2 * n2:
        matches += 1
        tmp = np.array([n1, n2, x1, x2, y1, y2])
        tmp = list(tmp.argsort())
        s = "".join(str(j) for j in tmp)
        if orders.get(s) is None:
            orders[s] = 1
        else:
            orders[s] += 1

print(matches / times)
print(sorted(all_orders.items(), key=lambda x: x[0]))
print(sorted(orders.items(), key=lambda x: x[0]))
order_change = {}
for k in orders.keys():
    t = ""
    for i in range(len(k)):
        if k[i] == '0':
            t += "n1"
        elif k[i] == '1':
            t += "n2"
        elif k[i] == '2':
            t += "x1"
        elif k[i] == '3':
            t += "x2"
        elif k[i] == '4':
            t += "y1"
        elif k[i] == '5':
            t += "y2"
    order_change[t] = orders[k]
orders = order_change
standard = [1388, 1041, 694, 347, 173.5, 86.75]  # times = 1000000
# standard = [13880, 10410, 6940, 3470, 1735, 867.5]  # times = 10000000
a1 = []
a2 = []
a3 = []
a4 = []
a5 = []
a6 = []

for i in range(len(orders.items())):
    item = int(list(orders.items())[i][1])
    distance = sys.maxsize
    for j in range(len(standard)):
        distance = min(distance, abs(item - standard[j]))
    for j in range(len(standard)):
        d = abs(item - standard[j])
        if d == distance:
            eval("a" + str(j + 1) + ".append(list(orders.items())[i])")
print(len(a1), a1)
print(len(a2), a2)
print(len(a3), a3)
print(len(a4), a4)
print(len(a5), a5)
print(len(a6), a6)
print(len(list(orders.keys())))
print(len(list(all_orders.keys())))
print(len(list(orders.keys())) / len(list(all_orders.keys())))

# 10 [('231504', 1256), ('032514', 1353), ('132504', 1290), ('152403', 1366), ('052413', 1373), ('053412', 1236), ('251403', 1428), ('054312', 1222), ('042513', 1230), ('023514', 1253)]
# 10 [('054231', 1039), ('321504', 1049), ('351402', 1019), ('241503', 1131), ('154302', 1049), ('013524', 1055), ('123504', 1008), ('153402', 1014), ('142503', 1208), ('451302', 1057)]
# 5 [('025413', 529), ('054321', 731), ('045312', 680), ('053421', 719), ('043512', 745)]
# 12 [('035412', 314), ('312504', 369), ('352401', 264), ('045231', 377), ('103524', 347), ('541302', 338), ('213504', 334), ('024513', 508), ('145302', 330), ('341502', 330), ('034512', 309), ('143502', 351)]
# 19 [('154230', 233), ('254130', 256), ('014523', 236), ('452301', 220), ('254301', 220), ('015423', 219), ('125403', 210), ('035241', 191), ('354201', 235), ('453201', 239), ('302514', 160), ('521403', 211), ('154320', 143), ('124503', 240), ('045321', 189), ('043521', 179), ('203514', 158), ('253401', 234), ('421503', 244)]
# 21 [('352410', 87), ('431502', 108), ('254310', 86), ('015432', 88), ('453120', 126), ('135402', 108), ('531402', 112), ('354120', 120), ('025431', 39), ('253410', 79), ('015342', 120), ('014532', 71), ('153420', 123), ('453210', 83), ('452310', 68), ('134502', 128), ('354210', 83), ('034521', 70), ('025341', 59), ('024531', 25), ('035421', 70)]
# 77
# 720
# 0.10694444444444444

# 10 [('n1y1x1y2x2n2', 1412), ('n1y1x1y2n2x2', 1265), ('y1n1x1y2x2n2', 1439), ('y1x1n1n2y2x2', 1252), ('n1y1y2x2x1n2', 1249), ('n1y1n2x1y2x2', 1264), ('y1x1n1y2x2n2', 1351), ('y1n1x1n2y2x2', 1346), ('n1y1x1n2y2x2', 1251), ('n1y1y2x1x2n2', 1261)]
# 10 [('y1n1y2x1x2n2', 1053), ('y1x1n1y2n2x2', 1162), ('n1n2y1x1y2x2', 1044), ('y1n1x1y2n2x2', 1140), ('y1n1n2x1y2x2', 1075), ('n1y2x2y1x1n2', 1029), ('y1x1y2x2n1n2', 1080), ('y1n1y2x2x1n2', 1080), ('y1x1n2n1y2x2', 1041), ('y1x1y2n1x2n2', 1077)]
# 5 [('n1y1n2y2x1x2', 530), ('n1y1y2x1n2x2', 618), ('n1y2y1x1x2n2', 704), ('n1y1y2x2n2x1', 732), ('n1y2y1x2x1n2', 633)]
# 11 [('n1y1y2n2x2x1', 314), ('y1x1y2n1n2x2', 334), ('n1y1y2n2x1x2', 322), ('y1n1y2x2n2x1', 329), ('n1y1n2y2x2x1', 485), ('y1n1y2x1n2x2', 367), ('y1n2x1n1y2x2', 345), ('y1x1y2x2n2n1', 345), ('n2n1y1x1y2x2', 349), ('y1n2n1x1y2x2', 367), ('n1y2x2y1n2x1', 326)]
# 19 [('n1n2y1y2x1x2', 222), ('y2x2n1y1x1n2', 254), ('n1y2y1x2n2x1', 178), ('y1y2n1x2x1n2', 236), ('n1y2y1x1n2x2', 171), ('y1y2x2x1n1n2', 236), ('y1y2x1n1x2n2', 225), ('y2n1x2y1x1n2', 213), ('y1n1n2y2x2x1', 237), ('y1x1n2y2n1x2', 221), ('n1y2x2n2y1x1', 165), ('n1n2y1y2x2x1', 235), ('y1x1n2y2x2n1', 212), ('y1y2n1x1x2n2', 237), ('y1n1n2y2x1x2', 216), ('y1y2x2n1x1n2', 202), ('n2y1x1n1y2x2', 188), ('n2y1n1x1y2x2', 195), ('y1y2x1x2n1n2', 233)]
# 22 [('n1y2n2x2y1x1', 63), ('n1y2y1n2x1x2', 47), ('y2y1n1x2x1n2', 81), ('y2y1x2n1x1n2', 81), ('y2y1x1x2n1n2', 79), ('y1x1y2n2n1x2', 124), ('y1x1y2n2x2n1', 119), ('n1n2y2x2y1x1', 112), ('y2y1n1x1x2n2', 73), ('y1n1y2n2x1x2', 120), ('y2y1x1n1x2n2', 73), ('y2x2y1n1x1n2', 124), ('y2x2y1x1n1n2', 120), ('y2n1y1x1x2n2', 121), ('y2n1y1x2x1n2', 106), ('n1y2n2y1x2x1', 36), ('y2y1x2x1n1n2', 75), ('y1n1y2n2x2x1', 96), ('n1n2y2y1x2x1', 82), ('n1y2n2y1x1x2', 38), ('n1n2y2y1x1x2', 84), ('n1y2y1n2x2x1', 60)]

# # 10 [
# ('n1y1x1y2x2n2', 1412), o
# ('n1y1x1y2n2x2', 1265),
# ('y1n1x1y2x2n2', 1439), o
# ('y1x1n1n2y2x2', 1252),
# ('n1y1y2x2x1n2', 1249),
# ('n1y1n2x1y2x2', 1264),
# ('y1x1n1y2x2n2', 1351), o
# ('y1n1x1n2y2x2', 1346),
# ('n1y1x1n2y2x2', 1251),
# ('n1y1y2x1x2n2', 1261)]
# # 10 [('y1n1y2x1x2n2', 1053), ('y1x1n1y2n2x2', 1162), ('n1n2y1x1y2x2', 1044), ('y1n1x1y2n2x2', 1140), ('y1n1n2x1y2x2', 1075), ('n1y2x2y1x1n2', 1029), ('y1x1y2x2n1n2', 1080), ('y1n1y2x2x1n2', 1080), ('y1x1n2n1y2x2', 1041), ('y1x1y2n1x2n2', 1077)]
# # 5 [('n1y1n2y2x1x2', 530), ('n1y1y2x1n2x2', 618), ('n1y2y1x1x2n2', 704), ('n1y1y2x2n2x1', 732), ('n1y2y1x2x1n2', 633)]
# # 11 [('n1y1y2n2x2x1', 314), ('y1x1y2n1n2x2', 334), ('n1y1y2n2x1x2', 322), ('y1n1y2x2n2x1', 329), ('n1y1n2y2x2x1', 485), ('y1n1y2x1n2x2', 367), ('y1n2x1n1y2x2', 345), ('y1x1y2x2n2n1', 345), ('n2n1y1x1y2x2', 349), ('y1n2n1x1y2x2', 367), ('n1y2x2y1n2x1', 326)]
# # 19 [('n1n2y1y2x1x2', 222), ('y2x2n1y1x1n2', 254), ('n1y2y1x2n2x1', 178), ('y1y2n1x2x1n2', 236), ('n1y2y1x1n2x2', 171), ('y1y2x2x1n1n2', 236), ('y1y2x1n1x2n2', 225), ('y2n1x2y1x1n2', 213), ('y1n1n2y2x2x1', 237), ('y1x1n2y2n1x2', 221), ('n1y2x2n2y1x1', 165), ('n1n2y1y2x2x1', 235), ('y1x1n2y2x2n1', 212), ('y1y2n1x1x2n2', 237), ('y1n1n2y2x1x2', 216), ('y1y2x2n1x1n2', 202), ('n2y1x1n1y2x2', 188), ('n2y1n1x1y2x2', 195), ('y1y2x1x2n1n2', 233)]
# # 22 [('n1y2n2x2y1x1', 63), ('n1y2y1n2x1x2', 47), ('y2y1n1x2x1n2', 81), ('y2y1x2n1x1n2', 81), ('y2y1x1x2n1n2', 79), ('y1x1y2n2n1x2', 124), ('y1x1y2n2x2n1', 119), ('n1n2y2x2y1x1', 112), ('y2y1n1x1x2n2', 73), ('y1n1y2n2x1x2', 120), ('y2y1x1n1x2n2', 73), ('y2x2y1n1x1n2', 124), ('y2x2y1x1n1n2', 120), ('y2n1y1x1x2n2', 121), ('y2n1y1x2x1n2', 106), ('n1y2n2y1x2x1', 36), ('y2y1x2x1n1n2', 75), ('y1n1y2n2x2x1', 96), ('n1n2y2y1x2x1', 82), ('n1y2n2y1x1x2', 38), ('n1n2y2y1x1x2', 84), ('n1y2y1n2x2x1', 60)]
