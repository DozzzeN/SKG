import numpy as np

times = 1000000

matches0 = 0
matches1 = 0
matches2 = 0
matches3 = 0
matches4 = 0
matches5 = 0
matches6 = 0
matches7 = 0
matches8 = 0
matches9 = 0
matches10 = 0
matches11 = 0
matches12 = 0
matches13 = 0
matches14 = 0
matches15 = 0
matches16 = 0
matches17 = 0
matches18 = 0
matches19 = 0
matches20 = 0
matches21 = 0
matches22 = 0
matches23 = 0
matches24 = 0
matches25 = 0
matches26 = 0
matches27 = 0
matches28 = 0
matches29 = 0
matches30 = 0
matches31 = 0
matches32 = 0
matchesall1 = 0

for t in range(times):
    n1 = np.random.rand()
    n2 = np.random.rand()
    x1 = np.random.rand()
    x2 = np.random.rand()
    y1 = np.random.rand()
    y2 = np.random.rand()

    # 0.0263
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2:
        matches14 += 1

    # 0.01943
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 < y2:
        matches16 += 1

    # 0.00693
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2:
        matches15 += 1

    a = y2
    b = y1 + n1 - n2
    c = 2 * y2 + n2 - n1 - 1
    # 311
    # 0.00212
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and 2 * y2 + n2 - 2 > 0:
        matches17 += 1

    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and 2 * y2 + n2 - 2 < 0:
        matches18 += 1

    # 0.00032
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and 2 * y2 + n2 - 2 < 0 and y1 + n1 - n2 < 0 and n2 - y1 < y2 + n2 - 1:
        matches18 += 1

    # 0.002922
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and 2 * y2 + n2 - 2 < 0 and 2 - 2 * y2 > 1:
        matches19 += 1

    # matches23+matches26+matches29
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and y2 > 0 and 0 > y1 + n1 - n2 and 2 * y2 + n2 - n1 - 1 > y2:
        matches20 += 1

    # 13/11520
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and y1 + n1 - n2 < 0 and 2 * y2 + n2 - n1 - 1 > y2 \
            and 2 * y2 + n2 - 2 > 0 and n2 - y1 > y2 + n2 - 1 and y2 + n2 - 1 < 1 and n2 - y1 < 1:
        matches21 += 1

    # 1/3840
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and y1 + n1 - n2 < 0 and 2 * y2 + n2 - n1 - 1 > y2 \
            and 2 * y2 + n2 - 2 > 0 and n2 - y1 < y2 + n2 - 1 and y2 + n2 - 1 < 1:
        matches22 += 1

    # matches21+matches22
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and y1 + n1 - n2 < 0 and 2 * y2 + n2 - n1 - 1 > y2 \
            and 2 * y2 + n2 - 2 > 0:
        matches23 += 1

    # 119/46080
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and y1 + n1 - n2 < 0 and 2 * y2 + n2 - n1 - 1 > y2 \
            and 2 * y2 + n2 - 2 < 0 and n2 - y1 > y2 + n2 - 1 and y2 < 1 / 2:
        matches24 += 1

    # 61/46080
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and y1 + n1 - n2 < 0 and 2 * y2 + n2 - n1 - 1 > y2 \
            and 2 * y2 + n2 - 2 < 0 and n2 - y1 > y2 + n2 - 1 and y2 > 1 / 2:
        matches25 += 1

    # matches24+matches25
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and y1 + n1 - n2 < 0 and 2 * y2 + n2 - n1 - 1 > y2 \
            and 2 * y2 + n2 - 2 < 0 and n2 - y1 > y2 + n2 - 1:
        matches26 += 1

    # 1/5120
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and y1 + n1 - n2 < 0 and 2 * y2 + n2 - n1 - 1 > y2 \
            and 2 * y2 + n2 - 2 < 0 and n2 - y1 < y2 + n2 - 1 and y2 > 1 / 2:
        matches27 += 1

    # 1/7680
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and y1 + n1 - n2 < 0 and 2 * y2 + n2 - n1 - 1 > y2 \
            and 2 * y2 + n2 - 2 < 0 and n2 - y1 < y2 + n2 - 1 and y2 < 1 / 2:
        matches28 += 1

    # matches27+matches28
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and y1 + n1 - n2 < 0 and 2 * y2 + n2 - n1 - 1 > y2 \
            and 2 * y2 + n2 - 2 < 0 and n2 - y1 < y2 + n2 - 1:
        matches29 += 1

    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and y2 > 0 and 0 > y1 + n1 - n2 and 2 * y2 + n2 - n1 - 1 < y2 \
            and 2 * y2 + n2 - n1 > 1 and y2 + n2 - 1 < 0 and 2 * y2 + n2 - 1 > n2 - y1:
        matches30 += 1

print("matches14 %.8f" % (matches14 / times))
print("matches16 %.8f" % (matches16 / times))
print("matches15 %.8f" % (matches15 / times))
print("matches17 %.8f" % (matches17 / times))
print("matches18 %.8f" % (matches18 / times))
print("matches19 %.8f" % (matches19 / times))
print("matches20 %.8f" % (matches20 / times), (matches23 + matches26 + matches29) / times)
print("matches21 %.8f" % (matches21 / times))
print("matches22 %.8f" % (matches22 / times))
print("matches23 %.8f" % (matches23 / times))
print("matches24 %.8f" % (matches24 / times))
print("matches25 %.8f" % (matches25 / times))
print("matches26 %.8f" % (matches26 / times))
print("matches27 %.8f" % (matches27 / times))
print("matches28 %.8f" % (matches28 / times))
print("matches29 %.8f" % (matches29 / times))
print("matches30 %.8f" % (matches30 / times))
print("matches31 %.8f" % (matches31 / times))
