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

    if x1 > y1 and x2 + n2 > y1 + n1 and x2 > y2 and x1 + n1 < y2 + n2 \
            and x1 + n1 < x2 + n2 and x1 + x2 < 2 * y2 + n2 - n1:
        matches1 += 1

    # 1/192
    # 2
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 > 1 \
            and x2 > y2 and y2 > y1 + n1 - n2 and 2 * y2 + n2 - n1 - 1 > 1 \
            and n1 < 2 * y2 + n2 - 2 and n1 < y2 + n2 - y1:
        matches2 += 1

    # 11/11520
    # 21
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 > 1 \
            and x2 > y2 and y2 > y1 + n1 - n2 and 2 * y2 + n2 - n1 - 1 > 1 \
            and 2 * y2 + n2 - 2 < y2 - y1 + n2 and y2 - y1 + n2 < 1:
        matches3 += 1

    # 211
    # 1/3840
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 > 1 \
            and x2 > y2 and y2 > y1 + n1 - n2 and 2 * y2 + n2 - n1 - 1 > 1 \
            and 2 * y2 + n2 - 2 < y2 - y1 + n2 and y2 - y1 + n2 < 1 and n1 < 2 * y2 + n2 - 2 \
            and n2 > 2 - 2 * y2 and y1 > y2 and y2 > 1 / 2:
        matches4 += 1

    # 212
    # 1/1440
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 > 1 \
            and x2 > y2 and y2 > y1 + n1 - n2 and 2 * y2 + n2 - n1 - 1 > 1 \
            and 2 * y2 + n2 - 2 < y2 - y1 + n2 and y2 - y1 + n2 < 1 and n1 < 2 * y2 + n2 - 2 \
            and n2 > 2 - 2 * y2 and n2 < 1 + y1 - y2 and y1 < y2 and y1 > 1 - y2 and y2 > 1 / 2:
        matches5 += 1

    # 49/11520
    # 22
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 > 1 \
            and x2 > y2 and y2 > y1 + n1 - n2 and 2 * y2 + n2 - n1 - 1 > 1 \
            and 2 * y2 + n2 - 2 < y2 - y1 + n2 and 2 * y2 + n2 - 2 < 1 and y2 - y1 + n2 > 1:
        matches6 += 1

    # 7/3840
    # 221
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 > 1 \
            and x2 > y2 and y2 > y1 + n1 - n2 and 2 * y2 + n2 - n1 - 1 > 1 \
            and 2 * y2 + n2 - 2 < y2 - y1 + n2 and 2 * y2 + n2 - 2 < 1 and y2 - y1 + n2 > 1 \
            and n2 > 2 - 2 * y2 and 2 - 2 * y2 > 1 + y1 - y2:
        matches7 += 1

    # 222
    # 7/2880
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 > 1 \
            and x2 > y2 and y2 > y1 + n1 - n2 and 2 * y2 + n2 - n1 - 1 > 1 \
            and 2 * y2 + n2 - 2 < y2 - y1 + n2 and 2 * y2 + n2 - 2 < 1 and y2 - y1 + n2 > 1 \
            and n2 > 2 - 2 * y2 and 2 - 2 * y2 < 1 + y1 - y2:
        matches8 += 1

    # 1/192
    # 3
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 > 1 \
            and x2 > y2 and y2 > y1 + n1 - n2 and 2 * y2 + n2 - n1 - 1 < 1 and 2 * y2 + n2 - n1 - 1 > 0 \
            and x2 < 2 * y2 + n2 - n1 - 1 and n1 < y2 + n2 - 1 and y2 + n2 - 1 > 0 and y2 + n2 - 1 < 1 \
            and n1 > 2 * y2 + n2 - 2:
        matches9 += 1

    # 31
    # 1/384
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 > 1 \
            and x2 > y2 and y2 > y1 + n1 - n2 and 2 * y2 + n2 - n1 - 1 < 1 and 2 * y2 + n2 - n1 - 1 > 0 \
            and x2 < 2 * y2 + n2 - n1 - 1 and n1 < y2 + n2 - 1 and y2 + n2 - 1 > 0 and y2 + n2 - 1 < 1 \
            and n1 > 2 * y2 + n2 - 2 and 2 * y2 + n2 - 2 > 0 and 2 * y2 + n2 - 2 < 1:
        matches10 += 1

    # 321
    # 1/768
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 > 1 \
            and x2 > y2 and y2 > y1 + n1 - n2 and 2 * y2 + n2 - n1 - 1 < 1 and 2 * y2 + n2 - n1 - 1 > 0 \
            and x2 < 2 * y2 + n2 - n1 - 1 and n1 < y2 + n2 - 1 and y2 + n2 - 1 > 0 and y2 + n2 - 1 < 1 \
            and 2 * y2 + n2 - 2 < 0 and n2 > 1 - y2 and n2 < 2 - 2 * y2 and y2 > 1 / 2:
        matches11 += 1

    # 322
    # 1/768
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 > 1 \
            and x2 > y2 and y2 > y1 + n1 - n2 and 2 * y2 + n2 - n1 - 1 < 1 and 2 * y2 + n2 - n1 - 1 > 0 \
            and x2 < 2 * y2 + n2 - n1 - 1 and n1 < y2 + n2 - 1 and y2 + n2 - 1 > 0 and y2 + n2 - 1 < 1 \
            and 2 * y2 + n2 - 2 < 0 and n2 > 1 - y2 and y2 < 1 / 2:
        matches12 += 1

    # 1/96 0.01041530
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 > 1 \
            and x2 > y2:
        matches13 += 1

    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2:
        matches14 += 1

    # 1/480
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and x2 > y1 + n1 - n2 and x2 > 2 * y2 + n2 - n1 - 1 \
            and y1 + n1 - n2 < y2 and 2 * y2 + n2 - n1 - 1 > y2 \
            and 2 * y2 + n2 - n1 > 1 and n1 > 2 * y2 + n2 - 2 and 2 * y2 + n2 - 2 > 0 \
            and n1 < y2 + n2 - 1 and n2 > 2 - 2 * y2 and 2 - 2 * y2 > 1 - y2:
        matches15 += 1

    # 13/11520
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and x2 > y1 + n1 - n2 and x2 > 2 * y2 + n2 - n1 - 1 \
            and y1 + n1 - n2 < 0 and 2 * y2 + n2 - n1 - 1 > y2 and 2 * y2 + n2 - n1 - 1 < 1 \
            and 2 * y2 + n2 - 2 > 0 and n1 < y2 + n2 - 1 and y2 + n2 - 1 < n2 - y1 \
            and n2 > 2 - 2 * y2 and n2 > y1 and y1 < 2 - 2 * y2:
        matches16 += 1

    # 1/3840
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and x2 > y1 + n1 - n2 and x2 > 2 * y2 + n2 - n1 - 1 \
            and y1 + n1 - n2 < 0 and 2 * y2 + n2 - n1 - 1 > y2 and 2 * y2 + n2 - n1 - 1 < 1 \
            and 2 * y2 + n2 - 2 > 0 and n1 < y2 + n2 - 1 and y2 + n2 - 1 > n2 - y1 \
            and n2 > 2 - 2 * y2 and n2 > y1 and y1 < 2 - 2 * y2 and y2 > 1 / 2:
        matches17 += 1

    # 61/46080
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and x2 > y1 + n1 - n2 and x2 > 2 * y2 + n2 - n1 - 1 \
            and y1 + n1 - n2 < 0 and 2 * y2 + n2 - n1 - 1 > y2 and 2 * y2 + n2 - n1 - 1 < 1 \
            and 2 * y2 + n2 - 2 < 0 and n1 < y2 + n2 - 1 and n1 < n2 - y1 and y2 + n2 - 1 < n2 - y1 \
            and n2 > 1 - y2 and n2 < 2 - 2 * y2 and y1 < 1 - y2 and y2 > 1 / 2:
        matches18 += 1

    # 1/5120
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and x2 > y1 + n1 - n2 and x2 > 2 * y2 + n2 - n1 - 1 \
            and y1 + n1 - n2 < 0 and 2 * y2 + n2 - n1 - 1 > y2 and 2 * y2 + n2 - n1 - 1 < 1 \
            and 2 * y2 + n2 - 2 < 0 and n1 < y2 + n2 - 1 and n1 < n2 - y1 and y2 + n2 - 1 > n2 - y1 \
            and n2 > y1 and n2 < 2 - 2 * y2 and y1 > 1 - y2 and y2 > 1 / 2:
        matches19 += 1

    #
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and x2 > y1 + n1 - n2 and x2 > 2 * y2 + n2 - n1 - 1 \
            and y1 + n1 - n2 < 0 and 2 * y2 + n2 - n1 - 1 > y2 and 2 * y2 + n2 - n1 - 1 < 1 \
            and 2 * y2 + n2 - 2 < 0 and n1 < y2 + n2 - 1 and n1 < n2 - y1 and y2 + n2 - 1 > n2 - y1 \
            and n2 > y1 and n2 < 2 - 2 * y2 and n2 < 1 and y1 > 1 - y2 and y2 < 1 / 2:
        matches20 += 1

    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and x2 > y1 + n1 - n2 and x2 > 2 * y2 + n2 - n1 - 1 \
            and y1 + n1 - n2 < 0 and 2 * y2 + n2 - n1 - 1 > y2 and 2 * y2 + n2 - n1 - 1 < 1 \
            and 2 * y2 + n2 - 2 < 0 and n1 < y2 + n2 - 1 and n1 < n2 - y1 and y2 + n2 - 1 > n2 - y1:
        matches21 += 1

    # 181920
    # if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
    #         and x2 > y2 and x2 > y1 + n1 - n2 and x2 > 2 * y2 + n2 - n1 - 1 \
    #         and y1 + n1 - n2 < 0 and 2 * y2 + n2 - n1 - 1 > y2 and 2 * y2 + n2 - n1 - 1 < 1 \
    #         and 2 * y2 + n2 - 2 < 0 and n1 < y2 + n2 - 1 and n1 < n2 - y1:
    #     matches21 += 1

print("matches1 %.8f" % (matches1 / times))
print("matches2 %.8f" % (matches2 / times))
print("matches3 %.8f" % (matches3 / times))
print("matches4 %.8f" % (matches4 / times))
print("matches5 %.8f" % (matches5 / times))
print("matches6 %.8f" % (matches6 / times))
print("matches7 %.8f" % (matches7 / times))
print("matches8 %.8f" % (matches8 / times))
print("matches9 %.8f" % (matches9 / times))
print("matches10 %.8f" % (matches10 / times))
print("matches11 %.8f" % (matches11 / times))
print("matches12 %.8f" % (matches12 / times))
print("matches13 %.8f" % (matches13 / times))
print("matches14 %.8f" % (matches14 / times))
print("matches15 %.8f" % (matches15 / times))
print("matches16 %.8f" % (matches16 / times))
print("matches17 %.8f" % (matches17 / times))
print("matches18 %.8f" % (matches18 / times))
print("matches19 %.8f" % (matches19 / times))
print("matches20 %.8f" % (matches20 / times))
print("matches21 %.8f" % (matches21 / times))
