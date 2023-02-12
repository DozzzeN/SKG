import numpy as np

times = 1000000

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

    if x1 > y1 and x1 < x2 + n2 - n1 \
            and x2 > y1 + n1 - n2 and x2 < y2 \
            and n1 > y2 + n2 - 1 and n1 > n2 - y1 \
            and y2 + n2 - 1 > 0 and n2 - y1 > 0 and y2 + n2 - 1 > n2 - y1:
        matches1 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 and x2 > y1 + n1 - n2 and x2 < y2 and n1 > y2 + n2 - 1 and \
            n2 > 1 + y1 - y2 and y1 > 1 - y2 and y2 > 1 / 2:
        matches2 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 and x2 > y1 + n1 - n2 and x2 < y2 and n1 > y2 + n2 - 1 and n1 < y2 + n2 - y1 and \
            n2 > y1 and y1 > y2 and y2 > 1 / 2:
        matches3 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 and x2 > y1 + n1 - n2 and x2 < y2 and n1 > y2 + n2 - 1 and n1 < y2 + n2 - y1 and \
            n2 > y1 and y1 > 1 - y2 and y2 < 1 / 2:
        matches4 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 and x2 > y1 + n1 - n2 and x2 < y2 and n1 > y2 + n2 - 1 and n1 < y2 + n2 - y1 and \
            n2 > y1 and n2 < 1 + y1 - y2 and y1 > 1 - y2 and y1 < y2 and y2 > 1 / 2:
        matches5 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 and x2 > y1 + n1 - n2 and x2 < y2 and n1 > y2 + n2 - 1 and n1 < y2 + n2 - y1 \
            and n2 > 1 - y2 and n2 < y1:
        matches6 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 \
            and x2 > y1 + n1 - n2 and x2 < y2 \
            and n1 > y2 + n2 - 1 and n1 > n2 - y1 \
            and y2 + n2 - 1 > 0 and n2 - y1 > 0 and n2 - y1 > y2 + n2 - 1:
        matches7 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 and x2 > y1 + n1 - n2 and x2 < y2 and n1 > n2 - y1 \
            and n2 > 1 + y1 - y2 and y1 < 1 - y2 and y2 > 1 / 2:
        matches8 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 and x2 > y1 + n1 - n2 and x2 < y2 and n1 > n2 - y1 \
            and n2 > 1 + y1 - y2 and y1 < y2 and y2 < 1 / 2:
        matches9 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 and x2 > y1 + n1 - n2 and x2 < y2 and n1 > n2 - y1 and n1 < y2 + n2 - y1 \
            and n2 > 1 - y2 and n2 < 1 + y1 - y2 and y1 < 1 - y2 and y2 > 1 / 2:
        matches10 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 and x2 > y1 + n1 - n2 and x2 < y2 and n1 > n2 - y1 and n1 < y2 + n2 - y1 \
            and n2 > 1 - y2 and n2 < 1 + y1 - y2 and y1 < y2 and y2 < 1 / 2:
        matches11 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 and x2 > y1 + n1 - n2 and x2 < y2 and n1 > n2 - y1 and n1 < y2 + n2 - y1 \
            and n2 > 1 - y2 and y1 > y2 and y1 < 1 - y2 and y2 < 1 / 2:
        matches12 += 1

    if x1 > y1 and x1 < x2 + n2 - n1 and x2 > y1 + n1 - n2 and x2 < y2 and n1 > n2 - y1 and n1 < y2 + n2 - y1 \
            and n2 - y1 > y2 + n2 - 1 and n2 > y1 and n2 < 1 - y2 and y1 < 1 - y2:
        matches13 += 1

    if x1 > y1 and x1 < x2 + n2 - n1 and x2 > y1 + n1 - n2 and x2 < y2 and n1 < y2 + n2 - y1 \
            and n2 - y1 > y2 + n2 - 1 and 0 > n2 - y1 and n2 < y1 and n2 > y1 - y2 and y1 < 1 - y2:
        matches14 += 1

    if x1 > y1 and x1 < x2 + n2 - n1 and x2 > y1 + n1 - n2 and x2 < y2 and n1 < y2 + n2 - y1 \
            and n2 - y1 > y2 + n2 - 1 and 0 > n2 - y1 and n2 < y1 and n2 > y1 - y2 and y1 < 1 - y2 and y1 > y2 and y2 / 1 / 2:
        matches15 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 and x2 > y1 + n1 - n2 and x2 < y2 and n1 < y2 + n2 - y1 \
            and n2 - y1 > y2 + n2 - 1 and 0 > n2 - y1 and n2 < y1 and y1 < y2 and y2 < 1 / 2:
        matches16 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 and x2 > y1 + n1 - n2 and x2 < y2 and n1 < y2 + n2 - y1 \
            and n2 - y1 > y2 + n2 - 1 and 0 > n2 - y1 and n2 < y1 and y1 < 1 - y2 and y2 > 1 / 2:
        matches17 += 1

    if x1 > y1 and x1 < x2 + n2 - n1 and x2 > y1 + n1 - n2 and x2 < y2 and n1 < y2 + n2 - y1 \
            and y2 + n2 - 1 > n2 - y1 and 0 > y2 + n2 - 1 and n2 < 1 - y2 and n2 > y1 - y2 and y1 > y2 and y2 > 1 / 2:
        matches18 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 and x2 > y1 + n1 - n2 and x2 < y2 and n1 < y2 + n2 - y1 \
            and y2 + n2 - 1 > n2 - y1 and 0 > y2 + n2 - 1 and n2 < 1 - y2 and n2 > y1 - y2 and y1 > 1 - y2 and y2 < 1 / 2:
        matches19 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 and x2 > y1 + n1 - n2 and x2 < y2 and n1 < y2 + n2 - y1 \
            and y2 + n2 - 1 > n2 - y1 and 0 > y2 + n2 - 1 and n2 < 1 - y2 and y1 > 1 - y2 and y1 < y2 and y2 > 1 / 2:
        matches20 += 1

    if x1 > y1 and x1 < x2 + n2 - n1 and x2 > y1 + n1 - n2 and x2 < y2 and n1 < y2 + n2 - y1 \
            and y2 + n2 - 1 > n2 - y1 and 0 > y2 + n2 - 1 and n2 < 1 - y2:
        matches21 += 1

    if x1 > y1 and x1 < x2 + n2 - n1 \
            and x2 > y1 + n1 - n2 and x2 < y2 and x2 + n2 - n1 < 1 and x2 + n2 - n1 > 0 \
            and y1 + n1 - n2 > 0 and y1 + n1 - n2 < 1 and 1 + n1 - n2 > y2:
        matches22 += 1

    if x1 > y1 and x1 < x2 + n2 - n1 \
            and x2 < y2 and x2 + n2 - n1 < 1 and x2 + n2 - n1 > 0 \
            and y1 + n1 - n2 < 0 and 1 + n1 - n2 > y2:
        matches23 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 and x2 < y2 and y2 + n2 - 1 > 0 and n1 > y2 + n2 - 1 \
            and n1 < n2 - y1 and n2 > 1 - y2 and y1 < 1 - y2:
        matches24 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 and x2 < y2 and y2 + n2 - 1 < 0 \
            and n1 < n2 - y1 and n2 < 1 - y2 and n2 > y1 and y1 < 1 - y2:
        matches25 += 1

    if x1 > y1 and x1 < x2 + n2 - n1 and x2 < 1 + n1 - n2 and y1 + n1 - n2 < 0 and 1 + n1 - n2 < y2 \
            and x2 + n2 - n1 < 1 and x2 + n2 - n1 > 0 and n1 < y2 + n2 - 1:
        matches26 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 and x2 < 1 + n1 - n2 and y1 + n1 - n2 < 0 and 1 + n1 - n2 < y2 \
            and x2 + n2 - n1 < 1 and x2 + n2 - n1 > 0 and n1 < y2 + n2 - 1 and y2 + n2 - 1 < n2 - y1 \
            and y2 + n2 - 1 > 0:
        matches27 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 and x2 < 1 + n1 - n2 and y1 + n1 - n2 < 0 and 1 + n1 - n2 < y2 \
            and x2 + n2 - n1 < 1 and x2 + n2 - n1 > 0 and n1 < n2 - y1 and y2 + n2 - 1 > n2 - y1 \
            and n2 - y1 > 0:
        matches28 += 1

    if x1 > y1 and x1 < x2 + n2 - n1 and y1 < x2 + n2 - n1 and x2 + n2 - n1 < 1 and x2 + n2 - n1 > 0 \
            and x2 < 1 + n1 - n2 and x2 > y1 + n1 - n2 \
            and y1 + n1 - n2 > 0 and n1 < y2 + n2 - 1 and y2 + n2 - 1 < 1 \
            and n2 > 1 - y2 and n2 < y1 and y1 > 1 - y2:
        matches29 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 and y1 < x2 + n2 - n1 and x2 + n2 - n1 < 1 and x2 + n2 - n1 > 0 \
            and x2 < 1 + n1 - n2 and x2 > y1 + n1 - n2 \
            and y1 + n1 - n2 > 0 and n1 < y2 + n2 - 1 and y2 + n2 - 1 < 1 and n1 > n2 - y1 and n2 > y1 \
            and y1 > 1 - y2:
        matches30 += 1
    if x1 > y1 and x1 < x2 + n2 - n1 and x2 + n2 - n1 < 1 and x2 + n2 - n1 > 0 \
            and 1 + n1 - n2 < y2 and x2 < 1 + n1 - n2 and x2 > y1 + n1 - n2 \
            and y1 + n1 - n2 > 0:
        matches31 += 1

    if x1 > y1 and x1 < x2 + n2 - n1 and x2 + n2 - n1 < 1 and x2 + n2 - n1 > 0 \
            and x2 < y2 and x2 < 1 + n1 - n2 and x2 > y1 + n1 - n2:
        matches32 += 1

    if x1 > y1 and x2 + n2 - n1 > 1 and x2 < y2 and x2 + n2 > y1 + n1 and x1 + n1 < x2 + n2:
        matchesall1 += 1

print("matches1 %.8f" % (matches1 / times))
print("matches2 %.8f" % (matches2 / times))
print("matches3 %.8f" % (matches3 / times))
print("matches4 %.8f" % (matches4 / times))
print("matches5 %.8f" % (matches5 / times))
print("%.8f" % ((matches2 + matches3 + matches4 + matches5) / times))  # matches1

print("matches6 %.8f" % (matches6 / times))

print("matches7 %.8f" % (matches7 / times))
print("matches8 %.8f" % (matches8 / times))
print("matches9 %.8f" % (matches9 / times))
print("matches10 %.8f" % (matches10 / times))
print("matches11 %.8f" % (matches11 / times))
print("matches12 %.8f" % (matches12 / times))
print("%.8f" % ((matches8 + matches9 + matches10 + matches11 + matches12) / times))  # matches7

print("matches13 %.8f" % (matches13 / times))

print("matches14 %.8f" % (matches14 / times))
print("matches15 %.8f" % (matches15 / times))
print("matches16 %.8f" % (matches16 / times))
print("matches17 %.8f" % (matches17 / times))
print("%.8f" % ((matches15 + matches16 + matches17) / times))  # matches14

print("matches18 %.8f" % (matches18 / times))
print("matches19 %.8f" % (matches19 / times))
print("matches20 %.8f" % (matches20 / times))
print("matches21 %.8f" % (matches21 / times))
print("%.8f" % ((matches18 + matches19 + matches20) / times))  # matches21

print("matches22 %.8f" % (matches22 / times))
print("%.8f" % ((matches1 + matches6 + matches7 + matches13 + matches14 + matches21) / times))  # matches22

print("matches23 %.8f" % (matches23 / times))
print("matches24 %.8f" % (matches24 / times))
print("matches25 %.8f" % (matches25 / times))
print("%.8f" % ((matches24 + matches25) / times))  # matches23

print("matches26 %.8f" % (matches26 / times))
print("matches27 %.8f" % (matches27 / times))
print("matches28 %.8f" % (matches28 / times))
print("%.8f" % ((matches27 + matches28) / times))  # matches26

print("matches29 %.8f" % (matches29 / times))
print("matches30 %.8f" % (matches30 / times))
print("matches31 %.8f" % (matches31 / times))
print("%.8f" % ((matches29 + matches30) / times))  # matches31

print("matches32 %.8f" % (matches32 / times))
print("%.8f" % ((matches22 + matches23 + matches26 + matches31) / times))  # matches32

print("matchesall1 %.8f" % (matchesall1 / times))
print("all %.8f" % ((matchesall1 + matches22 + matches23 + matches26 + matches31) / times))  # all
