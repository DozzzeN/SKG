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
matches191 = 0
matches192 = 0
matches20 = 0
matches21 = 0
matches211 = 0
matches212 = 0
matches2121 = 0
matches2122 = 0
matches22 = 0
matches221 = 0
matches222 = 0
matches2221 = 0
matches2222 = 0
matches23 = 0
matches231 = 0
matches2311 = 0
matches2312 = 0
matches232 = 0
matches24 = 0
matches25 = 0
matches26 = 0
matches27 = 0
matches28 = 0
matches29 = 0
matches30 = 0
matches31 = 0
matches311 = 0
matches312 = 0
matches32 = 0
matches321 = 0
matches322 = 0
matches33 = 0
matches34 = 0
matches35 = 0
matches36 = 0

for t in range(times):
    n1 = np.random.rand()
    n2 = np.random.rand()
    x1 = np.random.rand()
    x2 = np.random.rand()
    y1 = np.random.rand()
    y2 = np.random.rand()

    # 0.02635680
    # matches14=matches15+matches16
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2:
        matches14 += 1

    # 0.01938550
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 < y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2:
        matches16 += 1

    # 0.00697130
    # matches15=matches17+matches18
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2:
        matches15 += 1

    # 0.00120610
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and y1 + n1 - n2 > 0:
        matches17 += 1

    # 0.00576520
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and y1 + n1 - n2 < 0:
        matches18 += 1

    # 0.00072090
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and y1 + n1 - n2 > 0 and 2 * y2 + n2 - 2 > 0:
        matches19 += 1

    # 0.00067820 1/1440
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and y1 + n1 - n2 > 0 and 2 * y2 + n2 - 2 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - 2 \
            and 2 * y2 + n2 - y1 - n1 > 1:
        matches191 += 1

    # 0.00004270
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and y1 + n1 - n2 > 0 and 2 * y2 + n2 - 2 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - 2 \
            and 2 * y2 + n2 - y1 - n1 < 1:
        matches192 += 1

    # 0.00048520
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and y1 + n1 - n2 > 0 and 2 * y2 + n2 - 2 < 0:
        matches20 += 1

    a = 2 * y2 + n2 - 2
    b = n2 - y1
    # 0.00025520 1/3888
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and a > b and b > 0 and y1 + n1 - n2 > 0 and 2 * y2 + n2 - 2 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - 2 \
            and 2 * y2 + n2 - y1 - n1 > 1:
        matches21 += 1

    # 0.00023440 41/174960
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and a > b and b > 0 and y1 + n1 - n2 > 0 and 2 * y2 + n2 - 2 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - 2 \
            and 2 * y2 + n2 - y1 - n1 > 1 and 2 * y2 + n2 - y1 - 1 > y2 + n2 - 1:
        matches211 += 1

    # 0.00001380 1/65610
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and a > b and b > 0 and y1 + n1 - n2 > 0 and 2 * y2 + n2 - 2 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - 2 \
            and 2 * y2 + n2 - y1 - n1 > 1 and 2 * y2 + n2 - y1 - 1 < y2 + n2 - 1 and y2 > 2 / 3:
        matches2121 += 1

    # 0.00000760 1/131220
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and a > b and b > 0 and y1 + n1 - n2 > 0 and 2 * y2 + n2 - 2 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - 2 \
            and 2 * y2 + n2 - y1 - n1 > 1 and 2 * y2 + n2 - y1 - 1 < y2 + n2 - 1 and y2 < 2 / 3:
        matches2122 += 1

    # 0.00002080 1/43740
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and a > b and b > 0 and y1 + n1 - n2 > 0 and 2 * y2 + n2 - 2 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - 2 \
            and 2 * y2 + n2 - y1 - n1 > 1 and 2 * y2 + n2 - y1 - 1 < y2 + n2 - 1:
        matches212 += 1

    # 0.00010650 1/9720
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and a > 0 and 0 > b and y1 + n1 - n2 > 0 and 2 * y2 + n2 - 2 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - 2 \
            and 2 * y2 + n2 - y1 - n1 > 1:
        matches22 += 1

    # 0.00008450 1/11664
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and a > 0 and 0 > b and y1 + n1 - n2 > 0 and 2 * y2 + n2 - 2 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - 2 \
            and 2 * y2 + n2 - y1 - n1 > 1 and 2 * y2 + n2 - y1 - 1 > y2 + n2 - 1:
        matches221 += 1

    # 0.00001890 1/58320
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and a > 0 and 0 > b and y1 + n1 - n2 > 0 and 2 * y2 + n2 - 2 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - 2 \
            and 2 * y2 + n2 - y1 - n1 > 1 and 2 * y2 + n2 - y1 - 1 < y2 + n2 - 1:
        matches222 += 1

    # 0.00000240 1/524880
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and a > 0 and 0 > b and y1 + n1 - n2 > 0 and 2 * y2 + n2 - 2 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - 2 \
            and 2 * y2 + n2 - y1 - n1 > 1 and 2 * y2 + n2 - y1 - 1 < y2 + n2 - 1 and y2 < 2 / 3:
        matches2221 += 1

    # 0.00001610 1/65610
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and a > 0 and 0 > b and y1 + n1 - n2 > 0 and 2 * y2 + n2 - 2 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - 2 \
            and 2 * y2 + n2 - y1 - n1 > 1 and 2 * y2 + n2 - y1 - 1 < y2 + n2 - 1 and y2 > 2 / 3:
        matches2222 += 1

    # 0.00032870 13/38880
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and b > a and a > 0 and y1 + n1 - n2 > 0 and 2 * y2 + n2 - 2 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - 2 \
            and 2 * y2 + n2 - y1 - n1 > 1:
        matches23 += 1

    # 0.00030110 85/279936
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and b > a and a > 0 and y1 + n1 - n2 > 0 and 2 * y2 + n2 - 2 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - 2 \
            and 2 * y2 + n2 - y1 - n1 > 1 and 2 * y2 + n2 - y1 - 1 > y2 + n2 - 1:
        matches231 += 1

    # 0.00006390 283/4199040
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and b > a and a > 0 and y1 + n1 - n2 > 0 and 2 * y2 + n2 - 2 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - 2 \
            and 2 * y2 + n2 - y1 - n1 > 1 and 2 * y2 + n2 - y1 - 1 > y2 + n2 - 1 \
            and y2 < 2 / 3 and y1 > 1 - y2:
        matches2311 += 1

    # 0.00023720 31/131220
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and b > a and a > 0 and y1 + n1 - n2 > 0 and 2 * y2 + n2 - 2 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - 2 \
            and 2 * y2 + n2 - y1 - n1 > 1 and 2 * y2 + n2 - y1 - 1 > y2 + n2 - 1 \
            and y2 > 2 / 3 and y1 > 1 - y2:
        matches2312 += 1

    # 0.00003100 43/1399680
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and b > a and a > 0 and y1 + n1 - n2 > 0 and 2 * y2 + n2 - 2 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - 2 \
            and 2 * y2 + n2 - y1 - n1 > 1 and 2 * y2 + n2 - y1 - 1 < y2 + n2 - 1:
        matches232 += 1

    a = 2 * y2 + n2 - y1 - 1
    b = n2 - y1
    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and a > 0 and 0 > b and y1 + n1 - n2 > 0 and 2 * y2 + n2 - y1 - 1 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - y1 - 1 \
            and 2 * y2 + n2 - y1 - n1 < 1:
        matches31 += 1

    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and a > 0 and 0 > b and y1 + n1 - n2 > 0 and 2 * y2 + n2 - y1 - 1 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - y1 - 1 \
            and 2 * y2 + n2 - y1 - n1 < 1 and 1 - y2 < 1 + y1 - 2 * y2:
        matches312 += 1

    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and a > b and b > 0 and y1 + n1 - n2 > 0 and 2 * y2 + n2 - y1 - 1 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - y1 - 1 \
            and 2 * y2 + n2 - y1 - n1 < 1:
        matches32 += 1

    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and a > b and b > 0 and y1 + n1 - n2 > 0 and 2 * y2 + n2 - y1 - 1 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - y1 - 1 \
            and 2 * y2 + n2 - y1 - n1 < 1 and 1 > 2 + y1 - 2 * y2:
        matches321 += 1

    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and a > b and b > 0 and y1 + n1 - n2 > 0 and 2 * y2 + n2 - y1 - 1 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - y1 - 1 \
            and 2 * y2 + n2 - y1 - n1 < 1 and 1 < 2 + y1 - 2 * y2:
        matches322 += 1

    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and b > a and a > 0 and y1 + n1 - n2 > 0 and 2 * y2 + n2 - y1 - 1 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - y1 - 1 \
            and 2 * y2 + n2 - y1 - n1 < 1:
        matches33 += 1

    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and b > 0 and 0 > a and y1 + n1 - n2 > 0 and 2 * y2 + n2 - y1 - 1 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - y1 - 1 \
            and 2 * y2 + n2 - y1 - n1 < 1:
        matches34 += 1

    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and 0 > a and a > b and y1 + n1 - n2 > 0 and 2 * y2 + n2 - y1 - 1 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - y1 - 1 \
            and 2 * y2 + n2 - y1 - n1 < 1:
        matches35 += 1

    if x1 > y1 and 2 * y2 + n2 - n1 - x2 < 1 and x1 < 2 * y2 + n2 - n1 - x2 \
            and x2 > y2 and 2 * y2 + n2 - n1 - 1 > y2 and x2 > 2 * y2 + n2 - n1 - 1 and x2 > y1 + n1 - n2 \
            and 0 > b and b > a and y1 + n1 - n2 > 0 and 2 * y2 + n2 - y1 - 1 > 0 and n1 > n2 - y1 and n1 > 2 * y2 + n2 - y1 - 1 \
            and 2 * y2 + n2 - y1 - n1 < 1:
        matches36 += 1

print("matches14 %.8f" % (matches14 / times))
print("matches16 %.8f" % (matches16 / times))
print("matches15 %.8f" % (matches15 / times))
print("all %.8f" % ((matches17 + matches18) / times), matches15 / times)
print("matches17 %.8f" % (matches17 / times))
print("matches18 %.8f" % (matches18 / times))
print("matches19 %.8f" % (matches19 / times))
print("matches191 %.8f" % (matches191 / times))
print("matches192 %.8f" % (matches192 / times))
print("all %.8f" % ((matches191 + matches192) / times), matches19 / times)
print("matches20 %.8f" % (matches20 / times))
print("all %.8f" % ((matches19 + matches20) / times), matches17 / times)
print("matches21 %.8f" % (matches21 / times))
print("matches211 %.8f" % (matches211 / times))
print("matches212 %.8f" % (matches212 / times))
print("matches2121 %.8f" % (matches2121 / times))
print("matches2122 %.8f" % (matches2122 / times))
print("all %.8f" % ((matches2121 + matches2122) / times), matches212 / times)
print("all %.8f" % ((matches211 + matches212) / times), matches21 / times)
print("matches22 %.8f" % (matches22 / times))
print("matches221 %.8f" % (matches221 / times))
print("matches222 %.8f" % (matches222 / times))
print("matches2221 %.8f" % (matches2221 / times))
print("matches2222 %.8f" % (matches2222 / times))
print("all %.8f" % ((matches2221 + matches2222) / times), matches222 / times)
print("all %.8f" % ((matches221 + matches222) / times), matches22 / times)
print("matches23 %.8f" % (matches23 / times))
print("matches231 %.8f" % (matches231 / times))
print("matches2311 %.8f" % (matches2311 / times))
print("matches2312 %.8f" % (matches2312 / times))
print("all %.8f" % ((matches2311 + matches2312) / times), matches231 / times)
print("matches232 %.8f" % (matches232 / times))
print("all %.8f" % ((matches231 + matches232) / times), matches23 / times)
print("all %.8f" % ((matches21 + matches22 + matches23) / times), matches191 / times)
print("matches31 %.8f" % (matches31 / times))
print("matches312 %.8f" % (matches312 / times))
print("matches32 %.8f" % (matches32 / times))
print("matches33 %.8f" % (matches33 / times))
print("matches34 %.8f" % (matches34 / times))
print("matches35 %.8f" % (matches35 / times))
print("matches36 %.8f" % (matches36 / times))
print("matches321 %.8f" % (matches321 / times))
print("matches322 %.8f" % (matches322 / times))
print("all %.8f" % ((matches321 + matches321) / times), matches32 / times)
print("all %.8f" % ((matches31 + matches32 + matches33 + matches34 + matches35 + matches36) / times),
      matches192 / times)
