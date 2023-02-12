import itertools

import numpy as np

times = 1000000

# a and b为真的所有情况
bools = [[0, 0], [0, 3], [1, 1], [1, 2], [2, 1], [2, 2], [3, 0], [3, 3]]
matches = 0
matches1 = 0
matches11 = 0
matches111 = 0
matches112 = 0
matches12 = 0
matches2 = 0
matches21 = 0
matches211 = 0
matches212 = 0
matches22 = 0
matches221 = 0
matches222 = 0
matches3 = 0
matches31 = 0
matches311 = 0
matches312 = 0
matches32 = 0
matches4 = 0
matches41 = 0
matches411 = 0
matches412 = 0
matches42 = 0
matches421 = 0
matches422 = 0
a = 0
b = 10
ma = 0
mb = 100

for t in range(times):
    x1 = np.random.uniform(a, b)
    x2 = np.random.uniform(a, b)
    y1 = np.random.uniform(a, b)
    y2 = np.random.uniform(a, b)
    m1 = np.random.uniform(ma, mb)
    m2 = np.random.uniform(ma, mb)

    z11 = np.abs(x1 - y1)
    z21 = np.abs(m1 - m2)
    z12 = np.abs(x2 - y2)
    z22 = np.abs(m1 - m2)

    _a = z11 < z21
    _b = z12 < z22

    a1 = x1 > y1 and m1 > m2
    a2 = x1 < y1 and m1 < m2
    a3 = x1 > y1 and m1 < m2
    a4 = x1 < y1 and m1 > m2
    a1234 = [a1, a2, a3, a4]

    b1 = x2 > y2 and m1 > m2
    b2 = x2 < y2 and m1 < m2
    b3 = x2 > y2 and m1 < m2
    b4 = x2 < y2 and m1 > m2
    b1234 = [b1, b2, b3, b4]

    # if _a and _b and a1 and b1:
    #     matches += 1

    # 341/3000
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2:
        matches += 1

    # 19/6000
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 < b and m1 - m2 + y2 < b:
        matches1 += 1

    # 1/6000
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 < b and m1 - m2 + y2 < b and m2 + b - a > mb and ma < mb - (b - a):
        matches11 += 1

    # 1/6000
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 < b and m1 - m2 + y2 < b and m2 + b - a > mb and ma < mb - (b - a) \
            and mb - (b - 1) > ma:
        matches111 += 1

    # 0
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 < b and m1 - m2 + y2 < b and m2 + b - a > mb and ma < mb - (b - a) \
            and mb - (b - 1) < ma:
        matches112 += 1

    # 3/1000
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 < b and m1 - m2 + y2 < b and m2 + b - a < mb:
        matches12 += 1

    # 7/3000
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 < b and m1 - m2 + y2 > b:
        matches2 += 1

    # 1/12000
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 < b and m1 - m2 + y2 > b and m2 + b - a > mb and ma < mb - (b - a):
        matches21 += 1

    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 < b and m1 - m2 + y2 > b and m2 + b - a > mb and ma < mb - (b - a) \
            and b - (m1 - m2) > a:
        matches211 += 1

    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 < b and m1 - m2 + y2 > b and m2 + b - a > mb and ma < mb - (b - a) \
            and b - (m1 - m2) < a:
        matches212 += 1

    # 9/4000
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 < b and m1 - m2 + y2 > b and m2 + b - a < mb:
        matches22 += 1

    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 < b and m1 - m2 + y2 > b and m2 + b - a < mb \
            and b - (m1 - m2) > a:
        matches221 += 1

    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 < b and m1 - m2 + y2 > b and m2 + b - a < mb \
            and b - (m1 - m2) < a:
        matches222 += 1

    # 7/3000
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 > b and m1 - m2 + y2 < b:
        matches3 += 1

    # 1/12000
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 > b and m1 - m2 + y2 < b and m2 + b - a > mb and ma < mb - (b - a):
        matches31 += 1

    # 1/12000
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 > b and m1 - m2 + y2 < b and m2 + b - a > mb and ma < mb - (b - a) \
            and b - (m1 - m2) > a:
        matches311 += 1

    # 0
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 > b and m1 - m2 + y2 < b and m2 + b - a > mb and ma < mb - (b - a) \
            and b - (m1 - m2) < a:
        matches312 += 1

    # 9/4000
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 > b and m1 - m2 + y2 < b and m2 + b - a < mb and b - (m1 - m2) > a:
        matches32 += 1

    # 127/1200
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 > b and m1 - m2 + y2 > b:
        matches4 += 1

    # 1/12000
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 > b and m1 - m2 + y2 > b and m2 + b - a > mb and ma < mb - (b - a):
        matches41 += 1

    # 1/12000
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 > b and m1 - m2 + y2 > b and m2 + b - a > mb and ma < mb - (b - a) \
            and b - (m1 - m2) > a:
        matches411 += 1

    # 0
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 > b and m1 - m2 + y2 > b and m2 + b - a > mb and ma < mb - (b - a) \
            and b - (m1 - m2) < a:
        matches412 += 1

    # 423/4000
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 > b and m1 - m2 + y2 > b and m2 + b - a < mb:
        matches42 += 1

    # 9/2000
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 > b and m1 - m2 + y2 > b and m2 + b - a < mb \
            and b - (m1 - m2) > a:
        matches421 += 1

    # 81/800
    if x1 > y1 and x2 > y2 and m1 > m2 and x1 - y1 < m1 - m2 and x2 - y2 < m1 - m2 \
            and m1 - m2 + y1 > b and m1 - m2 + y2 > b and m2 + b - a < mb \
            and b - (m1 - m2) < a:
        matches422 += 1

print("matches", matches / times, (matches1 + matches2 + matches3 + matches4) / times)
print("matches1", matches1 / times, (matches11 + matches12) / times)
print("matches11", matches11 / times)
print("matches111", matches111 / times)
print("matches112", matches112 / times)
print("matches12", matches12 / times)
print("matches2", matches2 / times, (matches21 + matches22) / times)
print("matches21", matches21 / times)
print("matches211", matches211 / times)
print("matches212", matches212 / times)
print("matches22", matches22 / times)
print("matches221", matches221 / times)
print("matches222", matches222 / times)
print("matches3", matches3 / times, (matches31 + matches32) / times)
print("matches31", matches31 / times)
print("matches311", matches311 / times)
print("matches312", matches312 / times)
print("matches32", matches32 / times)
print("matches4", matches4 / times, (matches41 + matches42) / times)
print("matches41", matches41 / times)
print("matches411", matches411 / times)
print("matches412", matches412 / times)
print("matches42", matches42 / times)
print("matches421", matches421 / times)
print("matches422", matches422 / times)
