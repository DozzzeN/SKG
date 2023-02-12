import math

from scipy.special import comb

m = 76
for m in range(1, 128):
    times1 = float(math.factorial(m))
    times2 = 0
    for mr in range(0, m + 1):
        times2 += comb(m, mr) * comb(2 * m, mr)
    # print(comb(3 * m, m))
    # print(times2)
    if times1 > times2:
        print(m)

# SA-SKG的64bits相当于128bits加密强度
print(comb(3 * 64, 64))
print(float(pow(2, 128)))
# BM-SKG的64bits相当于256bits加密强度
print(float(math.factorial(64)))
print(float(pow(2, 256)))
