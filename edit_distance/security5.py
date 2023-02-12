import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

halfKeyLen = 64
Np = 224
NrBound = 128
points224 = np.zeros(NrBound + 1)
empty224 = np.zeros(224 - 128 + 1)
lowers = np.zeros(NrBound + 1)
for Nr in range(len(lowers)):
    lowers[Nr] = pow(2, Nr * 2)
flag = True
for Nr in range(len(empty224)):
    tmp = 0
    for _Nr in range(0, Nr + 1):
        tmp += comb(halfKeyLen * 2, _Nr) * comb(Np, _Nr)
    empty224[Nr] = tmp
for Nr in range(0, 224 - 128):
    points224[Nr] = None
for Nr in range(224 - 128, NrBound + 1):
    tmp = 0
    for _Nr in range(224 - 128, Nr + 1):
        tmp += comb(halfKeyLen * 2, _Nr) * comb(Np, _Nr)
    points224[Nr] = tmp
    if flag and points224[Nr] <= lowers[Nr]:
        print(Nr)
        flag = False

Np = 192
NrBound = 128
points192 = np.zeros(NrBound + 1)
flag = True
for Nr in range(0, 192 - 128):
    points192[Nr] = None
for Nr in range(192 - 128, NrBound + 1):
    tmp = 0
    for _Nr in range(192 - 128, Nr + 1):
        tmp += comb(halfKeyLen * 2, _Nr) * comb(Np, _Nr)
    points192[Nr] = tmp
    if flag and points192[Nr] < lowers[Nr]:
        print(Nr)
        flag = False

Np = 160
NrBound = 128
points160 = np.zeros(NrBound + 1)
flag = True
for Nr in range(0, 160 - 128):
    points160[Nr] = None
for Nr in range(160 - 128, NrBound + 1):
    tmp = 0
    for _Nr in range(160 - 128, Nr + 1):
        tmp += comb(halfKeyLen * 2, _Nr) * comb(Np, _Nr)
    points160[Nr] = tmp
    if flag and points160[Nr] < lowers[Nr]:
        print(Nr)
        flag = False

Np = 128
NrBound = 128
points128 = np.zeros(NrBound + 1)
flag = True
for Nr in range(0, NrBound + 1):
    tmp = 0
    for _Nr in range(0, Nr + 1):
        tmp += comb(halfKeyLen * 2, _Nr) * comb(Np, _Nr)
    points128[Nr] = tmp
    if flag and points128[Nr] < lowers[Nr]:
        print(Nr)
        flag = False

Np = 64
NrBound = 64
points64 = np.zeros(NrBound + 1)
flag = True
for Nr in range(0, NrBound + 1):
    tmp = 0
    for _Nr in range(0, Nr + 1):
        tmp += comb(halfKeyLen * 2, _Nr) * comb(Np, _Nr)
    points64[Nr] = tmp
    if flag and points64[Nr] < lowers[Nr]:
        print(Nr)
        flag = False

plt.close()
plt.figure()
plt.grid(linestyle="--")
plt.xlabel("$N_r$", fontsize=12, fontweight='bold')
plt.ylabel("The number of guesses", fontsize=12, fontweight='bold')
plt.yscale('log')
# plt.scatter(range(len(points224)), points224, s=5, marker=".", color="b", linewidth=2, label="$N_{guess}$, $N_p=224$")
# plt.scatter(range(len(points192)), points192, s=5, marker=">", color="g", linewidth=2, label="$N_{guess}$, $N_p=192$")
# plt.scatter(range(len(points160)), points160, s=5, marker="<", color="k", linewidth=2, label="$N_{guess}$, $N_p=160$")
# plt.scatter(range(len(points128)), points128, s=5, marker="v", color="c", linewidth=2, label="$N_{guess}$, $N_p=128$")
# plt.scatter(range(len(points64)), points64, s=5, marker="^", color="m", linewidth=2, label="$N_{guess}$, $N_p=64$")
# plt.scatter(range(len(lowers)), lowers, s=5, marker="+", color="r", linewidth=2, label="$2^{2N_r}$")

plt.plot(range(len(points224)), points224, linestyle='--', color="b", linewidth=2, label="$N_{guess}$, $N_p=224$")
plt.plot(range(len(points192)), points192, linestyle='-.', color="g", linewidth=2, label="$N_{guess}$, $N_p=192$")
plt.plot(range(len(points160)), points160, linestyle='--', color="k", linewidth=2, label="$N_{guess}$, $N_p=160$")
plt.plot(range(len(points128)), points128, linestyle=':', color="c", linewidth=2, label="$N_{guess}$, $N_p=128$")
plt.plot(range(len(points64)), points64, linestyle='--', color="m", linewidth=2, label="$N_{guess}$, $N_p=64$")
plt.plot(range(len(lowers)), lowers, color="r", linewidth=2, label="$2^{2N_r}$")

plt.legend(loc='lower right')
leg_text = plt.gca().get_legend().get_texts()
plt.setp(leg_text, fontsize=10, fontweight='bold')
plt.savefig('./evaluations/npAllPlot.pdf', format='pdf')
# plt.show()
