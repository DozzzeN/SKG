import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

M = 128
lowers = np.zeros(M + 1)
for m in range(len(lowers)):
    lowers[m] = pow(2, m)

Mp = 128
points128 = np.zeros(M + 1)
flag = True
for m in range(M + 1):
    tmp = 0
    for Mr in range(0, m + 1):
        tmp += comb(m, Mr) * comb(Mp, Mr)
    points128[m] = tmp
    if flag and points128[m] < lowers[m]:
        print("128", m)
        flag = False

Mp = 96
points96 = np.zeros(M + 1)
flag = True
for m in range(M + 1):
    tmp = 0
    for Mr in range(0, m + 1):
        tmp += comb(m, Mr) * comb(Mp, Mr)
    points96[m] = tmp
    if flag and points96[m] < lowers[m]:
        print("96", m)
        flag = False

Mp = 64
points64 = np.zeros(M + 1)
flag = True
for m in range(M + 1):
    tmp = 0
    for Mr in range(0, m + 1):
        tmp += comb(m, Mr) * comb(Mp, Mr)
    points64[m] = tmp
    if flag and points64[m] < lowers[m]:
        print("64", m)
        flag = False

Mp = 48
points48 = np.zeros(M + 1)
flag = True
for m in range(M + 1):
    tmp = 0
    for Mr in range(0, m + 1):
        tmp += comb(m, Mr) * comb(Mp, Mr)
    points48[m] = tmp
    if flag and points48[m] < lowers[m]:
        print("48", m)
        flag = False

Mp = 32
points32 = np.zeros(M + 1)
flag = True
for m in range(M + 1):
    tmp = 0
    for Mr in range(0, m + 1):
        tmp += comb(m, Mr) * comb(Mp, Mr)
    points32[m] = tmp
    if flag and points32[m] < lowers[m]:
        print("32", m)
        flag = False


plt.close()
plt.figure()
plt.grid(linestyle="--")
plt.xlabel("$M$", fontsize=12, fontweight='bold')
plt.ylabel("The number of guesses", fontsize=12, fontweight='bold')
plt.yscale('log')
# plt.scatter(range(len(points224)), points224, s=5, marker=".", color="b", linewidth=2, label="$N_{guess}$, $N_p=224$")
# plt.scatter(range(len(points192)), points192, s=5, marker=">", color="g", linewidth=2, label="$N_{guess}$, $N_p=192$")
# plt.scatter(range(len(points160)), points160, s=5, marker="<", color="k", linewidth=2, label="$N_{guess}$, $N_p=160$")
# plt.scatter(range(len(points128)), points128, s=5, marker="v", color="c", linewidth=2, label="$N_{guess}$, $N_p=128$")
# plt.scatter(range(len(points64)), points64, s=5, marker="^", color="m", linewidth=2, label="$N_{guess}$, $N_p=64$")
# plt.scatter(range(len(lowers)), lowers, s=5, marker="+", color="r", linewidth=2, label="$2^{2N_r}$")

x0 = 104
y0 = points32[x0]
plt.scatter(x0, y0, color="red")
# plt.plot([x0, x0], [y0, 0], "r--")
plt.annotate("$(%d,%.2e)$" % (x0, y0), xy=(x0, y0), xytext=(-60, -40), textcoords='offset points', fontsize=10,
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
plt.plot(range(len(points128)), points128, linestyle='--', color="b", linewidth=2, label="$M_{guess}^{SA}$, $M_p=128$")
plt.plot(range(len(points96)), points96, linestyle='-', color="m", linewidth=2, label="$M_{guess}^{SA}$, $M_p=96$")
plt.plot(range(len(points64)), points64, linestyle=':', color="c", linewidth=2, label="$M_{guess}^{SA}$, $M_p=64$")
plt.plot(range(len(points48)), points48, linestyle='-.', color="g", linewidth=2, label="$M_{guess}^{SA}$, $M_p=48$")
plt.plot(range(len(points32)), points32, linestyle='--', color="k", linewidth=2, label="$M_{guess}^{SA}$, $M_p=32$")
plt.plot(range(len(lowers)), lowers, color="r", linewidth=2, label="$2^M$")
plt.ylim(1.66223336e-11, 1.268869e+97)

plt.legend(loc='upper left')
leg_text = plt.gca().get_legend().get_texts()
plt.setp(leg_text, fontsize=10, fontweight='bold')
plt.savefig('./evaluations/nAllPlot_ed.pdf', format='pdf')
plt.show()