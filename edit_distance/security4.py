import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

halfKeyLen = 64
# 第一张图
Np = 224
NrBound = 128
points = np.zeros(NrBound + 1)
lowers = np.zeros(NrBound + 1)
flag = True
for Nr in range(224 - 128, NrBound + 1):
    tmp = 0
    for _Nr in range(224 - 128, Nr + 1):
        tmp += comb(halfKeyLen * 2, _Nr) * comb(Np, _Nr)
    points[Nr] = tmp
    lowers[Nr] = pow(2, Nr * 2)
    if flag and points[Nr] <= lowers[Nr]:
        print(Nr)
        flag = False

plt.close()
plt.figure()
plt.grid(linestyle="--")
plt.xlabel("$N_r$", fontsize=12, fontweight='bold')
plt.ylabel("The number of guesses", fontsize=12, fontweight='bold')
plt.yscale('log')
plt.plot(range(len(points)), points, color="blue", linewidth=2, label="$N_{guess}$")
plt.plot(range(len(points)), lowers, color="green", linewidth=2, label="$2^{2N_r}$")
# x0 = 124
# y0 = points[x0]
# plt.scatter(x0, y0, color="red")
# plt.plot([x0, x0], [y0, 0], "r--")
# plt.annotate("$(%d,%.2e)$" % (x0, y0), xy=(x0, y0), xytext=(-100, -60), textcoords='offset points', fontsize=10,
#              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
plt.legend(loc='upper left')
leg_text = plt.gca().get_legend().get_texts()
plt.setp(leg_text, fontsize=12, fontweight='bold')
plt.savefig('./evaluations/np224.pdf', format='pdf')
# plt.show()

# 第二张图
Np = 192
NrBound = 128
points = np.zeros(NrBound + 1)
lowers = np.zeros(NrBound + 1)
flag = True
for Nr in range(192 - 128, NrBound + 1):
    tmp = 0
    for _Nr in range(192 - 128, Nr + 1):
        tmp += comb(halfKeyLen * 2, _Nr) * comb(Np, _Nr)
    points[Nr] = tmp
    lowers[Nr] = pow(2, Nr * 2)
    if flag and points[Nr] < lowers[Nr]:
        print(Nr)
        flag = False

plt.close()
plt.figure()
plt.grid(linestyle="--")
plt.xlabel("$N_r$", fontsize=12, fontweight='bold')
plt.ylabel("The number of guesses", fontsize=12, fontweight='bold')
plt.yscale('log')
plt.plot(range(len(points)), points, color="blue", linewidth=2, label="$N_{guess}$")
plt.plot(range(len(points)), lowers, color="green", linewidth=2, label="$2^{2N_r}$")
# x0 = 117
# y0 = points[x0]
# plt.scatter(x0, y0, color="red")
# plt.plot([x0, x0], [y0, 0], "r--")
# plt.annotate("$(%d,%.2e)$" % (x0, y0), xy=(x0, y0), xytext=(-100, -60), textcoords='offset points', fontsize=10,
#              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
plt.legend(loc='upper left')
leg_text = plt.gca().get_legend().get_texts()
plt.setp(leg_text, fontsize=12, fontweight='bold')
plt.savefig('./evaluations/np192.pdf', format='pdf')
# plt.show()

# 第二张图
Np = 160
NrBound = 128
points = np.zeros(NrBound + 1)
lowers = np.zeros(NrBound + 1)
flag = True
for Nr in range(160 - 128, NrBound + 1):
    tmp = 0
    for _Nr in range(160 - 128, Nr + 1):
        tmp += comb(halfKeyLen * 2, _Nr) * comb(Np, _Nr)
    points[Nr] = tmp
    lowers[Nr] = pow(2, Nr * 2)
    if flag and points[Nr] < lowers[Nr]:
        print(Nr)
        flag = False

plt.close()
plt.figure()
plt.grid(linestyle="--")
plt.xlabel("$N_r$", fontsize=12, fontweight='bold')
plt.ylabel("The number of guesses", fontsize=12, fontweight='bold')
plt.yscale('log')
plt.plot(range(len(points)), points, color="blue", linewidth=2, label="$N_{guess}$")
plt.plot(range(len(points)), lowers, color="green", linewidth=2, label="$2^{2N_r}$")
# x0 = 109
# y0 = points[x0]
# plt.scatter(x0, y0, color="red")
# plt.plot([x0, x0], [y0, 0], "r--")
# plt.annotate("$(%d,%.2e)$" % (x0, y0), xy=(x0, y0), xytext=(-100, -60), textcoords='offset points', fontsize=10,
#              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
plt.legend(loc='upper left')
leg_text = plt.gca().get_legend().get_texts()
plt.setp(leg_text, fontsize=12, fontweight='bold')
plt.savefig('./evaluations/np160.pdf', format='pdf')
# plt.show()

# 第三张图
Np = 128
NrBound = 128
points = np.zeros(NrBound + 1)
lowers = np.zeros(NrBound + 1)
flag = True
for Nr in range(0, NrBound + 1):
    tmp = 0
    for _Nr in range(0, Nr + 1):
        tmp += comb(halfKeyLen * 2, _Nr) * comb(Np, _Nr)
    points[Nr] = tmp
    lowers[Nr] = pow(2, Nr * 2)
    if flag and points[Nr] < lowers[Nr]:
        print(Nr)
        flag = False

plt.close()
plt.figure()
plt.grid(linestyle="--")
plt.xlabel("$N_r$", fontsize=12, fontweight='bold')
plt.ylabel("The number of guesses", fontsize=12, fontweight='bold')
plt.yscale('log')
plt.plot(range(len(points)), points, color="blue", linewidth=2, label="$N_{guess}$")
plt.plot(range(len(points)), lowers, color="green", linewidth=2, label="$2^{2N_r}$")
# x0 = 98
# y0 = points[x0]
# plt.scatter(x0, y0, color="red")
# plt.plot([x0, x0], [y0, 0], "r--")
# plt.annotate("$(%d,%.2e)$" % (x0, y0), xy=(x0, y0), xytext=(-100, -60), textcoords='offset points', fontsize=10,
#              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
plt.legend(loc='upper left')
leg_text = plt.gca().get_legend().get_texts()
plt.setp(leg_text, fontsize=12, fontweight='bold')
plt.savefig('./evaluations/np128.pdf', format='pdf')
# plt.show()

# 第四张图
Np = 64
NrBound = 64
points = np.zeros(NrBound + 1)
lowers = np.zeros(NrBound + 1)
flag = True
for Nr in range(0, NrBound + 1):
    tmp = 0
    for _Nr in range(0, Nr + 1):
        tmp += comb(halfKeyLen * 2, _Nr) * comb(Np, _Nr)
    points[Nr] = tmp
    lowers[Nr] = pow(2, Nr * 2)
    if flag and points[Nr] < lowers[Nr]:
        print(Nr)
        flag = False

plt.close()
plt.figure()
plt.grid(linestyle="--")
plt.xlabel("$N_r$", fontsize=12, fontweight='bold')
plt.ylabel("The number of guesses", fontsize=12, fontweight='bold')
plt.yscale('log')
plt.plot(range(len(points)), points, color="blue", linewidth=2, label="$N_{guess}$")
plt.plot(range(len(points)), lowers, color="green", linewidth=2, label="$2^{2N_r}$")
# x0 = 64
# y0 = points[x0]
# plt.scatter(x0, y0, color="red")
# plt.plot([x0, x0], [y0, 0], "r--")
# plt.annotate("$(%d,%.2e)$" % (x0, y0), xy=(x0, y0), xytext=(-100, -60), textcoords='offset points', fontsize=10,
#              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
plt.legend(loc='upper left')
leg_text = plt.gca().get_legend().get_texts()
plt.setp(leg_text, fontsize=12, fontweight='bold')
plt.savefig('./evaluations/np64.pdf', format='pdf')
# plt.show()