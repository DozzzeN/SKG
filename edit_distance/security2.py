import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

halfKeyLen = 64
points = np.zeros(halfKeyLen * 4 + 1)
lowers = np.zeros(halfKeyLen * 4 + 1)
for Np in range(0, halfKeyLen * 4 + 1):
    lower = pow(2, Np * 2)
    guess = 0
    lowers[Np] = lower
    for Na in range(0, halfKeyLen * 2 + 1):
        guess += comb(halfKeyLen * 2, Na) * comb(Np, Na)
    points[Np] = guess
    if guess >= lower:
        print(Np)
print(np.unravel_index(np.argmax(points), points.shape))
print(np.log2(points[256]))

reduce = np.zeros(240 + 1)
halfKeyLen = 64
for Np in range(45, 241):
    guess = 0
    for Na in range(10, 126):
        guess += comb(halfKeyLen * 2, Na) * comb(Np, Na)
    reduce[Np] = guess

plt.close()
plt.figure()
plt.grid(linestyle="--")
plt.xlabel("$N_p$", fontsize=12, fontweight='bold')
plt.ylabel("The number of guesses", fontsize=12, fontweight='bold')
plt.plot(range(len(points)), np.log2(points), color="blue", linewidth=2, label="$log_2(N_{guess})$")
plt.plot(range(len(points)), np.log2(lowers), color="green", linewidth=2, label="$log_2(2^{N_p})$")
x0 = 123
y0 = np.log2(points[x0])
plt.scatter(x0, y0, color="red")
plt.plot([x0, x0], [y0, 0], "r--")
plt.annotate("$(%d,%.2f)$" % (x0, y0), xy=(x0, y0), xytext=(+5, -30), textcoords='offset points', fontsize=12,
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
plt.legend(loc='upper left')
leg_text = plt.gca().get_legend().get_texts()
plt.setp(leg_text, fontsize=12, fontweight='bold')
plt.savefig('./evaluations/guess.pdf', format='pdf')
plt.show()

halfKeyLen = 64
lower = pow(2, halfKeyLen * 2)
points = np.zeros(halfKeyLen * 4 + 1)
lowers = np.zeros(halfKeyLen * 4 + 1)
for Np in range(0, halfKeyLen * 4 + 1):
    guess = 0
    lowers[Np] = lower
    for Na in range(0, halfKeyLen * 2 + 1):
        guess += comb(halfKeyLen * 2, Na) * comb(Np, Na)
    points[Np] = guess
    if guess >= lower:
        print(Np)
print(np.unravel_index(np.argmax(points), points.shape))
print(np.log2(points[256]))
plt.close()
plt.figure()
plt.grid(linestyle="--")
plt.xlabel("$N_p$", fontsize=12, fontweight='bold')
plt.ylabel("The number of guesses", fontsize=12, fontweight='bold')
plt.plot(range(len(points)), np.log2(points), color="blue", linewidth=2, label="$log_2(N_{guess})$")
plt.plot(range(len(points)), np.log2(lowers), color="green", linewidth=2, label="$log_2(2^N)$")
x0 = 40
y0 = np.log2(points[40])
plt.scatter(x0, y0, color="red")
plt.plot([x0, x0], [y0, 0], "r--")
plt.annotate("$(%d,%.2f)$" % (x0, y0), xy=(x0, y0), xytext=(+30, -30), textcoords='offset points', fontsize=12,
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
plt.legend(loc='upper left')
leg_text = plt.gca().get_legend().get_texts()
plt.setp(leg_text, fontsize=12, fontweight='bold')
plt.show()
