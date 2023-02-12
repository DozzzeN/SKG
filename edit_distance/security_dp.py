import math

import matplotlib.pyplot as plt
import numpy as np

M = 64
points = np.zeros(M + 1)
lowers = np.zeros(M + 1)
for m in range(len(lowers)):
    lowers[m] = pow(2, m)
    points[m] = math.factorial(m)

plt.close()
plt.figure()
plt.grid(linestyle="--")
plt.xlabel("$M$", fontsize=12, fontweight='bold')
plt.ylabel("The number of guesses", fontsize=12, fontweight='bold')
plt.yscale('log')

plt.plot(range(len(points)), points, linestyle='--', color="k", linewidth=2, label="$M_{guess}^{BM}$")
plt.plot(range(len(lowers)), lowers, color="r", linewidth=2, label="$2^M$")
plt.ylim(1.66223336e-11, 1.268869e+97)

plt.legend(loc='upper left')
leg_text = plt.gca().get_legend().get_texts()
plt.setp(leg_text, fontsize=10, fontweight='bold')
plt.savefig('./evaluations/nAllPlot_dp.pdf', format='pdf')
plt.show()