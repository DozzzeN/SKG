import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

halfKeyLen = 64
lower = pow(2, halfKeyLen * 2)
guess2 = 0
points = np.zeros([halfKeyLen * 4 + 1, halfKeyLen * 2 + 1])
lowers = np.zeros([halfKeyLen * 4 + 1, halfKeyLen * 2 + 1])
for Na in range(0, halfKeyLen * 2 + 1):
    for Np in range(0, halfKeyLen * 4 + 1):
        lowers[Np][Na] = math.log10(pow(2, Na * 2))
        tmp = comb(halfKeyLen * 2, Na) * comb(Np, Na)
        if tmp != 0:
            tmp = math.log10(tmp)
        points[Np][Na] = tmp
        guess2 += tmp
print(np.unravel_index(np.argmax(points), points.shape))
x = np.arange(0, halfKeyLen * 2 + 1, 1)
y = np.arange(0, halfKeyLen * 4 + 1, 1)
plt.close()
plt.figure()
X, Y = np.meshgrid(x, y)
ax = plt.axes(projection='3d')
ax.set_xlabel("$N_a$")
ax.set_ylabel("$N_p$")
ax.set_zlabel("The number of guesses")
# ax.set_zticks(np.power(10, [0, 20, 40, 60, 80, 100]))
p = ax.plot_surface(X, Y, points, color='b', alpha=1, cmap='rainbow')
ax.plot_surface(X, Y, lowers, color='r', alpha=1)

# plt.colorbar(p, shrink=0.5)
plt.savefig('./evaluations/guess-3D.pdf', format='pdf')
plt.show()
print(math.pow(2, 256))
