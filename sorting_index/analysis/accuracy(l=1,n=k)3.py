import math
import sys

import matplotlib.pyplot as plt
from scipy.spatial import distance
import numpy as np

ubs = []
flbs = []
fubs = []
rs = []
s = 6
e = 10

for k in range(s, e):
    l = 1
    n = l * k

    ubs.append(math.factorial(k) / np.power(k, k))

    flbs.append(np.power(2, n) * math.factorial(n) / math.factorial(2 * n) * np.power(1 / 2, n - 1) * math.factorial(n))
    if n % 2 == 0:
        fubs.append(np.power(2, n) * math.factorial(n) / math.factorial(2 * n) * \
              np.power(1 / 2, int((n - 2) / 2)) * math.factorial(n))
    else:
        fubs.append(np.power(2, n) * math.factorial(n) / math.factorial(2 * n) * \
              np.power(1 / 2, int((n - 1) / 2)) * math.factorial(n))

    rs.append(math.factorial(n) / math.factorial(2 * n) * math.sqrt(2) / 4 * \
    (math.pow(1 + math.sqrt(2), n + 1) - math.pow(1 - math.sqrt(2), n + 1)) / np.power(2, n - 1) * math.factorial(n))

plt.figure()
plt.plot(np.arange(s, e), flbs, "red")
plt.plot(np.arange(s, e), fubs, "blue")
plt.plot(np.arange(s, e), ubs, "black")
plt.plot(np.arange(s, e), rs, "cyan")
plt.show()
