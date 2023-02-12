import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import gamma

x = np.linspace(0, 10, 1000)
plt.plot(x, gamma(x), label='Factorial')
plt.plot(x, 2 ** x, label='$2^x$')
plt.plot(x, (1.618 ** x - (-0.618 ** x)) / math.sqrt(5), label='$Fibonacci$')

plt.legend()
plt.show()
