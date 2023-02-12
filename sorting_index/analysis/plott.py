import matplotlib.pyplot as plt
import numpy as np

y = []
for i in range(100000):
    y.append(np.random.normal(0, 1))

plt.figure()
plt.hist(y, bins=50)
plt.show()