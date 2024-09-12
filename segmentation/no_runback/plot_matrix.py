import numpy as np
from matplotlib import pyplot as plt

N = 8
transformMatrix = np.random.normal(0, 1, (N, N))
plt.figure()
plt.imshow(transformMatrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()

replace = np.eye(N)
replace[0, 0] = 0.5
replace[1, 1] = 1.5
transformMatrix = transformMatrix @ replace
plt.figure()
plt.imshow(transformMatrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()
