# https://www.geeksforgeeks.org/box-cox-transformation-using-python/
# Python3 code to show Box-cox Transformation
# of non-normal data
import math

import numpy as np
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

# generate non-normal data (exponential)
from scipy.io import loadmat

# original_data = np.random.normal(loc=0, scale=5, size=1000)
original_data = loadmat("../data/data_mobile_outdoor_1.mat")['A'][:, 0]
original_data = original_data - min(original_data) + 0.01
# original_data = original_data[0:1000]
# original_data = np.random.rayleigh(1, 1000)

# transform training data & save lambda value
fitted_data, fitted_lambda = stats.boxcox(original_data)
# fitted_data = 1 - np.exp(-0.5 * np.square(original_data))

plt.figure()
plt.hist(original_data)
plt.figure()
plt.hist(fitted_data)
plt.show()
