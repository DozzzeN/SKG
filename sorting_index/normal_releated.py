import numpy as np
from matplotlib import pyplot as plt

n = 1000000

isShow = False

a = []
b = []
c = []
d = []
chi2 = []
expo = []
lap = []
for i in range(n):
    a.append(np.random.normal(0, 1))
    b.append(np.random.normal(0, 1))
    c.append(np.random.normal(0, 1))
    d.append(np.random.normal(0, 1))
    chi2.append(np.random.chisquare(1) / 2)
    expo.append(np.random.exponential(1))
    lap.append(np.random.laplace(0, 1))
p1 = (np.array(a) + np.array(b)) / 2  # N(0,1/2)
p1 = np.square(p1)  # chi-square(1)/2
p2 = (np.array(c) + np.array(d)) / 2
p2 = np.square(p2)
print("mean", np.mean(p1), np.mean(chi2))
print("var", np.var(p1), np.var(chi2))

if isShow:
    plt.figure()
    plt.hist(p1, bins=50)
    plt.show()

    plt.figure()
    plt.hist(chi2, bins=50)
    plt.show()

q1 = np.array(p1) + np.array(p2)  # exp(1)
print()
print("mean", np.mean(q1), np.mean(expo))
print("var", np.var(q1), np.var(expo))

if isShow:
    plt.figure()
    plt.hist(q1, bins=50, color="r")
    plt.show()

    plt.figure()
    plt.hist(expo, bins=50, color="r")
    plt.show()

p3 = (np.array(a) - np.array(b)) / 2  # N(0,1/2)
p3 = np.square(p3)  # chi-square(1)/2
p4 = (np.array(c) - np.array(d)) / 2
p4 = np.square(p4)
print()
print("mean", np.mean(p3), np.mean(chi2))
print("var", np.var(p3), np.var(chi2))

q2 = np.array(p3) + np.array(p4)  # exp(1)
print()
print("mean", np.mean(q2), np.mean(expo))
print("var", np.var(q2), np.var(expo))

r = q1 - q2  # laplace(0, 1)
print()
print("mean", np.mean(r), np.mean(lap))
print("var", np.var(r), np.var(lap))

if isShow:
    plt.figure()
    plt.hist(r, bins=50, color="k")
    plt.show()

    plt.figure()
    plt.hist(lap, bins=50, color="k")
    plt.show()

