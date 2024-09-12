import numpy as np
from scipy.linalg import circulant
from scipy.optimize import leastsq

PB = circulant([1, 2.2, 1.7, 2])
PA = circulant([0.9, 1.9, 2.2, 1.3])

xA = np.array([1, 0, 1, 1])
bA = np.dot(PA, xA)


def residuals(x, tmpMulA1, PA):
    return tmpMulA1 - np.dot(PA, x)


xB = leastsq(residuals, np.random.binomial(1, 0.5, len(PB)), args=(bA, PB))[0]
print("xA", xA)
# xB中有些元素接近0.5，导致舍入误差
print("xB", xB)

# xB将这些接近0.5的元素（第一位）随机舍入为0或1，得到eB
eB = np.array([0, 0, 1, 1])
# Bob可以利用自己的测量值隐藏eB，得到PBeB，并将其发送给Alice
PBeB = np.dot(PB, eB)

# Alice通过求解方程组得到eA，分析eA与xA的差异，得到那些被Bob舍入的元素（或者Bob直接将那些元素的位置索引发送给Alice）
eA = leastsq(residuals, np.random.binomial(1, 0.5, len(PA)), args=(PBeB, PA))[0]
print("eA", eA)

# Alice利用自己的部分测量值sPA（第一位+新的一位）重新隐藏真实的密钥xA，得到sxA，并将其发送给Bob
sPB = circulant([1, 2.2])
sPA = circulant([0.9, 1.9])
sxA = np.array([1, 0])
sbA = np.dot(sPA, sxA)
# Bob通过求解方程组得到sxB，将其整合入xB，得到xB'
sxB = leastsq(residuals, np.random.binomial(1, 0.5, len(sPB)), args=(sbA, sPB))[0]
print("sxA", sxA)
print("sxB", sxB)
