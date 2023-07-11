import numpy as np
from scipy.optimize import leastsq

A = np.array([[1, 4], [2, 5], [3, 6]])
y = np.array([1, 1, 1])

x = np.random.normal(0, 1, 3)
Q, R = np.linalg.qr(A)

print(Q.T @ Q)
print(np.linalg.norm(Q))
print(Q)

R_hat = np.array([[2.2361, -0.4472], [0, 3.2864]])
Q_hat = np.array([[0.4781, 0.8783], [0.8783, -0.4781]])
M = np.array([[-2, 3], [1, -1]])
print("R", Q_hat @ R_hat @ np.linalg.inv(M))

print("R", R)
# equal
print(np.linalg.norm(R), np.linalg.norm(A))

y_hat = Q.T @ y
print("y_hat", y_hat)
y_hat = np.array([1.6036, 0.6547])

def residuals(x, A, y):
    return y - np.dot(A, x)


xe = leastsq(residuals, np.random.binomial(1, 0.5, 2), args=(A, y))[0]
xe = [round(i) for i in xe]
re = residuals(xe, A, y)
print(np.linalg.norm(re))

xe = leastsq(residuals, np.random.binomial(1, 0.5, 2), args=(R, y_hat))[0]
print(xe)
print(np.linalg.norm(y_hat))

print(Q_hat.T @ y_hat)
xe = leastsq(residuals, np.random.binomial(1, 0.5, 2), args=(R_hat, Q_hat.T @ y_hat))[0]
print(xe)
