import numpy as np
from scipy.linalg import circulant


def unitary_invariant_norm(A):
    if len(A) == 0 or len(A[0]) == 0:
        return 0
    else:
        # max(np.linalg.svd(A)[1])
        return np.linalg.norm(A, ord=2)


def Phi(F):
    if len(F) == 0 or len(F[0]) == 0:
        return 0
    else:
        f = np.linalg.svd(F)[1]
        return np.max(f / (np.sqrt(1 + f ** 2)))


#  perturbation of matrix inverses
A = np.random.randn(3, 3)
# A = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])  # rank(A) = 2
E = 0.01 * np.random.randn(3, 3)
B = A + E
kappa_bar = unitary_invariant_norm(A) * unitary_invariant_norm(np.linalg.pinv(B))
# equation 2.16
print(unitary_invariant_norm(np.linalg.pinv(B) - np.linalg.pinv(A)) / unitary_invariant_norm(np.linalg.pinv(A))
      <= kappa_bar * unitary_invariant_norm(E) / unitary_invariant_norm(A))
# equation 2.18
print(unitary_invariant_norm(np.linalg.pinv(A)) * unitary_invariant_norm(E) < 1)
kappa = unitary_invariant_norm(A) * unitary_invariant_norm(np.linalg.pinv(A))
gamma = 1 - kappa * unitary_invariant_norm(E) / unitary_invariant_norm(A)  # gamma > 0
# equation 2.19
print(unitary_invariant_norm(np.linalg.pinv(B)) <= unitary_invariant_norm(np.linalg.pinv(A)) / gamma)

# Projections
Astar = np.linalg.pinv(A)
PA = A @ Astar
PA = np.diag(np.linalg.svd(PA)[1])
RA = Astar @ A
RA = np.diag(np.linalg.svd(RA)[1])
U, S, V, = np.linalg.svd(A)
# U.T @ A @ V.T = np.diag(S)
U = U.T
V = V.T
A11 = np.diag(S)
E11 = U @ E @ V
B11 = U @ B @ V
if np.linalg.matrix_rank(A) != len(S):
    E12 = E11[:np.linalg.matrix_rank(A), np.linalg.matrix_rank(A):]
    E21 = E11[np.linalg.matrix_rank(A):, :np.linalg.matrix_rank(A)]
    E22 = E11[np.linalg.matrix_rank(A):, np.linalg.matrix_rank(A):]
    A11 = A11[:np.linalg.matrix_rank(A), :np.linalg.matrix_rank(A)]
    E11 = E11[:np.linalg.matrix_rank(A), :np.linalg.matrix_rank(A)]
    B11 = B11[:np.linalg.matrix_rank(A), :np.linalg.matrix_rank(A)]
else:
    E12 = np.zeros((np.linalg.matrix_rank(A), 0))
    E21 = np.zeros((0, np.linalg.matrix_rank(A)))
    E22 = np.zeros((0, 0))
print(np.allclose(B11, A11 + E11))
print(np.allclose(Astar, V @ np.linalg.pinv(A11) @ U))

print(np.isclose(unitary_invariant_norm(PA @ A @ RA), unitary_invariant_norm(A11)))
print(np.isclose(unitary_invariant_norm(PA @ E @ RA), unitary_invariant_norm(E11)))
PAorth = np.eye(3) - PA
RAorth = np.eye(3) - RA
Bstar = np.linalg.pinv(B)
PB = B @ Bstar
RB = Bstar @ B
PBorth = np.eye(3) - PB
RBorth = np.eye(3) - RB
# theorem 2.3
print(np.isclose(unitary_invariant_norm(PA @ PBorth), unitary_invariant_norm(PB @ PAorth)))
print(np.isclose(unitary_invariant_norm(PB - PA), unitary_invariant_norm(PB @ RAorth)))
print(unitary_invariant_norm(PB - PA) < 1, np.linalg.matrix_rank(A) == np.linalg.matrix_rank(B))
# theorem 2.4
print(np.allclose(PB @ PAorth, B.T @ RB @ E.T @ PAorth))
print(np.allclose(RBorth @ RA, -Astar @ E @ RBorth))
# theorem 2.5
# acute
print("isAcute:", np.linalg.matrix_rank(A) == np.linalg.matrix_rank(B) == np.linalg.matrix_rank(PA @ B @ RA))

# The pseudo-inverse
# theorem 3.1
print(unitary_invariant_norm(Bstar - Astar) >= 1 / unitary_invariant_norm(E))
print(unitary_invariant_norm(Bstar) >= 1 / unitary_invariant_norm(E))
# theorem 3.2
print(np.allclose(Bstar - Astar,
                  -Bstar @ PB @ E @ RA @ Astar + Bstar @ PB @ PAorth - RBorth @ RA @ Astar))
print(np.allclose(Bstar - Astar,
                  -Bstar @ PB @ E @ RA @ Astar + np.linalg.pinv(B.T @ B) @ RB @ E.T @ PAorth
                  - RBorth @ E.T @ PA @ np.linalg.pinv(A @ A.T)))
# theorem 3.3
print(unitary_invariant_norm(Bstar - Astar) <= 3 * max(unitary_invariant_norm(Astar) ** 2,
                                                       unitary_invariant_norm(Bstar) ** 2) * unitary_invariant_norm(E))
# theorem 3.4
print(unitary_invariant_norm(Bstar - Astar) <= unitary_invariant_norm(Astar) *
      unitary_invariant_norm(Bstar) * unitary_invariant_norm(E))
# 3.9
print(unitary_invariant_norm(Bstar - Astar) / unitary_invariant_norm(Bstar) <= kappa * unitary_invariant_norm(E)
      / unitary_invariant_norm(A))
# theorem 3.7
F21 = E21 @ np.linalg.pinv(B11)
F12 = np.linalg.pinv(B11) @ E12
print(np.allclose(Bstar[:np.linalg.matrix_rank(A), :np.linalg.matrix_rank(A)],
                  np.linalg.pinv(B11)))
print(np.allclose(Bstar[:np.linalg.matrix_rank(A), :np.linalg.matrix_rank(A)],
                  np.linalg.pinv(np.hstack((np.eye(np.linalg.matrix_rank(A)), F12))) @ np.linalg.pinv(B11)
                  @ np.linalg.pinv(np.vstack((np.eye(np.linalg.matrix_rank(A)), F21)))))
# theorem 3.8
kappa_bar = unitary_invariant_norm(A) / unitary_invariant_norm(np.linalg.pinv(B11))
print(unitary_invariant_norm(Bstar - Astar) / unitary_invariant_norm(Astar) <= kappa_bar * unitary_invariant_norm(E11)
      / unitary_invariant_norm(A) + Phi(kappa_bar + E12 / unitary_invariant_norm(A)) +
      Phi(kappa_bar * E21 / unitary_invariant_norm(A)))
print(unitary_invariant_norm(Bstar - Astar) / unitary_invariant_norm(Astar) <= kappa_bar *
      (unitary_invariant_norm(E11) + unitary_invariant_norm(E12) + unitary_invariant_norm(E21)) /
      unitary_invariant_norm(A))

# Projection
print(unitary_invariant_norm(PB - PA) <= kappa_bar * unitary_invariant_norm(E21) / unitary_invariant_norm(A) /
      np.sqrt(1 + (kappa_bar * unitary_invariant_norm(E21) / unitary_invariant_norm(A)) ** 2))
print(
    unitary_invariant_norm(PB - PA) <= kappa_bar * unitary_invariant_norm(PAorth @ E @ RA) / unitary_invariant_norm(A) /
    np.sqrt(1 + (kappa_bar * unitary_invariant_norm(PAorth @ E @ RA) / unitary_invariant_norm(A)) ** 2))

# The linear least squares problem
# theorem 5.2
print("5.2")
b = np.random.randn(3, 1)
x = Astar @ b
h = Bstar @ b - x
b1 = b
b2 = []
kappa_bar = unitary_invariant_norm(A) * unitary_invariant_norm(np.linalg.pinv(B))
eta = unitary_invariant_norm(A) * unitary_invariant_norm(x) / unitary_invariant_norm(b1)
print(unitary_invariant_norm(h) / unitary_invariant_norm(x) <=
      kappa_bar * unitary_invariant_norm(E11) / unitary_invariant_norm(A) +
      Phi(kappa_bar * E12 / unitary_invariant_norm(A)) +
      kappa_bar ** 2 * unitary_invariant_norm(E12) / unitary_invariant_norm(A) * (
              eta * unitary_invariant_norm(b2) / unitary_invariant_norm(b1) +
              unitary_invariant_norm(E21) / unitary_invariant_norm(A)))
r = b - A @ x
xhat = Bstar @ b
rhat = b - B @ xhat
rbar = b - A @ xhat
print(np.allclose(rhat, PBorth @ b))
print(unitary_invariant_norm(rhat - r) <= unitary_invariant_norm(PB - PA) * unitary_invariant_norm(b))
print(unitary_invariant_norm(r) <= unitary_invariant_norm(rbar) <= unitary_invariant_norm(r) +
      unitary_invariant_norm(E) * (unitary_invariant_norm(x) + unitary_invariant_norm(xhat)))
# theorem 5.3
print("5.3")
rhat = b - A @ xhat
print(unitary_invariant_norm(rhat) ** 2, unitary_invariant_norm(r) ** 2)
epsilon = np.sqrt(unitary_invariant_norm(rhat) ** 2 - unitary_invariant_norm(r) ** 2)
E = epsilon * xhat.T / unitary_invariant_norm(xhat) ** 2
print(np.isclose(unitary_invariant_norm(E), epsilon / unitary_invariant_norm(xhat)))
print(unitary_invariant_norm(b - (A + E) @ xhat))
