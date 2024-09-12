import numpy as np
import scipy.linalg
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from scipy.linalg import circulant, matrix_balance, toeplitz


# a = np.array([[1, 0, -1],
#               [0, 1, 0],
#               [1, 0, 1]])
# b = np.random.normal(0, 1, (10, 10))
# b1 = circulant(np.random.normal(0, 1, 10))
# balanced = matrix_balance(b)[0]
# print(np.linalg.cond(b))
# print(np.linalg.cond(b1))
# print(np.linalg.cond(balanced))
# # a = circulant(np.random.normal(0, 1, 10))
# u, s, v = np.linalg.svd(a)
# print(s)

np.random.seed(0)
a = np.random.normal(0, 1, 5)
# a = toeplitz(a)
a = circulant(a)
# u, s, v = np.linalg.svd(a)
# u, s, v = np.linalg.svd(a @ a.T)
d = np.diag(1 / np.sqrt(s))
k = 1
v_max = v.T[:, :k]
u_max = u[:, :k]
a = u @ d @ np.diag(s) @ d @ v
u, s, v = np.linalg.svd(a @ a.T)
a_pca = a @ v_max
# u_max = v_max
a_pca1 = u_max * s[: k]
a_pca2 = v_max * s[: k]
print(a_pca, a_pca1, a_pca2)
# print(a)
a_rec = u_max * s[: k] @ u_max.T
a_guess = u_max @ u_max.T
# a与a_rec差别很大：无法通过最大的特征向量恢复出原始数据
print(pearsonr(a.flatten(), a_rec.flatten())[0])
print(pearsonr(a.flatten(), a_guess.flatten())[0])
print(pearsonr(a[:, 0], a_rec[:, 0])[0])
print(pearsonr(a[:, 0], a_guess[:, 0])[0])
print(pearsonr(a[0], a_rec[0])[0])
print(pearsonr(a[0], a_guess[0])[0])
# print(a_pca, a_pca1, a_pca2)

exit()

def common_pca(ha, hb, k):
    ha = ha - np.mean(ha, axis=0)
    hb = hb - np.mean(hb, axis=0)
    rha = np.cov(ha, rowvar=False)
    eig_value = np.linalg.eig(rha)[0]
    eig_vector = np.linalg.eig(rha)[1]
    max_k_vectors_a = eig_vector[:, np.argsort(eig_value)[-k:]]
    ha_pca = max_k_vectors_a.T @ ha.T
    hb_pca = max_k_vectors_a.T @ hb.T
    return np.array(ha_pca), np.array(hb_pca)


def private_pca(ha, hb, k):
    ha = ha - np.mean(ha, axis=0)
    hb = hb - np.mean(hb, axis=0)
    rha = np.cov(ha, rowvar=False)
    rhb = np.cov(hb, rowvar=False)
    eig_value_a = np.linalg.eig(rha)[0]
    eig_vector_a = np.linalg.eig(rha)[1]
    eig_value_b = np.linalg.eig(rhb)[0]
    eig_vector_b = np.linalg.eig(rhb)[1]
    max_k_vectors_a = eig_vector_a[:, np.argsort(eig_value_a)[-k:]]
    max_k_vectors_b = eig_vector_b[:, np.argsort(eig_value_b)[-k:]]
    ha_pca = max_k_vectors_a.T @ ha.T
    hb_pca = max_k_vectors_b.T @ hb.T
    return np.array(ha_pca), np.array(hb_pca)


n = 10
k = 6
hau = np.random.normal(0, 1, (n, n))
hbu = hau + np.random.normal(0, 0.1, (n, n))
print(pearsonr(hau.flatten(), hbu.flatten())[0])
ya, yb = common_pca(hau, hbu, k)
print(pearsonr(ya.flatten(), yb.flatten())[0])
ya_prv, yb_prv = private_pca(hau, hbu, k)
print(pearsonr(ya_prv.flatten(), yb_prv.flatten())[0])

k = 2
pca = PCA(n_components=k)
# hau = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2, 1.6], [1, 1.1], [1.5, 1.6],
#                 [1.1, 0.9]])
# hau = circulant([1, 2, 3])
hau = np.random.normal(0, 1, (10, 2))
x_pca_prime = pca.fit_transform(hau)
print(pca.explained_variance_ratio_)
# 利用特征值分解手动实现
hau1 = hau - np.mean(hau, axis=0)
rha1 = np.cov(hau1, rowvar=False)
eig_value = np.linalg.eig(rha1)[0]
eig_vector = np.linalg.eig(rha1)[1]
max_vector = eig_vector[:, np.argsort(eig_value)[-1:-(k + 1):-1]]
max_vector = max_vector.reshape(k, k)
# x_pca = max_vector.T @ hau1.T
x_pca = hau1 @ max_vector
print()
print(x_pca)
print(x_pca_prime)

# 利用SVD手动实现
print("hau1", hau1)
U, S, Vt = np.linalg.svd(hau1, full_matrices=False)
U = U[:, : k]
hau_recover = U @ np.diag(np.random.normal(0, 1, 2)) @ Vt[: k, :]
print(pearsonr(hau_recover.flatten(), hau1.flatten())[0])
U *= S[: k]
print(hau1 @ Vt.T[:, : k])
print(U)
hau_recover = U @ np.diag(np.random.normal(0, 1, 2)) @ Vt[: k, :]
print("hau_recover", hau_recover)
print(U @ np.array(Vt.T[:, : k]).T)
# print(pearsonr(hau_recover.flatten(), hau1.flatten())[0])

# projected_var = np.var(x_pca_prime, axis=0).sum()
# print(projected_var)
# random_proj = np.random.rand(2, 2)
# X_random = np.dot(hau, random_proj.T)
# print(np.var(X_random, axis=0).sum())
# print(np.trace(np.array(hau1 @ max_vector @ np.array(hau1 @ max_vector).T)))
# max_vector = np.random.normal(0, 1, (2, 1))
# print(np.trace(np.array(hau1 @ max_vector @ np.array(hau1 @ max_vector).T)))
