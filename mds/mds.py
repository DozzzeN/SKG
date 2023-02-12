import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.manifold import MDS


class MyMDS:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, data):
        m, n = data.shape
        dist = np.zeros((m, m))
        disti = np.zeros(m)
        distj = np.zeros(m)
        B = np.zeros((m, m))
        for i in range(m):
            # 第i个元素与第j个元素的距离
            dist[i] = np.sum(np.square(data[i] - data), axis=1).reshape(1, m)
        for i in range(m):
            disti[i] = np.mean(dist[i, :])
            distj[i] = np.mean(dist[:, i])
        distij = np.mean(dist)
        for i in range(m):
            for j in range(m):
                B[i, j] = -0.5 * (dist[i, j] - disti[i] - distj[j] + distij)
        lamda, V = np.linalg.eigh(B)
        index = np.argsort(-lamda)[:self.n_components]
        diag_lamda = np.sqrt(np.diag(-np.sort(-lamda)[:self.n_components]))
        V_selected = V[:, index]
        Z = V_selected.dot(diag_lamda)
        return Z


iris = load_iris()

# clf1 = MyMDS(2)
# iris_t1 = clf1.fit(iris.data)
# plt.scatter(iris_t1[:, 0], iris_t1[:, 1], c=iris.target)
# plt.title('Using My MDS')
# plt.show()
#
# clf2 = MDS(2)
# clf2.fit(iris.data)
# iris_t2 = clf2.fit_transform(iris.data)
# plt.scatter(iris_t2[:, 0], iris_t2[:, 1], c=iris.target)
# plt.title('Using sklearn MDS')
# plt.show()

# # 测试自己的数据
coords = [
    [-6, 1, -5, 0],
    [-5, 1, -5, 0],
    [-7, -1, -5, 0],
    [-6, -2, -5, 0],
    [-5, -2, -5, 0],

    [-1, 2, -2, 3],
    [-3, 2, -2, 3],
    [-3, 4, -2, 3],
    [-1, 3, -2, 3],
    [-2, 1, -2, 3],

    [2, -1, 3, -2],
    [3, -1, 3, -2],
    [1, -2, 3, -2],
    [2, -3, 3, -2],
    [3, -3, 3, -2],

    [4, 1, 5, 0],
    [5, 2, 5, 0],
    [5, -1, 5, 0],
    [6, -1, 5, 0],
    [6, 2, 5, 0],

    [-3, -1, -1, -2],
    [-2, -3, -1, -2],
    [0, -2, -1, -2],
    [-2, 0, -1, -2],
    [-1, 0, -1, -2]
]
coords = np.array(coords)
res = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
plt.scatter(coords[:, 0], coords[:, 1], c=res)
plt.title('Coords1')
plt.show()
plt.scatter(coords[:, 1], coords[:, 2], c=res)
plt.title('Coords2')
plt.show()

clf3 = MyMDS(2)
iris_t3 = clf3.fit(coords)
plt.scatter(iris_t3[:, 0], iris_t3[:, 1], c=res)
plt.title('res')
plt.show()

clf4 = MDS(2)
clf4.fit(coords)
iris_t2 = clf4.fit_transform(coords)
plt.scatter(iris_t2[:, 0], iris_t2[:, 1], c=res)
plt.title('sklearn res')
plt.show()