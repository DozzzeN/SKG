import osqp
import numpy as np
from scipy import sparse

# 定义线性系统
A = sparse.csr_matrix([[1., 2.], [3., 4.]])
b = np.array([1, 1])

# 转换为标准形式
P = sparse.block_diag([A.T @ A, sparse.csc_matrix((2, 2))])
q = np.concatenate([-A.T @ b, np.zeros(2)])

# 构建OSQP问题
prob = osqp.OSQP()
prob.setup(P=P, q=q, alpha=1.0)
prob.update_settings(scaled_termination=True)
res = prob.solve()

# 解向量前半部分即为x
x = res.x[:2]
print(x)
