import numpy as np

for i in range(10):
    print(i)
    a = np.random.randint(0, 1, (3, 3))
    b = np.random.randint(0, 1, (3, 3))
    try:
        np.linalg.cholesky(a)
    except:
        print("a not positive definite")
    try:
        np.linalg.cholesky(b)
    except:
        print("b not positive definite")
    sigma = 1
    rho = 1
    KKT_matrix = np.block([[a, b.T], [b, -np.eye(3) * rho]])
    try:
        np.linalg.cholesky(KKT_matrix)
    except:
        print("KKT matrix not positive definite")
    print(np.linalg.matrix_rank(a), np.linalg.matrix_rank(b), np.linalg.matrix_rank(KKT_matrix))

