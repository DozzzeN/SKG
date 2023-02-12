import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])

ham = []
for x in a:
    index_a = np.where(a == x)[0][0]
    err = np.where(b == x)
    index_b = np.where(b == x)[0][0]
    print(index_a, index_b)
    ham.append(np.abs(index_a - index_b))

print(ham)

a = [12.58378775, 12.2859848, 11.64121736, 12.50178824, 9.50391816, 9.58861573, 9.46127459, 12.25449839, 12.73959942,
     12.76004837, 12.74863575, 9.48160511, 9.56005422, 9.44755524, 12.64927593, 12.72727809]
b = 9.56005422

print(np.where(np.array(a) == np.array(b))[0][0])
