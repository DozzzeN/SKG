from random import randint

from algorithm import dtw_metric

a = [1, 2, 3, 4, 5, 6, 7]
b = [24, 25, 26, 27]

c = [1]
d = [3]
# print(dtw_metric(a, b))
# print(dtw_metric(c, a))


x1 = randint(0, 100)
x2 = randint(0, 100)
x3 = randint(0, 100)
x4 = randint(0, 100)
x5 = randint(0, 100)
x6 = randint(0, 100)
x7 = randint(0, 100)
x8 = randint(0, 100)
print(dtw_metric([x1, x2], [x5, x6]))
print(dtw_metric([x2, x1], [x6, x5]))
print([x1, x2, x3, x4], [x5, x6, x7, x8])

print(dtw_metric([x1, x2, x3, x4], [x5, x6, x7, x8]))
print(dtw_metric([x4, x3, x1, x2], [x8, x7, x5, x6]))

print(dtw_metric([x1, x2, x4, x3], [x5, x6, x8, x7]))

print(dtw_metric([x3, x4, x1, x2], [x7, x8, x5, x6]))
print(dtw_metric([x2, x1, x4, x3], [x6, x5, x8, x7]))
