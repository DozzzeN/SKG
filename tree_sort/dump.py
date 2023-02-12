import numpy as np
from scipy.stats import pearsonr

from algorithm import moving_sum, successive_moving_sum, dtw_metric

a_rectify = [3904.8216100703803, 3947.5988753358497, 3962.334181307614, 3969.746264374093, 3984.478372218144,
             3997.1240637044048, 4038.663982051852, 4017.224327544175]
b_rectify = [3832.6149443660197, 3869.401082150765, 3885.4218694076862, 3890.1978445644827, 3902.35907614227,
             3918.287881517802, 3962.5598717425573, 3940.564449620109]

print(np.argsort(a_rectify))
print(np.argsort(b_rectify))

a = np.array([3904.8216100703803, 3949.075438078899, 3947.5988753358497, 3958.2047387517614, 3985.5202471492585,
              3962.334181307614, 3969.746264374093, 3984.478372218144, 3953.2966012963334, 3997.1240637044048,
              4038.663982051852, 3971.654746230098, 3912.370598266089, 3986.9026916216458, 4017.224327544175,
              3971.7003357168014])

b = np.array([3832.6149443660197, 3872.8072966408995, 3869.401082150765, 3875.677068600472, 3904.583462390668,
              3885.4218694076862, 3890.1978445644827, 3902.35907614227, 3872.0691916520655, 3918.287881517802,
              3962.5598717425573, 3894.5247743112764, 3835.249589187616, 3913.7738906341056, 3940.564449620109,
              3894.534912914469])

a_retain = [3949.075438078899, 3958.2047387517614, 3985.5202471492585, 3953.2966012963334, 3971.654746230098,
            3912.370598266089, 3986.9026916216458, 3971.7003357168014]
b_retain = [3872.8072966408995, 3875.677068600472, 3904.583462390668, 3872.0691916520655, 3894.5247743112764,
            3835.249589187616, 3913.7738906341056, 3894.534912914469]

a_retain = a_retain - np.mean(a_retain)
b_retain = b_retain - np.mean(b_retain)
# a = (a - np.mean(a)) * (a - np.mean(a) * np.std(a, ddof=1))
# b = (b - np.mean(b)) * (b - np.mean(b) * np.std(b, ddof=1))

# a_copy = a.copy()
# a = []
# step = 2
# for i in range(len(a_copy)):
#     tmp = 0
#     for j in range(step):
#         tmp += a_copy[(i + j) % len(a_copy)]
#     a.append(tmp)
#
# b_copy = b.copy()
# b = []
# for i in range(len(b_copy)):
#     tmp = 0
#     for j in range(step):
#         tmp += b_copy[(i + j) % len(b_copy)]
#     b.append(tmp)

# print(a)
# print(b)
print("keys", np.argsort(a))
print("keys", np.argsort(b))
print("keys blots", np.argsort(a_retain))
print("keys blots", np.argsort(b_retain))

print("correlation")
print(pearsonr(a, b)[0])
print(pearsonr(a_rectify, b_rectify)[0])
print(pearsonr(a_retain, b_retain)[0])

print("dtw")
print(dtw_metric(a, b) / len(a))
print(dtw_metric(a_rectify, b_rectify) / len(a_rectify))
print(dtw_metric(a_retain, b_retain) / len(a_retain))

print("keys blots", np.argsort(moving_sum(a_retain, 3)))
print("keys blots", np.argsort(moving_sum(b_retain, 3)))

print("keys blots", np.argsort(successive_moving_sum(a_retain, 3)))
print("keys blots", np.argsort(successive_moving_sum(b_retain, 3)))

a_diff = []
b_diff = []
for i in range(len(a) - 1):
    a_diff.append(round(a[i + 1] - a[i], 2))
    b_diff.append(round(b[i + 1] - b[i], 2))

print(a_diff)
print(b_diff)

print(np.argsort(a_diff))
print(np.argsort(b_diff))

a_sort = np.argsort(a)
b_sort = np.argsort(b)

# print(a_sort)
# print(b_sort)

print(a[8], a[1], a[8] - a[1])
print(b[1], b[8], b[1] - b[8])

a_diff = []
b_diff = []
for i in range(len(a_sort) - 1):
    a_diff.append([i, round(a[a_sort[i + 1]] - a[a_sort[i]], 2)])
    b_diff.append([i, round(b[b_sort[i + 1]] - b[b_sort[i]], 2)])

print(a_diff)
print(b_diff)