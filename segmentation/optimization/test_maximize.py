import numpy as np


def min_distance(solution1, solution2):
    total = 0
    for i in range(len(solution1)):
        min_dist = float("inf")
        for j in range(len(solution2)):
            if i == j:
                continue
            # 计算欧式距离
            min_dist = min(min_dist, np.square(np.array(solution1[i]) - np.array(solution2[j])))
        total += min_dist
    return total


def sum_distance(solution1, solution2):
    total = 0
    for i in range(len(solution1)):
        for j in range(len(solution2)):
            # 计算欧式距离
            distance = np.square(np.array(solution1[i]) - np.array(solution2[j]))
            total += distance
    return total


def compute_distance(solution):
    res = len(solution) * 2 * sum([i ** 2 for i in solution]) - 2 * (sum(solution)) ** 2
    return res


def all_distance(solution1, solution2):
    total = []
    for i in range(len(solution1)):
        for j in range(len(solution2)):
            if i == j:
                continue
            # 计算欧式距离
            # distance = np.sqrt(np.sum(np.square(np.array(solution1[i]) - np.array(solution2[j]))))
            distance = np.square(np.array(solution1[i]) - np.array(solution2[j]))
            total.append(distance)
    return total


# n = 8
# data = list(range(n))
# data = data - np.mean(data)
# data = (np.array(data) - np.min(data)) / (np.max(data) - np.min(data))
# print(data)
# print(sum_distance(data, data))
# print(sum(all_distance(data, data)))
# print(n * 2 * (sum([i ** 2 for i in data])) - 2 * (sum(data)) ** 2)
# exit()

# 等差数列
def arithmetic_sequence(a, d, n):
    return np.array([a + d * i for i in range(n)])


# 等比数列
def geometric_sequence(a, r, n):
    return np.array([a * r ** i for i in range(n)])


# 高斯分布
def normal_sequence(n):
    return np.random.normal(0, 1, n)


# 指数分布
def exponential_sequence(n):
    return np.random.exponential(1, n)


# 二项分布
def binomial_sequence(n, trials, p):
    return np.random.binomial(trials, p, n)


# 泊松分布
def poisson_sequence(n, lam):
    return np.random.poisson(lam, n)


# 生成测试数据
n = 10
solution1 = arithmetic_sequence(1, 1, n)
solution2 = geometric_sequence(1, 10, n)
solution3 = normal_sequence(n)
solution4 = exponential_sequence(n)
solution5 = binomial_sequence(n, 10, 0.5)
solution6 = poisson_sequence(n, 1)
solution1 = solution1 - np.mean(solution1)
solution2 = solution2 - np.mean(solution2)
solution3 = solution3 - np.mean(solution3)
solution4 = solution4 - np.mean(solution4)
solution5 = solution5 - np.mean(solution5)
solution6 = solution6 - np.mean(solution6)
solution1 = (solution1 - np.min(solution1)) / (np.max(solution1) - np.min(solution1))
solution2 = (solution2 - np.min(solution2)) / (np.max(solution2) - np.min(solution2))
solution3 = (solution3 - np.min(solution3)) / (np.max(solution3) - np.min(solution3))
solution4 = (solution4 - np.min(solution4)) / (np.max(solution4) - np.min(solution4))
solution5 = (solution5 - np.min(solution5)) / (np.max(solution5) - np.min(solution5))
solution6 = (solution6 - np.min(solution6)) / (np.max(solution6) - np.min(solution6))
print("solution1", solution1)
print("solution2", solution2)
print("solution3", solution3)
print("solution4", solution4)
print("solution5", solution5)
print("solution6", solution6)

print("sum_distance")
print(sum(all_distance(solution1, solution1)))
print(sum(all_distance(solution1, solution2)))
print(sum(all_distance(solution1, solution3)))
print(sum(all_distance(solution1, solution4)))
print(sum(all_distance(solution1, solution5)))
print(sum(all_distance(solution1, solution6)))
# print(compute_distance(solution1) / sum(all_distance(solution1, solution1)))
# print(compute_distance(solution2) / sum(all_distance(solution2, solution2)))
# print(compute_distance(solution3) / sum(all_distance(solution3, solution3)))
# print(compute_distance(solution4) / sum(all_distance(solution4, solution4)))

print("min_distance")
print(min_distance(solution1, solution1))
print(min_distance(solution2, solution2))
print(min_distance(solution3, solution3))
print(min_distance(solution4, solution4))
print(min_distance(solution5, solution5))
print(min_distance(solution6, solution6))

# print(np.mean(all_distance(solution1, solution1)))
# print(np.mean(all_distance(solution1, solution2)))
# print(np.mean(all_distance(solution1, solution3)))
# print(np.mean(all_distance(solution1, solution4)))
# print(compute_distance(solution1) / np.mean(all_distance(solution1, solution1)))
# print(compute_distance(solution2) / np.mean(all_distance(solution2, solution2)))
# print(compute_distance(solution3) / np.mean(all_distance(solution3, solution3)))
# print(compute_distance(solution4) / np.mean(all_distance(solution4, solution4)))

print("maximize")
# data = [1] * n
# data = [0] * 5 + [1] * 5
data = [0] * (n - 1) + [1]
data = data - np.mean(data)
data = (data - np.min(data)) / (np.max(data) - np.min(data))
print(data)
print(compute_distance(data))
print(sum(all_distance(data, data)))
print(min_distance(data, data))

# import numpy as np
# import matplotlib.pyplot as plt
#
# # Define the function to be plotted
# def f(a1, a2, a3, n):
#     return n * (a1**2 + a2**2 + a3**2) - (a1 + a2 + a3)**2
#
# # Generate data points
# a1 = np.linspace(0, 1, 100)
# a2 = np.linspace(0, 1, 100)
# a1, a2 = np.meshgrid(a1, a2)
# a3 = 1 / 2  # fix a3 to a constant value to visualize in 3D
#
# # Compute the function values
# n = 3  # fix n to a constant value
# Z = f(a1, a2, a3, n)
#
# # Plotting the function
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(a1, a2, Z, cmap='viridis')
#
# # Set labels
# ax.set_xlabel('a1')
# ax.set_ylabel('a2')
# ax.set_zlabel('n*(a1^2 + a2^2 + a3^2) - (a1 + a2 + a3)^2')
# ax.set_title('Plot of n*(a1^2 + a2^2 + a3^2) - (a1 + a2 + a3)^2')
#
# plt.tight_layout()
# plt.savefig('maximize_3D.svg', dpi=1200, bbox_inches='tight')

# plt.show()
