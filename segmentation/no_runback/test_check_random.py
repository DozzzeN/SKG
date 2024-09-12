import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 生成随机矩阵
matrix = np.random.normal(loc=0, scale=1, size=(100, 100))

# 将矩阵展平成一维数组
data = matrix.flatten()

# 计算均值和标准差
mean = np.mean(data)
std = np.std(data)
print(f"Mean: {mean}, Standard Deviation: {std}")

# 绘制直方图
plt.hist(data, bins=30, density=True, alpha=0.6, color='g')

# 拟合正态分布并绘制
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mean, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title("Histogram with Normal Distribution Fit")
plt.show()

# 绘制QQ图
stats.probplot(data, dist="norm", plot=plt)
plt.title("QQ Plot")
plt.show()

# 正态性检验
shapiro_test = stats.shapiro(data)
ks_test = stats.kstest(data, 'norm', args=(mean, std))
anderson_test = stats.anderson(data, dist='norm')

# 检验与随机正态分布的接近度
# 值越大越好，p值越大越好
print(f"Shapiro-Wilk Test: Statistic={shapiro_test.statistic}, p-value={shapiro_test.pvalue}")
# 值越小越好，p值越大越好
print(f"Kolmogorov-Smirnov Test: Statistic={ks_test.statistic}, p-value={ks_test.pvalue}")
# 值小于临界值则好
print(f"Anderson-Darling Test: Statistic={anderson_test.statistic}, Critical Values={anderson_test.critical_values}")
