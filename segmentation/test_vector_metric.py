import numpy as np
from scipy.stats import skew, kurtosis

# 假设一个N*1的数据向量
N = 100
data = np.random.randn(N, 1)

# 均值
mean = np.mean(data)

# 方差
variance = np.var(data)

# 标准差
std_dev = np.std(data)

# 中位数
median = np.median(data)

# 四分位数
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

# 偏度 偏度是衡量数据分布对称性的统计量。正偏度表示分布右尾长，负偏度表示分布左尾长。
skewness = skew(data)

# 峰度 峰度是衡量概率分布的尾部重量和峰度的统计量。它表示数据在平均值附近有多尖锐。
# 高峰度值表示数据在均值附近的集中度更高，低峰度值表示数据分布更加平坦。
kurt = kurtosis(data)

# 范围
range_ = np.ptp(data)

# 计算变异系数 变异系数是标准差与均值的比值，用于衡量数据的离散程度，相对与均值的波动程度。
cv = std_dev / mean

# 定义幂次p
p = 3

# 计算幂次和 幂次和是指将数据中的每个元素进行某个幂次运算后，再求和。它是描述数据分布特性的一种方法。
sum_of_powers = np.sum(data ** p)

# 输出结果
print("均值:", mean)
print("方差:", variance)
print("标准差:", std_dev)
print("中位数:", median)
print("第一四分位数 (Q1):", Q1)
print("第三四分位数 (Q3):", Q3)
print("四分位距 (IQR):", IQR)
print("偏度:", skewness)
print("峰度:", kurt)
print("范围:", range_)
print("变异系数 (CV):", cv)
print(f"{p}次幂次和:", sum_of_powers)

import numpy as np
from scipy.stats import kurtosis

# 初始数据（较高峰度）
data_initial = np.random.normal(0, 1, 1000)

# 峰度变化后的数据（更平坦的分布）
data_changed = np.random.uniform(-3, 3, 1000)

# 计算峰度
kurt_initial = kurtosis(data_initial)
kurt_changed = kurtosis(data_changed)

print("初始峰度:", kurt_initial)
print("变化后峰度:", kurt_changed)

import matplotlib.pyplot as plt
import seaborn as sns

# 绘制初始数据分布
sns.histplot(data_initial, kde=True, color='blue', label='Initial Data')
plt.axvline(np.mean(data_initial), color='blue', linestyle='dashed', linewidth=2)

# 绘制变化后数据分布
sns.histplot(data_changed, kde=True, color='red', label='Changed Data')
plt.axvline(np.mean(data_changed), color='red', linestyle='dashed', linewidth=2)

plt.legend()
plt.title('Data Distribution Before and After')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
