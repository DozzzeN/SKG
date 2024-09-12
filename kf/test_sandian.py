import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
df = pd.read_excel('副本t-SNE数据 %Rank-EL+%Rank-BA+Affinity.xlsx')

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(df['affinity'], df['rank'], marker='o', color='blue')

# 标注每个点的名称
for i, txt in enumerate(df['name']):
    plt.annotate(txt, (df['affinity'][i], df['rank'][i]), textcoords="offset points", xytext=(0,10), ha='center')

# 添加标题和轴标签
plt.title('Scatter Plot of Affinity vs Rank')
plt.xlabel('Affinity')
plt.ylabel('Rank')

# 显示图形
plt.grid(True)
plt.show()
