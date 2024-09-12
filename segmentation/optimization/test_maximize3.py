import numpy as np
from deap import base, creator, tools, algorithms


# 定义目标函数
def objective(K, A, n):
    K = np.array(K).reshape(n, n)
    B = A @ K
    B = B - np.mean(B)
    B = (B - np.min(B)) / (np.max(B) - np.min(B))
    sum_bi_squared = np.sum(B ** 2)
    sum_bi = np.sum(B)
    return -(n * sum_bi_squared - sum_bi ** 2),


# 参数设置
n = 32

# 随机生成向量 A
A = np.random.normal(0, 1, (1, n))
A = A - np.mean(A)
A = (A - np.min(A)) / (np.max(A) - np.min(A))

# 遗传算法参数
POP_SIZE = 50
NGEN = 100
CXPB = 0.7
MUTPB = 0.2

# 初始化遗传算法
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.normal, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n * n)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", objective, A=A, n=n)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# 遗传算法主循环
population = toolbox.population(n=POP_SIZE)
algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, verbose=False)

# 找到最优解
top_ind = tools.selBest(population, 1)[0]
optimal_K = np.array(top_ind).reshape(n, n)

# 打印最优的矩阵 K
print("Optimal matrix K:")
print(optimal_K)

# 计算矩阵 B = AK
B = A @ optimal_K
B = B - np.mean(B)
B = (B - np.min(B)) / (np.max(B) - np.min(B))

print("Matrix B = AK:")
print(B)

# 打印最大化后的目标函数值
objective_value = -objective(top_ind, A, n)[0]
print(f"Maximized objective value: {objective_value}")
