import time
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw_ndim
from dtw import accelerated_dtw
import gurobipy as gp
from gurobipy import GRB

def dtw_metric(data1, data2):
    if np.ndim(data1) == 1:
        distance = lambda x, y: np.abs(x - y)
        data1 = np.array(data1)
        data2 = np.array(data2)
        return accelerated_dtw(data1, data2, dist=distance)[0]
    else:
        return dtw_ndim.distance(data1, data2)

def compute_min_dtw(data1, data2):
    min_dtw = np.inf
    for i in range(len(data1)):
        for j in range(len(data2)):
            if i == j:
                continue
            min_dtw = min(min_dtw, dtw_metric(data1[i], data2[j]))
    return min_dtw

def segment_sequence(data, segment_lengths):
    segments = []
    start_index = 0
    for length in segment_lengths:
        end_index = start_index + length
        segments.append(data[start_index:end_index])
        start_index = end_index
    return segments

def optimize_with_gurobi(A, segment_method):
    n = len(A)
    model = gp.Model()
    model.setParam('OutputFlag', 0)

    K = model.addVars(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="K")
    B = [gp.LinExpr(A[i] * K[i]) for i in range(n)]

    mean_B = sum(B) / n
    B = [B[i] - mean_B for i in range(n)]

    B_min = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="B_min")
    B_max = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="B_max")

    model.addConstrs((B[i] >= B_min for i in range(n)), name="MinConstr")
    model.addConstrs((B[i] <= B_max for i in range(n)), name="MaxConstr")

    # B = [(B[i] - B_min) / (B_max - B_min) for i in range(n)]
    B_segments = segment_sequence(B, segment_method)

    def compute_dtw_for_model(B_segments):
        dtw_sum = 0
        for i in range(len(B_segments)):
            for j in range(len(B_segments)):
                if i != j:
                    dtw_sum += dtw_metric(B_segments[i], B_segments[j])
        return dtw_sum

    min_dtw = compute_dtw_for_model(B_segments)
    model.setObjective(-min_dtw, GRB.MINIMIZE)

    model.optimize()
    optimal_K = [K[i].X for i in range(n)]
    return np.diag(optimal_K)

# 参数设置
n = 16
np.random.seed(0)
A_bck = np.random.normal(0, 1, n)
A = A_bck - np.mean(A_bck)
A = (A - np.min(A)) / (np.max(A) - np.min(A))
segment_method = [4, 4, 4, 4]

initial_K = np.random.normal(0, 1, n)

print(f"Original value: {compute_min_dtw(segment_sequence(A, segment_method), segment_sequence(A, segment_method))}\n")

start_time = time.time_ns()
optimal_K = optimize_with_gurobi(A, segment_method)
elapsed_time = (time.time_ns() - start_time) / 1e6
print(f"Gurobi Optimizer, Time: {elapsed_time} ms")

B = A @ optimal_K
B = B - np.mean(B)
B = (B - np.min(B)) / (np.max(B) - np.min(B))
B = np.array(segment_sequence(B, segment_method))

print(f"Maximized objective value: {compute_min_dtw(B, B)}\n")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.stem(A.flatten(), use_line_collection=True)
plt.title("Vector A")
plt.xlabel("Index")
plt.ylabel("Value")

plt.subplot(1, 2, 2)
plt.stem(B.flatten(), use_line_collection=True)
plt.title("Vector B")
plt.xlabel("Index")
plt.ylabel("Value")
plt.tight_layout()
plt.show()

optimal_K = optimal_K - np.mean(optimal_K)
optimal_K = (optimal_K - np.min(optimal_K)) / (np.max(optimal_K) - np.min(optimal_K))
plt.figure()
plt.imshow(optimal_K, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Visualization of Matrix K')
plt.show()
