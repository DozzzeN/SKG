import cvxpy as cp
import numpy as np
import time
import scs

# 定义矩阵A和向量y
n = 256
np.random.seed(0)
A = np.random.normal(0, 4, (n, n * 2)).view(np.complex128)
np.random.seed(1)
# s = np.random.randint(0, 2, n) + 1.j * np.random.randint(0, 2, n)
s = np.random.randint(0, 2, n)
y = A @ s
# A = np.array([[2, 3], [1, 4], [3, 2]])
# s = np.array([1, 2])
# y = np.array([8, 9, 7])

# 定义扰动矩阵A'
np.random.seed(0)
A_perturbed = A + np.random.normal(0, 0.01, (n, n * 2)).view(np.complex128)  # 这里使用正态分布扰动矩阵A
# A_perturbed = A

# Construct a CVXPY problem
# x = cp.Variable(n, integer=True)
x = cp.Variable(n)
# x = cp.Variable(n, complex=True)
objective = cp.Minimize(cp.sum_squares(A_perturbed @ x - y))
# cons = [cp.real(x) >= 0, cp.real(x) <= 1, cp.imag(x) >= 0, cp.imag(x) <= 1]
cons = []
prob = cp.Problem(objective, cons)
start = time.time()
prob.solve()
# prob.solve(solver=cp.OSQP, scaling=False)  # 默认的求解器
print("time", time.time() - start)
# prob.solve(solver=cp.GUROBI, verbose=True, TimeLimit=20)  # 使用gurobi求解整数规划问题，设置求解时间限制为20s
# prob.solve(solver=cp.CPLEX, verbose=True, cplex_params={"timelimit": 20})  # 使用CPLEX求解整数规划问题，设置求解时间限制为20s
# prob.solve(solver=cp.SCIP, verbose=True, scip_params={"limits/time": 20})  # 使用SCIP求解整数规划问题，设置求解时间限制为20s
# prob.solve(solver=cp.SCS, verbose=True, max_iters=5000)
# prob.solve(solver=cp.CVXOPT, verbose=True)
# prob.solve(solver=cp.ECOS, verbose=True)
# prob.solve(solver=cp.PROXQP, verbose=True)
# prob.solve(solver=cp.MOSEK, verbose=True)
# prob.solve(solver=cp.CLARABEL, verbose=True)
# prob.solve(solver=cp.NAG, verbose=True)
# prob.solve(solver=cp.XPRESS, verbose=True)
# 安装后找不到DDL，故放弃之
# prob.solve(solver=cp.COPT, verbose=True, copt_params={"max_time": 20})  # 使用COPT求解整数规划问题，设置求解时间限制为20s
print(prob.solver_stats.solver_name)
print("Status: ", prob.status)
print("The optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("The real x is")
print(s)
x = np.array([round(abs(x[i].value)) for i in range(n)])
print("A solution x is")
print(x)
print("Objective value:", objective.value)
print(np.sum(np.abs(x - s)))
print(cp.installed_solvers())
print(prob.solver_stats.num_iters)
