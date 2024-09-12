import cvxpy as cp
import numpy as np
import time
import scs

# 定义矩阵A和向量y
n = 256
np.random.seed(0)
B = np.random.normal(0, 4, (n, n))
np.random.seed(1)
C = np.random.normal(0, 4, (n, n))
A = B + 1.j * C
np.random.seed(2)
y = np.random.randint(0, 2, n)
np.random.seed(3)
z = np.random.randint(0, 2, n)
x = y + 1.j * z
b = A @ x
d = np.real(b)
e = np.imag(b)
print(np.allclose(B @ y - C @ z, d))
print(np.allclose(B @ z + C @ y, e))

# 定义扰动矩阵A'
np.random.seed(0)
A_perturbed = A + np.random.normal(0, 0.01, (n, n * 2)).view(np.complex128)  # 这里使用正态分布扰动矩阵A
# A_perturbed = A
Ap_real = np.real(A_perturbed)
Ap_imag = np.imag(A_perturbed)
BigA = np.vstack((np.hstack((Ap_real, -Ap_imag)), np.hstack((Ap_imag, Ap_real))))
BigX = np.array([y, z]).reshape(2 * n)
Bigb = np.array([d, e]).reshape(2 * n)
# Construct a CVXPY problem
isComplex = False
if isComplex:
    sol = cp.Variable(n, complex=True)
    objective = cp.Minimize(cp.sum_squares(A_perturbed @ sol - b))
    cons = [cp.real(sol) >= 0, cp.real(sol) <= 1, cp.imag(sol) >= 0, cp.imag(sol) <= 1]
else:
    sol = cp.Variable(2 * n)
    objective = cp.Minimize(cp.sum_squares(BigA @ sol - Bigb))
    cons = [sol >= 0, sol <= 1]
# cons = []
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
print(sol.value)
if isComplex:
    sol = np.array([round(sol[i].value) for i in range(n)])
else:
    sol = np.array([round(abs(sol[i].value)) for i in range(n)]) + 1.j * np.array(
        [round(abs(sol[i].value)) for i in range(n, 2 * n)])
print("Objective value:", objective.value)
print("Error", np.sum(np.abs(sol - x)))
print(cp.installed_solvers())
print(prob.solver_stats.num_iters)
