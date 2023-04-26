import gurobipy as gp

# create a new Gurobi model
model = gp.Model()

# define variables
x = model.addVar(lb=-100, ub=100, name="x")
y = model.addVar(lb=-100, ub=100, name="y")

# define objective function
obj = x**2 - 2*x*y - y**2
model.setObjective(obj, sense=gp.GRB.MINIMIZE)

# add constraints
model.addConstr(12*x + y <= 0, name="c1")
model.addConstr(x + 2*y >= 0, name="c2")

# set solver parameters
params = {"NonConvex": 2, "OutputFlag": 0}  # set OutputFlag to 0 to suppress solver output

# optimize the model
for key, value in params.items():
    model.setParam(key, value)
model.optimize()

# print results
if model.status == gp.GRB.OPTIMAL:
    print("Optimal value:", model.objVal)
    print("x:", x.x)
    print("y:", y.x)
else:
    print("Optimization failed.")
