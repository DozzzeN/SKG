import pyomo.environ as pyo

# Create a concrete Pyomo model
model = pyo.ConcreteModel()

# Define the variables
model.x = pyo.Var(within=pyo.Reals, bounds=(-100, 100))
model.y = pyo.Var(within=pyo.Reals, bounds=(-100, 100))

# Define the objective function
model.obj = pyo.Objective(expr=model.x**2 - 2*model.x*model.y - model.y**2, sense=pyo.minimize)

# Define the constraints
model.con1 = pyo.Constraint(expr=12*model.x + model.y <= 0)
model.con2 = pyo.Constraint(expr=model.x + 2*model.y >= 0)

# Select a solver that can handle non-convex problems
solver = pyo.SolverFactory('baron')

# Solve the problem
results = solver.solve(model)

# Print the solution
print(pyo.value(model.x))
print(pyo.value(model.y))
