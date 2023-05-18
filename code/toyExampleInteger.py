import gurobipy as gp
from gurobipy import GRB

m = gp.Model("A Linear Programming model")


#variables

indices = [1, 2]
x = m.addVars(indices, name = 'x',)
y = m.addVars(indices, name = 'y', vtype = GRB.BINARY)

#objective
m.setObjective( x[1] - x[2] + y[1] + y[2] , GRB.MAXIMIZE)

#constraints
cons1 = m.addConstr( -x[1] - x[2] - y[2]   >= -2)
cons2 = m.addConstr( x[1] + x[2] - y[1]   >= 1)

#solve
m.optimize()
