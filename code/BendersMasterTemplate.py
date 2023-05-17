import gurobipy as gp
from gurobipy import GRB

m = gp.Model(#fill)
y = m.addVars(#fill)
z = m.addVar(#fill, ub=10000)
m.setObjective(#fill)
