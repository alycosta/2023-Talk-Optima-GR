import gurobipy as gp
from gurobipy import GRB

s = gp.Model(#fill)
s.Params.InfUnbdInfo = 1 #allow to get dual ray
x = s.addVars(#fill)
cons1 = s.addConstr(#fill, name="cons1")
cons2 = s.addConstr(#fill, name="cons2")

