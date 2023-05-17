import gurobipy as gp
from gurobipy import GRB

#master
m = gp.Model()
y = m.addVars(2, vtype = GRB.BINARY, name="y")
z = m.addVar(name="z", ub=1000)
m.setObjective(y[0]  + y[1] + z, GRB.MAXIMIZE)
m.Params.OutputFlag= 0

#sub
s = gp.Model()
s.Params.InfUnbdInfo = 1
s.Params.OutputFlag= 0
x = s.addVars(2, name="x")
cons1 = s.addConstr(x[0] + x[1] <= 0, name="cons1")
cons2 = s.addConstr(-x[0] - x[1] <= 0, name="cons2")
s.setObjective(x[0] - x[1],GRB.MAXIMIZE)

#Benders loop

LB = -100000
UB = 100000

while(UB - LB >= 0.00001):

  #optimize Master
  m.optimize()
  UB = m.objVal

  #update sub
  cons1.rhs = 2 - y[1].x
  cons2.rhs = -1 - y[0].x

  #optimize sub
  s.optimize()

  #generate cuts
  if s.status == 3:
    print("Infeasibility constraint")
    u1 = cons1.getAttr('FarkasDual')
    u2 = cons2.getAttr('FarkasDual')
    expr = u1*(2 - y[1]) + u2*(-1  - y[0])
    m.addConstr(expr >= 0)
    print(expr, ">=", "0")
  else:
    print("Feasibility constraint")
    u1 = cons1.getAttr('Pi')
    u2 = cons2.getAttr('Pi')
    expr = u1*(2 - y[1]) + u2*(-1  - y[0])
    print(expr, ">=", "z")
    m.addConstr( expr >= z)
    if   y[0].x  + y[1].x  + s.objVal > LB:
      LB = y[0].x  + y[1].x  + s.objVal

  print(LB, " ", UB)

# # Display optimal values of decision variables
for v in m.getVars():
    if v.x > 1e-6:
        print(v.varName, v.x)

# Display optimal solution value
print('Total profit: ', m.objVal)
