import gurobipy as gp
from gurobipy import GRB


#Initial Subproblem:
def initial_sub():
  s = gp.Model()
  s.Params.InfUnbdInfo = 1
  s.Params.OutputFlag= 0
  x = s.addVars(2, name="x")
  cons1 = s.addConstr(x[0] + x[1] <= 0, name="cons1")
  cons2 = s.addConstr(-x[0] -x[1] <= 0, name="cons2")
  s.setObjective(x[0] - x[1],GRB.MAXIMIZE)
  return s  

#Update subproblem
def update_subproblem(s,y1,y2):
  cons1 = s.getConstrByName("cons1")
  cons2 = s.getConstrByName("cons2")
  cons1.rhs = 2 - y2
  cons2.rhs = -1 - y1

def generate_cuts(model,where):
  if where == GRB.Callback.MIPSOL:
    valsy = model.cbGetSolution(model._y)
    print(valsy)
    update_subproblem(s,valsy[0],valsy[1])   
    s.optimize()    
    if s.status == 3:
      print("Infeasibility constraint")
      cons1 = s.getConstrByName("cons1")
      cons2 = s.getConstrByName("cons2")
      u1 = cons1.getAttr('FarkasDual')
      u2 = cons2.getAttr('FarkasDual')
      expr = u1*(2 - y[1]) + u2*(-1  - y[0])
      print(expr, ">=", "0")
      model.cbLazy( expr >= 0)
    else:
      print("Feasibility constraint")
      cons1 = s.getConstrByName("cons1")
      cons2 = s.getConstrByName("cons2")
      u1 = cons1.getAttr('Pi')
      u2 = cons2.getAttr('Pi')
      expr = u1*(2 - y[1]) + u2*(-1  - y[0])
      print(expr, ">=", "z")
      model.cbLazy( expr >= m._z)

s = initial_sub()
s.write("model.lp")

#master
m = gp.Model()
y = m.addVars(2, vtype = GRB.BINARY, name="y")
z = m.addVar(name="z", ub=1000)

m.setObjective(y[0]  + y[1] + z, GRB.MAXIMIZE)

m.Params.LazyConstraints = 1
m._y = y
m._z = z
m.Params.OutputFlag= 0
m.optimize(generate_cuts)

s.write("model.lp")


# Display optimal values of decision variables
for v in m.getVars():
    if v.x > 1e-6:
        print(v.varName, v.x)

# Display optimal solution value
print('Total profit: ', m.objVal)
