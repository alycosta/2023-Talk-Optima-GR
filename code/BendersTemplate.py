import gurobipy as gp
from gurobipy import GRB

#Benders loop
LB = -100000; UB = 100000

while(UB - LB >= 0.00001):
  #optimize Master
  m.optimize()
  UB = m.objVal

  #update the right hand side of the subproblem
  cons1.rhs = #fill
  cons2.rhs = #fill

  #optimize sub
  s.optimize()

  #generate cuts
  if s.status == 3:
    print("Infeasibility cut")
    # get the dual ray
    u1 = cons1.getAttr('FarkasDual') 
    u2 = cons2.getAttr('FarkasDual') 
    m.addConstr(#fill)  #add the new cut
  else:
    print("Feasibility cut")
    # get the dual ray
    u1 = cons1.getAttr('Pi')
    u2 = cons2.getAttr('Pi')
    m.addConstr( #fill)

    #fill (update the best lower bound if necessary).

  print(LB, " ", UB)

#Print solution
