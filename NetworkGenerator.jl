using Plots,Random
struct Data
    n::Int # number of nodes
    m::Int # number of edges
    k::Int # number of commodities/OD-pairs
    x::Vector{Float64}; y::Vector{Float64} # node coordinates
    frm::Vector{Int} # length m, edges sorted in increasing frm node order
    to::Vector{Int}
    outE::Vector{Vector{Int}} # edge indices of out edges from each node
    inE::Vector{Vector{Int}}  # edge indices of incoming edges to each node
    vcost::Vector{Int} # variable cost
    fcost::Vector{Int} # fixed cost
    cap::Vector{Int} # capacity of edge
    org::Vector{Int} # origin
    dst::Vector{Int} # destination
    W::Vector{Int} # demand org[i]->dst[i]
end
function randomData(n::Int, m::Int, k::Int)
    nODpts = max(round(Int,2*sqrt(k))+1, min(k + k÷2, n ÷ 5))
    width=150; height=100;  offset=2*(width+height)÷(nODpts+1)
    x = rand(3:width-2, n); y = rand(3:height-2, n)
    for i=1:nODpts
        if offset*i <= width; y[i]=1; x[i]=offset*i;
        elseif offset*i <= width+height; x[i]=width; y[i]=offset*i-width;
        elseif offset*i <= 2*width+height; y[i]=height; x[i]=2*width+height-offset*i;
        else x[i]=1; y[i]=2*(width+height)-offset*i; end
    end
    org = zeros(Int, k); dst = zeros(Int, k); W = zeros(Int, k)
    for i=1:k
        while true
            if i <= nODpts; org[i]=i; dst[i]=(i+rand(nODpts÷4:(3*nODpts)÷4)) % nODpts+1; 
            else org[i]=rand(1:nODpts); dst[i]=(nODpts+rand(nODpts÷4:(3*nODpts)÷4)) % nODpts+1; end
            if org[i] > dst[i]; org[i], dst[i] = dst[i], org[i]; end
            # flow from bottom left to top right
            if x[dst[i]] ==1 || y[dst[i]]==1; org[i], dst[i] = dst[i], org[i]; end
            if (org[i],dst[i]) ∈ Set([(org[j],dst[j]) for j=1:i-1]); continue; end
            break
        end
        W[i]=rand(1:5)
    end
    isDstOnly = [false for i=1:n];
    for i=1:k; if dst[i] ∉ org; isDstOnly[dst[i]]=true; end; end
    d = [ round(Int64,sqrt((x[i]-x[j])^2 + (y[i]-y[j])^2)) for i=1:n, j=1:n ]
    haveEdge = zeros(Bool, n, n)
    frm = zeros(Int64,m); to = zeros(Int64,m)
    for i=1:n, j=1:n
        if (x[j] <= x[i] && y[j]<=y[i]) || isDstOnly[i]
            haveEdge[i,j] = true; # don't generate these
        end
    end

    for i=1:m
        if i ≤ n &&  ! isDstOnly[i] # arc to nearest neighbour to towards top/right
            tmp, j = minimum( (d[i,k],k) for k=1:n if !haveEdge[i,k] )
            haveEdge[i,j] = true
            frm[i] = i; to[i] = j
        else
            while true
                frm[i] = rand(1:n)
                if isDstOnly[frm[i]]; continue; end
                Ns = [j for j=1:n if !haveEdge[frm[i],j]]
                if length(Ns) == 0; continue; end
                sort!(Ns, by= j -> d[frm[i],j] + (j ∈ to[1:i] ? 50 : 0) )
                to[i] = Ns[rand(1:min(5,length(Ns)))]
                haveEdge[frm[i],to[i]] = true
                break
            end
        end
    end
    vcost = [d[frm[i],to[i]] for i=1:m]
    # arcs/path = max(1, n- m ÷ n),  totalFLow=sum(W)
    # flow/arc = totalFLow*arcs/path ÷ m
    # cap/arc = sum(W)*max(1-n-m÷n, 1) ÷ m
    avg_cap = max(2, (sum(W)*max(n-m÷n, 1)) ÷ m )
    cap  = [ 2 + avg_cap÷2 + rand(0:avg_cap) for i=1:m]
    vcost = [ d[frm[i],to[i]] for i=1:m]
    fcost = [ round(Int64, cap[i]*vcost[i]*rand()) for i=1:m ]
    outE = [findall(frm .== i) for i in 1:n]
    inE = [findall(to .== i) for i in 1:n]
    return Data(n, m, k, x, y, frm, to, outE, inE, vcost, fcost, cap, org, dst, W)
end

@doc """Network design that is roughly grid-shaped.
        All origins at bottom & left, destinations top & right.
        expect n to be number of nodes corresponding approximately to 
        a 3x2 grid (or some multiple thereof) + origin destination pairs 
        expect m approximately 4 x n"""
function randomGridFCNF(n::Int, m::Int, k::Int)
    nODpts = max(round(Int,2*sqrt(k))+1, min(k + k÷2, n ÷ 5))
    width=150; height=100;  offset=2*(width+height)÷(nODpts+1)
    ################### define grid of nodes #####################
    ny = round(Int, ceil(sqrt((n-nODpts)/1.5)))
    nx = round(Int, ceil( (n-nODpts)/ny) )
    while nx*ny < n-nODpts;
        nx += 1;
        if nx*ny < n-nODpts; ny += 1; end
    end
    dx = width/(nx+1); dy = height/(ny+1)
    x = -dx/3 .+ rand(n) .* 2dx/3; y = -dy/3 .+ rand(n) .* 2dy/3
    grid = zeros(Int, nx, ny); nGrid = n-nODpts
    println("n=$n, nOD=$nODpts, nGrid=$nGrid nx*ny=$nx x $ny = $(nx*ny)")
    for i=1:(nx*ny-nGrid); grid[rand(1:nx), rand(1:ny)] = -1; end
    i=nODpts
    for ix=1:nx, iy=1:ny; 
        if grid[ix,iy] != -1; 
            i+= 1; grid[ix,iy]=i; 
            x[i] += ix*dx; y[i] += iy*dy;
            if i==n; break; end
        end
    end
    @assert(i==n)
    for i=1:nODpts
        if offset*i <= width; y[i]=1; x[i]=offset*i;
        elseif offset*i <= width+height; x[i]=width; y[i]=offset*i-width;
        elseif offset*i <= 2*width+height; y[i]=height; x[i]=2*width+height-offset*i;
        else x[i]=1; y[i]=2*(width+height)-offset*i; end
    end

    ################### origin & destinations ###########################
    org = zeros(Int, k); dst = zeros(Int, k); W = zeros(Int, k)
    Os = [i for i=1:nODpts if min(x[i],y[i])==1]; 
    Ds = [i for i=1:nODpts if min(x[i],y[i])!=1]; Random.shuffle!(Ds)
    @assert length(Os) > 0 && length(Ds) >0
    for i=1:k
        for iter=1:100
            if iter==100; error("failed to find OD $i"); end
            org[i] = ( i <= length(Os)) ? Os[i] : Os[rand(1:length(Os))]
            dst[i] = ( i <= length(Ds)) ? Ds[i] : Ds[rand(1:length(Ds))]
            if (org[i],dst[i]) ∈ Set([(org[j],dst[j]) for j=1:i-1]); continue; end
            break
        end
        W[i]=rand(1:5)
    end
    d = [ round(Int64,sqrt((x[i]-x[j])^2 + (y[i]-y[j])^2)) for i=1:n, j=1:n ]
    
    ################### define arcs #####################
    allowed = ones(Bool, n, n)
    println("origins: ", org')
    println("destin.: ", dst')
    println("x: ", x[1:nODpts]', "  y: ",y[1:nODpts]')
    for i=1:k, j=1:n; 
        allowed[j,org[i]] = allowed[dst[i],j] = false; 
    end
    nODarcs = max(2*nODpts, m-4*nGrid+2*(nx+ny))
    frm = zeros(Int,m); to = zeros(Int,m)
    indeg = zeros(Int,n); outdeg = zeros(Int,n)
    outNeigh(i;X=1.5*dx,Y=1.5*dy) = [j for j=1:n if allowed[i,j] && 
                    abs(x[i]-x[j])<X && abs(y[i]-y[j])<Y
                    && (x[i] <= x[j] || y[i]<=y[j])]
    inNeigh(j;X=2*dx,Y=2*dy) = [i for i=1:n if allowed[i,j] && 
                  abs(x[i]-x[j])<X && abs(y[i]-y[j])<Y
                  && (x[i] <= x[j] || y[i]<=y[j])]
    for i=1:m
        if i<=nODarcs;
            ii = (i <= 2*nODpts) ? (i-1)÷2+1 : rand(1:nODpts)
            α=1.5; jj=-1
            for iter=1:100 
                if x[ii] < 2;     cand = outNeigh(ii,X=√α*dx,Y=α*dy)
                elseif y[ii]<2;   cand = outNeigh(ii,X=α*dx,Y=√α*dy)
                elseif x[ii]>width-1; cand = inNeigh(ii,X=√α*dx,Y=α*dy)
                else;             cand = inNeigh(ii, X=α*dx,Y=√α*dy)
                end
                if ! isempty(cand)
                     jj = cand[rand(1:length(cand))]; 
                     if ! (x[ii]==1 || y[ii]==1); ii,jj = jj,ii; end
                     break; 
                elseif iter==100; 
                    println("allowed:",[j for j=1:n if allowed[ii,j]])
                    println("close:", [j for j=1:n if 
                               abs(x[ii]-x[j])<α*dx && abs(y[ii]-y[j])<√α*dy])
                    error("failed to find $i-th OD arc $((x[ii],y[ii]))");
                end
                α *= 1.3
            end
        else # grid arcs
            ii = jj = -1; α=1.5
            for iter=1:100
                if iter %5==0; α*=1.3; end
                if iter==100; error("failed to find $i-th grid arc"); end
                cand = [j for j=nODpts+1:n if min(outdeg[j],indeg[j]) == 0]
                if !isempty(cand); ii = cand[rand(1:length(cand))]; 
                elseif i <= nODarcs+2*nGrid;
                    ii = (iter < 2) ? 
                        argmin( min(outdeg[j],indeg[j]) for j=nODpts+1:n) :
                        grid[ mod1( i+iter,nx),  mod1((i+iter)÷nx,ny)];
                else
                    ii = grid[rand(1:nx), rand(1:ny)]
                end
                if ii <= 0; ii =  argmin( min(outdeg[j],indeg[j]) for j=nODpts+1:n) end
                doOut = ( abs(outdeg[ii]-indeg[ii]) >= 2 || 
                          min(outdeg[ii],indeg[ii])==0) ? 
                        indeg[ii]>outdeg[ii] : rand()<0.5 
                cand = doOut ? outNeigh(ii,X=α*dx,Y=α*dy) : inNeigh(ii,X=α*dx,Y=α*dy)
                if isempty(cand); continue; end
                if length(cand) > 3; sort!(cand, by=j->d[ii,j]) end
                jj = cand[rand(1:min(3,length(cand)))]
                if ! doOut; ii,jj = jj,ii; end
                break
            end
        end
        frm[i] = ii; to[i] = jj
        outdeg[ii] += 1; indeg[jj] += 1
        allowed[ii,jj] = false; allowed[jj,ii]=false; # no parallel arcs
    end

    vcost = [d[frm[i],to[i]] for i=1:m]
    # arcs/path = (nx+ny)÷2,  totalFLow=sum(W)
    # flow/arc = totalFLow*arcs/path ÷ m
    # cap/arc = sum(W)*(nx+ny)÷2 ÷ m
    avg_cap = max(2, (3*sum(W)*(nx+ny)) ÷ (2*m) )
    ## was: avg_cap = max(2, (sum(W)*max(n-m÷n, 1)) ÷ m )
    cap  = [ 3 + (3*avg_cap)÷4 + rand(0:avg_cap) for i=1:m]
    vcost = [ d[frm[i],to[i]] for i=1:m]
    fcost = [ round(Int64, cap[i]*vcost[i]*rand()) for i=1:m ]
    outE = [findall(frm .== i) for i in 1:n]
    inE = [findall(to .== i) for i in 1:n]
    return Data(n, m, k, x, y, frm, to, outE, inE, vcost, fcost, cap, org, dst, W)
end

function plotNetwork(d::Data,extraCol...)
    scatter(d.x, d.y, label="", markersize=2, color=:black)
    for i=1:d.k
        scatter!([d.x[d.org[i]], d.x[d.dst[i]]], [d.y[d.org[i]], d.y[d.dst[i]]], label="$i", markersize=D.W[i]+2,alpha=0.5) 
    end
    for i=1:d.m
        plot!([d.x[d.frm[i]], d.x[d.to[i]]], [d.y[d.frm[i]], d.y[d.to[i]]], 
              label="", color=:black, arrow=true, linewdith=D.cap[i]-3)
    end
    for (S,col) in extraCol
        if eltype(typeof(S)) == Int
            scatter!([d.x[i] for i=S], [d.y[i] for i=S], label="", 
                    markersize=8, color=col,alpha=0.7,linewidth=0)
        else
            for itm in S
                i,j = itm
                plot!([d.x[i], d.x[j]], [d.y[i], d.y[j]], 
                      label="", color=col, arrow=true, linewdith=4)
            end
        end
    end
    plot!(fmt=:png, legend=false, size=(600,400))
end

######################################################################
#%% 
using JuMP
import CPLEX as SOLVER
if false 
    function solveFullProblem(P::Data;LPrelax=true)
        N=1:P.n; A=1:P.m; K=1:P.k
        mip=Model(SOLVER.Optimizer);  #set_silent(master)
        @variable(mip, x[a=A,k=K]>=0)
        if LPrelax; @variable(mip, 0 <= y[a=A] <= 1);
        else;       @variable(mip, y[a=A], Bin); end
        @objective(mip, Min, sum( P.vcost[a]*sum(P.W[k]*x[a,k] for k=K) + P.fcost[a]*y[a] for a=A))
        @constraint(mip,flow[i=N,k=K],  sum(x[e,k] for e=P.outE[i]) - sum(x[e,k] for e=P.inE[i])
                                        == ( (i==P.org[k]) ? 1 : (i==P.dst[k] ? -1 : 0) ) )
        @constraint(mip,cap[a=A], sum(P.W[k]*x[a,k] for k=K) ≤ P.cap[a] * y[a] )
        @constraint(mip,fix[a=A,k=K], x[a,k] ≤ y[a])
        optimize!(mip)
        if termination_status(mip) != MOI.OPTIMAL
            return Inf,Float64[]
        end
        return objective_value(mip),value.(y) # objective and edges fixed
    end
    obj,y=solveFullProblem(dat)
    println("LP relaxation bound = ",obj)
end
#%%

#################################### Make data set ####################################
seed = 3526636178
Random.seed!(seed)
dat = randomGridFCNF(45, 120,20)


if false
    D = randomGridFCNF(32,100,10)

    seed = 1952470204
    if true
        for seed = [961940350,1952470204,3526636178,3398230935,2060017258,4273355581,2540217882,2331243338,2030331200,3132453327]
            Random.seed!(seed)
            #D = randomGridFCNF(45, 120,20) #randomData(15, 50, 10)
            D = randomGridFCNF(40, 90,12) #randomData(15, 50, 10)
            plotNetwork(D)
            sec = @elapsed (obj,yyy) = solveFullProblem(D)
            display(plot!(title="seed=$seed, sec=$sec, obj=$obj"))
        end
    end
    @time solveFullProblem(dat)
    #dat = randomData(15, 50, 10)
    plotNetwork(dat)
end
