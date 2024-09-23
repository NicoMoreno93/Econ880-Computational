
#keyword-enabled structure to hold model primitives
@with_kw struct Primitives
    β::Float64       = 0.9932 #discount rate
    q_guess::Float64 = 0.9937#27 
    α::Float64      = 1.5 #RRA
    a_min::Float64  = -2
    a_max::Float64  = 5
    na::Int64       = 1000 #number of capital grid points
    Zb::Float64     = 0.5
    Zg::Float64     = 1
    nz::Int64       = 2  
    ntot::Int64     = na*nz # Total Dimension of State-Space 
    Π::Matrix{Float64}  = [0.97 .03 ; 0.5 0.5]
    a_grid::Array{Float64,1}  = collect(range(a_min, length = na, stop = a_max)) #asset grid
    z_grid::Array{Float64,1}  = collect(range(Zg, length = nz, stop = Zb)) #productivity grid
    AA_grid::Array{Float64,3} = repeat(a_grid',na,1,nz) #zeros(na,na,nz)
    max_iter::Float64 = 100000 # Max number of iterations for excess demand
end

#structure that holds model results
mutable struct Results
    val_func::Array{Float64, 2} #value function 
    pol_func::Array{Float64, 2} #policy function
    distr::Array{Float64, 2}
    Trans_mat::Array{Float64, 2}
    #others::Array{Float64, 2}
    budget::Array{Float64,3} 
    c::Array{Float64,3} 
    q_new::Float64
    ED::Matrix{Float64}
end

#function for initializing model primitives and results
function Initialize()
    prim = Primitives() #initialize primtiives
    @unpack nz, ntot, na, q_guess, AA_grid, a_grid, z_grid = prim
    val_func  = zeros(na,nz) #initial value function guess
    pol_func  = zeros(na,nz) #initial policy function guess
    distr     = zeros(ntot,1) #initial policy function guess
    Trans_mat = zeros(ntot,ntot)
    budget    = zeros(na,na,nz)
    budget[:,:,1] = repeat(a_grid .+ z_grid[1],1,na)
    budget[:,:,2] = repeat(a_grid .+ z_grid[2],1,na) 
    #c::Array{Float64,3} = zeros(na,na,nz)
    c             = budget - q_guess.*AA_grid # Use Budget grid to find all consumptions
    c[c.<0]      .= 0       # Replace negative consumptions with zeros
    ED            = fill(100,1,1)
    q_new         = q_guess 
    res = Results(val_func, pol_func, distr, Trans_mat,budget,c,q_new,ED) #initialize results struct
    prim, res #return deliverables
end

function Value_Function(prim::Primitives,res::Results)
    @unpack val_func,c = res #unpack value function 
    @unpack a_grid,AA_grid, β, α, Π, z_grid, na , nz = prim 
    Eval_func = copy(AA_grid)
    v_next    = copy(val_func) #next guess of value function to fill
    Eval_func[:,:,1] .= (v_next*Π[1,:])'
    Eval_func[:,:,2] .= (v_next*Π[2,:])'
    val           = (c.^(1-α).-1)./(1-α) .+ β.*Eval_func # Calculate all possible values
    v_next        = reshape(maximum(val,dims=2),na,nz)
    max_index0    = argmax(val,dims=2)
    max_index     = reshape([i[2] for i in max_index0],na,nz) #cartesian_indices = max_index0[1]
    res.pol_func  = [a_grid[max_index[:,1]] a_grid[max_index[:,2]]] #reshape(AA_grid[max_index],na,nz)
    v_next #return next guess of value function
end

function Bellman_Operator(prim::Primitives, res::Results; tol::Float64 = 1e-4, err::Float64 = 100.0)
    # Initialize while
    n = 0 #counter
    # Update Consumption Grid since q changes
    c             = res.budget - res.q_new.*prim.AA_grid # Use Budget and Assets grids to find all consumptions
    c[c.<0]      .= 0       # Replace negative consumptions with zeros
    res.c         = c       # Update Consumption in Results structure
    while err>tol #begin iteration
        v_next = Value_Function(prim, res) # Find New VF
        # Sup Norm:
        err = norm(v_next.-res.val_func,Inf)
        res.val_func = v_next # Update VF
        n+=1
    end
    println("Value function converged in ", n, " iterations.")
end

#solve the model
function Solve_model(prim::Primitives, res::Results)
   Bellman_Operator(prim, res) #in this case, all we have to do is the value function iteration!
end

function EndoMat_create(prim::Primitives, res::Results) 
    @unpack val_func, pol_func = res
    @unpack a_grid, ntot, Π , β  = prim
    mat_aux=zeros(prim.ntot,prim.ntot)
    for ii_s = 1:prim.nz
        for ii_a = 1:prim.na
            a_tomorrow =  res.pol_func[ii_a,ii_s]   
            for kk_s = 1:prim.nz 
                for kk_a = 1:prim.na
                    if a_tomorrow == a_grid[kk_a]
                        row = ii_a + prim.na*(ii_s-1)
                        col = kk_a + prim.na*(kk_s-1)
                        mat_aux[row,col] = Π[ii_s,kk_s]
                    end
                end
            end
        end
    end
    res.Trans_mat = mat_aux';
    return res
end

function Tstar_Operator(prim::Primitives, res::Results; tol::Float64 = 1e-6, err::Float64 = 100.0)
    EndoMat_create(prim, res)
    Trans_mat  = res.Trans_mat
    res.distr  = ones(prim.ntot,1)/prim.ntot #should I take the last one?
    it=0
    distr0 = res.distr
    while err>tol
        dist_mu_iter = Trans_mat*distr0
        #Supnorm:
        err = norm(dist_mu_iter.-distr0,Inf)
        distr0 = dist_mu_iter./sum(dist_mu_iter)
        it+=1
    end
    res.distr = distr0
    #println("Tstar converged in ", it, " iterations.") 
    return res 
end

function Excess_Demand(prim::Primitives,res::Results)
    endo_tmat  = ones(1,prim.ntot)
    dist_aux   = ones(prim.ntot,1)

    mat_aux    = res.pol_func
    endo_tmat  = reshape(mat_aux,1,:) 
    dist_aux   = res.distr

    ED = endo_tmat*dist_aux
    res.ED =  ED #endo_tmat*res.distr
   return prim, res
end

function Huggett_Solve(prim::Primitives, res::Results; tol::Float64 = 1e-3)  
    it_count = 0;
    fs = it_count+10
    ED = sum(res.ED)
    while abs(ED)>tol 
        Bellman_Operator(prim, res)
        Tstar_Operator(prim, res)
        Excess_Demand(prim, res)   
        ED = sum(res.ED)
        q  = res.q_new
        if abs(ED)>tol 
            if ED>0
                q = min(q+(1-q)/fs,1)
            elseif ED<0
                q = max(q-(1-q)/fs,prim.β)
            end
            res.q_new = q
        end
        if it_count>fs/5
            fs = fs*10
        end   
        it_count = it_count+1
        println("Iteration # ", it_count)
    end
    println(it_count, ", ED is: ", res.ED) 
    println("Done!")
    res
end