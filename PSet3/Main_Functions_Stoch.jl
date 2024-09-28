#Keyword-enabled Structure holding model Primitives2
@with_kw struct Primitives2
    β::Float64        = 0.8  # Discount factor 
    θ::Float64        = 0.64 # Production function curvature
    A_::Float64       = 1/200 # Labor disutility parameter
    ce::Float64       = 5     # Fixed entry cost
    cf::Float64       = 10    # Fixed production cost
    ns::Int64         = 5     # Number of states
    # ntot::Int64       = na*nz # Total Dimension of State-Space 
    max_iter::Float64  = 100000 # Max number of iterations for excess demand
    α::Matrix{Float64}  = [1.0 2.0] # Variances of Type I EV shock
    γ_e::Float64       = MathConstants.eulergamma
    s_grid::Matrix{Float64}    = [3.98e-4 3.58 6.82 12.18 18.79] #productivity grid
    ν_s::Matrix{Float64}       = [0.37 0.4631 0.1102 0.0504 0.0063]
    F_s::Matrix{Float64}       = [0.6598 0.2600 0.0416 0.0331 0.0055;
                                  0.1997 0.7201 0.0420 0.0326 0.0056;
                                  0.2000 0.2000 0.5555 0.0344 0.0101;
                                  0.2000 0.2000 0.2502 0.3397 0.0101;
                                  0.2000 0.2000 0.2500 0.3400 0.0100]
    Pers_vec::Vector{Float64}  = diag(F_s) # Focus on how aborbent states are                             
    mu_guess::Matrix{Float64}  = [0.37 0.4631 0.1102 0.0504 0.0063]'#ones(ns,1)./ns # Initial Distribution
    P_guess::Float64           = 1           # Initial Price
end

# Structure holding Model Results
mutable struct Results2
    N_S::Float64 # Households Labor Decision 
    n_D::Array{Float64,2} # Firm's individual Labor Demand
    L_D::Matrix{Float64} # Aggregate Labor Demand 
    W_func::Array{Float64, 2} # Firm's Value function 
    U_s::Array{Float64, 2} # Firm's Value function
    x_prime::Matrix{Float64} # Firm's Dynamic & Discrete Choice
    mu_mat::Array{Float64, 2} # Firms Distribution across states
    M_entry::Float64       # Mass of entrants
    P_new::Float64         # Equilibrium Price (Pinned down by MEC)
    EC::Matrix{Float64}    # Entry Condition
    LMC::Matrix{Float64}   # Labor Market Clearing Condition
    Π::Matrix{Float64}     # Aggregate Profits
    π_s::Matrix{Float64}   # Vector of Profits by State
    α_indic::Int64         # Indicator of Alpha
end

#Initializing model results
function Initialize2()
    prim = Primitives2() #initialize primtiives
    @unpack ns, mu_guess, A_, P_guess, ce, ν_s = prim
    mu_mat    = copy(mu_guess)   # Set up initial distribution 
    M_entry   = 1.0        # Provide a first guess for entry
    Π         = ones(1,1).*0.0  # Initial level of aggregate profits
    π_s       = ones(ns,1)*Π    # Vector of initial level of profits by state
    P_new     = P_guess    # Start of Prices
    N_S       = P_new/A_ - sum(Π) # First guess for N supplied
    # mean_s    = sum(s_grid)/ns # Find a naive average productivity level
    # n_D       = (θ*P_new.*s_grid).^(1/(1-θ))# Start L_D at "average" L_D (not really, Jensen's inequality)
    n_D       = ones(ns,1)#.*L_D # Guess labor demands are all L_D
    L_D       = ν_s*n_D
    local E_W = P_new*ce./sum(ν_s) # Reverse engineering of Eqn 3, assuming all W(s|p) equal. sum(v_s) =1
    W_func    = ones(ns,1).*E_W # Start by assuming all firms have same value function
    U_s       = ones(ns,1).*E_W 
    x_prime   = zeros(ns,1)
    EC        = ones(1,1).*ce
    LMC       = ones(1,1).*0
    α_indic   = 1
    res = Results2(N_S, n_D,L_D, W_func,U_s,x_prime, mu_mat, M_entry,P_new,EC, LMC, Π, π_s,α_indic) #initialize results struct
    prim, res #return deliverables
end

function Value_Function(prim::Primitives2, res::Results2)
    @unpack β, α , γ_e, ns, θ, ce, cf, F_s, s_grid= prim
    @unpack n_D, P_new, W_func,U_s, π_s, x_prime,α_indic = res
    # Static Problem:
    α = sum(α[:,α_indic])
    n_D = ((θ*P_new*s_grid).^(1/(1-θ)))'
    π_s = P_new*s_grid'.*(n_D.^θ) - n_D .- P_new*cf
    res.n_D = n_D
    res.π_s = π_s
    # Dynamic Problem:
    E_W = copy(W_func)
    W_func_prime = copy(W_func)
    U_s_prime   = copy(U_s)
    for sss =1:ns
        E_W[sss,1]   = sum(F_s[sss,:]'*U_s_prime) # Compute conditional expectations
    end
    W_func0       = π_s + β.*E_W # If the firm stays
    W_func1       = π_s   # If the firm exits next period
    W_func_det    = hcat(W_func0,W_func1) # This would be the Deterministic Choice
    U_s_prime     = 1/α*(γ_e .+ log.(sum(exp.(convert(Matrix{BigFloat},α*W_func_det)), dims =2)))
    x_aux         = exp.(convert(Matrix{BigFloat},α*W_func_det))./sum(exp.(convert(Matrix{BigFloat},α*W_func_det)), dims =2) # Probability of Choosing 
    res.x_prime   = x_aux[:,2:2] #maximum(x_aux, dims=2) 
    x_indic       = argmax(x_aux,dims=2) # Find out what's the modal action
    # W_func_prime  = W_func_det[x_indic]
    return U_s_prime
end

function Bellman_Operator(prim::Primitives2, res::Results2; tol::Float64 = 1e-8, err::Float64 = 100.0)
    # Initialize while
    ii = 0 #counter
    while err>tol #begin iteration
        U_s_prime = Value_Function(prim, res) # Find New VF
        # Sup Norm:
        err = norm(U_s_prime.-res.U_s,Inf) # Inf Indicates sup norm (Max of Abs term by term)
        res.U_s    = U_s_prime # Update VF
        ii+=1
    end
    println("Value function converged in ", ii, " iterations.")
    W_func = copy(res.U_s)
    return W_func
end

function Entry_Condition(prim::Primitives2, res::Results2)
    W_func = Bellman_Operator(prim,res)
    res.W_func = W_func
    @unpack ce, ν_s = prim
    @unpack P_new = res
    EC1 = ν_s*W_func/P_new .- ce
    return EC1
end

function Finding_Price(prim::Primitives2, res::Results2; tol::Float64 = 1e-6, err::Float64 = 100.0)
    @unpack A_ = prim
    @unpack EC, W_func, Π, P_new = res
    ii= 1 # Counter
    # Define the initial bounds for the bisect algorithmÑ
    P_auxUB = P_new+0.5  # For this P, EC1 is positive
    P_auxLB = P_new/3    # For this P, EC1 is negative, so Root is inside 
    while err > tol
        EC1 = Entry_Condition(prim, res)
        #Calculate distance wrt zero:
        err = norm(EC1,Inf) 
        if err > tol
            if sum(EC1) > 0
                P_auxUB = res.P_new
            elseif sum(EC1) < 0
                P_auxLB = res.P_new 
            end
            res.P_new = (P_auxLB + P_auxUB)/2
        end
        
        ii = ii +1
        println("Iteration # ", ii-1)
        res.EC    = EC1
    end
    println("Terminal EC is: ", res.EC)
    println("Equilibrium Price: ", res.P_new)  
    println("Done!")
end

function New_Trans_Mat(prim::Primitives2, res::Results2)
    @unpack F_s, ν_s = prim
    @unpack M_entry, x_prime = res
    # Integrate out the conditional state to get unconditional dF(s')
    Uncond_F = F_s.*(1 .-x_prime) # Multiply by (1-x') to get actual dF(s')
    Mat_B    = I - Uncond_F' # Mat_B multiplies mu on LHS (Uncond_F must be transposed to multiply density of current states) 
    # Solve for mu, inverting B, and pre-multiply second term of eqn(4):
    M_entry_V = M_entry*Uncond_F'*ν_s'
    mu_mat = inv(Mat_B)*M_entry_V
    mu_mat_aux = Uncond_F'*mu_mat + M_entry*Uncond_F'*ν_s'
    mu_mat - mu_mat_aux
    return mu_mat 
end

function Market_Clearing(prim::Primitives2, res::Results2)
    @unpack ν_s, A_ = prim
    @unpack Π, n_D, LMC, π_s, M_entry, P_new = res
    LMC1 = copy(LMC)
    # Update Distribution of Firms:
    mu_mat = New_Trans_Mat(prim,res)
    # Calculate Profits and new N_S :
    Π          = π_s'*mu_mat + M_entry*ν_s*π_s
    res.N_S    = 1/A_ - sum(Π)  # This N_S is initially updated in Finding_Price(), later in Finding_Entry()
    # Computing actual L_D and then LMC:
    res.L_D = n_D'*mu_mat + M_entry*ν_s*n_D
    LMC1    = res.L_D .- res.N_S
    return LMC1,mu_mat 
end

function Finding_Entry(prim::Primitives2, res::Results2; tol::Float64 = 1e-5, err::Float64 = 100.0)
    @unpack M_entry = res
    ii= 1 # Counter
    # Define the initial bounds for the bisect algorithmÑ
    M_auxLB = M_entry - 0.5 # For this M, LMC is negative
    M_auxUB = M_entry + 10  # For this M, LMC is Positve, so Root is inside interval 
    while err > tol
        LMC1, mu_mat  = Market_Clearing(prim, res)
        #Calculate distance wrt zero:
        err = norm(LMC1,Inf)
        if err>tol
            if sum(LMC1) > 0
                M_auxUB = res.M_entry #max(res.M_entry - init_step*(ii+1)/ii, 0)
            elseif sum(LMC1) < 0
                M_auxLB = res.M_entry #res.M_entry + init_step*(ii+1)/ii
            end
            res.M_entry = (M_auxLB + M_auxUB)/2
        end
        ii = ii +1
        println("Iteration # ", ii)
        res.mu_mat = mu_mat
        res.LMC = LMC1
    end
    println("Terminal LMC is: ", res.LMC) 
    println("Mass of Entrants is: ", res.M_entry) 
    println("Done!")
end

function HR_Solve_Stoch(prim::Primitives2, res::Results2)
    Finding_Price(prim,res)
    Finding_Entry(prim,res)
end