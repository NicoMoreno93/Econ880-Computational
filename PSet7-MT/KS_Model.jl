#=
This file contains skeleton code for solving the Krusell-Smith model.

Table of Contents:
1. Setup model
    - 1.1. Primitives struct
    - 1.2. Results struct
2. Generate shocks
    - 2.1. Shocks struct
    - 2.2. Simulations struct
    - 2.3. functions to generate shocks
3. Solve HH problem
    - 3.1 utility function
    - 3.2 Bellman operator
    - 3.3 VFI algorithm
4. Solve model
    - 4.1 Simulate capital path
    - 4.2 Estimate regression
    - 4.3 Solve model
=#

######################### Part 1 - setup model #########################


@with_kw struct Primitives
    β::Float64 = 0.99           # discount factor
    α::Float64 = 0.36           # capital share
    δ::Float64 = 0.025          # depreciation rate
    ē::Float64 = 0.3271         # labor productivity

    z_grid::Vector{Float64} = [1.01, .99]      # grid for TFP shocks
    z_g::Float64 = z_grid[1]
    z_b::Float64 = z_grid[2]
    nz::Int64 = length(z_grid)

    ϵ_grid::Vector{Float64} = [1, 0]           # grid for employment shocks
    nϵ::Int64 = length(ϵ_grid)

    nk::Int64 = 31
    k_min::Float64 = 0.001
    k_max::Float64 = 15.0
    k_grid::Vector{Float64} = range(k_min, stop=k_max, length=nk) # grid for capital, start coarse

    nK::Int64 = 17
    K_min::Float64 = 11.0
    K_max::Float64 = 15.0
    K_grid::Vector{Float64} = range(K_min, stop=K_max, length=nK) # grid for aggregate capital, start coarse

end

@with_kw mutable struct Results
    Z::AbstractArray{Float64}                      # aggregate shocks
    E::Matrix{Float64}                      # employment shocks

    V::Array{Float64, 4}                    # value function, dims (k, ϵ, K, z)
    k_policy::Array{Float64, 4}             # capital policy, similar to V

    a₀::Float64                             # constant for capital LOM, good times
    a₁::Float64                             # coefficient for capital LOM, good times
    b₀::Float64                             # constant for capital LOM, bad times
    b₁::Float64                             # coefficient for capital LOM, bad times
    R²::Float64                             # R² for capital LOM

    K_path::Vector{Float64}                 # path of capital

end


######################### Part 2 - generate shocks #########################

@with_kw struct Shocks
    #parameters of transition matrix:
    d_ug::Float64 = 1.5 # Unemp Duration (Good Times)
    u_g::Float64 = 0.04 # Fraction Unemp (Good Times)
    d_g::Float64 = 8.0  # Duration (Good Times)
    u_b::Float64 = 0.1  # Fraction Unemp (Bad Times)
    d_b::Float64 = 8.0  # Duration (Bad Times)
    d_ub::Float64 = 2.5 # Unemp Duration (Bad Times)

    #transition probabilities for aggregate states
    pgg::Float64 = (d_g-1.0)/d_g
    pbg::Float64 = 1.0 - pgg #(d_g-1.0)/d_g
    pbb::Float64 = (d_b-1.0)/d_b
    pgb::Float64 = 1.0 - pbb #(d_b-1.0)/d_b

    #transition probabilities for aggregate states and staying unemployed
    pgg00::Float64 = (d_ug-1.0)/d_ug
    pbb00::Float64 = (d_ub-1.0)/d_ub
    pbg00::Float64 = 1.25*pbb00
    pgb00::Float64 = 0.75*pgg00

    #transition probabilities for aggregate states and becoming employed
    pgg01::Float64 = (u_g - u_g*pgg00)/(1.0-u_g)
    pbb01::Float64 = (u_b - u_b*pbb00)/(1.0-u_b)
    pbg01::Float64 = (u_b - u_g*pbg00)/(1.0-u_g)
    pgb01::Float64 = (u_g - u_b*pgb00)/(1.0-u_b)

    #transition probabilities for aggregate states and becoming unemployed
    pgg10::Float64 = 1.0 - (d_ug-1.0)/d_ug
    pbb10::Float64 = 1.0 - (d_ub-1.0)/d_ub
    pbg10::Float64 = 1.0 - 1.25*pbb00
    pgb10::Float64 = 1.0 - 0.75*pgg00

    #transition probabilities for aggregate states and staying employed
    pgg11::Float64 = 1.0 - (u_g - u_g*pgg00)/(1.0-u_g)
    pbb11::Float64 = 1.0 - (u_b - u_b*pbb00)/(1.0-u_b)
    pbg11::Float64 = 1.0 - (u_b - u_g*pbg00)/(1.0-u_g)
    pgb11::Float64 = 1.0 - (u_g - u_b*pgb00)/(1.0-u_b)

    # Markov Transition Matrix
    Mgg::Array{Float64,2} = [pgg11 pgg01
                            pgg10 pgg00]

    Mbg::Array{Float64,2} = [pgb11 pgb01
                            pgb10 pgb00]

    Mgb::Array{Float64,2} = [pbg11 pbg01
                            pbg10 pbg00]

    Mbb ::Array{Float64,2} = [pbb11 pbb01
                             pbb10 pbb00]

    M::Array{Float64,2} = [pgg*Mgg pgb*Mgb
                          pbg*Mbg pbb*Mbb]

    # aggregate transition matrix
    Mzz::Array{Float64,2} = [pgg pbg
                            pgb pbb]
end


@with_kw struct Simulations
    T::Int64 = 11_000           # number of periods to simulate
    N::Int64 = 5_000            # number of agents to simulate
    seed::Int64 = 234          # seed for random number generator

    V_tol::Float64 = 1e-9       # tolerance for value function iteration
    V_max_iter::Int64 = 10_000  # maximum number of iterations for value function

    burn::Int64 = 1_000         # number of periods to burn for regression
    reg_tol::Float64 = 5e-2#1e-6     # tolerance for regression coefficients
    reg_max_iter::Int64 = 10_000 # maximum number of iterations for regression
    λ::Float64 = 0.5            # update parameter for regression coefficients

    K_initial::Float64 = 12.5   # initial aggregate capital
end


function sim_Markov(current_index::Int64, Π::Matrix{Float64})
    #=
    Simulate the next state index given the current state index and Markov transition matrix

    Args
    current_index (Int): index current state
    Π (Matrix): Markov transition matrix, rows must sum to 1
    
    Returns
    next_index (Int): next state index
    =#
    
    # Generate a random number between 0 and 1
    rand_num = rand()

    # Get the cumulative sum of the probabilities in the current row
    cumulative_sum = cumsum(Π[current_index, :])

    # Find the next state index based on the random number
    next_index = searchsortedfirst(cumulative_sum, rand_num)

    return next_index
end


function DrawShocks(prim::Primitives, sho::Shocks, sim::Simulations)
    #=
    Generate a sequence of aggregate shocks

    Args
    prim (Primitives): model parameters
    sho (Shocks): shock parameters
    sim (Simulations): simulation parameters

    Returns
    Z (Vector): matrix of aggregate shocks, length T
    E (Matrix): matrix of employment shocks, size N x T
    =#
    @unpack_Primitives prim
    @unpack_Simulations sim
    @unpack_Shocks sho
    
    # Start by setting the seed
    Random.seed!(seed) 

    # # Generate a path for the aggregate exogenous state Z:
    Z = zeros(T,1) # Pre-allocate
    Z[1,1] = z_g # We can assume z_b too, as the burning periods eliminate initial condition dependence
    current_index = 1 # Good Z corresponds to first row of Trans Mat
    Π_z = copy(Mzz')
    for jj=1:(T-1)
        next_index = sim_Markov(current_index,Π_z) 
        Z[jj+1,1] = z_grid[next_index]
        current_index = next_index
    end
    # # Generate a path for the idiosyncratic exogenous states e:
    E = zeros(N,T) # Pre-allocate 
    E[:,1] .= ϵ_grid[1]  # Again, we can assume z_g and e=1 for all 5000
    current_index = 1 
    for rr=1:(N-1) 
        for cc=1:(T-1)
            # Because Aggregate uncertainty "comes first", idiosyncratic probabilities depend on z in t and t+1
            z0 = Z[cc,1]
            z1 = Z[cc+1,1] 
            if z0 == z_g && z1 == z_g
                Π_zz = copy(Mgg)
            elseif z0 == z_g && z1 == z_b
                Π_zz = copy(Mgb)
            elseif z0 == z_b && z1 == z_g
                Π_zz = copy(Mbg)
            else
                Π_zz = copy(Mbb)
            end
            if cc >1
                current_index = findall(E[rr,cc] .== ϵ_grid)[1]
            else
                current_index = 1
            end
            next_index = sim_Markov(current_index,Π_zz) 
            E[rr+1,cc+1] = ϵ_grid[next_index]
            current_index = next_index
        end
    end
    return Z, E
end


function Initialize()
    prim = Primitives()
    sho = Shocks()
    sim = Simulations()
    Z, E = DrawShocks(prim, sho, sim)
    
    V = zeros(prim.nk, prim.nϵ, prim.nK, prim.nz)
    k_policy = zeros(prim.nk, prim.nϵ, prim.nK, prim.nz)

    a₀ = 0.095
    a₁ = 0.5999 
    b₀ = 0.085
    b₁ = 0.5999
    R² = 0.0

    K_path = zeros(sim.T)
    res = Results(Z, E, V, k_policy, a₀, a₁, b₀, b₁, R², K_path)
    return prim,sho,sim, res
end

######################### Part 3 - HH Problem #########################

function u(c::Float64; ε::Float64 = 1e-16)
    #Define the utility function, with stitching function for numerical optimization
    if c > ε
        return log(c)
    else # a linear approximation for stitching function
        # ensures smoothness for numerical optimization
        return log(ε) - (ε - c) / ε
    end
end


function Bellman(prim::Primitives, res::Results, sho::Shocks)
    #= 
    Solve the Bellman equation for the household problem

    Args
    prim (Primitives): model parameters
    res (Results): results struct
    sho (Shocks): shock parameters

    Returns
    V_next (Array): updated value function
    k_next (Array): updated capital policy function
    =#

    @unpack_Primitives prim
    @unpack_Results res
    @unpack_Shocks sho

    V_next = zeros(nk, nϵ, nK, nz)
    k_next = zeros(nk, nϵ, nK, nz)

    #linear interpolation for employed in good times value function
    interpg1 = interpolate(V[:, 1, : ,1], BSpline(Linear())) # Performs the interpolation, but this is only for gaps "inside" the grid
    # extrap1 = extrapolate(interpg1, Line())              # gives linear extrapolation off grid
    Vg1_interp = scale(interpg1, range(k_min, k_max, nk),range(K_min, stop=K_max, length=nK)) # has to be scaled on increasing range object. Add K_grid as dimension
    
    #linear interpolation for employed in bad times value function
    interpb1 = interpolate(V[:, 1, : ,2], BSpline(Linear())) # Performs the interpolation, but this is only for gaps "inside" the grid
    # extrap1 = extrapolate(interpb1, Line())              # gives linear extrapolation off grid
    Vb1_interp = scale(interpb1, range(k_min, k_max, nk),range(K_min, stop=K_max, length=nK)) # has to be scaled on increasing range object. Add K_grid as dimension

    #linear interpolation for unemployed in good times value function
    interpg0 = interpolate(V[:, 2, : ,1], BSpline(Linear()))
    # extrap0 = extrapolate(interpg0, Line())
    Vg0_interp = scale(interpg0, range(k_min, k_max, nk),range(K_min, stop=K_max, length=nK))
        
    #linear interpolation for unemployed in bad times value function
    interpb0 = interpolate(V[:, 2, : ,2], BSpline(Linear()))
    # extrap0 = extrapolate(interp0, Line())
    Vb0_interp = scale(interpb0, range(k_min, k_max, nk),range(K_min, stop=K_max, length=nK))

    # Create the Bellman function
    for (z_index,z) in enumerate(z_grid)
        for (KK_index,K) in enumerate(K_grid)
            # The aggregate state directly affects total labor. It also affects how we forecast K:
            if z_index ==1
                L = ē*(1-u_g)
                KK_prime = a₀ + a₁*log(K)
            else
                L = ē*(1-u_b)
                KK_prime = b₀ + b₁*log(K)
            end
            KK_prime = clamp(exp(KK_prime),K_min,K_max)
            # Change Prices as Z and Capital fluctuate
            w = (1-α)*z* (K / L) ^ α 
            r = α*z* (L / K) ^ (1-α)
            for (ϵ_index, ϵ) in enumerate(ϵ_grid)
                p = M[ϵ_index + nϵ*(z_index-1), :]
                for (k_index, k) in enumerate(k_grid)
                    budget = w * ϵ + (r + 1.0 - δ) * k
                    obj(k_prime) = -(u(budget - k_prime) + β *(p[1]*Vg1_interp(k_prime,KK_prime) + p[2]*Vg0_interp(k_prime,KK_prime) + p[3]*Vb1_interp(k_prime,KK_prime) +  p[4]*Vb0_interp(k_prime,KK_prime)))
                    # This looks for a min, hence the negative in front of u in prev line!
                    res = optimize(obj, k_min, min(budget,k_max)) #1st arg = V,2nd=lower_bound,3rd=upper bound

                    if res.converged
                        V_next[k_index, ϵ_index,KK_index,z_index] = -res.minimum
                        k_next[k_index, ϵ_index,KK_index,z_index] = res.minimizer
                    else
                        error("Optimization did not converge")
                    end
                end
            end
        end
    end
    return V_next, k_next
end


function VFI(prim::Primitives, res::Results, sho::Shocks; tol, max_iter)
    #=
    Iterate on the value function until convergence

    Args
    prim (Primitives): model parameters
    res (Results): results struct
    sim (Simulations): simulation parameters
    =#
    error = 100 * tol
    iter = 0

    while error > tol && iter < max_iter
        V_next, k_next = Bellman(prim, res,sho)
        error = norm(V_next - res.V,Inf)
        res.V = V_next
        res.k_policy = k_next
        iter += 1
        println("VF iteration # ",iter)
    end

    if iter == max_iter
        println("Maximum iterations reached in VFI")
    elseif error < tol
        println("Converged in VFI after $iter iterations")
    end
    return res
end


########################### Part 4 - Solve model ###########################


function SimulateCapitalPath(prim::Primitives, res::Results, sim::Simulations)
    #=
    Simulate the path of K
    Args
    prim (Primitives): model parameters
    res (Results): results struct
    sim (Simulations): simulation parameters

    Returns
    K_path (Vector): path of capital
    =#
    @unpack_Primitives prim
    @unpack_Results res
    @unpack_Simulations sim
    # Initialize Capitals' Paths:
    K_path[1]     = K_initial
    k_panel       = zeros(size(E))
    k_panel[:,1] .= K_path[1]
    # Since we interpolated for (k,K) we need to do it now again:
    interp_k_policy0 = interpolate(k_policy, BSpline(Linear())) 
    # interp_k_policy0 = extrapolate(interp_k_policy0, Line()) 
    interp_k_policy = scale(interp_k_policy0, range(k_min, k_max, nk),1:nϵ,range(K_min, stop=K_max, length=nK),1:nz)
    # Simulating the economy
    for tt =1:(T-1)
        # Extract current aggregate states:
        K_t = K_path[tt]
        z_t = Z[tt]
        zz = findall(z_grid.== z_t)
        for nn =1:N
            # Extract current states for person nn:
            k_t = k_panel[nn,tt] 
            ε_t = E[nn,tt]
            # Find the shocks positions:
            εε = findall(ϵ_grid.== ε_t)
            k_panel[nn,tt+1] = interp_k_policy(k_t,εε,K_t,zz)[1]
        end
        K_path[tt+1] = sum(k_panel[:,tt+1])/N
    end
    # res.K_path = K_path
    return K_path
end


function EstimateRegression(prim::Primitives, res::Results, sim::Simulations)
    #=
    Estimate the law of motion for capital with log-log regression

    Args
    prim (Primitives): model parameters
    res (Results): results struct
    sim (Simulations): simulation parameters

    Returns
    a₀ (Float): constant for capital LOM, good times
    a₁ (Float): coefficient for capital LOM, good times
    b₀ (Float): constant for capital LOM, bad times
    b₁ (Float): coefficient for capital LOM, bad times
    R² (Float): R² for capital LOM
    =#
    @unpack_Primitives prim
    @unpack_Results res
    @unpack_Simulations sim
    # Drop the burning periods:
    K_path = K_path[burn+1:end]
    K_tt   = K_path[2:end]
    K_lag  = K_path[1:end-1]
    # Sort the data:
    auxZ    = Z[burn+1:end]
    auxZ    = auxZ[1:end-1]
    good_times = findall(auxZ.== z_g)
    bad_times  = findall(auxZ.== z_b)
    n_good     = size(good_times)[1]
    n_bad      = size(bad_times)[1]
    K_good   = K_tt[good_times]
    K_bad    = K_tt[bad_times]
    K_good_l = K_lag[good_times]
    K_bad_l  = K_lag[bad_times]
    # Run the AR(1) regresions:
    Y_g = log.(K_good)
    X_g = [ones(n_good,1) log.(K_good_l)]
    beta_g = inv(X_g'X_g)*X_g'Y_g
    Y_b = log.(K_bad)
    X_b = [ones(n_bad,1) log.(K_bad_l)]
    beta_b = inv(X_b'X_b)*X_b'Y_b
    # Find K_hat, Residuals and R²:
    Y_hat = [X_g*beta_g ; X_b*beta_b]
    Res_  = [Y_g ;Y_b] - Y_hat
    SSYg   = sum((Y_g .- sum(Y_g)./(n_good+n_bad-2)).^2)
    SSYb   = sum((Y_b .- sum(Y_b)./(n_good+n_bad-2)).^2)
    R²    = 1- sum(Res_.^2)/ (SSYg + SSYb)
    # Organize a bit:
    a₀ = beta_g[1]
    a₁ = beta_g[2]
    b₀ = beta_b[1]
    b₁ = beta_b[2]
    return a₀,a₁,b₀,b₁,R²
end


function Solve_KS_Model()
    #=
    Solve the Krusell-Smith model

    Returns
    res (Results): results struct
    =#
    prim,sho,sim, res = Initialize()
    @unpack V_tol, V_max_iter, reg_tol, reg_max_iter, λ = sim
    iter_reg = 0
    err_ = 1000
    while err_>reg_tol && iter_reg <= reg_max_iter
        res = VFI(prim, res, sho; tol= V_tol,max_iter=V_max_iter)
        K_path = SimulateCapitalPath(prim, res, sim)
        res.K_path = K_path 
        new_a₀,new_a₁,new_b₀,new_b₁,new_R² = EstimateRegression(prim, res, sim)
        err_ = abs(res.a₀-new_a₀) + abs(res.a₁-new_a₁) + abs(res.b₀-new_b₀) + abs(res.b₁-new_b₁)
        if err_>reg_tol
            res.a₀ = λ*new_a₀ + (1-λ)*res.a₀ 
            res.a₁ = λ*new_a₁ + (1-λ)*res.a₁
            res.b₀ = λ*new_b₀ + (1-λ)*res.b₀
            res.b₁ = λ*new_b₁ + (1-λ)*res.b₁
        else
            res.a₀ = new_a₀ 
            res.a₁ = new_a₁
            res.b₀ = new_b₀
            res.b₁ = new_b₁
        end
        res.R² = new_R²
        iter_reg += 1
        println("KS iteration # ",iter_reg)
        println("Norm # ",err_)
        println("a₀ = ",res.a₀)
        println("a₁ = ",res.a₁)
        println("b₀ = ",res.b₀)
        println("b₁ = ",res.b₁)
        println("R² = ",res.R²)
    end
    if iter_reg == reg_max_iter
        println("Maximum iterations reached in KS Algorithm")
    elseif err_ < reg_tol
        println("Converged in KS after",iter_reg," iterations")
    end
    return prim, res
end

