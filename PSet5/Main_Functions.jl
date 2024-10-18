# Author: Nicolas Moreno
#--------------------------------------------#
#           PREAMBLE!
#--------------------------------------------#

#keyword-enabled structure to hold model primitives
@with_kw struct Primitives
    β::Float64       = 0.97 # Discount rate
    δ::Float64       = 0.06 # Depreciation rate
    α::Float64       = 0.36  # Capital's share
    γ::Float64       = 0.42 # Consumption's weight in U()
    σ::Float64       = 2    # Relative Risk Aversion
    θ::Float64       = 0.11 # Labor Income Tax
    N_j::Int64       = 66   # Total Number of Cohorts
    N_r::Int64       = 46   # Retirement Age
    nn_r::Int64      = N_j-N_r+1 # Years of Retirement
    nn_w::Int64      = N_r-1 # Years Working
    n_p::Float64     = 0.011 # Population Growth
    a_min::Float64   = 0     # Minimum Asset Level
    a_max::Float64   = 75    # Maximum Asset Level
    na::Int64        = 1000  # Number of capital grid points
    Zb::Float64      = 0.5   # Lowest Productivity
    Zg::Float64      = 3.0   # Highest Productivity
    nz::Int64        = 2    # Number of Productivity Levels 
    ntot::Int64      = na*nz # Total Dimension of State-Space per Working Cohort 
    Π::Matrix{Float64}  = [0.9261 1-0.9261;1-0.9811 0.9811] # Transition Matrix for z
    π::Matrix{Float64}  = [0.2037 0.7963] # Ergodic Distribution [P(High),P(Low)]
    a_grid::Array{Float64,1}  = collect(range(a_min, length = na, stop = a_max)) #asset grid
    z_grid::Array{Float64,1}  = collect(range(Zg, length = nz, stop = Zb)) #productivity grid
    AA_grid::Array{Float64,3} = repeat(a_grid',na,1,nz) #zeros(na,na,nz)
    η_j::Vector{Float64}      = map(x->parse(Float64,x), readlines("ef.txt"))
    e_mat::Matrix{Float64}    = z_grid.*η_j'
    max_iter::Float64 = 100000 # Max number of iterations for excess demand
    pers_::Float64 = 0.85 # Persistence of the updating
end

#structure that holds model results
mutable struct Results
    W::Float64
    R::Float64
    B::Float64
    V_R::AbstractArray{Float64} # Value function of Retirees
    V_W::AbstractArray{Float64} # Value function of Workers
    Pol_Func_R::AbstractArray{Float64} # Policy Function of Retirees
    Pol_Func_W::AbstractArray{Float64} # Policy Function of Workers
    L_Func_W::AbstractArray{Float64} # Policy Function of Workers
    cR::AbstractArray{Float64} # Consumption of Retirees
    μ::AbstractArray{Float64} # Policy Function of Retirees
    L::Float64
    K::Float64
    # budget::Array{Float64,3} 
    # c::Array{Float64,3} 
    # q_new::Float64
    # ED::Matrix{Float64}
end

# Initialize Results and Primitives
function Initialize()
    prim = Primitives() #initialize primtiives
    @unpack na,nz, ntot,N_j, N_r,nn_r, nn_w, AA_grid, a_grid,α ,δ,η_j = prim
    W = 1.459#1.05
    R = 0.023#0.05
    B = 0.226#0.2
    V_R  = zeros(na,nn_r) #initial value function guess
    V_W  = zeros(na,nz,nn_w) #initial value function guess
    Pol_Func_R  = zeros(na,nn_r) #initial policy function guess
    Pol_Func_W  = zeros(na,nz,nn_w) #initial policy function guess
    L_Func_W = zeros(na,nz,nn_w) #initial policy function guess
    budgetR    = zeros(na,na) 
    # budgetW    = zeros(na,na,nz) 
    budgetR = repeat((1+R).*a_grid .+ B,1,na)
    # budgetW[:,:,1] = repeat((1+R).*a_grid .+ z_grid[1],1,na)
    # budgetW[:,:,2] = repeat((1+R).*a_grid .+ z_grid[2],1,na) 
    #c::Array{Float64,3} = zeros(na,na,nz)
    # budgetR = repeat((1+R).*a_grid .+ B,1,na)
    cR             = budgetR - AA_grid[:,:,1] # Use Budget grid to find all consumptions
    cR[cR.<0]     .= 0       # Replace negative consumptions with zeros
    μ = zeros(na,nz,N_j)
    # Since we have guesses for wages and R (Normalize Y and P to 1):
    # R/P = MPK - δ = αY/K - δ
    # R = α/K - δ --> K = α/(R + δ)
    # W = (1-α)/L --> L = (1-α)/W
    L = (1-α)/W
    K = α/(R + δ)
    res = Results(W, R , B ,V_R,V_W,Pol_Func_R,Pol_Func_W,L_Func_W,cR,μ,L,K) #initialize results struct
    prim, res #return deliverables
end

#--------------------------------------------#
#           CAKE EATING!
#--------------------------------------------#

# A Cake Eating Problem #1:
function Retirees_Eating_Cake(prim::Primitives,res::Results)
    @unpack V_R, cR,Pol_Func_R,R,B= res 
    @unpack a_grid,AA_grid, β, σ ,γ, na,nn_r  = prim 
    # # At age 66, future value is zero. Thus, optimal choice is a'=0
    Pol_Func_R[:,nn_r] = zeros(na,1)
    budgetR        = repeat((1+R).*a_grid .+ B,1,na)
    cR             = budgetR - AA_grid[:,:,1] # Use Budget grid to find all consumptions
    cR[cR.<0]     .= 0       # Replace negative Consumption
    cR_nn_r = cR[:,1] # If a'=0, then c_R is (1+r)*a +b, which is first column of consumption grid
    V_R[:,nn_r] = (cR_nn_r.^(γ*(1-σ)))./(1-σ) # Then V_R is simply current consumption's utility
    # # Initialize an auxiliary Value Function=current consumptions:
    val = (cR.^(γ*(1-σ)))./(1-σ)
    # Start Iterating backwards from Age 65
    for jj =1:nn_r-1
        V_aux            = val .+ β.*V_R[:,end-(jj-1)]' # Knowing Next period's V(), add it along columns
        V_R[:,end-jj]    = maximum(V_aux,dims=2) # Value Function one step backwards
        max_index0       = argmax(V_aux,dims=2)
        max_index        = reshape([i[2] for i in max_index0],na,1) #cartesian_indices = max_index0[1]
        Pol_Func_R[:,end-jj]  = a_grid[max_index,1] 
    end
    return V_R, Pol_Func_R
end

# A Cake Eating Problem #2:
function Workers_Eating_Cake(prim::Primitives,res::Results)
    V_R, Pol_Func_R = Retirees_Eating_Cake(prim,res)
    @unpack V_W, Pol_Func_W,L_Func_W, R, B, W = res 
    @unpack β, σ ,γ, θ, Π ,e_mat = prim
    @unpack a_grid,AA_grid,z_grid, na, nz, ntot, nn_w  = prim 
    # # Last Period Working (The Only "Deterministic" Period for workers)
    e_aux = reshape(e_mat[:,nn_w]',1,1,nz)
    if γ == 1
        L = fill(1,na,na,nz)
    else
        L   = (γ*(1−θ).*e_aux.*W .− (1 − γ)*((1 + R).*a_grid .− AA_grid))./((1 − θ)*W.*e_aux)
        L[L.<0]     .= 0.0 
        L[L.>1]     .= 1
    end
    cW  = W*(1−θ).*e_aux.*L .+ (1 + R).*a_grid .− AA_grid
    cW[cW.<0]     .= 0.0 
    val = ((cW.^γ.*(1 .-L).^(1-γ)).^(1-σ))./(1-σ)
    V_aux       = val .+ β.*V_R[:,1]'
    V_W[:,:,nn_w] = reshape(maximum(V_aux,dims=2),na,nz)
    max_index0    = argmax(V_aux,dims=2)
    max_index     = reshape([i[2] for i in max_index0],na,nz) #cartesian_indices = max_index0[1]
    if nz==1
        Pol_Func_W[:,:,nn_w]  .= a_grid[max_index[:,1]]
    else
        Pol_Func_W[:,:,nn_w]  = [a_grid[max_index[:,1]] a_grid[max_index[:,2]]] #
    end
    L_Func_W[:,:,nn_w]    = reshape(L[max_index0],na,nz)
    # # Start Iterating backwards from Age 45
    Eval_func = copy(AA_grid)
    for jj =1:nn_w-1
        # Solve the Static Problem of each Cohort
        e_aux = reshape(e_mat[:,end-jj]',1,1,nz)
        if γ == 1
            L = fill(1,na,na,nz)
        else
            L   = (γ*(1−θ).*e_aux.*W .− (1 − γ)*((1 + R).*a_grid .− AA_grid))./((1 − θ)*W.*e_aux)
            L[L.<0]  .= 0.0 
            L[L.>1]  .= 1
        end
        cW  = W*(1−θ).*e_aux.*L .+ (1 + R).*a_grid .− AA_grid
        cW[cW.<0]     .= 0.0 
        val = ((cW.^γ.*(1 .-L).^(1-γ)).^(1-σ))./(1-σ)
        # Compute the Expected Value of Future Value Function:
        if nz >1
            Eval_func[:,:,1] .= (V_W[:,:,end-(jj-1)]*Π[1,:])'
            Eval_func[:,:,2] .= (V_W[:,:,end-(jj-1)]*Π[2,:])'
        else
            Eval_func = copy(V_W[:,:,end-(jj-1)])'
        end
        V_aux            = val .+ β.*Eval_func  # Knowing Next period's V(), add it along columns
        V_W[:,:,end-jj]  = maximum(V_aux,dims=2) # Value Function one step backwards, matrix with z's by column
        max_index0       = argmax(V_aux,dims=2)
        max_index        = reshape([i[2] for i in max_index0],na,nz) #cartesian_indices = max_index0[1]
        if nz==1
            Pol_Func_W[:,:,end-jj]  = a_grid[max_index[:,1]] #
        else
            Pol_Func_W[:,:,end-jj]  = [a_grid[max_index[:,1]] a_grid[max_index[:,2]]] #
        end
        L_Func_W[:,:,end-jj] = reshape(L[max_index0],na,nz)
    end
    return V_W, Pol_Func_W, V_R, Pol_Func_R,L_Func_W 
end

#--------------------------------------------#
#           DISTRIBUTIONS
#--------------------------------------------#
function EndoMat_create(prim::Primitives, res::Results) 
    V_W, Pol_Func_W,V_R, Pol_Func_R,L_Func_W = Workers_Eating_Cake(prim,res)
    @unpack a_grid, na, nz,N_j,nn_w, ntot,n_p, Π, π , β  = prim
    mat_aux = zeros(na,nz,N_j)  
    pop_growth = [1; (1+n_p).^(-collect(1:N_j-1))]
    pop_size   = pop_growth./sum(pop_growth)
    if nz ==1
        mat_aux[1,1,1] = 1
    else
        mat_aux[1,1,1] = π[1]#*pop_size[1]
        mat_aux[1,2,1] = π[2]#*pop_size[1]
    end
    
    for jj=1:(N_j-1)
        if jj<=nn_w
            pol_func = Pol_Func_W[:,:,jj]
            for ii_a = 1:na
                for ii_s = 1:nz
                    a_tomorrow =  pol_func[ii_a,ii_s]   
                    for kk_s = 1:nz 
                        row1 = argmin(abs.(a_tomorrow .- a_grid)) #+ na*(kk_s-1) # Position in grid of future asset
                        if jj == nn_w
                            mat_aux[row1,kk_s,jj+1] += mat_aux[ii_a,ii_s,jj] # Exogenous Probability of Transitioning there
                        else
                            mat_aux[row1,kk_s,jj+1] += Π[ii_s,kk_s].*mat_aux[ii_a,ii_s,jj] # Exogenous Probability of Transitioning there
                        # mat_aux[row,col,jj+1] = Π[ii_s,kk_s] # Exogenous Probability of Transitioning there
                        end
                    end
                end
            end
            # if jj == nn_w
            #     mat_aux[:,1:nz,jj+1] .= mat_aux[:,nz,jj+1]
            # end
        else
            pol_func = Pol_Func_R[:,jj-nn_w]
            for ii_a = 1:na
                a_tomorrow =  pol_func[ii_a]   
                row1 = argmin(abs.(a_tomorrow .- a_grid)) #+ na*(kk_s-1) # Position in grid of future asset
                # if jj == nn_w+1
                    # mat_aux[row1:row1,:,jj+1] .+= sum(mat_aux[ii_a,:,jj]) # Exogenous Probability of Transitioning there
                # else
                    mat_aux[row1,1,jj+1] += mat_aux[ii_a,1,jj] # Exogenous Probability of Transitioning there
                # end
            end
        end
        pol_func = nothing
    end
    # nz=2
    pop_size = reshape(pop_size,1,1,N_j)
    mat_aux[:,2,nn_w] .= 0
    res.μ = mat_aux.*pop_size # mat_aux';
    res.V_W = V_W;
    res.V_R = V_R;
    res.Pol_Func_W = Pol_Func_W;
    res.Pol_Func_R = Pol_Func_R;
    res.L_Func_W = L_Func_W;
    return res,pop_size
end

#--------------------------------------------#
#      MARKET CLEARING AND PRICES UPDATE
#--------------------------------------------#
function Market_Clearing(prim::Primitives, res::Results, tol::Float64 = 1e-2) 
    @unpack θ,δ,α,η_j  = prim
    @unpack a_grid, na, nz, N_j, nn_w, N_r,z_grid,pers_,e_mat  = prim
    err_MC = 100 
    it_count = 0
    while abs(err_MC)>tol
        res,pop_size = EndoMat_create(prim,res)
        @unpack W,R,B,K,L,μ,L_Func_W = res
        K_new = 0
        for jj=1:N_j
            local nzK = nz
            if jj >nn_w
                nzK =1
            end
            for ii_s = 1:nzK
                for ii_a = 1:na
                    K_new = K_new + μ[ii_a,ii_s,jj]*a_grid[ii_a]
                end
            end
        end
        L_new = 0
        for jj=1:nn_w
            for ii_s = 1:nz
                for ii_a = 1:na
                    L_new = L_new + μ[ii_a,ii_s,jj]*e_mat[ii_s,jj]*L_Func_W[ii_a,ii_s,jj]
                end
            end
        end
        Agg_quantities = [K-K_new, L-L_new]
        err_MC = norm(Agg_quantities,Inf)
        if abs(err_MC)>tol
            res.K = pers_*K + (1-pers_)*K_new
            res.L = pers_*L + (1-pers_)*L_new
            Y = (res.K^α)*(res.L^(1-α))
            res.W = (1-α)*Y/res.L
            res.R = α*Y/res.K - δ
            res.B = (θ*res.W*res.L)./sum(pop_size[1,1,N_r:end])
        end
        it_count = it_count+1
        println("Iteration # ", it_count," Norm is: ",err_MC)
    end
    println("Norm is: ",err_MC) 
    println("Aggregate Capital is: ",res.K)
    println("Aggregate Labor is: ",res.L)
    println("Wages are: ",res.W)
    println("Interest Rate is: ",res.R)
    println("Social Security Benefits are: ",res.B)
    println("Market Clearing Done!")
    return res
end

#--------------------------------------------#
#           WELFARE CALCULATIONS
#--------------------------------------------#
function Compute_Welfare(prim::Primitives, res::Results) 
    @unpack V_R, V_W, μ, K = res
    @unpack a_grid, na, nz,nn_r  = prim
    # Aggregate Welfare Calculation:
    Value_Vec = cat(V_W,reshape([V_R; V_R],na,nz,nn_r),dims=3)
    Welfare   = sum(replace(Value_Vec.*μ,NaN=>0))
    # Wealth's Coefficient of Variation:
    Wealth_Mean     = K     # By Mkt Clearing, Total Wealth equals Capital Stock, and Population = 1
    Wealth_StdDev   = sum(μ.*(a_grid .- K).^ 2)^0.5 # (Asset grid - K)^2 are squared deviations, then weighted sum using μ. Square root to find Std.Dev 
    Wealth_CV       = Wealth_StdDev/Wealth_Mean # Std Dev over Mean gives us Coefficient of Variation
    println("Aggregate Welfare is: ",Welfare)
    println("Wealth's Coefficient of Variation is: ", Wealth_CV)
    println("Done!")
    return Welfare, Wealth_CV
end

