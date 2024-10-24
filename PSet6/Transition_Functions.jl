# Author: Nicolas Moreno
#--------------------------------------------#
#           PREAMBLE!
#--------------------------------------------#

#keyword-enabled structure to hold model primitives
@with_kw struct TP_Primitives
    T_::Int64        = 30+1   # Total Number of Periods for Transition + Initial SS
    K0_path::Array{Float64,1} = collect(range(3.37230586033587, length = T_, stop = 4.62094705131206)) # Initial K_path spanning from Kss with Social to Kss without Social 
    change_T::Int64 = 2#21 # Max number of iterations for excess demand
    max_iter::Float64 = 100000 # Max number of iterations for excess demand
    tol_K::Float64 = 1e-2 # Persistence of the updating
    tol_T::Float64 = 1e-3 # Persistence of the updating
end

#structure that holds model results
mutable struct TP_Results
    K_path::AbstractArray{Float64} # Capital path (Updated in each iteration "i")
    L_path::AbstractArray{Float64} # Aggregate Labor path (Updated in each iteration "i")
    W_path::AbstractArray{Float64} # Wages path (Updated in each iteration "i")
    R_path::AbstractArray{Float64} # Real Interest Rate paths (Updated in each iteration "i")
    B_path::AbstractArray{Float64} # Social Security Benefits Path
    θ_path::AbstractArray{Float64} # Social Security Tax Path
    V_R_path::AbstractArray{Float64} # Value function of Retirees
    V_W_path::AbstractArray{Float64} # Value function of Workers
    PF_R_path::AbstractArray{Float64} # Policy Function of Retirees
    PF_W_path::AbstractArray{Float64} # Policy Function of Workers
    LF_W_path::AbstractArray{Float64} # Policy Function of Workers
    cR_path::AbstractArray{Float64} # Consumption of Retirees
    μ_path::AbstractArray{Float64} # Policy Function of Retirees
    μ_ss::AbstractArray{Float64} # Policy Function of Retirees
    # L::Float64
    # K::Float64
end

# Initialize TP_Results and TP_Primitives
function Initialize2(res1,res2)
    TP_prim = TP_Primitives() #initialize primitives
    @unpack T_,K0_path,change_T = TP_prim
    @unpack na,nz,nn_r,nn_w,N_j,N_r,δ,α,θ = prim
    @unpack L,K,μ = res1
    K_0 =K
    L_0 = L
    μ_0 = μ
    @unpack L, K, μ, V_R,V_W,Pol_Func_R,Pol_Func_W,L_Func_W = res2
    L_1 = L
    K_1 = K
    μ_1 = μ
    K_path = log.(collect(range(exp(K_0), length=T_ , exp(K_1)))) # Improve the guess, let's do concave from start
    # collect(range(K_0, length = T_, stop = K_1)) 
    L_path = collect(range(L_0, length = T_, stop = L_1))
    Y_path = K_path.^α.*L_path.^(1-α) 
    W_path = (1-α)*Y_path./L_path
    R_path = α*Y_path./K_path .- δ
    θ_path = [θ*ones(change_T-1,1)*0; zeros(T_-(change_T-1),1)]
    μ_prom = (μ_0[:,1,N_r:end].+ μ_1[:,1,N_r:end])/2
    B_path = (θ_path.*W_path.*L_path)./sum(μ_prom)    
    # Pre-allocation of Hypermatrices:
    V_R_path   = zeros(na,nn_r,T_) #initial value function guess
    V_W_path   = zeros(na,nz,nn_w,T_) #initial value function guess1
    PF_R_path  = zeros(na,nn_r,T_) #initial policy function guess
    PF_W_path  = zeros(na,nz,nn_w,T_) #initial policy function guess
    LF_W_path  = zeros(na,nz,nn_w,T_) #initial policy function guess
    cR_path    = zeros(na,na) 
    μ_path     = zeros(na,nz,N_j,T_)
    μ_ss       = cat(μ_0,μ_1,dims=3)
    # Refine Guess by ensuring it ends with new SS
    V_R_path[:,:,T_]     = V_R
    V_W_path[:,:,:,T_]   = V_W
    PF_R_path[:,:,T_]    = Pol_Func_R
    PF_W_path[:,:,:,T_]  = Pol_Func_W
    LF_W_path[:,:,:,T_]  = L_Func_W
    μ_path[:,:,:,1]     = μ_0
    TP_res = TP_Results(K_path, L_path, W_path, R_path, B_path, θ_path,V_R_path,V_W_path,PF_R_path,PF_W_path,LF_W_path,cR_path,μ_path,μ_ss); #initialize TP_Results struct
    TP_prim, TP_res; #return deliverables
end

#--------------------------------------------#
#           CAKE EATING!
#--------------------------------------------#

# A Cake Eating Problem with Changing Cakes #1:
function Retirees_Eating_Cake2(TP_prim::TP_Primitives,TP_res::TP_Results,prim::Primitives)
    @unpack T_  = TP_prim     
    @unpack a_grid,AA_grid, β, σ ,γ, na,nn_r  = prim 
    @unpack cR_path,R_path, B_path = TP_res 
    for tt=1:(T_-1)
        # These steps before next loop are always the "same", because they initialize backward induction through generations using death as a starting point
        budgetR  = repeat((1+R_path[end-tt]).*a_grid .+ B_path[end-tt],1,na) # This is the key line, this changes along transition (R and B)
        cR_path  = clamp.(budgetR - AA_grid[:,:,1],0,Inf) # Use Budget grid to find all consumptions
        cR_nn_r  = cR_path[:,1] # If a'=0, then c_R = (1+r)*a +b, which is first column of consumption grid
        TP_res.V_R_path[:,nn_r,T_-tt] = (cR_nn_r.^(γ*(1-σ)))./(1-σ) # Then V_R is simply current consumption's utility
        # Initialize the auxiliary Instant Utility= u(current consumptions):
        val = (cR_path.^(γ*(1-σ)))./(1-σ)
        # Start Iterating backwards from Age 65
        for jj =1:nn_r-1
            V_aux            = val .+ β.*TP_res.V_R_path[:,end-(jj-1),T_-(tt-1)]' # Knowing Next period's V(), add it along columns
            TP_res.V_R_path[:,end-jj,T_-tt]    = maximum(V_aux,dims=2) # Value Function one step backwards
            max_index0       = argmax(V_aux,dims=2)
            max_index        = reshape([i[2] for i in max_index0],na,1) #cartesian_indices = max_index0[1]
            TP_res.PF_R_path[:,end-jj,T_-tt]  = a_grid[max_index,1] 
        end
        println("Period Completed for Retirees ",T_-tt)
    end
    # TP_res.V_R_path  = V_R_path
    # TP_res.PF_R_path = PF_R_path
    V_R1 = TP_res.V_R_path[:,1,:]
    return V_R1;
end

# A Cake Eating Problem with Changing Cakes  #2:
function Workers_Eating_Cake2(TP_prim::TP_Primitives,TP_res::TP_Results,prim::Primitives)
    V_R1 = Retirees_Eating_Cake2(TP_prim,TP_res,prim)
    @unpack T_  = TP_prim  ;
    @unpack β, σ ,γ, Π ,e_mat,a_grid,AA_grid, na, nz, nn_w = prim;
    @unpack W_path, R_path,θ_path = TP_res; 
    Eval_func = copy(AA_grid)
    e_mat_aux = reshape(e_mat',1,nn_w,nz)
    for tt=1:(T_-1)
        # # Last Period Working (The Only "Deterministic" Period for workers)
        e_aux = e_mat_aux[:,nn_w:nn_w,:]
        L   = clamp.((γ*(1−θ_path[end-tt]).*e_aux.*W_path[end-tt] .− (1 − γ)*((1 + R_path[end-tt]).*a_grid .− AA_grid))./((1 − θ_path[end-tt])*W_path[end-tt].*e_aux),0,1)
        cW  = clamp.(W_path[end-tt]*(1−θ_path[end-tt]).*e_aux.*L .+ (1 + R_path[end-tt]).*a_grid .− AA_grid,0,Inf)
        # Initialize the auxiliary Instant Utility= u(current consumptions): 
        val = ((cW.^γ.*(1 .-L).^(1-γ)).^(1-σ))./(1-σ)
        V_aux       = val .+ β.*V_R1[:,T_-(tt-1)]'
        TP_res.V_W_path[:,:,nn_w,T_-tt] = reshape(maximum(V_aux,dims=2),na,nz)
        max_index0    = argmax(V_aux,dims=2)
        max_index     = reshape([i[2] for i in max_index0],na,nz) #cartesian_indices = max_index0[1]
        TP_res.PF_W_path[:,:,nn_w,T_-tt]  = [a_grid[max_index[:,1]] a_grid[max_index[:,2]]] #
        TP_res.LF_W_path[:,:,nn_w,T_-tt]  = reshape(L[max_index0],na,nz)
        # # Start Iterating backwards from Age 45
        for jj =1:nn_w-1
            # Solve the Static Problem of each Cohort
            e_aux = e_mat_aux[:,(end-jj):(end-jj),:]
            L   = clamp.((γ*(1−θ_path[end-tt]).*e_aux.*W_path[end-tt] .− (1 − γ)*((1 + R_path[end-tt]).*a_grid .− AA_grid))./((1 − θ_path[end-tt])*W_path[end-tt].*e_aux),0,1)
            cW  = clamp.(W_path[end-tt]*(1−θ_path[end-tt]).*e_aux.*L .+ (1 + R_path[end-tt]).*a_grid .− AA_grid,0,Inf)
            val = ((cW.^γ.*(1 .-L).^(1-γ)).^(1-σ))./(1-σ)
            # Compute the Expected Value of Future Value Function:
            Eval_func[:,:,1] .= (TP_res.V_W_path[:,:,end-(jj-1),T_-(tt-1)] *Π[1,:])' 
            Eval_func[:,:,2] .= (TP_res.V_W_path[:,:,end-(jj-1),T_-(tt-1)] *Π[2,:])'
            V_aux             = val .+ β.*Eval_func  # Knowing Next period's V(), add it along columns
            TP_res.V_W_path[:,:,end-jj,T_-tt]  = maximum(V_aux,dims=2) # Value Function one step backwards, matrix with z's by column
            max_index0       = argmax(V_aux,dims=2)
            max_index        = reshape([i[2] for i in max_index0],na,nz) #cartesian_indices = max_index0[1]
            TP_res.PF_W_path[:,:,end-jj,T_-tt]  = [a_grid[max_index[:,1]] a_grid[max_index[:,2]]] #
            TP_res.LF_W_path[:,:,end-jj,T_-tt] = reshape(L[max_index0],na,nz)
        end
        println("Period Completed for Workers ",T_-tt)
    end
    # return V_W_path, PF_W_path,V_R_path, PF_R_path,LF_W_path;
end

#--------------------------------------------#
#           DISTRIBUTIONS
#--------------------------------------------#
function EndoMat_create_TP(TP_prim::TP_Primitives, TP_res::TP_Results,prim::Primitives) 
    Workers_Eating_Cake2(TP_prim,TP_res,prim);
    @unpack a_grid, na, nz,N_j,nn_w,n_p, Π, π   = prim
    @unpack T_  = TP_prim
    @unpack μ_path,PF_R_path,PF_W_path  = TP_res
    pop_growth = [1; (1+n_p).^(-collect(1:N_j-1))]
    pop_size   = pop_growth./sum(pop_growth)
    pop_size   = reshape(pop_size,1,1,N_j)
    μ_path[:,:,:,1] = μ_path[:,:,:,1]./pop_size
    μ_path[:,:,:,2:end] .= 0 
    μ_path[1,1,1,:] .= π[1] # Ensuring that every distribution draws new generations from ergodic distribution
    μ_path[1,2,1,:] .= π[2] # Ensuring that every distribution draws new generations from ergodic distribution
    for tt=1:T_-1    
        for jj=1:(N_j-1) 
            if jj<=nn_w
                for ii_a = 1:na
                    for ii_s = 1:nz
                        a_tomorrow =  PF_W_path[ii_a,ii_s,jj,tt]   # Chosen future asset 
                        for kk_s = 1:nz 
                            row1 = argmin(abs.(a_tomorrow .- a_grid)) # Position in grid of future asset
                            if jj == nn_w
                                μ_path[row1,kk_s,jj+1,tt+1] += 1.0*μ_path[ii_a,ii_s,jj,tt] # Exogenous Probability of Transitioning there times how many people are in previous a
                            else
                                μ_path[row1,kk_s,jj+1,tt+1] += Π[ii_s,kk_s].*μ_path[ii_a,ii_s,jj,tt] # Exogenous Probability of Transitioning there times how many people are in previous a
                            end
                        end
                    end
                end
            else
                for ii_a = 1:na
                    a_tomorrow = PF_R_path[ii_a,jj-nn_w,tt]   # Chosen future asset
                    row1 = argmin(abs.(a_tomorrow .- a_grid)) # Position in grid of future asset
                    μ_path[row1,1,jj+1,tt+1] += μ_path[ii_a,1,jj,tt] # Exogenous Probability of Transitioning there times how many people are in previous a
                end
            end
        end
        μ_path[:,2,nn_w+1,tt+1] .= 0
    end
    μ_path  = reshape(pop_size,1,1,N_j,1).*μ_path[:,:,:,:] # Normalize again the size of age cohorts
    println("Transition Path of Transition Functions is stored")
    TP_res.μ_path = μ_path
    return μ_path,pop_size;
end

#--------------------------------------------#
#      MARKET CLEARING AND PRICES UPDATE
#--------------------------------------------#
function Shooting_Forward(TP_prim::TP_Primitives, TP_res::TP_Results,prim::Primitives) 
    @unpack T_,tol_K= TP_prim;
    @unpack θ,δ,α,η_j  = prim;
    @unpack a_grid, na, nz, N_j, nn_w, N_r,pers_,e_mat  = prim;
    @unpack μ_ss,K_path, L_path, W_path, R_path, θ_path = TP_res;
    K_path_new = zeros(size(K_path))
    L_path_new = zeros(size(L_path))
    err_MC = 100 
    it_count = 0
    # μ_path,pop_size = EndoMat_create_TP(TP_prim,TP_res,prim)
    while err_MC>tol_K
        μ_path,pop_size = EndoMat_create_TP(TP_prim,TP_res,prim)
        @unpack LF_W_path= TP_res
        for tt=1:T_
            μ = μ_path[:,:,:,tt]
            K_path_new[tt] = sum(reshape(a_grid,na,1,1,).*μ) 
            L_path_new[tt] = sum(reshape(e_mat,1,nz,nn_w).*LF_W_path[:,:,:,tt].*μ[:,:,1:nn_w])
        end 
        # @unpack K_path, L_path = TP_res;
        Agg_quantities = [(TP_res.K_path - K_path_new)./TP_res.K_path ; (TP_res.L_path - L_path_new)./TP_res.L_path]
        Indic_Diff = [argmax((TP_res.K_path - K_path_new)./TP_res.K_path); argmax((TP_res.L_path - L_path_new)./TP_res.L_path)]
        err_MC = norm(Agg_quantities,Inf)
        if err_MC>tol_K
            display(plot([L_path_new TP_res.L_path],
                        label = ["Demand" "Supply"],
                        title = "Labor",
                        legend = :bottomright))
                display(plot([K_path_new TP_res.K_path],
                             label = ["Demand" "Supply"],
                             title = "Capital",
                             legend = :bottomright))
            TP_res.K_path = 0.5*K_path + (1-0.5)*K_path_new
            TP_res.L_path = 0.5*L_path + (1-0.5)*L_path_new
            TP_res.W_path = (1-α).*(TP_res.K_path.^α).*(TP_res.L_path.^(1-α))./TP_res.L_path
            TP_res.R_path = α.*(TP_res.K_path.^α).*(TP_res.L_path.^(1-α))./TP_res.K_path .- δ
            TP_res.B_path = (θ_path.*TP_res.W_path.*TP_res.L_path)./sum(pop_size[1,1,N_r:end])
        end
        K_path_new .= 0
        L_path_new .= 0
        it_count = it_count+1
        println("Path Updating, Iteration # ", it_count," Norm is: ",err_MC)
        println("Largest difference is # ", Indic_Diff)
    end
    println("Norm is: ",err_MC) 
    println("Aggregate Paths and Solutions are stored")
    println("Shooting Algorithm Done!")
    return TP_res
end

function Adjusting_T(TP_prim::TP_Primitives, TP_res::TP_Results,prim::Primitives,res2) 
    @unpack K = res2
    K_1 = K
    err_T = 100 
    it_count = 0
    # μ_path,pop_size = EndoMat_create_TP(TP_prim,TP_res,prim)
    while err_T>tol_T
        TP_res = Shooting_Forward(TP_prim, TP_res,prim)
        @unpack K_path = TP_res;
        err_T = norm(K_path[end]-K_1,Inf)
        @unpack T_ = TP_prim
        T_0 = T_
        TP_prim = TP_Primitives(T_=T_0+1)
        it_count = it_count+1
        println("Horizon Updating, Iteration # ", it_count," Norm is: ",err_T)
    end
    println("Norm is: ",err_T) 
    println("Everything converged!")
    return TP_res
end

#--------------------------------------------#
#           WELFARE CALCULATIONS
#--------------------------------------------#
function Compute_Welfare2(prim::Primitives, TP_res::TP_Results,res1) 
    @unpack V_W_path = TP_res
    @unpack V_W , μ = res1
    @unpack γ, σ  = prim
    # Consumption Equivalent Variation at time 0:
    CEV = (V_W_path[:,:,:,1]./V_W_path).^(1/(γ*(1-σ)))
    # Voting  Share:
    Vote_yes   = CEV.>1
    Vote_share =  sum(Vote_yes.*μ)*100
    println("Consumption Equivalent Variation for first generation today: ",CEV[1])
    println("Share of Population in favor (%): ", Vote_share)
    println("Done!")
    return CEV, Vote_share
end

