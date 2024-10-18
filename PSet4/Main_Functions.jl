# Author: Nicolas Moreno
#--------------------------------------------#
#           PREAMBLE!
#--------------------------------------------#

#Keyword-enabled Structure holding model primitives
@with_kw struct Primitives
    T_::Int64         = 200  # Time Length of True data
    ρ_o::Float64      = 0.5  # True Persistence
    σ_o::Float64      = 1.0  # True Variance of residuals
    x_o::Float64      = 0.0  # True Initial value
    σ_yε0::Float64    = 0.7  # Initial Guesses
    ρ_y0::Float64     = 0.25 # Initial Guesses
    n_m::Int64        = 3    # Number of moments
    H_::Int64         = 10   # Number of Vectors of length T for simulation
    i_T::Int64        = 4    # Newey-West Lag Length
    s_step::Float64   = 1e-10
    d_true::Normal{Float64} = Normal(0, σ_o) # True distribution of Residuals
    # ε_true::Vector{Float64} = rand(d_true,T_) # Generate the vector of true disturbances
    seed::Int64       = 234
end

mutable struct Results
    x_t::Matrix{Float64}    # Vector of True AR
    d_simul::Normal{Float64} # Distribution for Simulation
    ε_true::Array{Float64}
    ε_Simul::Matrix{Float64}
    y_t::Array{Float64,2}    # Vector of True AR
    σ_yε::Float64
    σ_y::Float64
    ρ_y::Float64
    W_mat::Array{Float64,2}
    T_moms::Array{Float64,1}
    ∇g_TH::Matrix{Float64} 
    Ω_b::Array{Float64}
    Std_ε::AbstractArray{Float64}
    AGS::AbstractArray{Float64}
    test_J::Float64
    bs::AbstractArray{Float64}
end

function Initialize()
    prim = Primitives() #initialize primtiives
    @unpack T_, ρ_o, σ_o, n_m, H_,x_o,seed,d_true, σ_yε0,ρ_y0  = prim
    x_t = zeros(T_,1) # Pre-allocate the vector of true data
    d_simul = Normal(0, σ_o) # True distribution of Residuls
    if !ismissing(seed)
        Random.seed!(seed)
    end
    ε_true = rand(d_true,T_) 
    ε_Simul = rand(d_simul,T_,H_)
    y_t = zeros(T_,H_) # Pre-allocate the matrix for Simulations
    σ_yε = σ_yε0;
    ρ_y = ρ_y0;
    σ_y = ((σ_yε^2)/(1-ρ_y^2))^0.5;
    W_mat = diagm(ones(n_m))
    μ_t = x_o
    σ_t = (σ_o^2/(1-ρ_o^2))^0.5
    ϕ_t = ρ_o
    T_moms = [μ_t; σ_t; ϕ_t]
    ∇g_TH = zeros(n_m,2)
    Ω_b   = zeros(2,2)
    Std_ε = zeros(2,1)
    AGS   = copy(∇g_TH')
    test_J = 0.0
    bs = zeros(2,2)
    res = Results(x_t,d_simul,ε_true,ε_Simul,y_t, σ_yε,σ_y, ρ_y,W_mat,T_moms,∇g_TH,Ω_b,Std_ε,AGS,test_J,bs ) #initialize results struct
    prim, res #return deliverables
end

# Generate True Data
function True_Data(prim::Primitives, res::Results)
    @unpack T_, ρ_o, σ_o, x_o = prim
    @unpack x_t,ε_true= res
    x_t[1,1] = x_o # Intial value of the AR
    for tt = 2:T_
    x_t[tt,1] =  ρ_o*x_t[tt-1,1] + ε_true[tt,1]
    end
    res.x_t = x_t
    return x_t
end

# Generate Simulated Data
function Simulated_Data(prim::Primitives, res::Results,ρ_y,σ_yε)
    @unpack T_, H_ = prim
    # @unpack ρ_y,σ_yε , y_t, ε_Simul= res
    @unpack  y_t, ε_Simul= res
    y_t[1,:] .= 0.0 # Intial values for the Simulated AR
    for hh = 1: H_
        for tt = 2:T_
            y_t[tt,hh] =  ρ_y*y_t[tt-1,hh] + σ_yε*ε_Simul[tt,hh]
        end
    end
    return y_t
end

#--------------------------------------------#
#           STANDARD ESTIMATIONS
#--------------------------------------------#
function Moment_Calculator(prim::Primitives, res::Results,x_t)
    μ_x = mean(x_t)
    σ_x = var(x_t, corrected=true)
    autocov_x = mean(x_t[2:end,:].*x_t[1:end-1,:]) -mean(x_t[2:end,:]).*mean(x_t[1:end-1,:])
    ϕ_x = autocov_x/(var(x_t[2:end,:], corrected=true)*var(x_t[1:end-1,:], corrected=true))^0.5
    Moment_Vec = [μ_x; σ_x ;ϕ_x]
    return Moment_Vec
end

function Obj_Function(b, prim::Primitives, res::Results,indic_)
    @unpack W_mat = res
    x_t = True_Data(prim, res)
    Mom_X = Moment_Calculator(prim, res,x_t)
    # res.ρ_y  = sum(b[1])
    # res.σ_yε = sum(b[2])
    ρ_y  = sum(b[1])
    σ_yε = sum(b[2])
    y_t   = Simulated_Data(prim, res,ρ_y,σ_yε)
    Mom_Y = Moment_Calculator(prim, res,y_t)
    g_TH  = Mom_X - Mom_Y
    if indic_ == 1
        g_TH  = g_TH[1:2,1:1]
        # W_mat = W_mat[1:2,1:2] 
    elseif indic_ ==2
        g_TH  = g_TH[2:3,1:1]
        # W_mat = W_mat[2:3,2:3]
    end

    # Compute the Objective Function for GMM:
    try 
        g_TH'*W_mat*g_TH
    catch
        W_mat = W_mat[1:2,1:2]
    finally
        res.test_J = sum(g_TH'*W_mat*g_TH)
    end 
    J_TH = res.test_J 
    res.test_J = 0
    return J_TH
end

function Estimation(prim::Primitives, res::Results,indic_)
    @unpack ρ_y0,σ_yε0,n_m = prim
    # Initial Guess
    b0  = [ρ_y0, σ_yε0]
    b_aux = copy(b0)
    # Plot JTH for different b's, in the just identified case.
    # indic_    = 3
    rho_vec   = collect(range(0.35, length = 100, stop= 0.65)) 
    sigma_vec = collect(range(0.8, length = 100, stop= 1.2))
    J_TH_MAT  = zeros(100,100)
    for rr = 1:100
        for ss=1:100
            b_aux[1,1] = rho_vec[rr,1]
            b_aux[2,1] = sigma_vec[ss,1]
            J_TH = Obj_Function(b_aux,prim, res,indic_)
            J_TH_MAT[rr,ss] = sum(J_TH)
        end
    end
    surface(rho_vec,sigma_vec,J_TH_MAT, title="Objective Function",  colormap=:hsv,alpha=0.55,camera=(40,25),xlabel="\$ρ\$",ylabel="\$σ\$",zlabel="\$J_{TH}\$")
    if res.W_mat == diagm(ones(n_m))
        savefig("ObjFunc_3D_initW_" *string(indic_)* ".pdf")
    end
    # Solve for b
    LB_ = [0.2,0.1]
    UB_ = [0.9999,4.0]
    result_ = optimize(b -> Obj_Function(b,prim,res,indic_), LB_, UB_, b0)
    b1 = result_.minimizer
    J_TH_opt = result_.minimum
    # Update Estimates:
    res.ρ_y  = b1[1]
    res.σ_yε = b1[2]
    return b1, J_TH_opt
end

function Weighting_Update(prim::Primitives, res::Results,indic_)
    @unpack i_T,T_,H_, n_m = prim
    @unpack ρ_y,σ_yε = res 
    y_t   = Simulated_Data(prim, res,ρ_y,σ_yε)
    Mom_Y = Moment_Calculator(prim, res, y_t)
    if indic_ == 1
        Mom_Y  = Mom_Y[1:2,1:1]
    elseif indic_ ==2
        Mom_Y  = Mom_Y[2:3,1:1]
    end

    if indic_ !=3
        n_m = n_m-1
    end
    Γ_Mat = zeros(n_m,n_m,i_T+1)
    S_yTH = zeros(n_m,n_m)
    Mom_t  = zeros(n_m,n_m,H_)

    for jj =1:i_T+1
        aux_y    = y_t[jj:end,:] # Matrix of current y_t
        auxL_y   = y_t[1:end-(jj-1),:] # Matrix of current y_t
        aux_Mom_t = zeros(n_m,n_m,T_-(jj-1))
        for hh=1:H_
            aux_μ  = mean(aux_y[:,hh])
            auxL_μ = copy(aux_μ)
            # auxL_μ = mean(auxL_y[:,hh])
            aux_σ  = var(aux_y[:,hh], corrected=true)^0.5
            auxL_σ = copy(aux_σ)
            # auxL_σ = var(auxL_y[:,hh], corrected=true)^0.5
            for tt=jj:size(aux_y)[1]               
                aux_m1   = aux_y[tt,hh]
                auxL_m1  = auxL_y[tt,hh]
                aux_m2   = (aux_y[tt,hh]-aux_μ)^2
                auxL_m2  = (auxL_y[tt,hh]-auxL_μ)^2
                aux_m31  = (aux_y[tt,hh]-aux_μ)*(aux_y[tt-(jj-1),hh]-aux_μ)  
                auxL_m31 = (auxL_y[tt,hh]-auxL_μ)*(auxL_y[tt-(jj-1),hh]-auxL_μ)
                aux_m3   = aux_m31./aux_σ
                auxL_m3  = auxL_m31./auxL_σ
                if indic_ ==1
                    aux_Mom1 = [aux_m1;aux_m2 ]
                    aux_Mom2 = [auxL_m1;auxL_m2]
                elseif indic_ ==2
                    aux_Mom1 = [aux_m2 ;aux_m3]
                    aux_Mom2 = [auxL_m2;auxL_m3]
                else
                aux_Mom1 = [aux_m1;aux_m2;  aux_m3]
                aux_Mom2 = [auxL_m1;auxL_m2;  auxL_m3]
                end
                aux_Mom_t[:,:,tt] = (aux_Mom1- Mom_Y)*(aux_Mom2- Mom_Y)'
            end
            Mom_t[:,:,hh] = sum(aux_Mom_t,dims=3)
        end
        Γ_Mat[:,:,jj] = sum(Mom_t,dims=3)./((T_-(jj-1))*H_)
    end
    # S_yTH = Γ_Mat[:,:,1]
    for jj = 2:i_T+1
        S_yTH = S_yTH + (1- jj/(i_T+1)).*(Γ_Mat[:,:,jj] + Γ_Mat[:,:,jj]')
    end
    S_yTH = Γ_Mat[:,:,1] + S_yTH
    S_TH  = (1+1/H_).*S_yTH #Finally Compute the Asymptotic VarCov Matrix
    res.W_mat = inv(S_TH)
    return S_TH
end

function Numeric_Jacobian_VarCov(prim::Primitives, res::Results,indic_)
    @unpack n_m,s_step,T_ = prim
    @unpack ρ_y,σ_yε,∇g_TH,W_mat = res 
    y_t   = Simulated_Data(prim, res,ρ_y,σ_yε)
    Mom_Y = Moment_Calculator(prim, res, y_t)
    ρ_y2 = ρ_y - s_step
    y_t_step   = Simulated_Data(prim, res,ρ_y2,σ_yε)
    Mom_Y_step1 = Moment_Calculator(prim, res, y_t_step)
    σ_yε2 =σ_yε - s_step
    y_t_step2   = Simulated_Data(prim, res,ρ_y,σ_yε2)
    Mom_Y_step2 = Moment_Calculator(prim, res, y_t_step2)
    if indic_ == 1
        Mom_Y  = Mom_Y[1:2,1:1]
        Mom_Y_step1  = Mom_Y_step1[1:2,1:1]
        Mom_Y_step2  = Mom_Y_step2[1:2,1:1]
        ∇g_TH = ∇g_TH[1:2,:]
    elseif indic_ ==2
        Mom_Y  = Mom_Y[2:3,1:1]
        Mom_Y_step1  = Mom_Y_step1[2:3,1:1]
        Mom_Y_step2  = Mom_Y_step2[2:3,1:1]
        ∇g_TH = ∇g_TH[2:3,:]
    end
    ∇g_TH[:,1] = -(Mom_Y-Mom_Y_step1)./s_step
    ∇g_TH[:,2] = -(Mom_Y-Mom_Y_step2)./s_step
    S_TH = inv(W_mat)
    Ω_b = (1/T_).*inv(∇g_TH'*W_mat*∇g_TH)
    Std_ε = 0.0
    AGS = -inv(∇g_TH'*inv(S_TH)*∇g_TH)*(∇g_TH'*inv(S_TH))
    return ∇g_TH,Ω_b,Std_ε,AGS
end

function J_Test(prim::Primitives, res::Results,J_TH_opt)
    @unpack T_,H_ = prim
    test_J = T_*(H_/(1+H_))*J_TH_opt
    return test_J 
end

function SMM_Algorithm(prim::Primitives, res::Results)
    @unpack n_m = prim
    Result_Structure = Dict()
    for ii=1:n_m
        indic_ = ii
        b1, J_TH_opt1 = Estimation(prim, res,indic_)
        res.bs[:,1] = b1
        S_TH = Weighting_Update(prim, res,indic_)
        b2, J_TH_opt2 = Estimation(prim, res,indic_)
        res.bs[:,2] = b2
        ∇g_TH,Ω_b,Std_ε,AGS = Numeric_Jacobian_VarCov(prim, res,indic_)
        res.∇g_TH = ∇g_TH
        res.Ω_b = Ω_b
        res.AGS = AGS
        try
            Std_ε = diag(Ω_b).^0.5
        catch
            Std_ε = [0 ;0]
        finally
            res.Std_ε = Std_ε
        end
        test_J = J_Test(prim, res,J_TH_opt2)
        res.test_J = test_J
        # final_res = Dict()
        # final_res["res"] = deepcopy(res)
        # final_res["post_estim"] = deepcopy(res)
        Result_Structure["RES_" * string(ii)] = deepcopy(res)
        # prim, res = Initialize()
        # Re-Initialize key parameters
        # res.W_mat = diagm(ones(n_m))
        # res.ρ_y = prim.ρ_y0  
        # res.σ_yε = prim.σ_yε0 
        prim, res = Initialize()  
    end
    Result_Structure
end

#--------------------------------------------#
#           BOOTSTRAPPING EXERCISE
#--------------------------------------------#

function Bootstrap(prim::Primitives, res::Results,n_draws)
    seed_vec = collect(1235:(1234+n_draws))
    prim,res = Initialize()
    indic_ = 3
    b_MAT = zeros(2,n_draws)
    for ee=1:n_draws
        # prim = Primitives(seed=seed_vec[ee])
        @unpack T_, H_, d_true = prim
        @unpack d_simul = res
        seed=seed_vec[ee]
        Random.seed!(seed)
        res.ε_true = rand(d_true,T_) 
        res.ε_Simul = rand(d_simul,T_,H_)
        # prim,res = Initialize()
        b1, J_TH_opt = Estimation(prim, res,indic_)
        b_MAT[:,ee] = b1
    end
    return b_MAT
end

#--------------------------------------------#
#           MA EXERCISE
#--------------------------------------------#

function MA_Estimation(prim::Primitives, res::Results,z_t,MA_ORDER)
    model   = fit(MA{MA_ORDER}, vec(z_t.-mean(z_t))) # Demean Data to make sure it doesn't estimate a relevant constant term
    params_ = coef(model)
    std_samp = params_[1]
    theta_v  = params_[3:2+MA_ORDER]
    # # println(resMA.results.coef_table)
    # params_ = resMA.results.coef_table.coef
    # powers_ =1
    # theta_v= params_[1:MA_ORDER]#.^powers_
    # std_samp = (sum(((z_t.-mean(z_t)).^2))./(prim.T_-1)).^0.5
    return theta_v, std_samp
end

function Simulated_MAs(b,prim::Primitives, res::Results,MA_ORDER)
    @unpack T_, H_ = prim
    ρ_y1  = b[1]
    σ_yε1 = b[2]
    y_t   = Simulated_Data(prim, res,ρ_y1,σ_yε1)
    theta_MAT = zeros(MA_ORDER,H_)
    std_VEC   = zeros(1,H_)
    for hh= 1:H_
        y_z = y_t[:,hh]
        theta_v, std_samp = MA_Estimation(prim,res,y_z,MA_ORDER)
        theta_MAT[:,hh] .= theta_v
        std_VEC[1,hh]   = std_samp
    end
    mean_theta_v  = mean(theta_MAT, dims=2)
    mean_std_samp = mean(std_VEC)
    return mean_theta_v,mean_std_samp
end

function II_criterion(b, prim::Primitives, res::Results,MA_ORDER,theta_v_hat, std_samp_hat)
    mean_theta_v,mean_std_samp = Simulated_MAs(b,prim,res,MA_ORDER)
    II_crit = sum((mean_theta_v.-theta_v_hat)'*(mean_theta_v.-theta_v_hat) .+ (mean_std_samp.-std_samp_hat)^2)
end

function Indirect_Inference(prim::Primitives, res::Results,MA_ORDER)
    @unpack T_, H_ = prim
    @unpack x_t = res
    theta_v_hat, std_samp_hat = MA_Estimation(prim,res,x_t,MA_ORDER)
    ρ_y  = prim.ρ_y0  
    σ_yε = prim.σ_yε0
    b0 = [ρ_y,σ_yε]
    result_ = optimize(b -> II_criterion(b ,prim,res,MA_ORDER,theta_v_hat, std_samp_hat), b0)
    b1 = result_.minimizer
    II_crit_Opt = result_.minimum
    return b1,II_crit_Opt
end

#--------------------------------------------#
#           ACCESSORY
#--------------------------------------------#

function log_output(expr)
    result = eval(expr)  # Evaluate the expression
    println(result)      # Print the result (captures all output)
    return result        # Return the result for further use if needed
end