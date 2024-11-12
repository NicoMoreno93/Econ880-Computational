# Author: Nicolas Moreno
# Date: 07/11/2024
#--------------------------------------------#
#         Q1: Analytical Calculations
#--------------------------------------------#
# Function for Evaluating Log-likelihood:
function Log_Likelihood(β,X_MAT, Y_V)
    n_i = size(Y_V,1)
    LL_β = 0.0 # Initial value for Log-likelihood
    Λ_Xβ = 0.0 # Initial value for Prob(Y=1)
    for ii = 1:n_i
        Λ_Xβ = exp.(X_MAT[ii:ii,:]*β)./(1 .+ exp.(X_MAT[ii:ii,:]*β)) # Prob=1, Slide 7 
        y_i  = Y_V[ii,1][1]
        LL_β += log(Λ_Xβ[1]^y_i*(1-Λ_Xβ[1])^(1-y_i)) # Log-like, Slide 8
    end
    return LL_β
end
# Analytical Score of the Log-likelihood:
function Score_L(β,X_MAT, Y_V)
    n_i = size(Y_V,1)
    g_β = zeros(size(X_MAT[1,:],1),1) # Initial value for Log-likelihood
    Λ_Xβ = 0.0 # Initial value for Prob(Y=1)
    for ii = 1:n_i
        x_i = X_MAT[ii:ii,:]
        y_i  = Y_V[ii,1][1]
        Λ_Xβ = exp.(x_i*β)./(1 .+ exp.(x_i*β))  
        g_β += ((y_i-Λ_Xβ[1])*x_i)'# Score, Slide 8
    end
    return g_β
end
# Analytical Hessian of the Log-likelihood:
function Hessian_L(β,X_MAT)
    n_i = size(X_MAT,1)
    H_β = zeros(size(X_MAT[1,:],1),size(X_MAT[1,:],1)) # Initial value for Log-likelihood
    Λ_Xβ = 0.0 # Initial value for Prob(Y=1)
    for ii = 1:n_i
        x_i = X_MAT[ii:ii,:]
        Λ_Xβ = exp.(x_i*β)./(1 .+ exp.(x_i*β))  
        H_β += -(Λ_Xβ[1]*(1-Λ_Xβ[1])).*(x_i'x_i)# Score, Slide 8
    end
    return H_β
end
#--------------------------------------------#
#         Q2: Numerical Functions
#--------------------------------------------#
# Numerical Gradient of the Log-likelihood:
function Numerical_Gradient(β,X_MAT,Y_V,n_x,ϵ,ϵ_V)
    ∇g_β = zeros(n_x,1)
    for kk = 1:n_x
        β_1  = β .+ ϵ_V[:,kk]
        β_2  = β .- ϵ_V[:,kk]
        LL_β1 = Log_Likelihood(β_1,X_MAT, Y_V)
        LL_β2 = Log_Likelihood(β_2,X_MAT, Y_V)
        ∇g_β[kk,1] = (LL_β1-LL_β2)./(2*ϵ)
    end
    return ∇g_β
end
# Numerical Hessian of the Log-likelihood:
function Numerical_Hessian(β,X_MAT,Y_V,n_x,ϵ,ϵ_V)
    ∇H_β = zeros(n_x,n_x)
    for cc =1:n_x
        β_1  = β .+ ϵ_V[:,cc]
        β_2  = β .- ϵ_V[:,cc]
        ∇g_β1 = Numerical_Gradient(β_1,X_MAT,Y_V,n_x,ϵ,ϵ_V)
        ∇g_β2 = Numerical_Gradient(β_2,X_MAT,Y_V,n_x,ϵ,ϵ_V)
        ∇H_β[:,cc] = (∇g_β1.-∇g_β2)./(2*ϵ)
    end
    return ∇H_β
end
#--------------------------------------------#
#         Q3: Newton Algorithm
#--------------------------------------------#
function Newton_Routine(β₀,X_MAT,Y_V,n_x,ϵ,ϵ_V,tol,opt_)
    err_new = 100
    β₁ = copy(β₀)
    iter_ = 0
    while err_new > tol
        if opt_ == 0 # Use Analytical functions
            ∇g_β = Score_L(β₀,X_MAT, Y_V)
            ∇H_β = Hessian_L(β₀,X_MAT)
        else
            ∇g_β = Numerical_Gradient(β₀,X_MAT,Y_V,n_x,ϵ,ϵ_V)
            ∇H_β = Numerical_Hessian(β₀,X_MAT,Y_V,n_x,ϵ,ϵ_V)
        end
        β₁   = β₀ - inv(∇H_β)*∇g_β
        err_new = norm(β₁-β₀,Inf)
        β₀   = β₁
        iter_ += 1
        println("Newton Iteration: ",iter_)
    end
    β₁ = β₀
    println("Newton Algorithm Converged in ",iter_," iterations")
    return β₁
end
#--------------------------------------------#
#         Q4: BFGS and SIMPLEX
#--------------------------------------------#
