# Pset 1 - JF
# Author: Nicolas Moreno
# Date: 07/11/2024
# using Pkg; Pkg.add(["DataFrames", "StatFiles"])
using Parameters,LinearAlgebra, Missings, Serialization, DataFrames
First_Load = 0 # If 1, you activate it
#--------------------------------------------#
#            GETTING DATA READY              #
#--------------------------------------------#
# Load the Data:
if First_Load == 1
    using  StatFiles
    cd_path = pwd()
    cd(".\\Inputs") 
    df = DataFrame(load("Mortgage_performance_data.dta"))
    cd(cd_path)
    serialize("Database.jls", df)
    Mor_Data = df
else
    Mor_Data = deserialize("Database.jls");
end
# Organize Data for functions:
Y_V = Float64.(Mor_Data.i_close_first_year) # Dependent (binary) Variable
x_vars = ["i_large_loan", "i_medium_loan", "rate_spread", "i_refinance", "age_r", "cltv", "dti",
"cu","first_mort_r", "score_0", "score_1", "i_FHA", "i_open_year2", "i_open_year3", "i_open_year4","i_open_year5"] # List of Dependent variables
n_x = size(x_vars,1)+1 # Number of variables, including constant
X_MAT      = zeros(size(Y_V,1),n_x)
X_MAT[:,1] = ones(size(Y_V,1),1)
for jj =1:(n_x-1)
    aux_x = select(Mor_Data,x_vars[jj,1])
    X_MAT[:,jj+1] .= aux_x
end
#--------------------------------------------#
#            EVALUATING FUNCTIONS            #
#--------------------------------------------#
include("Main_Functions.jl") 
# # Defining some numerical parameters:
β = [-1; zeros(n_x-1,1)] # Initial vector of coefficients
ϵ = 1e-5 # Size of step for numeric derivatives
ϵ_V = Matrix{Float64}(I, n_x, n_x).*ϵ
tol = 1e-5 # Tolerance for Newton Algorithm
reflection_coeff  = 1
expansion_coeff   = 2
contraction_coeff = 0.5
shrinkage_coeff   = 0.5
# # Getting the Results for each question:
using PrettyTables
OUT_file = open("Results.txt", "w")
redirect_stdout(OUT_file)
# # Answer for 1:
    println("Question 1 Results:")
    println("Initial Log_Likelihood:")
    LL_β = Log_Likelihood(β,X_MAT, Y_V)
    pretty_table([LL_β])
    println("Analytical Score:")
    g_β = Score_L(β,X_MAT, Y_V);
    pretty_table(g_β)
    println("Analytical Hessian:")
    H_β = Hessian_L(β,X_MAT);
    pretty_table(H_β)
# # Answer for 2:
    println("Question 2 Results:")
    ∇g_β = Numerical_Gradient(β,X_MAT,Y_V,n_x,ϵ,ϵ_V);
    println("Difference in Gradients:")
    pretty_table(g_β - ∇g_β)
    println("Difference in Hessian:")
    ∇H_β = Numerical_Hessian(β,X_MAT,Y_V,n_x,ϵ,ϵ_V);
    pretty_table(H_β - ∇H_β)
# # Answer for 3:
    opt_ = 0 # When Zero, it uses analytical gradient and hessian
    @elapsed β₀= Newton_Routine(β,X_MAT,Y_V,n_x,ϵ,ϵ_V,tol,opt_)
    println("Estimates using Newton (Analytical):")
    pretty_table(β₀)
    opt_ = 1
    @elapsed β₁= Newton_Routine(β,X_MAT,Y_V,n_x,ϵ,ϵ_V,tol,opt_)
    println("Estimates using Newton (Numerical):")
    pretty_table(β₁)
# # Answer for 4:
using Optim
    # Optimizer Options for BFGS:
    options_BFGS = Optim.Options(iterations=10000,f_tol=1e-6,g_tol=1e-6, show_trace=true)
    @elapsed result1 = optimize(β -> -Log_Likelihood(β,X_MAT, Y_V),β,BFGS(),options_BFGS)
    println("Estimates using BFGS:")
    β₂ = result1.minimizer
    pretty_table(β₂)
    # Optimizer Options for Simplex (Nelder-Mead):
    # options_NM1 = Optim.FixedParameters(α = reflection_coeff, β = expansion_coeff, γ = contraction_coeff, δ = shrinkage_coeff)
    options_NM2 = Optim.Options(iterations=10000,f_tol = 1e-6,x_tol=1e-6,show_trace=true)
    @elapsed result2 = optimize(β -> -Log_Likelihood(β,X_MAT, Y_V),β,NelderMead(),options_NM2)
    β₃ = result2.minimizer
    println("Estimates using Simplex:")
    pretty_table(β₃)
flush(OUT_file)
close(OUT_file)
println("Everything is done!")
