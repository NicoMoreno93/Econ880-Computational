#import Pkg; Pkg.add("Distributed")
using Distributed
addprocs(2)
# import Pkg; Pkg.add("Plots")
# import Pkg; Pkg.add("Parameters")
# import Pkg; Pkg.add("SharedArrays")
@everywhere using Parameters, Plots, SharedArrays #import the libraries we want
include("parallel_functions.jl") #import the functions that solve our growth model
timer = time() # Initiate Timer
@everywhere prim, res = Initialize() #initialize primitive and results structs
@time Solve_model(prim, res) #solve the model!
@unpack val_func, pol_func = res
@unpack k_grid = prim
time2 = time() - timer # Save Time used for VFI

##############Make plots
#value function
plot(k_grid, val_func, title="Value Function V(K)",labels=["Low Productivity(Z=0.2)" "High Productivity(Z=1.25)"],color=["blue" "black"],legend=:bottomright)
xlabel!("Initial Capital(K)")
ylabel!("Value V(K)")
savefig("Value_Functions.png")

#policy functions
plot(k_grid, pol_func, title="Policy Functions g(K)",labels=["Low Productivity(z=0.2)" "High Productivity(z=1.25)"],color=["blue" "black"],linestyle=:solid)
plot!(k_grid,k_grid,linestyle = :dot,linewidth = 2,linecolor= :red,labels = nothing)
xlabel!("Initial Capital")
ylabel!("Future Capital")
savefig("Policy_Functions.png")

#changes in policy function
pol_func_δ = pol_func.-k_grid
Plots.plot(k_grid, pol_func_δ, title="Optimal Savings",labels=["Low Productivity(z=0.2)" "High Productivity(z=1.25)"],color=["blue" "black"],legend = :bottomright)
xlabel!("Initial Capital")
ylabel!("Saving")
savefig("Policy_Functions_Changes.png")

println("All done!")
################################
