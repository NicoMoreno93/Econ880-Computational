using Parameters, Plots #import the libraries we want
include("02Growth_model.jl") #import the functions that solve our growth model
timer = time() # Initiate Timer
prim, res = Initialize() #initialize primitive and results structs
@elapsed Solve_model(prim, res) #solve the model!
@unpack val_func, pol_func = res
@unpack k_grid = prim
time2 = time() - timer # Save Time used for VFI

##############Make plots
#value function
Plots.plot(k_grid, val_func, title="Value Function V(K)", labels=["Low Productivity(Z=0.2)" "High Productivity(Z=1.25)"],color=["blue" "black"],legend = :bottomright)
xlabel!("Initial Capital(K)")
ylabel!("Value V(K)")
Plots.savefig("02_Value_Functions.png")

#policy functions
Plots.plot(k_grid, pol_func, title="Policy Functions",labels=["Low Productivity(z=0.2)" "High Productivity(z=1.25)"],color=["blue" "black"],legend = :topleft)
plot!(k_grid,k_grid,linestyle = :dot,linewidth = 2,linecolor= :red,labels = nothing) # Add 45° line
xlabel!("Initial Capital")
ylabel!("Future Capital")
Plots.savefig("02_Policy_Functions.png")

#changes in policy function
pol_func_δ = copy(pol_func).-k_grid
Plots.plot(k_grid, pol_func_δ, title="Optimal Savings",labels=["Low Productivity(z=0.2)" "High Productivity(z=1.25)"],color=["blue" "black"],legend = :bottomright)
xlabel!("Initial Capital")
ylabel!("Saving")
Plots.savefig("02_Policy_Functions_Changes.png")

println("All done!")
################################
