using Parameters, Plots, Colors,LinearAlgebra #import the libraries we want
include("Main_Functions.jl") #import the functions that solve our growth model
# timer = time() # Initiate Timer
prim, res = Initialize() #initialize primitive and results structs
#@elapsed Solve_model(prim, res) #solve the model!
@elapsed Huggett_Solve(prim, res) #solve the model!

@unpack val_func, pol_func = res
@unpack a_grid, na = prim
# time2 = time() - timer # Save Time used for VFI

# Finding A Upper Bar
crossing   = argmin(abs.(pol_func[:,1] .- a_grid))
a_UpperBar = a_grid[crossing]
wealth_distr = reshape(res.distr,na,prim.nz);

include("Inequality_Welfare.jl") 
wealth_distr=calculate_wealth_distr(res)

############## Make plots
#Custom Colors:
myGreen = RGB(0.1,0.4,0.2);
myBlue  = RGB(0.1,0.3,0.6);
myGold  = RGB(0.5,0.4,0.1);

# Value Functions
Plots.plot(a_grid, val_func, title="Value Function V(A,Z)", labels=["Employed" "Unemployed"],color=[myBlue myGold],legend = :bottomright)
plot!([a_UpperBar, a_UpperBar], [minimum(val_func), maximum(val_func)], label ="\$\\bar{a}\$="*string(a_grid[crossing]),linestyle=:dash, color=myGreen)
xlabel!("Initial Assets(A)")
ylabel!("Value V(A,Z)")
Plots.savefig("Value_Functions.png")

# Policy Functions
Plots.plot(a_grid, pol_func, title="Policy Functions",labels=["Employed" "Unemployed"],color=[myBlue myGold],legend = :topleft)
plot!(a_grid,a_grid,linestyle = :dot,linewidth = 0.8,linecolor= :black,labels = "45° Line (a'=a)") # Add 45° line
plot!([a_UpperBar, a_UpperBar], [minimum(pol_func), maximum(pol_func)], label ="\$\\bar{a}\$="*string(a_grid[crossing]),linestyle=:dash,color=myGreen)
xlabel!("Initial Assets")
ylabel!("Future Assets")
Plots.savefig("Policy_Functions.pdf")

# Wealth Distribution
plot([a_grid a_grid], wealth_distr , legend=:topright, labels=["Employed" "Unemployed"], title="Wealth distributions by Employment Status ", color=[myBlue myGold])
xlabel!("Wealth")
ylabel!("PDF(\$W\$)")
Plots.savefig("Wealth_Distributions.pdf")

#plot(wealth_distr[crossing:crossing+100,1])
# Changes in policy function
pol_func_δ = copy(pol_func).-a_grid
Plots.plot(a_grid, pol_func_δ, title="Optimal Savings",labels=["Employed" "Unemployed"],color=[myBlue myGold],legend = :bottomright)
xlabel!("Initial Assets")
ylabel!("Saving")
Plots.savefig("Policy_Functions_Changes.pdf")

#Lorenz Curve
LC = Lorenz_Curve_Computation(wealth_distr);
plot([LC[:,1] LC[:,1]], [LC[:,1] LC[:,2]], labels = ["45° Line" "Lorenz Curve"], title="Lorenz Curve", legend=:bottomright;
linewidth=2,color=[myBlue myGold])
xlabel!("Cumulative Distribution")
ylabel!("Cumulative Share of Pop.")
Plots.savefig("LorenzCurve.pdf")
#Calculate  GINI 
GINI=Gini_Computation(LC)

# welfare in complete markets
Welfare_CM = Complete_Markets_Welfare()

# Consumption equivalent
λ = Welfare_Computation(res, Welfare_CM)
plot(a_grid, λ, title="Consumption Equivalent",labels=["Employed" "Unemployed"],color=[myBlue myGold],legend = :bottomright)
xlabel!("Assets")
ylabel!("λ")
Plots.savefig("Welfare_Graph.pdf")

# Welfare in Incomplete markets
Welfare_IM = sum(res.distr .* reshape(res.val_func,prim.ntot,1))

# Welfare gain
Welfare_Gain = sum(res.distr .* reshape(λ,prim.ntot,1))

# fraction of population that prefers complete market
f = sum((reshape(λ,prim.ntot,1) .>= 0) .*res.distr)

println("All done!")




