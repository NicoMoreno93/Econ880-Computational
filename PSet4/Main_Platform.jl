# Author: Nicolas Moreno

using Parameters, Distributions,Optim, Plots,LinearAlgebra, Random  #, Colors ##import the libraries we want
include("Main_Functions.jl") #import the functions that solve our growth model
prim, res = Initialize() #initialize primitive and results structs
Result_Structure = SMM_Algorithm(prim,res)
# Extract Results
res1 = Result_Structure["RES_1"]
res2 = Result_Structure["RES_2"]
res3 = Result_Structure["RES_2"]
# Organize Results
using DataFrames
OUT_file = open("SMM_OUTPUT.txt", "w")
redirect_stdout(OUT_file)

println("Output for Question 4")
Table_bs1 = DataFrame()
Table_bs1.b1       = res1.bs[:,1]
Table_bs1.b2       = res1.bs[:,2]
Table_bs1.Std_ε    = res1.Std_ε 
log_output(Table_bs1)
println("∇g_TH")
log_output(res1.∇g_TH)
println("J-Test")
log_output(res1.test_J) 

println("Output for Question 5")
Table_bs2 = DataFrame()
Table_bs2.b1       = res2.bs[:,1]
Table_bs2.b2       = res2.bs[:,2]
Table_bs2.Std_ε    = res2.Std_ε 
log_output(Table_bs2)
println("∇g_TH")
log_output(res2.∇g_TH)
println("J-Test")
log_output(res2.test_J) 

println("Output for Question 6")
Table_bs3 = DataFrame()
Table_bs3.b1       = res3.bs[:,1]
Table_bs3.b2       = res3.bs[:,2]
Table_bs3.Std_ε    = res3.Std_ε 
log_output(Table_bs3)
println("∇g_TH")
log_output(res3.∇g_TH)
println("J-Test")
log_output(res3.test_J) 


# redirect_stdout(stdout) 
#--------------------------------------------#
#           BOOTSTRAPPING EXERCISE
#--------------------------------------------#
n_draws = 200
b_MAT = Bootstrap(prim, res,n_draws)
b_MAT
#--------------------------------------------#
#           MA EXERCISE
#--------------------------------------------#
# using StateSpaceModels
using ARCHModels
MA_ORDERV = [1 2 3]
Table_bsMA = DataFrame()
for kk in MA_ORDERV
    b1,II_crit_Opt = Indirect_Inference(prim, res,MA_ORDERV[kk])
    col_name = Symbol("b" * string(kk)) 
    Table_bsMA[!, col_name] = b1
end
println("Output for Question 7")
log_output(Table_bsMA)
flush(OUT_file)
close(OUT_file)

#--------------------------------------------#
#           MAKE PLOTS
#--------------------------------------------#
#Custom Colors:
using Colors
myGreen = RGB(0.1,0.4,0.2);
myBlue  = RGB(0,0.3,0.8);
myGold  = RGB(0.5,0.4,0.1);


# # BOOTSTRAPPING PLOTS
# RHO
Plots.plot(collect(1:n_draws), b_MAT[1,:], title=" Bootstrap for \$ \\hat{ρ}_{y} \$",label ="Draws",color=[myBlue])
Plots.plot!(collect(1:n_draws), res3.ρ_y.*ones(n_draws,1),color="black",label ="Point Estimate"  ,legend = :topleft)
xlabel!("Different Seeds")
ylabel!("Estimated \$ \\hat{ρ}_{y} \$")
Plots.savefig("Bootstrap_rho.pdf")

histogram(b_MAT[1,:],bins=20,color="cyan", fillalpha=0.3, label = "\$ \\hat{ρ}_{y} \$")
savefig("Hist_Bootstrap_rho.pdf")

# SIGMA
Plots.plot(collect(1:n_draws), b_MAT[2,:], title=" Bootstrap for \$ \\hat{σ}_{y} \$",label ="Draws",color=[myGreen])
Plots.plot!(collect(1:n_draws), res3.σ_yε.*ones(n_draws,1),color="black",label ="Point Estimate"  ,legend = :topleft)
xlabel!("Different Seeds")
ylabel!("Estimated \$ \\hat{σ}_{y} \$")
Plots.savefig("Bootstrap_sigma.pdf")

histogram(b_MAT[2,:],bins=20,color="teal", fillalpha=0.3, label = "\$ \\hat{σ}_{y} \$")
Plots.savefig("Hist_Bootstrap_sigma.pdf")


