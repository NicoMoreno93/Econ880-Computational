# Author: Nicolas Moreno

using Parameters,LinearAlgebra, Missings 
include("Main_Functions.jl") 
prim, res = Initialize()
#--------------------------------------------#
#       Run and Save Benchmark Scenario
#--------------------------------------------#
Result_Structure = Dict()
@elapsed res1 = Market_Clearing(prim, res)      
Welfare1, Wealth_CV1 = Compute_Welfare(prim, res1)
Result_Structure["RES_1"] = deepcopy(res1)
#--------------------------------------------#
#       Run Alternative Scenarios
#--------------------------------------------#
# # No Social Security:
prim = Primitives(θ=0)
@elapsed res2 = Market_Clearing(prim, res)
Welfare2, Wealth_CV2 = Compute_Welfare(prim, res2)
Result_Structure["RES_2"] = deepcopy(res2)
# # No Risk:
prim = Primitives(θ=0.11,Zg=0.5,pers_=0.9)
@elapsed res3 = Market_Clearing(prim, res) 
Welfare3, Wealth_CV3 = Compute_Welfare(prim, res3)
Result_Structure["RES_3"] = deepcopy(res3)
# # No Risk + No SS:
prim = Primitives(θ=0,Zg=0.5,pers_=0.9)
@elapsed res4 = Market_Clearing(prim, res) 
Welfare4, Wealth_CV4 = Compute_Welfare(prim, res4)
Result_Structure["RES_4"] = deepcopy(res4)
# # Exogenous Labor:
prim = Primitives(θ=0.11,Zg=3,γ=1,pers_=0.75)
@elapsed res5 = Market_Clearing(prim, res) 
Welfare5, Wealth_CV5 = Compute_Welfare(prim, res5)
Result_Structure["RES_5"] = deepcopy(res5)
# # Exogenous Labor + No SS:
prim = Primitives(θ=0,Zg=3,γ=1,pers_=0.75)
@elapsed res6 = Market_Clearing(prim, res) 
Welfare6, Wealth_CV6 = Compute_Welfare(prim, res6)
Result_Structure["RES_6"] = deepcopy(res6)

#--------------------------------------------#
############## MAKE PLOTS!
#--------------------------------------------#
using Plots,Colors
@unpack V_W, Pol_Func_W,V_R, Pol_Func_R = res1
@unpack a_grid, na = prim
#Custom Colors:
myGreen = RGB(0.1,0.4,0.2);
myBlue  = RGB(0.1,0.3,0.6);
myGold  = RGB(0.5,0.4,0.1);

# Value Functions
Plots.plot(a_grid, V_R[:,5], title="Value Functions \$V_{j}(a,z)\$", labels="\$V_{R,50}(a)\$",color=[myBlue],linewidth=3)
plot!(a_grid, V_W[:,1,1], labels="\$V_{W,20}(a,z_{H})\$",color=[myGreen],linestyles=:solid, linewidth=3)
plot!(a_grid, V_W[:,2,1], labels="\$V_{W,20}(a,z_{L})\$",color=[myGreen],linestyles=:dash,linewidth=2,legend = :bottomright)
xlabel!("Current Asset Holdings(a)")
ylabel!("Value V(a,z)")
Plots.savefig("Value_Functions.pdf")

# Policy Functions
Plots.plot(a_grid, Pol_Func_R[:,5], title="Policy Functions \$g_{j}(a,z)\$", labels="\$g_{R,50}(a)\$",color=[myBlue],linewidth=1)
plot!(a_grid,a_grid,linestyle = :dot,linewidth = 0.8,linecolor= :black,labels = "\$45° Line\$ \$(a'=a)\$") # Add 45° line
xlabel!("Current Asset Holdings (a)")
ylabel!("Future Asset Holdings \$g_{j}(a,z)\$")
Plots.savefig("Policy_Function_Retiree.pdf")

Plots.plot(a_grid, Pol_Func_W[:,1,1], title="Policy Functions \$g_{j}(a,z)\$",labels="\$g_{W,20}(a,z_{H})\$",color=[myGreen],linestyles=:solid, linewidth=1)
plot!(a_grid, Pol_Func_W[:,2,1], labels="\$g_{W,20}(a,z_{L})\$",color=[myGreen],linestyles=:dash,linewidth=1,legend = :bottomright)
plot!(a_grid,a_grid,linestyle = :dot,linewidth = 0.8,linecolor= :black,labels = "\$45° Line\$ \$(a'=a)\$") 
xlabel!("Current Asset Holdings (a)")
ylabel!("Future Asset Holdings \$g_{j}(a,z)\$")
Plots.savefig("Policy_Functions_Workers.pdf")
