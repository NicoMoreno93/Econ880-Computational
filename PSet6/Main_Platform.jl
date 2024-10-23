# Author: Nicolas Moreno
using Parameters,LinearAlgebra, Missings, Serialization
Run_SS = 0
#--------------------------------------------#
#       Run and Save Steady States
#--------------------------------------------#
include("Steady_State_Functions_2.jl") 
prim, res = Initialize();
if Run_SS == 1
# # With Social Security:
Result_Structure = Dict();
@elapsed res1 = Market_Clearing(prim, res)      
# Welfare1, Wealth_CV1 = Compute_Welfare(prim, res1)
Result_Structure["RES_1"] = deepcopy(res1);
# # No Social Security:
prim = Primitives(θ=0);
@elapsed res2 = Market_Clearing(prim, res)
Welfare2, Wealth_CV2 = Compute_Welfare(prim, res2)
Result_Structure["RES_2"] = deepcopy(res2);
# Save the dictionary to a file
serialize("Results_MutableDict.jls", Result_Structure)
res1 = Result_Structure["RES_1"];
res2 = Result_Structure["RES_2"];
end
#--------------------------------------------#
#       Run and Save Steady States
#--------------------------------------------#
if Run_SS ==0
# Load the dictionary with SS info from the file
Result_Structure = deserialize("Results_MutableDict.jls");
res1 = Result_Structure["RES_1"];
res2 = Result_Structure["RES_2"];
end
include("Transition_Functions.jl") 
TP_prim, TP_res = Initialize2(res1,res2);
TP_res = Shooting_Forward(TP_prim, TP_res,prim) 
return


#--------------------------------------------#
############## MAKE PLOTS!
#--------------------------------------------#
using Plots,Colors
# @unpack V_W, Pol_Func_W,V_R, Pol_Func_R = res1
# @unpack a_grid, na = prim
#Custom Colors:
myGreen = RGB(0.1,0.4,0.2);
myBlue  = RGB(0.1,0.3,0.6);
myGold  = RGB(0.5,0.4,0.1);

