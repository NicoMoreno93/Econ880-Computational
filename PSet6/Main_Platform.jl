# Author: Nicolas Moreno
# Date: 24/10/2024
using Parameters,LinearAlgebra, Missings, Serialization
Run_SS = 0 # If 1, you activate it
Run_TP = 0 # If 1, you activate it
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
#       Run and Save Transition Paths
#--------------------------------------------#
if Run_SS ==0
# Load the dictionary with SS info from the file
Result_Structure = deserialize("Results_MutableDict.jls");
res1 = Result_Structure["RES_1"];
res2 = Result_Structure["RES_2"];
end
using Plots
# Initialize Functions for Transition Path:
include("Transition_Functions.jl") 
TP_prim, TP_res = Initialize2(res1,res2);
if Run_TP == 1
    # Run MIT Shock from t=0:
    TP_res1 = Adjusting_T(TP_prim, TP_res,prim,res1,res2) ;
    # TP_res1 = Shooting_Forward(TP_prim, TP_res,prim) 
    Result_Structure["RES_TP1"] = deepcopy(TP_res1);
    # Run MIT Shock from T=21:
    TP_prim = TP_Primitives(change_T=21);
    TP_res2 = Adjusting_T(TP_prim, TP_res,prim,res1,res2); 
    # TP_res2 = Shooting_Forward(TP_prim, TP_res,prim) 
    Result_Structure["RES_TP2"] = deepcopy(TP_res2);
    serialize("Results_Transitions.jls", Result_Structure);
else
    Result_Structure = deserialize("Results_MutableDict.jls");
end
# # Welfare Calculations:
Welfare_Structure = Dict();
CEV1, Vote_share1 = Compute_Welfare2(prim, TP_res1,res1)
CEV2, Vote_share2 = Compute_Welfare2(prim, TP_res2,res1)
Welfare_Structure["Welf_1"] = deepcopy(CEV1);
Welfare_Structure["Welf_2"] = deepcopy(CEV2);
#--------------------------------------------#
############## MAKE PLOTS!
#--------------------------------------------#
using Colors
#Custom Colors:
myGreen = RGB(0.1,0.4,0.2);
myBlue  = RGB(0.1,0.3,0.6);
myGold  = RGB(0.5,0.4,0.1);

# # # Exercises 1 and 2
for ii =1:2
    TP_res = Result_Structure["RES_TP" * string(ii)]
    CEV    = Welfare_Structure["Welf_"* string(ii)]
    # # Transition Paths:
        # Capital
        Plots.plot([res1.K*ones(size(TP_res.K_path)) res2.K*ones(size(TP_res.K_path)) ], label = ["\$LR\$ \$with\$ \$Social\$ \$Security\$ " "\$LR\$ \$with\$ \$No\$ \$Social\$ \$Security\$"],color=[myBlue myGold],linestyle = :dot,linewidth = 1.5, title = "\$K's\$ \$Transition\$ \$Path\$", legend = :bottomright)
        plot!(TP_res.K_path, label = "\$Transition\$ \$Dynamic\$",color=[myGreen],linestyles=:solid,linewidth = 2 , legend = :right)
        xlabel!("\$Time\$")
        ylabel!("\$K's\$ \$Transition\$ \$Path\$")
        Plots.savefig("K_Transition_Path" * string(ii) * ".pdf")
        #Labor
        Plots.plot([res1.L*ones(size(TP_res.K_path)) res2.L*ones(size(TP_res.K_path)) ], label = ["\$LR\$ \$with\$ \$Social\$ \$Security\$ " "\$LR\$ \$with\$ \$No\$ \$Social\$ \$Security\$"],color=[myBlue myGold],linestyle = :dot,linewidth = 1.5, title = "\$Labor's\$ \$Transition\$ \$Path\$", legend = :bottomright)
        plot!(TP_res.L_path, label = "\$Transition\$ \$Dynamic\$",color=[myGreen],linestyles=:solid,linewidth = 2 , legend = :right)
        xlabel!("\$Time\$")
        ylabel!("\$Labor's\$ \$Transition\$ \$Path\$")
        Plots.savefig("L_Transition_Path" * string(ii) * ".pdf")
        # Real Interest Rate:
        Plots.plot([res1.R*ones(size(TP_res.K_path)) res2.R*ones(size(TP_res.K_path)) ], label = ["\$LR\$ \$with\$ \$Social\$ \$Security\$ " "\$LR\$ \$with\$ \$No\$ \$Social\$ \$Security\$"],color=[myBlue myGold],linestyle = :dot,linewidth = 1.5, title = "\$R's\$ \$Transition\$ \$Path\$", legend = :bottomright)
        plot!(TP_res.R_path, label = "\$Transition\$ \$Dynamic\$",color=[myGreen],linestyles=:solid,linewidth = 2 , legend = :right)
        xlabel!("\$Time\$")
        ylabel!("\$R's\$ \$Transition\$ \$Path\$")
        Plots.savefig("R_Transition_Path" * string(ii) * ".pdf")
        # Wages:
        Plots.plot([res1.W*ones(size(TP_res.K_path)) res2.W*ones(size(TP_res.K_path)) ], label = ["\$LR\$ \$with\$ \$Social\$ \$Security\$ " "\$LR\$ \$with\$ \$No\$ \$Social\$ \$Security\$"],color=[myBlue myGold],linestyle = :dot,linewidth = 1.5, title = "\$Wage's\$ \$Transition\$ \$Path\$", legend = :bottomright)
        plot!(TP_res.W_path, label = "\$Transition\$ \$Dynamic\$",color=[myGreen],linestyles=:solid,linewidth = 2 , legend = :right)
        xlabel!("\$Time\$")
        ylabel!("\$Wage's\$ \$Transition\$ \$Path\$")
        Plots.savefig("W_Transition_Path" * string(ii) * ".pdf")
    # # Consumption Equivalent Variation:
        Plots.plot(reshape(sum(CEV.*res1.μ,dims=(1,2)),prim.N_j), label = "\$Consumption\$ \$Equivalent\$ \$Variation\$",title = "\$Mean\$ \$Consumption\$ \$EV\$ \$by\$ \$Age\$",color=[myGreen],linestyles=:solid,linewidth = 2 , legend = :bottomleft)
        plot!([prim.N_r, prim.N_r], [minimum(sum(CEV.*res1.μ,dims=(1,2))), maximum(sum(CEV.*res1.μ,dims=(1,2)))], label ="\$Retirement\$ \$Age\$",linestyle=:dash, linecolor=:black,linewidth = 2)
        xlabel!("\$Generation\$")
        ylabel!("\$Mean\$ \$CEV\$")
        Plots.savefig("CEV" * string(ii) *".pdf")
        TP_res = nothing
        CEV = nothing
end