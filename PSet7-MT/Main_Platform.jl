# Author: Nicolas Moreno
# Date: 11/02/2024
using Parameters, LinearAlgebra, Random, Interpolations, Optim
# using Parameters,LinearAlgebra, Missings, Serialization
Run_Ai = 0 # If 1, you activate it
Run_KS = 1 # If 1, you activate it
#--------------------------------------------#
#       Run Aiyagari Steady State
#--------------------------------------------#
# # # SOLVE AIYAGARI MODEL:
if Run_Ai == 1
# include("Aiyagari.jl") 
@elapsed prim, res = Solve_Aiyagari_Model()
end
# # # PLOT THE RESULTS
using Plots,Colors
# Custom Colors:
myGreen = RGB(0.1,0.4,0.2);
myBlue  = RGB(0.1,0.3,0.6);
myGold  = RGB(0.5,0.4,0.1);
myRed   = RGB(0.6,0,0.25);
# # Decision Rules:
if Run_Ai == 1
    Kss_RA = ((1-0.99*(1-0.025))/(0.36*0.99))^(1/(-0.64))*0.3271*0.93 # Baseline's SS K
    Kss_Ay = res.K
    nk = prim.nk    
    # Capital
    Plots.plot(prim.k_grid,res.k_policy, label = ["\$Employed\$" "\$Unemployed\$"],color=[myBlue myGreen],linestyles=:solid,linewidth = 2 , legend = :topleft, title = "\$Capital\$ \$Policy\$ \$Functions\$")
    plot!(prim.k_grid,prim.k_grid,linestyle =:dot,linewidth = 0.8,linecolor= :black,label = "\$45° Line\$ \$(k'=k)\$") # Add 45° line
    plot!([Kss_RA, Kss_RA],[minimum(res.k_policy)-2e-1, maximum(res.k_policy)] , label = "\$Steady\$ \$State\$ \$Capital\$ \$(RA)\$ ",color=myGold,linestyle = :dot,linewidth = 1.5)
    plot!([Kss_Ay, Kss_Ay],[minimum(res.k_policy)-2e-1, maximum(res.k_policy)] , label = "\$Steady\$ \$State\$ \$Capital\$ \$(Aiyagari)\$",color= myRed,linestyle = :dot,linewidth = 1.5)
    xlabel!("\$Capital Today\$")
    ylabel!("\$Future\$ \$Capital\$")
    Plots.savefig("K_Policy_Func.pdf")
end
#--------------------------------------------#
#             Run Krusell-Smith 
#--------------------------------------------#
# # # SOLVE KS MODEL:
if Run_KS ==1
include("KS_Model.jl") 
# prim,sho,sim, res = Initialize();
@elapsed prim, res = Solve_KS_Model();
end

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