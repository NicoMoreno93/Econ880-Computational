using Parameters, Plots, Colors,LinearAlgebra #import the libraries we want
# Run the Deterministic Part:
include("Main_Functions.jl") #import the functions that solve our growth model
prim, res = Initialize() #initialize primitive and results structs
@elapsed HR_Solve(prim,res)
@unpack W_func, x_prime, mu_mat, M_entry, P_new,L_D,n_D = res
@unpack s_grid, ns, ν_s,F_s = prim
# Run the Stochastic Part:
include("Main_Functions_Stoch.jl") #import the functions that solve our growth model
prim1, res1 = Initialize2() #initialize primitive and results structs
Stoch = Dict()
for aa = 1:2
    res1.α_indic = aa
    @elapsed HR_Solve_Stoch(prim1,res1)
    Stoch["RES_" * string(aa)] = deepcopy(res1)
    prim1, res1 = Initialize2() #Update primitives and results structs
end

# Extract Results
res1 = Stoch["RES_1"]
res2 = Stoch["RES_2"]

############## Make plots
#Custom Colors:
myGreen = RGB(0.1,0.4,0.2);
myBlue  = RGB(0,0.3,0.8);
myGold  = RGB(0.5,0.4,0.1);

# Value Functions
Plots.plot(s_grid', [W_func, res1.U_s, res2.U_s], title="Value Functions \$W(s;p^{*})\$", labels = ["Deterministic" "\$ \\alpha = 1 \$" "\$ \\alpha = 2 \$" ],color=[myBlue myGold myGreen],legend = :topleft,xticks=s_grid')
xlabel!("Productivity Level (s)")
ylabel!("Value W(s;p)")
Plots.savefig("Value_Functions.pdf")

# Policy Functions
s_cutoff = minimum(s_grid'[findall(x_prime.<1)])
Plots.plot(s_grid', [x_prime res1.x_prime res2.x_prime], title="Policy Functions ", labels = ["Deterministic" "TV \$ \\alpha = 1 \$" "TV \$ \\alpha = 2 \$" ],color=[myBlue myGold myGreen],legend = :topright,xticks=s_grid')
plot!([s_cutoff, s_cutoff], [minimum(x_prime)-2e-2, maximum(x_prime)], label ="Exit Cutoff = "*string(s_cutoff),linestyle=:dash,color=:black)
xlabel!("Productivity Level (s)")
ylabel!("P(Exit Decision \$X'(s;p^{*}) = 1 \$)")
Plots.savefig("Policy_Functions.pdf")

# Distribution of Firms:
Plots.plot(s_grid', [mu_mat res1.mu_mat res2.mu_mat], title="Firms Size Distribution \$μ^{*}(s;p^{*})\$", labels = ["Deterministic" "TV \$ \\alpha = 1 \$" "TV \$ \\alpha = 2 \$" ],color=[myBlue myGold myGreen],legend = :topright,xticks=s_grid')
xlabel!("Productivity Level (s)")
ylabel!("\$μ^{*}(s;p^{*})\$")
Plots.savefig("Firms_Distribution.pdf")
println("Plots done!")

# ------------------------ #
#   Get Results for Table  #
# ------------------------ #

# Prices:
println("Equilibrium Prices are: ")
println("...Deterministic: ", round(P_new,digits=3))  
println("...TV1 : ", round(res1.P_new,digits=3))  
println("...TV2 : ", round(res2.P_new,digits=3))  
# Mass of Incumbents:
println("Mass of Incumbents: ")
println("...Deterministic: ", round(sum(mu_mat.-M_entry*(F_s.*(1 .-x_prime))'*ν_s'),digits=3))  
println("...TV1 : ", round(sum(res1.mu_mat .- res1.M_entry*(F_s.*(1 .- res1.x_prime))'*ν_s'),digits=3))  
println("...TV2 : ", round(sum(res2.mu_mat .- res2.M_entry*(F_s.*(1 .- res2.x_prime))'*ν_s'),digits=3)) 
# Mass of Entrants:
println("Mass of Entry: ")
println("...Deterministic: ", round(M_entry,digits=3))  
println("...TV1 : ", round(res1.M_entry,digits=3))  
println("...TV2 : ", round(res2.M_entry,digits=3))  
# Mass of Exits:
println("Mass of Exits: ")
println("...Deterministic: ", round(sum((F_s.*x_prime)'*mu_mat),digits=3))  
println("...TV1 : ", round(sum((F_s.*res1.x_prime)'*res1.mu_mat),digits=3))  
println("...TV2 : ", round(sum((F_s.*res2.x_prime)'*res2.mu_mat),digits=3))
# Aggregate Labor:
println("Aggregate Labor: ")
println("...Deterministic: ", round(sum(L_D),digits=3))  
println("...TV1 : ", round(sum(res1.L_D),digits=3))  
println("...TV2 : ", round(sum(res2.L_D),digits=3))
# Labor of Incumbents:
println("Incumbents' Labor: ")
println("...Deterministic: ", round(sum(n_D'*mu_mat),digits=3))  
println("...TV1 : ", round(sum(res1.n_D'*res1.mu_mat),digits=3))  
println("...TV2 : ", round(sum(res2.n_D'*res2.mu_mat),digits=3))
# Labor of Entrants:
println("Entrants' Labor: ")
println("...Deterministic: ", round(sum(L_D-n_D'*mu_mat),digits=3))  
println("...TV1 : ", round(sum(res1.L_D-res1.n_D'*res1.mu_mat),digits=3))  
println("...TV2 : ", round(sum(res2.L_D-res2.n_D'*res2.mu_mat),digits=3))
# Fraction of Entrants:
println("Entrants' Labor/Agg. Labor: ")
println("...Deterministic: ", round(1- sum(n_D'*mu_mat/L_D),digits=3))  
println("...TV1 : ", round(1- sum(res1.n_D'*res1.mu_mat/res1.L_D),digits=3))  
println("...TV2 : ", round(1- sum(res2.n_D'*res2.mu_mat/res2.L_D),digits=3))


println("All done!")
