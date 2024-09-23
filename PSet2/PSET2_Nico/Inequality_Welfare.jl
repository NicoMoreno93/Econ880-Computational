
function calculate_wealth_distr(res::Results)
    @unpack a_grid, z_grid, nz, na = prim
    
    w = zeros(na, nz)

    for i_s = 1:nz
        for i_a = 1:na

            i_w = argmin(abs.(a_grid[i_a] .+ z_grid[i_s] .- a_grid))

            w[i_w, i_s] = wealth_distr[i_a, i_s]
        end
    end
    w
end

function Lorenz_Curve_Computation(wealth_distr::Array{Float64, 2})
    @unpack a_grid, na = prim

    x = cumsum(wealth_distr[:,1] .+ wealth_distr[:,2])
    y = cumsum((wealth_distr[:,1] .+ wealth_distr[:,2]) .* a_grid)

    unique([x/x[na] y/y[na]]; dims = 1)
end


function Gini_Computation(LC::Array{Float64, 2})
    l = LC;
    widths = diff(l[:,1])
    heights = ((l[1:end-1,1] .+ l[2:end,1])./2 .- (l[1:end-1,2] .+ l[2:end,2])./2)
    a = sum(widths .* heights)

    l_pos = l[l[:,2].>0, :]
    widths = diff(l_pos[:,1])
    heights = (l_pos[1:end-1,2] .+ l_pos[2:end,2])./2
    b = sum(widths .* heights)

    a/(a+b)
end


function Complete_Markets_Welfare()
    @unpack α, β, z_grid, Π = Primitives()
    fixed_point_Π = [Π[2,1]/(1-Π[1,1]+Π[2,1]) 1-Π[2,1]/(1-Π[1,1]+Π[2,1])]
    c_IC = fixed_point_Π[1] * z_grid[1] + fixed_point_Π[2] * z_grid[2]
    ((c_IC)^(1 - α) - 1)/((1 - α) * (1 - β))
end

function Welfare_Computation(res::Results, w_fb::Float64)
    @unpack α, β = Primitives()

    numerator = w_fb + 1 /((1 - α)*(1 - β))
    denominator = res.val_func .+ (1 ./((1 .- α).*(1 .- β)))

    (numerator./denominator).^(1/(1 .- α)) .- 1
end