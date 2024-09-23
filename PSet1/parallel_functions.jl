#= 
This code is a parallelized version of the VFI code for the neoclassical growth model.
The main difference is that the Bellman operator is parallelized using the @distributed macro.
September 2024
=#
@everywhere @with_kw struct Primitives
    β::Float64      = 0.99 #discount rate
    δ::Float64      = 0.025 #depreciation rate
    α::Float64      = 0.36 #capital share
    k_min::Float64  = 0.01 #capital lower bound
    k_max::Float64  = 90.0 #capital upper bound
    Zb::Float64     = 0.2
    Zg::Float64     = 1.25
    nz::Int64       = 2
    nk::Int64       = 1000 #number of capital grid points
    k_grid::SharedVector{Float64} = SharedVector(collect(range(start=k_min, stop=k_max, length=nk)))
    z_grid::SharedVector{Float64} = SharedVector(collect(range(Zb, length = nz, stop = Zg))) #productivity grid
    Π::SharedMatrix{Float64}      = [0.926 0.074; 0.023 0.977]
end

@everywhere @with_kw mutable struct Results
    val_func::SharedArray{Float64}
    pol_func::SharedArray{Float64}
end

@everywhere function Initialize()
    prim = Primitives()
    val_func = SharedArray{Float64}(zeros(prim.nk, prim.nz))
    pol_func = SharedArray{Float64}(zeros(prim.nk, prim.nz))
    res = Results(val_func, pol_func)
    prim, res
end

@everywhere function Bellman(prim::Primitives, res::Results)
    @unpack_Results res
    @unpack_Primitives prim
    
    v_next = SharedArray{Float64}(zeros(nk,nz))

    @sync @distributed for (k_index, z_index) in collect(Iterators.product(1:nk,1:nz))    
            zy = z_grid[z_index]
            k = k_grid[k_index]
            # This is -Inf is just for the first iteration
            candidate_max = -Inf 
            budget = zy*k^α + (1-δ)*k
                       
            for kp_index in 1:nk
                c = budget - k_grid[kp_index]
                if c>0
                    val = log(c) + β*sum(val_func[kp_index,:].*Π[z_index,:])
                    if val > candidate_max
                        candidate_max = val
                        res.pol_func[k_index,z_index] = k_grid[kp_index]
                    end
                end
            end
            v_next[k_index,z_index] = candidate_max    
    end
    v_next
end

function V_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-6, err::Float64 = 100.0)
    n = 0
    while err > tol
        v_next = Bellman(prim, res)
        err = maximum(abs.(v_next .- res.val_func))
        res.val_func .= v_next
        n += 1
    end
    println("Value function converged in ", n, " iterations.")
end

#solve the model
function Solve_model(prim::Primitives, res::Results)
    V_iterate(prim, res) #in this case, all we have to do is the value function iteration!
end