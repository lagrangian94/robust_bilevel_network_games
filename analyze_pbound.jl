using Serialization, LinearAlgebra, Statistics
using JuMP, Mosek, MosekTools, Gurobi

include("network_generator.jl")
include("build_uncertainty_set.jl")
include("parallel_utils.jl")
include("strict_benders.jl")
include("nested_benders_trust_region.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model

nb1 = deserialize("nb_cut_pool_grid_3x3_S2.jls")

network = generate_grid_network(3, 3, seed=42)
S = 2; epsilon = 0.5; v = 1.0; γ_ratio = 0.10; ρ = 0.2
num_arcs = length(network.arcs) - 1
capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=42)
R, r_dict, xi_bar = build_robust_counterpart_matrices(capacities[1:end-1, :], epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)
ϕU = 1/epsilon; λU = ϕU
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, γ_ratio * length(interdictable_idx))
c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
w = round(ρ * γ * c_bar, digits=4)
πU = ϕU; yU = ϕU; ytsU = ϕU
d0 = zeros(num_arcs+1); d0[end] = 1.0

# ISP 구축
x0 = zeros(num_arcs); λ0 = 0.0; h0 = zeros(num_arcs); ψ0_0 = zeros(num_arcs); α0 = zeros(num_arcs)
leader_instances, follower_instances = initialize_isp(
    network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer,
    λ_sol=λ0, x_sol=x0, h_sol=h0, ψ0_sol=ψ0_0, α_sol=α0,
    πU=πU, yU=yU, ytsU=ytsU)

isp_data = Dict(:E => ones(num_arcs, num_arcs+1), :network => network, :ϕU => ϕU,
    :πU => πU, :yU => yU, :ytsU => ytsU, :λU => λU, :γ => γ, :w => w, :v => v,
    :uncertainty_set => uncertainty_set, :d0 => d0, :S => S)

opt_cuts = filter(c -> c[:type] == :opt_cut, nb1)

println("="^120)
println("P-BOUND PENALTY ANALYSIS — grid_3x3, S=1, ϕU=$ϕU, πU=$πU, yU=$yU, ytsU=$ytsU")
println("="^120)
println()

for c in opt_cuts
    xs = c[:x_sol]; λs = c[:λ_sol]; hs = c[:h_sol]; ψ0s = c[:ψ0_sol]
    αs = c[:α_sol]

    for s in 1:S
        U_s = Dict(:R => Dict(:1=>R[s]), :r_dict => Dict(:1=>r_dict[s]),
                    :xi_bar => Dict(:1=>xi_bar[s]), :epsilon => epsilon)
        isp_leader_optimize!(leader_instances[s][1], leader_instances[s][2];
            isp_data=isp_data, uncertainty_set=U_s,
            λ_sol=λs, x_sol=xs, h_sol=hs, ψ0_sol=ψ0s, α_sol=αs)
        isp_follower_optimize!(follower_instances[s][1], follower_instances[s][2];
            isp_data=isp_data, uncertainty_set=U_s,
            λ_sol=λs, x_sol=xs, h_sol=hs, ψ0_sol=ψ0s, α_sol=αs)
    end

    total_obj_l = 0.0; total_obj_f = 0.0
    total_P_l = 0.0; total_P_f = 0.0

    for s in 1:S
        vl = leader_instances[s][2]; vf = follower_instances[s][2]
        obj_l = objective_value(leader_instances[s][1])
        obj_f = objective_value(follower_instances[s][1])

        P_l = -ϕU * sum(value.(vl[:Phat1_Φ])) - πU * sum(value.(vl[:Phat1_Π])) -
               ϕU * sum(value.(vl[:Phat2_Φ])) - πU * sum(value.(vl[:Phat2_Π]))

        P_f = -ϕU * sum(value.(vf[:Ptilde1_Φ])) - πU * sum(value.(vf[:Ptilde1_Π])) -
               ϕU * sum(value.(vf[:Ptilde2_Φ])) - πU * sum(value.(vf[:Ptilde2_Π])) -
               yU * sum(value.(vf[:Ptilde1_Y])) - ytsU * sum(value.(vf[:Ptilde1_Yts])) -
               yU * sum(value.(vf[:Ptilde2_Y])) - ytsU * sum(value.(vf[:Ptilde2_Yts]))

        total_obj_l += obj_l; total_obj_f += obj_f
        total_P_l += P_l; total_P_f += P_f
    end

    total_obj = (total_obj_l + total_obj_f) / S
    total_P = (total_P_l + total_P_f) / S
    non_P = total_obj - total_P
    P_ratio = abs(total_P) / abs(non_P) * 100

    println("iter $(c[:iter]), x=$(findall(xs .> 0.5)), Q=$(round(c[:subprob_obj], digits=4))")
    println("  obj/S      = $(round(total_obj, digits=4))")
    println("  P-penalty/S= $(round(total_P, digits=4))  ($(round(P_ratio, digits=1))% of |non-P|)")
    println("  non-P/S    = $(round(non_P, digits=4))")

    for s in 1:S
        vl = leader_instances[s][2]; vf = follower_instances[s][2]
        Φ1_l = -ϕU*sum(value.(vl[:Phat1_Φ])); Π1_l = -πU*sum(value.(vl[:Phat1_Π]))
        Φ2_l = -ϕU*sum(value.(vl[:Phat2_Φ])); Π2_l = -πU*sum(value.(vl[:Phat2_Π]))
        Φ1_f = -ϕU*sum(value.(vf[:Ptilde1_Φ])); Π1_f = -πU*sum(value.(vf[:Ptilde1_Π]))
        Φ2_f = -ϕU*sum(value.(vf[:Ptilde2_Φ])); Π2_f = -πU*sum(value.(vf[:Ptilde2_Π]))
        Y1_f = -yU*sum(value.(vf[:Ptilde1_Y])); Yts1_f = -ytsU*sum(value.(vf[:Ptilde1_Yts]))
        Y2_f = -yU*sum(value.(vf[:Ptilde2_Y])); Yts2_f = -ytsU*sum(value.(vf[:Ptilde2_Yts]))
        println("  [s=$s] Leader  -ϕU·ΣP̂Φ: UB=$(round(Φ1_l,digits=4)) LB=$(round(Φ2_l,digits=4))  -πU·ΣP̂Π: UB=$(round(Π1_l,digits=4)) LB=$(round(Π2_l,digits=4))")
        println("       Follower -ϕU·ΣP̃Φ: UB=$(round(Φ1_f,digits=4)) LB=$(round(Φ2_f,digits=4))  -πU·ΣP̃Π: UB=$(round(Π1_f,digits=4)) LB=$(round(Π2_f,digits=4))  -yU·ΣP̃Y: UB=$(round(Y1_f,digits=4)) LB=$(round(Y2_f,digits=4))  -ytsU·ΣP̃Yts: UB=$(round(Yts1_f,digits=4)) LB=$(round(Yts2_f,digits=4))")
    end
    println()
end
