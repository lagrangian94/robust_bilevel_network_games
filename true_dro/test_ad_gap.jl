"""
test_ad_gap.jl — a≠d vs a=d insample comparison at fixed x*.
Polska S=20, ε̂=ε̃=0.1, β=0.95, x*=[3,34].
"""

using JuMP, Gurobi, Printf, LinearAlgebra

include("../network_generator.jl")
NG = NetworkGenerator
include("true_dro_data.jl")
include("true_dro_build_omp.jl")
include("true_dro_build_subproblem.jl")

net = NG.generate_polska_network()
num_arcs = length(net.arcs) - 1
S = 20; γ = 2
intd_arcs = fill(true, length(net.arcs))
net = NG.RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
    net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)
caps, _ = NG.generate_capacity_scenarios_uniform_model(length(net.arcs), S;
    interdictable_arcs=intd_arcs, seed=42)
intd_idx = findall(intd_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
q_hat = fill(1.0/S, S)

td = make_true_dro_data(net, caps, q_hat, 0.1, 0.1; w=w, lambda_U=10.0, gamma=γ, beta=0.95)

# 여러 x 후보 테스트
using Random
Random.seed!(123)

# γ=2이므로 interdictable arc 중 2개 조합을 여러 개 뽑기
all_intd = findall(intd_arcs[1:num_arcs])
x_candidates = Vector{Vector{Float64}}()

# 랜덤 1개
idx = sort(shuffle(all_intd)[1:γ])
x_rand = zeros(num_arcs)
for i in idx; x_rand[i] = 1.0; end
push!(x_candidates, x_rand)

function solve_sub_ad(td, x_bar; force_a_eq_d=false)
    sub_model, sub_vars = build_true_dro_subproblem(td, x_bar;
        optimizer=Gurobi.Optimizer, silent=false)
    set_optimizer_attribute(sub_model, "NonConvex", 2)
    if force_a_eq_d
        for s in 1:td.S
            @constraint(sub_model, sub_vars[:a][s] == sub_vars[:d][s])
        end
    end
    JuMP.optimize!(sub_model)
    Z0 = objective_value(sub_model)
    a_val = [value(sub_vars[:a][s]) for s in 1:td.S]
    d_val = [value(sub_vars[:d][s]) for s in 1:td.S]
    r_val = get(sub_vars, :_use_cvar, false) ? [value(sub_vars[:r][s]) for s in 1:td.S] : nothing
    return Z0, a_val, d_val, r_val
end

for (ci, x_bar) in enumerate(x_candidates)
    x_arcs = findall(x_bar .> 0.5)
    println("\n" * "="^76)
    @printf("Case %d: x = %s\n", ci, x_arcs)
    println("="^76)

    Z0_free, a_free, d_free, r_free = solve_sub_ad(td, x_bar; force_a_eq_d=false)
    Z0_tied, a_tied, d_tied, r_tied = solve_sub_ad(td, x_bar; force_a_eq_d=true)

    # α 추출
    function get_alpha(td, x_bar; force_a_eq_d=false)
        sub_model, sub_vars = build_true_dro_subproblem(td, x_bar;
            optimizer=Gurobi.Optimizer, silent=false)
        set_optimizer_attribute(sub_model, "NonConvex", 2)
        if force_a_eq_d
            for s in 1:td.S
                @constraint(sub_model, sub_vars[:a][s] == sub_vars[:d][s])
            end
        end
        JuMP.optimize!(sub_model)
        return [value(sub_vars[:α][k]) for k in 1:td.num_arcs]
    end

    α_free = get_alpha(td, x_bar; force_a_eq_d=false)
    α_tied = get_alpha(td, x_bar; force_a_eq_d=true)

    # nonzero α만 출력
    nz = findall(k -> abs(α_free[k]) > 1e-4 || abs(α_tied[k]) > 1e-4, 1:num_arcs)

    gap = Z0_free - Z0_tied
    α_diff = sum(abs.(α_free .- α_tied))

    @printf("  Z₀(free)=%.4f  Z₀(tied)=%.4f  gap=%.6f (%.4f%%)\n",
            Z0_free, Z0_tied, gap, gap/max(abs(Z0_tied),1e-10)*100)
    @printf("  ||a-d||₁(free)=%.6f\n", sum(abs.(a_free .- d_free)))
    @printf("  ||α_free-α_tied||₁=%.6f  →  %s\n", α_diff, α_diff < 1e-4 ? "α SAME" : "α DIFFERENT")

    if !isempty(nz)
        @printf("  nonzero α arcs: ")
        for k in nz
            @printf("k%d[%.2f/%.2f] ", k, α_free[k], α_tied[k])
        end
        println()
    end
end
