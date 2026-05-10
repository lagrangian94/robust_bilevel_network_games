# test_alpha_sweep.jl — α 분배 sweep으로 saturation 실험적 검증.
# Case A (x=[1,11], v-homo): α를 k1/k11에 어떻게 분배해도 MF^s 불변 → saturation
# Case B (x=[1,9], v-hetero): α 분배에 따라 MF^s 변화 → saturation 깨짐

using JuMP, Gurobi, Printf, LinearAlgebra, Random
include("../network_generator.jl"); NG = NetworkGenerator
include("true_dro_data.jl")

net = NG.generate_grid_network(3, 3; seed=42)
K = length(net.arcs) - 1; S = 7

caps, _ = NG.generate_capacity_scenarios_uniform_model(length(net.arcs), S;
    interdictable_arcs=net.interdictable_arcs, seed=42)
intd_idx = findall(net.interdictable_arcs[1:K])
w = round(maximum(caps[intd_idx, :]); digits=4)
q_hat = fill(1.0/S, S)

# 모든 arc interdictable
intd_all = fill(true, length(net.arcs))
net = NG.GridNetworkData(net.m, net.n, net.nodes, net.arcs, net.N,
    intd_all, net.arc_directions, net.arc_adjacency, net.node_arc_incidence)

# v_k^s ~ Bernoulli(0.75)
Random.seed!(77)
v_rand = zeros(K, S)
for k in 1:K, s in 1:S
    v_rand[k, s] = rand() < 0.75 ? 1.0 : 0.0
end

# ── node-arc incidence 준비 (source-removed) ──
nai = net.node_arc_incidence
nodes_no_s = [n for n in net.nodes if n != "s"]
node_idx = Dict(n => i for (i, n) in enumerate(nodes_no_s))
nv1 = length(nodes_no_s)
Ny = zeros(nv1, K)
Nts = zeros(nv1)
t_idx = node_idx["t"]
Nts[t_idx] = -1.0  # dummy arc: t → s
for k in 1:K
    from, to = net.arcs[k]
    if from != "s"; Ny[node_idx[from], k] = -1.0; end
    if to != "s"; Ny[node_idx[to], k] = 1.0; end
end

# ── max-flow LP for scenario s, given x and α ──
function maxflow_s(x_bar, α_val, s)
    model = Model(Gurobi.Optimizer); set_silent(model)
    @variable(model, y[1:K] >= 0)
    @variable(model, σ)
    for k in 1:K
        eff = caps[k, s] * (1.0 - v_rand[k, s] * x_bar[k]) + α_val[k]
        set_upper_bound(y[k], max(0.0, eff))
    end
    @constraint(model, [j=1:nv1], sum(Ny[j,k]*y[k] for k in 1:K) + Nts[j]*σ == 0)
    @objective(model, Max, σ)
    optimize!(model)
    return objective_value(model)
end

γ = 2; budget = γ * w  # = 18.0

println("grid3x3, S=$S, γ=$γ, w=$w, budget=γ·w=$budget")
println("v_rand pattern on key arcs:")
@printf("%5s %15s", "k", "arc")
for s in 1:S; @printf(" s%d", s); end; println()
for k in [1,2,3,9,11]
    @printf("%5d %15s", k, "$(net.arcs[k][1])→$(net.arcs[k][2])")
    for s in 1:S; @printf("  %d", Int(v_rand[k,s])); end; println()
end

# ── α sweep steps ──
α1_vals = 0.0:1.0:budget

for (label, x_list) in [
    ("A: x=[1,11] v-homo (k1,k11 both v=1 all s)", [1, 11]),
    ("B: x=[1,9] v-hetero (k9: v=[0,0,1,1,1,1,1])", [1, 9]),
]

x_bar = zeros(K)
for k in x_list; x_bar[k] = 1.0; end
k_a, k_b = x_list

println("\n" * "="^90)
println(label)
println("x = $x_list,  sweep α[$k_a] ∈ [0, $budget], α[$k_b] = $budget - α[$k_a]")
println("="^90)

# header
@printf("%6s %6s |", "α[$k_a]", "α[$k_b]")
for s in 1:S; @printf(" MF(s%d)", s); end
@printf("  min_MF  max_MF  spread\n")
println("-"^90)

for α1 in α1_vals
    α_val = zeros(K)
    α_val[k_a] = α1
    α_val[k_b] = budget - α1

    mfs = [maxflow_s(x_bar, α_val, s) for s in 1:S]
    mn = minimum(mfs); mx = maximum(mfs)
    @printf("%6.1f %6.1f |", α1, budget - α1)
    for s in 1:S; @printf(" %6.2f", mfs[s]); end
    @printf("  %6.2f  %6.2f  %6.2f\n", mn, mx, mx - mn)
end

end  # for loop
