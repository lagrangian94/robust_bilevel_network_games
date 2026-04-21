"""
analyze_polska.jl — 시나리오별 nominal interdiction 해 다양성 분석
  polska, Uniform(1,10) i.i.d., seed=42, S=20, γ=2
  각 시나리오 s에 대해 단독(S=1) nominal SP로 풀어서 x* 비교
"""

using JuMP, Gurobi, Printf, LinearAlgebra, Statistics

include("../../network_generator.jl")
NG = NetworkGenerator

include("../../build_nominal_sp.jl")

# ── network setup ──
net = NG.generate_polska_network()
γ = 2
intd_arcs = fill(true, length(net.arcs))
net = NG.RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
    net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)

num_arcs = length(net.arcs) - 1
S = 20
λU = 10.0

caps, _ = NG.generate_capacity_scenarios_uniform_model(length(net.arcs), S;
    interdictable_arcs=intd_arcs, seed=42)
intd_idx = findall(intd_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)

println("polska uniform: arcs=$num_arcs, intd=$(length(intd_idx)), γ=$γ, λU=$λU, w=$w")
println("Scenarios: S=$S\n")

# ── 시나리오별 단독 solve (nominal SP, S=1) ──
x_solutions = Vector{Vector{Int}}()
z_values = Float64[]

for s in 1:S
    cap_s = caps[:, s:s]  # (num_arcs+1) × 1
    xi_bar_vecs = [cap_s[1:num_arcs, 1]]
    uncertainty_set = Dict(
        :xi_bar => xi_bar_vecs,
        :R => zeros(0, 0),
        :r_dict_hat => Dict(),
        :epsilon_hat => 0.0,
    )

    ϕU = λU
    v_param = 1.0
    model, vars = build_full_2SP_model(net, 1, ϕU, λU, γ, w, v_param, uncertainty_set)
    set_optimizer_attribute(model, "OutputFlag", 0)
    optimize!(model)

    st = termination_status(model)
    if st != MOI.OPTIMAL
        @printf("  s=%2d: FAILED (%s)\n", s, st)
        push!(x_solutions, zeros(Int, num_arcs))
        push!(z_values, NaN)
        continue
    end

    x_int = round.(Int, value.(vars[:x]))
    push!(x_solutions, x_int)
    push!(z_values, objective_value(model))

    intd_arcs_s = findall(x_int .> 0)
    @printf("  s=%2d: Z₀=%8.4f, x=%s\n", s, objective_value(model), intd_arcs_s)
end

# ── 다양성 분석 ──
println("\n" * "="^60)
println("Solution diversity analysis")
println("="^60)

# unique solutions
unique_x = unique(x_solutions)
println("\nUnique x solutions: $(length(unique_x)) / $S")

for (i, ux) in enumerate(unique_x)
    count = sum(x == ux for x in x_solutions)
    scenarios = findall(x == ux for x in x_solutions)
    @printf("  Pattern %d (%d times, scenarios %s): arcs %s\n",
        i, count, scenarios, findall(ux .> 0))
end

# pairwise Hamming distance
println("\nPairwise Hamming distances:")
dists = Int[]
for i in 1:S, j in i+1:S
    d = sum(x_solutions[i] .!= x_solutions[j])
    push!(dists, d)
end
if !isempty(dists)
    @printf("  min=%d, max=%d, mean=%.2f, median=%.1f\n",
        minimum(dists), maximum(dists), mean(dists), sort(dists)[length(dists)÷2+1])
end

# Z0 분포
valid_z = filter(!isnan, z_values)
if !isempty(valid_z)
    println("\nZ₀ distribution:")
    @printf("  min=%.4f, max=%.4f, mean=%.4f, std=%.4f\n",
        minimum(valid_z), maximum(valid_z), mean(valid_z), std(valid_z))
end

# capacity 시나리오 간 variation 확인
println("\nCapacity scenario variation (interdictable arcs only):")
cap_intd = caps[intd_idx, :]
for a in 1:min(5, length(intd_idx))
    vals = cap_intd[a, :]
    @printf("  arc %d: min=%.2f, max=%.2f, std=%.2f\n",
        intd_idx[a], minimum(vals), maximum(vals), std(vals))
end
if length(intd_idx) > 5
    println("  ... ($(length(intd_idx)) arcs total)")
end

cv_per_arc = [std(cap_intd[a, :]) / max(mean(cap_intd[a, :]), 1e-8) for a in 1:size(cap_intd, 1)]
@printf("\nCoeff of variation across arcs: mean=%.3f, min=%.3f, max=%.3f\n",
    mean(cv_per_arc), minimum(cv_per_arc), maximum(cv_per_arc))
