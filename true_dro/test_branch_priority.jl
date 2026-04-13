"""
test_branch_priority.jl — BranchPriority on α 효과 검증.

동일한 bilinear subproblem instance (고정 x̄)에 대해:
  A: BranchPriority 없음 (default)
  B: α에 BranchPriority=100

여러 x̄에서 반복 → solve time, node count, objective 비교.
"""

using Revise
using JuMP
using Gurobi
using Printf
using LinearAlgebra
using Random
using Statistics

if !@isdefined(NetworkGenerator)
    include("../network_generator.jl")
end
using .NetworkGenerator

includet("true_dro_data.jl")
includet("true_dro_build_omp.jl")
includet("true_dro_build_subproblem.jl")


"""
    generate_x_candidates(td, n_random; seed=42)

테스트용 x̄ 후보 생성:
  1. x = 0 (no interdiction)
  2. x = greedy (가장 큰 capacity arc부터 γ개)
  3~n: random feasible x ∈ {0,1}^K, Σx ≤ γ, interdictable only
"""
function generate_x_candidates(td, n_random; seed=42)
    K = td.num_arcs
    γ = td.gamma
    rng = MersenneTwister(seed)

    candidates = Vector{Float64}[]

    # (1) x = 0
    push!(candidates, zeros(K))

    # (2) greedy: interdictable arcs sorted by mean capacity desc, pick top γ
    x_greedy = zeros(K)
    interdictable = findall(td.interdictable_arcs[1:K])
    mean_cap = [mean(td.xi_bar[k, :]) for k in 1:K]
    sorted_idx = sort(interdictable, by=k -> mean_cap[k], rev=true)
    for k in sorted_idx[1:min(γ, length(sorted_idx))]
        x_greedy[k] = 1.0
    end
    push!(candidates, x_greedy)

    # (3~) random feasible
    for _ in 1:n_random
        x_rand = zeros(K)
        pool = shuffle(rng, interdictable)
        for k in pool[1:min(γ, length(pool))]
            x_rand[k] = 1.0
        end
        push!(candidates, x_rand)
    end

    return candidates
end


"""
    solve_with_config(td, x_bar, time_limit; branch_priority_alpha=false, seed=0)

Bilinear subproblem을 빌드 + 풀고 결과 반환.
매번 새 모델 빌드 (캐시/warm-start 영향 제거).
Gurobi `Seed`로 internal randomness 통제.
"""
function solve_with_config(td, x_bar, time_limit;
                           branch_priority_alpha::Bool=false,
                           gurobi_seed::Int=0)
    K = td.num_arcs

    model, vars = build_true_dro_subproblem(td, x_bar;
        optimizer=Gurobi.Optimizer, silent=true)
    set_optimizer_attribute(model, "NonConvex", 2)
    set_optimizer_attribute(model, "TimeLimit", time_limit)
    set_optimizer_attribute(model, "Seed", gurobi_seed)
    set_optimizer_attribute(model, "Threads", 1)  # deterministic

    if branch_priority_alpha
        for k in 1:K
            MOI.set(model, Gurobi.VariableAttribute("BranchPriority"),
                    vars[:α][k], 100)
        end
    end

    optimize!(model)

    st = termination_status(model)
    has_sol = (st == MOI.OPTIMAL) ||
              (st == MOI.TIME_LIMIT && has_values(model))

    obj = has_sol ? objective_value(model) : NaN
    bound = has_dual_bound(model) ? dual_objective_value(model) : NaN
    gap_val = has_sol ? abs(bound - obj) / max(abs(obj), 1e-10) : NaN
    node_count = MOI.get(model, MOI.NodeCount())
    solve_time = solve_time_sec = MOI.get(model, MOI.SolveTimeSec())

    return (status=st, obj=obj, bound=bound, gap=gap_val,
            nodes=node_count, time=solve_time,
            branch_prio=branch_priority_alpha)
end


function has_dual_bound(model)
    try
        dual_objective_value(model)
        return true
    catch
        return false
    end
end


# ================================================================
# Main
# ================================================================
println("=" ^ 70)
println("BranchPriority on α: A/B Test")
println("=" ^ 70)

# --- Instance setup ---
print("Network (1=grid, 2=sioux_falls, 3=abilene, 4=polska) [4]: ")
net_str = strip(readline())
net_choice = isempty(net_str) ? 4 : parse(Int, net_str)

if net_choice == 1
    print("Grid m [3]: "); m = parse(Int, something(tryparse(Int, strip(readline())), 3) |> string)
    print("Grid n [3]: "); n = parse(Int, something(tryparse(Int, strip(readline())), 3) |> string)
    network = generate_grid_network(m, n; seed=42)
elseif net_choice == 2
    network = generate_sioux_falls_network()
elseif net_choice == 3
    network = generate_abilene_network()
else
    network = generate_polska_network()
end

print("S [10]: "); S = parse(Int, let s=strip(readline()); isempty(s) ? "10" : s end)
print("ε̂ [0.1]: "); ε_hat = parse(Float64, let s=strip(readline()); isempty(s) ? "0.1" : s end)
print("Time limit per solve (sec) [60]: "); tl = parse(Float64, let s=strip(readline()); isempty(s) ? "60" : s end)
print("Random x̄ 개수 [5]: "); n_rand = parse(Int, let s=strip(readline()); isempty(s) ? "5" : s end)
print("Gurobi seeds (comma-sep) [0]: "); seeds_str = strip(readline())
gurobi_seeds = isempty(seeds_str) ? [0] : parse.(Int, split(seeds_str, ","))

num_arcs = length(network.arcs) - 1
num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, 0.10 * num_interdictable)
capacities, _ = generate_capacity_scenarios_uniform_model(length(network.arcs), S; seed=42)
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
w = round(0.2 * γ * c_bar; digits=4)
λU = 1.0 / ε_hat
q_hat = fill(1.0 / S, S)

td = make_true_dro_data(network, capacities, q_hat, ε_hat, ε_hat;
                        w=w, lambda_U=λU, gamma=γ)

println("\n|A|=$num_arcs, S=$S, γ=$γ, w=$w, ε̂=$ε_hat")
println("Time limit: $(tl)s, Random x̄: $n_rand, Seeds: $gurobi_seeds")

# --- Generate x̄ candidates ---
x_candidates = generate_x_candidates(td, n_rand; seed=42)
println("x̄ candidates: $(length(x_candidates))")

# --- Run A/B tests ---
println("\n" * "-" ^ 70)
@printf("%-4s %-6s %-5s %-10s %-10s %-10s %-8s %-8s\n",
        "x#", "prio", "seed", "obj", "bound", "gap", "nodes", "time(s)")
println("-" ^ 70)

results = []

for (xi, x_bar) in enumerate(x_candidates)
    for gseed in gurobi_seeds
        for use_prio in [false, true]
            r = solve_with_config(td, x_bar, tl;
                                  branch_priority_alpha=use_prio,
                                  gurobi_seed=gseed)
            push!(results, (x_idx=xi, seed=gseed, r...))

            prio_str = use_prio ? "α=100" : "none"
            @printf("%-4d %-6s %-5d %-10.4f %-10.4f %-10.2e %-8d %-8.2f\n",
                    xi, prio_str, gseed, r.obj, r.bound, r.gap, r.nodes, r.time)
        end
    end
end

# --- Summary ---
println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)

# Paired comparison: same (x_idx, seed), with vs without prio
no_prio = filter(r -> !r.branch_prio, results)
yes_prio = filter(r -> r.branch_prio, results)

time_speedup = Float64[]
node_ratio = Float64[]
gap_diff = Float64[]

for np in no_prio
    yp = filter(r -> r.x_idx == np.x_idx && r.seed == np.seed, yes_prio)
    if !isempty(yp)
        yp = first(yp)
        if np.time > 0 && yp.time > 0
            push!(time_speedup, np.time / yp.time)
        end
        if np.nodes > 0 && yp.nodes > 0
            push!(node_ratio, np.nodes / yp.nodes)
        end
        if !isnan(np.gap) && !isnan(yp.gap)
            push!(gap_diff, np.gap - yp.gap)
        end
    end
end

if !isempty(time_speedup)
    @printf("Time speedup (no_prio/yes_prio): mean=%.2fx, median=%.2fx, range=[%.2f, %.2f]\n",
            mean(time_speedup), median(time_speedup),
            minimum(time_speedup), maximum(time_speedup))
end
if !isempty(node_ratio)
    @printf("Node ratio  (no_prio/yes_prio): mean=%.2fx, median=%.2fx, range=[%.2f, %.2f]\n",
            mean(node_ratio), median(node_ratio),
            minimum(node_ratio), maximum(node_ratio))
end
if !isempty(gap_diff)
    @printf("Gap diff (no - yes, positive=prio better): mean=%.2e, median=%.2e\n",
            mean(gap_diff), median(gap_diff))
end

n_prio_faster = count(x -> x > 1.0, time_speedup)
n_total = length(time_speedup)
@printf("\nα-priority faster: %d / %d (%.0f%%)\n",
        n_prio_faster, n_total, 100 * n_prio_faster / max(n_total, 1))
