"""
compare_vi_at_xsols.jl — Abilene Benders에서 나온 실제 x* 들로
bilinear subproblem을 VI 유/무로 각각 풀어서 비교.
"""

using Revise
using JuMP
using Gurobi
using Printf
using LinearAlgebra
using Infiltrator

include("../network_generator.jl")
using .NetworkGenerator

include("true_dro_data.jl")
include("true_dro_build_omp.jl")
include("true_dro_build_subproblem.jl")

# --- Abilene γ=3 setup (log_abilene_S10_20260415_2118152 재현) ---
function setup_abilene(γ::Int; S=10, ρ=0.2, seed=42, ε=0.1)
    network = generate_abilene_network()
    num_arcs = length(network.arcs) - 1
    capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), S;
        interdictable_arcs=network.interdictable_arcs, seed=seed)
    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
    w = round(ρ * γ * c_bar; digits=4)
    λU = 2.0
    q_hat = fill(1.0 / S, S)
    td = make_true_dro_data(network, capacities, q_hat, ε, ε;
                            w=w, lambda_U=λU, gamma=γ)
    return td
end

td = setup_abilene(3)
K = td.num_arcs

# --- Abilene γ=3 log에서 뽑은 unique x* 들 ---
x_list = [
    Float64[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    Float64[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    Float64[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    Float64[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    Float64[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    Float64[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    Float64[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    Float64[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    Float64[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    Float64[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    Float64[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    Float64[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # 최종 optimal
    Float64[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]
println("=" ^ 80)
println("Abilene γ=3, S=10: $(length(x_list))개 x*에서 VI 유/무 비교")
println("=" ^ 80)

# ===== Model 두 개 빌드: baseline / arcwise fixing =====
function build_and_configure(td; arcwise::Bool=false, time_limit::Float64=60.0)
    model, vars = build_true_dro_subproblem(td, zeros(td.num_arcs);
        optimizer=Gurobi.Optimizer, silent=true,
        add_objF_vi_arcwise=arcwise)
    set_optimizer_attribute(model, "NonConvex", 2)
    set_optimizer_attribute(model, "MIPGap", 1e-4)
    set_time_limit_sec(model, time_limit)
    return model, vars
end

model_A, vars_A = build_and_configure(td; arcwise=false)           # baseline
model_B, vars_B = build_and_configure(td; arcwise=true)            # arcwise fix

# ===== 각 x에 대해 두 모델에서 solve =====
function _solve_and_measure!(model, vars, td, x)
    update_true_dro_subproblem_objective!(model, vars, td, x)
    t_start = time()
    optimize!(model)
    t = time() - t_start
    st = termination_status(model)
    Z = (st == MOI.OPTIMAL || has_values(model)) ? objective_value(model) : NaN
    Zbnd = try
        objective_bound(model)
    catch
        NaN
    end
    gap = (isnan(Z) || isnan(Zbnd)) ? NaN : abs(Zbnd - Z) / max(abs(Z), 1e-10)
    nodes = try
        Int(MOI.get(model, Gurobi.ModelAttribute("NodeCount")))
    catch
        -1
    end
    return (Z=Z, Zbnd=Zbnd, gap=gap, t=t, nodes=nodes, st=st)
end

println("\n" * "-" ^ 130)
@printf("%4s | %-40s || %10s %8s %7s %8s || %10s %8s %7s %8s\n",
        "idx", "x (nonzero arcs)",
        "Z_A", "t_A(s)", "nd_A", "gap_A",
        "Z_B", "t_B(s)", "nd_B", "gap_B")
println("-" ^ 130)

results = []
for (i, x) in enumerate(x_list)
    nonzero = [k for k in 1:K if x[k] > 0.5]
    nonzero_str = string(nonzero)

    rA = _solve_and_measure!(model_A, vars_A, td, x)
    rB = _solve_and_measure!(model_B, vars_B, td, x)

    @printf("%4d | %-40s || %10.4f %8.2f %7d %7.2e || %10.4f %8.2f %7d %7.2e\n",
            i, nonzero_str,
            rA.Z, rA.t, rA.nodes, rA.gap,
            rB.Z, rB.t, rB.nodes, rB.gap)
    push!(results, (i=i, x=x,
                    Z_A=rA.Z, t_A=rA.t, node_A=rA.nodes, gap_A=rA.gap, st_A=rA.st,
                    Z_B=rB.Z, t_B=rB.t, node_B=rB.nodes, gap_B=rB.gap, st_B=rB.st))
end
println("-" ^ 130)

# ===== Summary =====
println("\n" * "=" ^ 70)
println("Summary   (A: baseline, B: arcwise fixing)")
println("=" ^ 70)
n = length(results)
total_tA = sum(r.t_A for r in results)
total_tB = sum(r.t_B for r in results)
total_nA = sum(r.node_A for r in results if r.node_A >= 0)
total_nB = sum(r.node_B for r in results if r.node_B >= 0)
max_Zdiff = maximum(abs(r.Z_A - r.Z_B) for r in results if !isnan(r.Z_A) && !isnan(r.Z_B))

@printf("Total wall time:  baseline = %.2f s,  arcwise = %.2f s  (ratio = %.2fx)\n",
        total_tA, total_tB, total_tB / max(total_tA, 1e-6))
@printf("Total B&B nodes:  baseline = %d,  arcwise = %d  (ratio = %.2fx)\n",
        total_nA, total_nB, total_nB / max(total_nA, 1))
@printf("Max |Z_A - Z_B|:  %.6e\n", max_Zdiff)

println("\nPer-x breakdown:")
@printf("  %-10s : arcwise faster in %d / %d cases\n", "speedup",
        sum(r.t_B < r.t_A for r in results), n)
@printf("  %-10s : arcwise fewer nodes in %d / %d cases\n", "nodes",
        sum(r.node_B < r.node_A for r in results if r.node_A > 0 && r.node_B > 0), n)
