"""
Compare Benders decomposition algorithms:
1. Strict Benders
2. Nested Benders (plain)
3. TR Nested Benders — 4 combinations: (outer_tr, inner_tr) = {(F,F), (T,F), (F,T), (T,T)}
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using Hypatia
using LinearAlgebra
using Infiltrator
using Plots
using Serialization
using Revise

includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("strict_benders.jl")
includet("nested_benders.jl")
includet("nested_benders_trust_region.jl")
includet("plot_benders.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary

# ===== Common Parameters =====
S = 10
ϕU = 10.0
λU = 10.0
γ = 2.0
w = 1.0
v = 1.0
seed = 42
epsilon = 0.5

# ===== JIT Warm-up =====
#
# Julia는 JIT(Just-In-Time) 컴파일러를 사용한다. 함수가 처음 호출될 때 Julia는
# 해당 함수의 인자 타입에 맞는 네이티브 머신코드를 생성(컴파일)한다.
# 이 과정은 한 번만 발생하며, 이후 동일 타입으로 호출하면 이미 컴파일된 코드를 재사용한다.
#
# 문제: 첫 번째로 실행되는 알고리즘이 JIT 컴파일 시간을 떠안게 되어,
# 실제 알고리즘 실행 시간보다 훨씬 느리게 측정된다.
# 예) Strict Benders가 첫 번째로 실행되면 23초, warm-up 후엔 2.8초.
#
# 해결: 실제 측정 전에 작은 인스턴스(3x3)로 모든 코드 경로를 한 번씩 실행하여
# JIT 컴파일을 완료시킨다. warm-up 실행의 결과는 버린다.
# 이후 실제 인스턴스 측정에서는 순수 알고리즘 실행 시간만 측정된다.
# 추가로 각 측정 전 GC.gc()를 호출하여 가비지 컬렉션이 측정 중에 개입하는 것을 방지한다.
#
println("="^80)
println("JIT WARM-UP (3x3 grid, S=1, results discarded)")
println("="^80)

warmup_S = 1
actual_S = S  # 실제 S를 보존
S = warmup_S  # solver 내부에서 전역 S, R, r_dict, xi_bar, epsilon을 참조하므로 임시로 변경

network = generate_grid_network(3, 3, seed=seed)
warm_cap, _ = generate_capacity_scenarios_uniform_model(length(network.arcs), warmup_S, seed=seed)
R, r_dict, xi_bar = build_robust_counterpart_matrices(warm_cap[1:end-1, :], epsilon)
warm_uset = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

# Warm-up: Strict Benders
wm1, wv1 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=false)
strict_benders_optimize!(wm1, wv1, network, ϕU, λU, γ, w, warm_uset; optimizer=Gurobi.Optimizer)

# Warm-up: Nested Benders
wm2, wv2 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
nested_benders_optimize!(wm2, wv2, network, ϕU, λU, γ, w, warm_uset; mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer, multi_cut=true)

# Warm-up: TR variants (4 combinations)
for (otr, itr) in [(false,false), (true,false), (false,true), (true,true)]
    wm, wv = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
    tr_nested_benders_optimize!(wm, wv, network, ϕU, λU, γ, w, warm_uset; mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer, multi_cut=true, outer_tr=otr, inner_tr=itr)
end

S = actual_S  # 실제 S 복원
println("Warm-up complete.\n")
# ===== Generate Network & Uncertainty Set =====
println("="^80)
println("GENERATING NETWORK AND UNCERTAINTY SET")
println("="^80)

network = generate_grid_network(3, 3, seed=seed)
print_network_summary(network)
capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
capacity_scenarios_regular = capacities[1:end-1, :]
R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

results = Dict{String, Any}()

# ===== 1. Strict Benders =====
println("\n" * "="^80)
println("1. STRICT BENDERS DECOMPOSITION")
println("="^80)

GC.gc()
model1, vars1 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=false)
t1_start = time()
result1 = strict_benders_optimize!(model1, vars1, network, ϕU, λU, γ, w, uncertainty_set; optimizer=Gurobi.Optimizer)
t1_end = time()
results["strict_benders"] = t1_end - t1_start
println("\n>> Strict Benders time: $(results["strict_benders"]) seconds")


# ===== 2. Nested Benders =====
println("\n" * "="^80)
println("2. NESTED BENDERS DECOMPOSITION")
println("="^80)

GC.gc()
model2, vars2 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
t2_start = time()
result2 = nested_benders_optimize!(model2, vars2, network, ϕU, λU, γ, w, uncertainty_set; mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer, multi_cut=true)
t2_end = time()
results["nested_benders"] = t2_end - t2_start
if haskey(result2, :solution_time)
    results["nested_benders_internal"] = result2[:solution_time]
end
println("\n>> Nested Benders time: $(results["nested_benders"]) seconds")


# ===== 3. TR Nested Benders — No TR (outer=false, inner=false) =====
println("\n" * "="^80)
println("3a. TR NESTED BENDERS — NO TR (outer=false, inner=false)")
println("="^80)

GC.gc()
model3a, vars3a = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
t3a_start = time()
result3a = tr_nested_benders_optimize!(model3a, vars3a, network, ϕU, λU, γ, w, uncertainty_set; mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer, multi_cut=true, outer_tr=false, inner_tr=false)
t3a_end = time()
results["tr_none"] = t3a_end - t3a_start
if haskey(result3a, :solution_time)
    results["tr_none_internal"] = result3a[:solution_time]
end
println("\n>> No TR time: $(results["tr_none"]) seconds")

# ===== 3b. TR Nested Benders — Outer TR only =====
println("\n" * "="^80)
println("3b. TR NESTED BENDERS — OUTER TR ONLY (outer=true, inner=false)")
println("="^80)

GC.gc()
model3b, vars3b = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
t3b_start = time()
result3b = tr_nested_benders_optimize!(model3b, vars3b, network, ϕU, λU, γ, w, uncertainty_set; mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer, multi_cut=true, outer_tr=true, inner_tr=false)
t3b_end = time()
results["tr_outer_only"] = t3b_end - t3b_start
if haskey(result3b, :solution_time)
    results["tr_outer_only_internal"] = result3b[:solution_time]
end
println("\n>> Outer TR only time: $(results["tr_outer_only"]) seconds")

# ===== 3c. TR Nested Benders — Inner TR only =====
println("\n" * "="^80)
println("3c. TR NESTED BENDERS — INNER TR ONLY (outer=false, inner=true)")
println("="^80)

GC.gc()
model3c, vars3c = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
t3c_start = time()
result3c = tr_nested_benders_optimize!(model3c, vars3c, network, ϕU, λU, γ, w, uncertainty_set; mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer, multi_cut=true, outer_tr=false, inner_tr=true)
t3c_end = time()
results["tr_inner_only"] = t3c_end - t3c_start
if haskey(result3c, :solution_time)
    results["tr_inner_only_internal"] = result3c[:solution_time]
end
println("\n>> Inner TR only time: $(results["tr_inner_only"]) seconds")

# ===== 3d. TR Nested Benders — Both (original) =====
println("\n" * "="^80)
println("3d. TR NESTED BENDERS — BOTH TR (outer=true, inner=true)")
println("="^80)

GC.gc()
model3d, vars3d = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
t3d_start = time()
result3d = tr_nested_benders_optimize!(model3d, vars3d, network, ϕU, λU, γ, w, uncertainty_set; mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer, multi_cut=true, outer_tr=true, inner_tr=true)
t3d_end = time()
results["tr_both"] = t3d_end - t3d_start
if haskey(result3d, :solution_time)
    results["tr_both_internal"] = result3d[:solution_time]
end
println("\n>> Both TR time: $(results["tr_both"]) seconds")

# ===== Summary =====
println("\n" * "="^80)
println("COMPARISON SUMMARY")
println("="^80)
println("  Strict Benders:              $(round(results["strict_benders"], digits=2)) sec")
println("  Nested Benders:              $(round(results["nested_benders"], digits=2)) sec")
println("  TR None (F,F):               $(round(results["tr_none"], digits=2)) sec")
println("  TR Outer only (T,F):         $(round(results["tr_outer_only"], digits=2)) sec")
println("  TR Inner only (F,T):         $(round(results["tr_inner_only"], digits=2)) sec")
println("  TR Both (T,T):               $(round(results["tr_both"], digits=2)) sec")
println("="^80)
