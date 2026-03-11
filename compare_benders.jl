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

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary,
                         RealWorldNetworkData, generate_sioux_falls_network, generate_nobel_us_network,
                         generate_abilene_network, generate_polska_network, print_realworld_network_summary

"""
결과 진행사항
 S=20, 3x3 grid 결과:

  Strict Benders:              34.96 sec
  Nested Benders:              82.85 sec
  TR None (F,F):               82.63 sec
  TR Outer only (T,F):        122.03 sec
  TR Inner only (F,T):         71.56 sec
  TR Both (T,T):               97.64 sec

  패턴이 일관됩니다:

  ┌────────────┬──────┬───────┬────────┐
  │            │ S=2  │ S=10  │  S=20  │
  ├────────────┼──────┼───────┼────────┤
  │ Strict     │ 2.87 │ 17.32 │ 34.96  │
  ├────────────┼──────┼───────┼────────┤
  │ Nested     │ 5.98 │ 39.77 │ 82.85  │
  ├────────────┼──────┼───────┼────────┤
  │ TR None    │ 6.00 │ 39.83 │ 82.63  │
  ├────────────┼──────┼───────┼────────┤
  │ Inner only │ 5.38 │ 34.33 │ 71.56  │
  ├────────────┼──────┼───────┼────────┤
  │ Outer only │ 6.23 │ 57.51 │ 122.03 │
  ├────────────┼──────┼───────┼────────┤
  │ Both       │ 6.28 │ 47.52 │ 97.64  │
  └────────────┴──────┴───────┴────────┘

  - Inner TR only가 모든 S에서 nested 계열 중 최고 (nested 대비 -13~14%)
  - Outer TR only는 모든 S에서 가장 느림 (+45~47%)
  - Strict Benders가 3x3에서는 여전히 가장 빠름 — 네트워크를 키워야 nested의 이점이 보일 거예요

✻ Baked for 9m 48s
● 5x5, S=2 결과:

  Strict Benders:              1152.20 sec
  Nested Benders:              1194.99 sec
  TR None (F,F):               1193.24 sec
  TR Outer only (T,F):         1138.56 sec
  TR Inner only (F,T):         1001.21 sec  ← 최고
  TR Both (T,T):               1022.95 sec

  5x5로 키우니 흥미로운 변화가 보여요:
  - Inner TR only가 여전히 최고 (1001초, -16% vs nested)
  - Outer TR only가 처음으로 nested보다 빠름 (1138 vs 1195) — 네트워크가 커지니 outer stabilization 효과가 드디어 나타남
  - Both도 nested보다 빠름 (1023 vs 1195)
  - Strict와 Nested가 비슷해짐 — 3x3에선 strict가 압도적이었는데 5x5에선 거의 동급
"""


# ===== Common Parameters =====
S = 1
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

# S = warmup_S  # solver 내부에서 전역 S, R, r_dict, xi_bar, epsilon을 참조하므로 임시로 변경
# network = generate_grid_network(3, 3, seed=seed)
# warm_cap, _ = generate_capacity_scenarios_uniform_model(length(network.arcs), warmup_S, seed=seed)
# R, r_dict, xi_bar = build_robust_counterpart_matrices(warm_cap[1:end-1, :], epsilon)
# warm_uset = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

# # Warm-up: Strict Benders
# wm1, wv1 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=false)
# strict_benders_optimize!(wm1, wv1, network, ϕU, λU, γ, w, warm_uset; optimizer=Gurobi.Optimizer)

# # Warm-up: Nested Benders
# wm2, wv2 = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
# nested_benders_optimize!(wm2, wv2, network, ϕU, λU, γ, w, warm_uset; mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer, multi_cut=true)

# # Warm-up: TR variants (4 combinations)
# for (otr, itr) in [(false,false), (true,false), (false,true), (true,true)]
#     wm, wv = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
#     tr_nested_benders_optimize!(wm, wv, network, ϕU, λU, γ, w, warm_uset; mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer, multi_cut=true, outer_tr=otr, inner_tr=itr)
# end

S = actual_S  # 실제 S 복원
println("Warm-up complete.\n")
# ===== Generate Network & Uncertainty Set =====
println("="^80)
println("GENERATING NETWORK AND UNCERTAINTY SET")
println("="^80)



# realworld_generators = [
#     # ("ABILENE",     generate_abilene_network),
#     # ("POLSKA",      generate_polska_network),
#     ("NOBEL-US",    generate_nobel_us_network),
#     # ("Sioux-Falls", generate_sioux_falls_network),
# ]

# realworld_S = S  # 위에서 설정한 S 사용 (또는 별도로 지정)

# realworld_results = Dict{String, Dict{String, Float64}}()

# for (net_name, gen_func) in realworld_generators
#     println("\n" * "="^80)
#     println("REAL-WORLD NETWORK: $net_name (S=$realworld_S)")
#     println("="^80)

#     rw_network = gen_func()
#     print_realworld_network_summary(rw_network)

#     # Generate capacity scenarios
#     rw_cap, _ = generate_capacity_scenarios_uniform_model(length(rw_network.arcs), realworld_S, seed=seed)
#     rw_cap_regular = rw_cap[1:end-1, :]
#     rw_R, rw_r_dict, rw_xi_bar = build_robust_counterpart_matrices(rw_cap_regular, epsilon)
#     rw_uset = Dict(:R => rw_R, :r_dict => rw_r_dict, :xi_bar => rw_xi_bar, :epsilon => epsilon)

#     net_results = Dict{String, Float64}()

#     # # --- Strict Benders ---
#     # println("\n  [Strict Benders]")
#     # GC.gc()
#     # m1, v1 = build_omp(rw_network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=false)
#     # t_start = time()
#     # strict_benders_optimize!(m1, v1, rw_network, ϕU, λU, γ, w, rw_uset; optimizer=Gurobi.Optimizer, outer_tr=true)
#     # net_results["strict_benders"] = time() - t_start
#     # println("    Time: $(round(net_results["strict_benders"], digits=2)) sec")

#     # --- Nested Benders ---
#     println("\n  [Nested Benders]")
#     GC.gc()
#     m2, v2 = build_omp(rw_network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
#     t_start = time()
#     nested_benders_optimize!(m2, v2, rw_network, ϕU, λU, γ, w, rw_uset; mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer, multi_cut=true)
#     net_results["nested_benders"] = time() - t_start
#     println("    Time: $(round(net_results["nested_benders"], digits=2)) sec")
# end
# @infiltrate
network = generate_grid_network(4, 4, seed=seed)
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

@infiltrate

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

# ==============================================================================
# REAL-WORLD NETWORK EXPERIMENTS
# ==============================================================================
#
# 아래 섹션을 주석 해제하여 real-world 네트워크 실험을 실행하세요.
# 네트워크 선택: generate_sioux_falls_network, generate_nobel_us_network,
#               generate_abilene_network, generate_polska_network
#
# 참고: real-world 네트워크는 GridNetworkData와 동일한 인터페이스를 제공합니다.
#   - source="s", sink="t"로 매핑됨 (dummy arc = ("t","s"))
#   - interdictable_arcs, arc_adjacency, node_arc_incidence 포함
# ==============================================================================

# realworld_generators = [
#     ("ABILENE",     generate_abilene_network),
#     ("POLSKA",      generate_polska_network),
#     ("NOBEL-US",    generate_nobel_us_network),
#     ("Sioux-Falls", generate_sioux_falls_network),
# ]
#
# realworld_S = S  # 위에서 설정한 S 사용 (또는 별도로 지정)
#
# realworld_results = Dict{String, Dict{String, Float64}}()
#
# for (net_name, gen_func) in realworld_generators
#     println("\n" * "="^80)
#     println("REAL-WORLD NETWORK: $net_name (S=$realworld_S)")
#     println("="^80)
#
#     rw_network = gen_func()
#     print_realworld_network_summary(rw_network)
#
#     # Generate capacity scenarios
#     rw_cap, _ = generate_capacity_scenarios_uniform_model(length(rw_network.arcs), realworld_S, seed=seed)
#     rw_cap_regular = rw_cap[1:end-1, :]
#     rw_R, rw_r_dict, rw_xi_bar = build_robust_counterpart_matrices(rw_cap_regular, epsilon)
#     rw_uset = Dict(:R => rw_R, :r_dict => rw_r_dict, :xi_bar => rw_xi_bar, :epsilon => epsilon)
#
#     net_results = Dict{String, Float64}()
#
#     # --- Strict Benders ---
#     println("\n  [Strict Benders]")
#     GC.gc()
#     m1, v1 = build_omp(rw_network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=false)
#     t_start = time()
#     strict_benders_optimize!(m1, v1, rw_network, ϕU, λU, γ, w, rw_uset; optimizer=Gurobi.Optimizer)
#     net_results["strict_benders"] = time() - t_start
#     println("    Time: $(round(net_results["strict_benders"], digits=2)) sec")
#
#     # --- Nested Benders ---
#     println("\n  [Nested Benders]")
#     GC.gc()
#     m2, v2 = build_omp(rw_network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
#     t_start = time()
#     nested_benders_optimize!(m2, v2, rw_network, ϕU, λU, γ, w, rw_uset; mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer, multi_cut=true)
#     net_results["nested_benders"] = time() - t_start
#     println("    Time: $(round(net_results["nested_benders"], digits=2)) sec")
#
#     # --- TR Nested Benders (4 combinations) ---
#     tr_configs = [
#         ("tr_none",       false, false),
#         ("tr_outer_only", true,  false),
#         ("tr_inner_only", false, true),
#         ("tr_both",       true,  true),
#     ]
#     for (label, otr, itr) in tr_configs
#         println("\n  [TR outer=$otr, inner=$itr]")
#         GC.gc()
#         m3, v3 = build_omp(rw_network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
#         t_start = time()
#         tr_nested_benders_optimize!(m3, v3, rw_network, ϕU, λU, γ, w, rw_uset; mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer, multi_cut=true, outer_tr=otr, inner_tr=itr)
#         net_results[label] = time() - t_start
#         println("    Time: $(round(net_results[label], digits=2)) sec")
#     end
#
#     realworld_results[net_name] = net_results
#
#     # --- Per-network summary ---
#     println("\n  " * "-"^60)
#     println("  $net_name SUMMARY (S=$realworld_S)")
#     println("  " * "-"^60)
#     println("    Strict Benders:   $(round(net_results["strict_benders"], digits=2)) sec")
#     println("    Nested Benders:   $(round(net_results["nested_benders"], digits=2)) sec")
#     println("    TR None (F,F):    $(round(net_results["tr_none"], digits=2)) sec")
#     println("    TR Outer (T,F):   $(round(net_results["tr_outer_only"], digits=2)) sec")
#     println("    TR Inner (F,T):   $(round(net_results["tr_inner_only"], digits=2)) sec")
#     println("    TR Both (T,T):    $(round(net_results["tr_both"], digits=2)) sec")
#     println("  " * "-"^60)
# end
#
# # ===== Cross-network comparison =====
# if !isempty(realworld_results)
#     println("\n" * "="^80)
#     println("CROSS-NETWORK COMPARISON (S=$realworld_S)")
#     println("="^80)
#     header = rpad("Network", 15) * join([rpad(a, 14) for a in ["Strict", "Nested", "TR(F,F)", "TR(T,F)", "TR(F,T)", "TR(T,T)"]])
#     println(header)
#     println("-"^99)
#     for (net_name, _) in realworld_generators
#         haskey(realworld_results, net_name) || continue
#         nr = realworld_results[net_name]
#         row = rpad(net_name, 15)
#         for key in ["strict_benders", "nested_benders", "tr_none", "tr_outer_only", "tr_inner_only", "tr_both"]
#             row *= rpad("$(round(nr[key], digits=2))s", 14)
#         end
#         println(row)
#     end
#     println("="^80)
# end
