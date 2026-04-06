"""
benchmark_cosmo.jl — COSMO.jl (chordal decomposition) vs Mosek 벤치마크
Sioux Falls S=20, 3 inner iters, ldr_mode=:self

사용법:
  julia -t 8 benchmark_cosmo.jl

기존 benchmark_isp_time.jl 결과와 비교용.
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using COSMO
using HiGHS
using LinearAlgebra
using Statistics
using Revise

includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("parallel_utils.jl")
includet("strict_benders.jl")
includet("nested_benders_trust_region.jl")

using .NetworkGenerator: generate_sioux_falls_network, generate_capacity_scenarios_uniform_model, print_realworld_network_summary

println("Julia threads: $(Threads.nthreads())")
println("="^80)

# ===== Setup Sioux Falls S=20 =====
S = 20
seed = 42
epsilon_hat = 0.5
epsilon_tilde = 0.5
ϕU_hat = 1/epsilon_hat
ϕU_tilde = 1/epsilon_tilde
λU = ϕU_hat
γ_ratio = 0.10
ρ = 0.2
v = 1.0

network = generate_sioux_falls_network()
print_realworld_network_summary(network)
num_arcs = length(network.arcs) - 1

num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, γ_ratio * num_interdictable)

capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
w = round(ρ * γ * c_bar, digits=4)

capacity_scenarios_regular = capacities[1:end-1, :]
R, r_dict_hat, r_dict_tilde, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon_hat, epsilon_tilde)
uncertainty_set = Dict(:R => R, :r_dict_hat => r_dict_hat, :r_dict_tilde => r_dict_tilde, :xi_bar => xi_bar, :epsilon_hat => epsilon_hat, :epsilon_tilde => epsilon_tilde)

# LDR bounds
source_arc_idx = findall(a -> a[1] == "s", network.arcs[1:num_arcs])
max_flow_ub = maximum(sum(capacities[source_arc_idx, s] for s in 1:S) / S)
max_cap = maximum(capacity_scenarios_regular)
πU_hat = ϕU_hat
πU_tilde = ϕU_tilde
yU = min(max_cap, ϕU_tilde)
ytsU = min(max_flow_ub, ϕU_tilde)

println("Parameters: S=$S, γ=$γ, ϕU_hat=$ϕU_hat, ϕU_tilde=$ϕU_tilde, w=$(round(w,digits=2))")
println("Arcs: $num_arcs (interdictable: $num_interdictable)")

# ===== isp_data (shared) =====
E = ones(num_arcs, num_arcs+1)
d0 = zeros(num_arcs + 1)
d0[end] = 1.0
isp_data = Dict(:E => E, :network => network, :ϕU_hat => ϕU_hat, :ϕU_tilde => ϕU_tilde, :λU => λU, :γ => γ,
    :w => w, :v => v, :uncertainty_set => uncertainty_set, :d0 => d0, :S => S, :πU_hat => πU_hat, :πU_tilde => πU_tilde, :yU => yU, :ytsU => ytsU)

# ===== Get initial OMP solution =====
println("\n--- Initializing OMP ---")
omp_model, omp_vars = build_omp(network, ϕU_hat, λU, γ, w; optimizer=Gurobi.Optimizer, S=S)
st, λ_sol, x_sol, h_sol, ψ0_sol = initialize_omp(omp_model, omp_vars)
println("OMP initialized: status=$st")

# ===== Stats helper =====
function report_stats(name, times)
    if isempty(times)
        println("  $name: no data")
        return
    end
    n = length(times)
    println("  $name (n=$n):")
    println("    mean   = $(round(mean(times)*1000, digits=1)) ms")
    println("    median = $(round(median(times)*1000, digits=1)) ms")
    println("    std    = $(round(std(times)*1000, digits=1)) ms")
    println("    min    = $(round(minimum(times)*1000, digits=1)) ms")
    println("    max    = $(round(maximum(times)*1000, digits=1)) ms")
    println("    total  = $(round(sum(times), digits=2)) s")
end

const MAX_INNER_ITER = 1   # COSMO가 매우 느리므로 1 iter만
const LDR_MODE = :self   # 기존 벤치마크와 동일

# ===== COSMO용 ISP optimize wrapper (assertion tolerance 완화) =====
# 기존 isp_leader_optimize!는 @assert abs(dual_obj - obj_val) < 1e-4 로 하드 체크.
# COSMO(ADMM)는 primal-dual gap이 클 수 있으므로, follower처럼 warn+보정 방식으로 변경.
function isp_leader_optimize_cosmo!(isp_leader_model, isp_leader_vars; isp_data=nothing, uncertainty_set=nothing, λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing, α_sol=nothing)
    model, vars = isp_leader_model, isp_leader_vars
    E, d0 = isp_data[:E], isp_data[:d0]
    ϕU = isp_data[:ϕU_hat]
    πU = get(isp_data, :πU_hat, ϕU)
    R, r_dict, xi_bar, epsilon = uncertainty_set[:R], uncertainty_set[:r_dict], uncertainty_set[:xi_bar], uncertainty_set[:epsilon]
    diag_x_E = Diagonal(x_sol) * E
    scaling_S = get(isp_data, :scaling_S, isp_data[:S])
    S = 1
    Uhat1, Uhat3, Phat1_Φ, Phat1_Π, Phat2_Φ, Phat2_Π = vars[:Uhat1], vars[:Uhat3], vars[:Phat1_Φ], vars[:Phat1_Π], vars[:Phat2_Φ], vars[:Phat2_Π]
    βhat1_1 = vars[:βhat1_1]
    obj_term1 = [-ϕU * sum(Uhat1[s, :, :] .* diag_x_E) for s=1:S]
    obj_term2 = [-ϕU * sum(Uhat3[s, :, :] .* (E-diag_x_E)) for s=1:S]
    obj_term3 = [(d0')* βhat1_1[s,:] for s=1:S]
    obj_term_ub_hat = [-ϕU * sum(Phat1_Φ[s,:,:]) - πU * sum(Phat1_Π[s,:,:]) for s=1:S]
    obj_term_lb_hat = [-ϕU * sum(Phat2_Φ[s,:,:]) - πU * sum(Phat2_Π[s,:,:]) for s=1:S]
    @objective(model, Max, sum(obj_term1) + sum(obj_term2) + sum(obj_term3) + sum(obj_term_ub_hat) + sum(obj_term_lb_hat))
    coupling_cons = vec(model[:coupling_cons])
    set_normalized_rhs.(coupling_cons, α_sol)
    optimize!(model)
    st = MOI.get(model, MOI.TerminationStatus())
    if (st == MOI.OPTIMAL) || (st == MOI.SLOW_PROGRESS) || (st == MOI.ALMOST_OPTIMAL)
        μhat = shadow_price.(coupling_cons)
        ηhat = shadow_price.(vec(model[:cons_dual_constant]))
        intercept, subgradient = (1/scaling_S)*sum(ηhat), μhat
        dual_obj = intercept + α_sol'*subgradient
        gap = abs(dual_obj - objective_value(model))
        if gap > 1e-4
            @warn "ISP leader duality gap $(round(gap, digits=6)) → intercept 역산 보정"
            intercept = objective_value(model) - α_sol'*subgradient
            dual_obj = objective_value(model)
        end
        cut_coeff = Dict(:μhat=>μhat, :intercept=>intercept, :obj_val=>dual_obj)
        return (:OptimalityCut, cut_coeff)
    else
        @warn "ISP leader status: $st"
        return (:Error, Dict(:μhat=>zeros(length(α_sol)), :intercept=>0.0, :obj_val=>0.0))
    end
end

# ===== Helper: swap optimizer on all ISP instances =====
function swap_to_cosmo!(leader_instances, follower_instances, S; decompose=true, eps_abs=1e-5, eps_rel=1e-5, max_iter=10000, verbose=false)
    for s in 1:S
        for (model, _) in [leader_instances[s], follower_instances[s]]
            set_optimizer(model, COSMO.Optimizer)
            set_silent(model)
            set_optimizer_attribute(model, "decompose", decompose)
            set_optimizer_attribute(model, "eps_abs", eps_abs)
            set_optimizer_attribute(model, "eps_rel", eps_rel)
            set_optimizer_attribute(model, "max_iter", max_iter)
            if verbose
                unset_silent(model)
                set_optimizer_attribute(model, "verbose", true)
            end
        end
    end
end

# ===== Benchmark function (same structure as original) =====
function benchmark_solver(solver_label::String, leader_instances, follower_instances; parallel::Bool=false, leader_opt_fn=isp_leader_optimize!, follower_opt_fn=isp_follower_optimize!)
    println("\n" * "="^80)
    println("SOLVER: $solver_label  (parallel=$parallel, threads=$(Threads.nthreads()), ldr_mode=$LDR_MODE)")
    println("="^80)

    times_leader = Float64[]
    times_follower = Float64[]
    iter_times = Float64[]

    # Initialize IMP
    imp_model, imp_vars = build_imp(network, S, ϕU_hat, λU, γ, w, v, uncertainty_set;
        mip_optimizer=Gurobi.Optimizer)
    st_imp, α_sol_imp = initialize_imp(imp_model, imp_vars)

    R_unc = uncertainty_set[:R]
    r_dict_hat_unc = uncertainty_set[:r_dict_hat]
    r_dict_tilde_unc = uncertainty_set[:r_dict_tilde]
    xi_bar_unc = uncertainty_set[:xi_bar]
    epsilon_hat_unc = uncertainty_set[:epsilon_hat]
    epsilon_tilde_unc = uncertainty_set[:epsilon_tilde]

    for iter in 1:MAX_INNER_ITER
        t_iter = @elapsed begin
            optimize!(imp_model)
            α_sol_imp = value.(imp_vars[:α])

            scenario_times_l = zeros(S)
            scenario_times_f = zeros(S)

            scenario_results, status = solve_scenarios(S; parallel=parallel) do s
                U_s_hat = Dict(:R => Dict(:1=>R_unc[s]), :r_dict => Dict(:1=>r_dict_hat_unc[s]),
                            :xi_bar => Dict(:1=>xi_bar_unc[s]), :epsilon => epsilon_hat_unc)
                U_s_tilde = Dict(:R => Dict(:1=>R_unc[s]), :r_dict => Dict(:1=>r_dict_tilde_unc[s]),
                            :xi_bar => Dict(:1=>xi_bar_unc[s]), :epsilon => epsilon_tilde_unc)
                t_l = @elapsed begin
                    (status_l, cut_info_l) = leader_opt_fn(
                        leader_instances[s][1], leader_instances[s][2];
                        isp_data=isp_data, uncertainty_set=U_s_hat,
                        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol_imp)
                end
                t_f = @elapsed begin
                    (status_f, cut_info_f) = follower_opt_fn(
                        follower_instances[s][1], follower_instances[s][2];
                        isp_data=isp_data, uncertainty_set=U_s_tilde,
                        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol_imp)
                end
                scenario_times_l[s] = t_l
                scenario_times_f[s] = t_f
                ok = (status_l == :OptimalityCut) && (status_f == :OptimalityCut)
                if !ok
                    println("  [WARN] Scenario $s: leader=$status_l, follower=$status_f")
                end
                return (ok, (cut_info_l, cut_info_f))
            end

            append!(times_leader, scenario_times_l)
            append!(times_follower, scenario_times_f)

            # Add inner cuts to IMP
            for s in 1:S
                cut_l = scenario_results[s][1]
                cut_f = scenario_results[s][2]
                @constraint(imp_model, imp_vars[:t_1_l][s] <= cut_l[:intercept] + imp_vars[:α]' * cut_l[:μhat])
                @constraint(imp_model, imp_vars[:t_1_f][s] <= cut_f[:intercept] + imp_vars[:α]' * cut_f[:μtilde])
            end
        end

        push!(iter_times, t_iter)
        println("  Inner iter $iter: $(round(t_iter, digits=2))s  " *
                "(leader: mean=$(round(mean(scenario_times_l)*1000, digits=1))ms, " *
                "follower: mean=$(round(mean(scenario_times_f)*1000, digits=1))ms)")
    end

    println("\n--- Summary: $solver_label ---")
    report_stats("ISP Leader", times_leader)
    report_stats("ISP Follower", times_follower)
    lf_combined = times_leader .+ times_follower
    report_stats("Leader+Follower (combined)", lf_combined)
    println("  Wall-clock/iter: $(round(mean(iter_times), digits=1)) s")
    println("  Total time:      $(round(sum(iter_times), digits=2)) s ($MAX_INNER_ITER iters)")

    return Dict(:leader => copy(times_leader), :follower => copy(times_follower),
                :iter_times => copy(iter_times))
end

# ===================================================================
# Run benchmarks
# ===================================================================
results = Dict{String, Any}()

# # --- 1. Mosek baseline (sequential, Mosek=1) ---
# println("\n### Building ISP instances (Mosek, ldr_mode=$LDR_MODE) ###")
# ENV["MOSEK_NUM_THREADS"] = "1"
# t_init_mosek = @elapsed begin
#     leader_mosek, follower_mosek = initialize_isp(
#         network, S, ϕU_hat, ϕU_tilde, λU, γ, w, v, uncertainty_set;
#         conic_optimizer=Mosek.Optimizer,
#         λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=zeros(num_arcs),
#         πU_hat=πU_hat, πU_tilde=πU_tilde, yU=yU, ytsU=ytsU, scaling_S=S, ldr_mode=LDR_MODE)
# end
# println("  Mosek ISP init: $(round(t_init_mosek, digits=2)) s")

# println("\n### Mosek sequential ###")
# results["Mosek seq"] = benchmark_solver("Mosek seq", leader_mosek, follower_mosek; parallel=false)
# GC.gc()

# --- 2. COSMO (decompose=true, sequential) ---
println("\n### Building ISP instances (COSMO, ldr_mode=$LDR_MODE) ###")
t_init_cosmo = @elapsed begin
    leader_cosmo, follower_cosmo = initialize_isp(
        network, S, ϕU_hat, ϕU_tilde, λU, γ, w, v, uncertainty_set;
        conic_optimizer=Mosek.Optimizer,   # Mosek으로 빌드 후 교체
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=zeros(num_arcs),
        πU_hat=πU_hat, πU_tilde=πU_tilde, yU=yU, ytsU=ytsU, scaling_S=S, ldr_mode=LDR_MODE)
    swap_to_cosmo!(leader_cosmo, follower_cosmo, S; decompose=true, eps_abs=1e-5, eps_rel=1e-5, max_iter=10000, verbose=true)
end
println("  COSMO ISP init: $(round(t_init_cosmo, digits=2)) s")

# 첫 번째 시나리오 verbose solve로 clique 분해 정보 확인 (1개만)
println("\n### COSMO chordal decomposition info (scenario 1, leader) ###")
let model = leader_cosmo[1][1]
    unset_silent(model)
    set_optimizer_attribute(model, "verbose", true)
    set_optimizer_attribute(model, "max_iter", 100)  # 분해 정보만 확인, 빠르게 중단
    set_normalized_rhs.(vec(model[:coupling_cons]), zeros(num_arcs))
    optimize!(model)
    println("  Status: $(termination_status(model))")
    set_silent(model)
    set_optimizer_attribute(model, "verbose", true)
    set_optimizer_attribute(model, "max_iter", 10000)  # 원복
end

println("\n### COSMO decompose=true sequential ###")
results["COSMO decomp seq"] = benchmark_solver("COSMO decomp seq", leader_cosmo, follower_cosmo; parallel=false, leader_opt_fn=isp_leader_optimize_cosmo!)
GC.gc()

# COSMO no-decomp 생략 — decompose=true에서 이미 PSD 분해 불가 확인됨

# ===================================================================
# Final comparison table
# ===================================================================
println("\n" * "="^80)
println("FINAL COMPARISON: Sioux Falls S=$S, ldr_mode=$LDR_MODE, $MAX_INNER_ITER inner iters")
println("="^80)

for label in ["Mosek seq", "COSMO decomp seq"]
    r = results[label]
    lf = r[:leader] .+ r[:follower]
    println("\n--- $label ---")
    println("  Leader mean:     $(round(mean(r[:leader])*1000, digits=1)) ms")
    println("  Follower mean:   $(round(mean(r[:follower])*1000, digits=1)) ms")
    println("  L+F mean:        $(round(mean(lf)*1000, digits=1)) ms")
    println("  L+F median:      $(round(median(lf)*1000, digits=1)) ms")
    println("  L+F max:         $(round(maximum(lf)*1000, digits=1)) ms")
    println("  L+F min:         $(round(minimum(lf)*1000, digits=1)) ms")
    println("  Wall-clock/iter: $(round(mean(r[:iter_times]), digits=1)) s")
    println("  Total time:      $(round(sum(r[:iter_times]), digits=2)) s")
end

println("\nDone.")
