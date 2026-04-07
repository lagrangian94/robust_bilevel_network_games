"""
benchmark_isp_time.jl — Sioux Falls S=20에서 ldr_mode별 개별 ISP solve time 벤치마크

사용법:
  julia -t 8 benchmark_isp_time.jl

IMP 3회 inner iteration만 돌리고, 개별 ISP leader/follower solve time 통계 출력.
ldr_mode = :self, :head, :both 순서로 비교.
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using LinearAlgebra
using Statistics
using Revise

includet("../network_generator.jl")
includet("../build_uncertainty_set.jl")
includet("../parallel_utils.jl")
includet("../strict_benders.jl")
includet("../nested_benders_trust_region.jl")

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

# ===== isp_data (shared across all modes) =====
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

# ===== Benchmark function =====
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

const MAX_INNER_ITER = 3

function benchmark_ldr_mode(ldr_mode::Symbol; parallel::Bool=false)
    println("\n" * "="^80)
    println("LDR MODE: $ldr_mode  (parallel=$parallel, threads=$(Threads.nthreads()))")
    println("="^80)

    times_leader = Float64[]
    times_follower = Float64[]
    times_init = Float64[]
    iter_times = Float64[]

    # Initialize ISP instances (timed)
    t_init = @elapsed begin
        leader_instances, follower_instances = initialize_isp(
            network, S, ϕU_hat, ϕU_tilde, λU, γ, w, v, uncertainty_set;
            conic_optimizer=Mosek.Optimizer,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=zeros(num_arcs),
            πU_hat=πU_hat, πU_tilde=πU_tilde, yU=yU, ytsU=ytsU, scaling_S=S, ldr_mode=ldr_mode)
    end
    println("  ISP initialization: $(round(t_init, digits=2)) s")

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

    # Run MAX_INNER_ITER inner iterations
    for iter in 1:MAX_INNER_ITER
        t_iter = @elapsed begin
            optimize!(imp_model)
            α_sol_imp = value.(imp_vars[:α])
            model_estimate = (sum(value.(imp_vars[:t_1_l])) + sum(value.(imp_vars[:t_1_f]))) / S

            dict_cut_info_l, dict_cut_info_f = Dict(), Dict()
            subprob_obj = 0.0

            # Per-scenario ISP solves (with individual timing)
            scenario_times_l = zeros(S)
            scenario_times_f = zeros(S)

            scenario_results, status = solve_scenarios(S; parallel=parallel) do s
                U_s_hat = Dict(:R => Dict(:1=>R_unc[s]), :r_dict => Dict(:1=>r_dict_hat_unc[s]),
                            :xi_bar => Dict(:1=>xi_bar_unc[s]), :epsilon => epsilon_hat_unc)
                U_s_tilde = Dict(:R => Dict(:1=>R_unc[s]), :r_dict => Dict(:1=>r_dict_tilde_unc[s]),
                            :xi_bar => Dict(:1=>xi_bar_unc[s]), :epsilon => epsilon_tilde_unc)
                t_l = @elapsed begin
                    (status_l, cut_info_l) = isp_leader_optimize!(
                        leader_instances[s][1], leader_instances[s][2];
                        isp_data=isp_data, uncertainty_set=U_s_hat,
                        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol_imp)
                end
                t_f = @elapsed begin
                    (status_f, cut_info_f) = isp_follower_optimize!(
                        follower_instances[s][1], follower_instances[s][2];
                        isp_data=isp_data, uncertainty_set=U_s_tilde,
                        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol_imp)
                end
                scenario_times_l[s] = t_l
                scenario_times_f[s] = t_f
                ok = (status_l == :OptimalityCut) && (status_f == :OptimalityCut)
                return (ok, (cut_info_l, cut_info_f))
            end

            append!(times_leader, scenario_times_l)
            append!(times_follower, scenario_times_f)

            for s in 1:S
                dict_cut_info_l[s] = scenario_results[s][1]
                dict_cut_info_f[s] = scenario_results[s][2]
                subprob_obj += scenario_results[s][1][:obj_val] + scenario_results[s][2][:obj_val]
            end
            subprob_obj /= S

            # Add inner cuts
            subgradient_l = [dict_cut_info_l[s][:μhat] for s in 1:S]
            subgradient_f = [dict_cut_info_f[s][:μtilde] for s in 1:S]
            intercept_l = [dict_cut_info_l[s][:intercept] for s in 1:S]
            intercept_f = [dict_cut_info_f[s][:intercept] for s in 1:S]
            @constraint(imp_model, [s=1:S], imp_vars[:t_1_l][s] <= intercept_l[s] + imp_vars[:α]'*subgradient_l[s])
            @constraint(imp_model, [s=1:S], imp_vars[:t_1_f][s] <= intercept_f[s] + imp_vars[:α]'*subgradient_f[s])
        end

        push!(iter_times, t_iter)
        println("  Inner iter $iter: $(round(t_iter, digits=2))s  " *
                "(leader: mean=$(round(mean(scenario_times_l)*1000, digits=1))ms, " *
                "follower: mean=$(round(mean(scenario_times_f)*1000, digits=1))ms)")
    end

    println("\n--- Summary: ldr_mode=$ldr_mode ---")
    report_stats("ISP Leader", times_leader)
    report_stats("ISP Follower", times_follower)
    report_stats("Leader+Follower (combined)", times_leader .+ times_follower)
    println("  Total iteration time: $(round(sum(iter_times), digits=2)) s ($(MAX_INNER_ITER) iters)")

    GC.gc()
    return Dict(:leader => copy(times_leader), :follower => copy(times_follower),
                :init => t_init, :iter_times => copy(iter_times))
end

# ===== Run benchmarks =====
results = Dict{Symbol, Any}()

# === :both, 8-par, Mosek=1 vs Mosek=3 ===
for mosek_t in [1, 3]
    ENV["MOSEK_NUM_THREADS"] = string(mosek_t)
    label = Symbol("mosek_t$(mosek_t)")
    println("\n### 8-par, Mosek=$mosek_t, ldr_mode=:both ###")
    results[label] = benchmark_ldr_mode(:both; parallel=true)
end

# ===== Final comparison =====
println("\n" * "="^80)
println("FINAL COMPARISON: Sioux Falls S=$S, 8 Julia threads parallel, ldr_mode=:self, $MAX_INNER_ITER inner iters")
println("="^80)
for (k, label) in [(:mosek_t1, "8-par Mosek=1"), (:mosek_t3, "8-par Mosek=3")]
    r = results[k]
    println("\n--- $label ---")
        println("  Julia threads:   $(Threads.nthreads())")
    println("  MOSEK_NUM_THREADS: $(ENV["MOSEK_NUM_THREADS"])")
    println("  ISP Init:        $(round(r[:init], digits=2)) s")
    println("  Leader mean:     $(round(mean(r[:leader])*1000, digits=1)) ms")
    println("  Follower mean:   $(round(mean(r[:follower])*1000, digits=1)) ms")
    println("  L+F mean:        $(round(mean(r[:leader] .+ r[:follower])*1000, digits=1)) ms")
    println("  L+F median:      $(round(median(r[:leader] .+ r[:follower])*1000, digits=1)) ms")
    println("  L+F max:         $(round(maximum(r[:leader] .+ r[:follower])*1000, digits=1)) ms")
    println("  L+F min:         $(round(minimum(r[:leader] .+ r[:follower])*1000, digits=1)) ms")
    println("  Wall-clock/iter: $(round(mean(r[:iter_times]), digits=1)) s")
    println("  Total time:      $(round(sum(r[:iter_times]), digits=2)) s")
end

println("\nDone.")
