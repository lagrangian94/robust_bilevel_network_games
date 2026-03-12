"""
test_primal_isp.jl — Validate primal ISP against dual ISP via strong duality.

Test plan:
1. Set up 3×3 grid, S=1
2. Get initial (x, h, λ, ψ0) from OMP solve
3. Get initial α from IMP
4. Build both dual ISP and primal ISP for same scenario
5. Solve both with same α
6. Check:
   - |primal_obj - dual_obj| < 1e-4 (strong duality)
   - |primal μ̂ - dual μ̂| < 1e-4 per component
   - |primal intercept - dual intercept| < 1e-4
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using Hypatia
using LinearAlgebra
using Infiltrator
using Revise

includet("network_generator.jl")
includet("build_uncertainty_set.jl")
includet("strict_benders.jl")
includet("nested_benders_trust_region.jl")
includet("build_primal_isp.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary

println("="^80)
println("PRIMAL ISP VALIDATION TEST")
println("="^80)

# ===== Parameters =====
S = 1
ϕU = 10.0
λU = 10.0
γ = 2.0
w = 1.0
v = 1.0
seed = 42
epsilon = 0.5

# ===== Generate Network & Uncertainty Set =====
println("\n[1] Generating 3×3 grid network...")
network = generate_grid_network(3, 3, seed=seed)
print_network_summary(network)

capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
capacity_scenarios_regular = capacities[1:end-1, :]
R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

num_arcs = length(network.arcs) - 1

# ===== Get initial (x, h, λ, ψ0) from OMP =====
println("\n[2] Solving OMP to get initial outer variables...")
omp_model, omp_vars = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
optimize!(omp_model)  # unbounded without cuts, but gives feasible point

# Use a reasonable initial point
x_sol = zeros(num_arcs)
# Set some interdictable arcs to 1
interdictable_indices = findall(network.interdictable_arcs)
for i in interdictable_indices[1:min(Int(γ), length(interdictable_indices))]
    x_sol[i] = 1.0
end
λ_sol = 1.0
h_sol = (λ_sol * w / num_arcs) * ones(num_arcs)
ψ0_sol = λ_sol * x_sol  # ψ0 = λ * x (when λ < λU)

println("  x = $x_sol")
println("  λ = $λ_sol")
println("  h = $h_sol")

# ===== Get initial α =====
println("\n[3] Setting initial α...")
imp_model, imp_vars = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set; mip_optimizer=Gurobi.Optimizer)
optimize!(imp_model)
α_sol = value.(imp_vars[:α])
println("  α = $α_sol")

# ===== Common data =====
E = ones(num_arcs, num_arcs+1)
d0 = zeros(num_arcs + 1)
d0[end] = 1.0
isp_data = Dict(
    :E => E,
    :ϕU => ϕU,
    :d0 => d0,
    :S => S,
    :w => w,
    :v => v,
    :uncertainty_set => uncertainty_set,
)

# ===== Build and solve DUAL ISP (current) =====
println("\n[4] Building dual ISP instances...")
dual_leader_instances, dual_follower_instances = initialize_isp(
    network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)

println("\n[5] Solving dual ISP...")
for s in 1:S
    U_s = Dict(:R => Dict(1=>R[s]), :r_dict => Dict(1=>r_dict[s]),
                :xi_bar => Dict(1=>xi_bar[s]), :epsilon => epsilon)

    println("\n  --- Scenario $s ---")
    # Dual ISP leader
    (status_dl, cut_dl) = isp_leader_optimize!(
        dual_leader_instances[s][1], dual_leader_instances[s][2];
        isp_data=isp_data, uncertainty_set=U_s,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
    println("  Dual leader: status=$status_dl, obj=$(cut_dl[:obj_val])")
    println("    intercept=$(cut_dl[:intercept])")
    println("    μhat=$(cut_dl[:μhat])")

    # Dual ISP follower
    (status_df, cut_df) = isp_follower_optimize!(
        dual_follower_instances[s][1], dual_follower_instances[s][2];
        isp_data=isp_data, uncertainty_set=U_s,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
    println("  Dual follower: status=$status_df, obj=$(cut_df[:obj_val])")
    println("    intercept=$(cut_df[:intercept])")
    println("    μtilde=$(cut_df[:μtilde])")
end

# ===== Build and solve PRIMAL ISP =====
println("\n[6] Building primal ISP instances...")
primal_leader_instances, primal_follower_instances = initialize_primal_isp(
    network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)

println("\n[7] Solving primal ISP...")
for s in 1:S
    println("\n  --- Scenario $s ---")
    # Primal ISP leader
    (status_pl, cut_pl) = primal_isp_leader_optimize!(
        primal_leader_instances[s][1], primal_leader_instances[s][2];
        isp_data=isp_data, α_sol=α_sol)
    println("  Primal leader: status=$status_pl, obj=$(cut_pl[:obj_val])")
    println("    intercept=$(cut_pl[:intercept])")
    println("    μhat=$(cut_pl[:μhat])")

    # Primal ISP follower
    (status_pf, cut_pf) = primal_isp_follower_optimize!(
        primal_follower_instances[s][1], primal_follower_instances[s][2];
        isp_data=isp_data, α_sol=α_sol)
    println("  Primal follower: status=$status_pf, obj=$(cut_pf[:obj_val])")
    println("    intercept=$(cut_pf[:intercept])")
    println("    μtilde=$(cut_pf[:μtilde])")
end

# ===== Comparison =====
println("\n" * "="^80)
println("VALIDATION RESULTS")
println("="^80)

all_pass = true
for s in 1:S
    U_s = Dict(:R => Dict(1=>R[s]), :r_dict => Dict(1=>r_dict[s]),
                :xi_bar => Dict(1=>xi_bar[s]), :epsilon => epsilon)

    # Re-solve to get results in same scope
    (_, cut_dl) = isp_leader_optimize!(
        dual_leader_instances[s][1], dual_leader_instances[s][2];
        isp_data=isp_data, uncertainty_set=U_s,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
    (_, cut_df) = isp_follower_optimize!(
        dual_follower_instances[s][1], dual_follower_instances[s][2];
        isp_data=isp_data, uncertainty_set=U_s,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol)
    (_, cut_pl) = primal_isp_leader_optimize!(
        primal_leader_instances[s][1], primal_leader_instances[s][2];
        isp_data=isp_data, α_sol=α_sol)
    (_, cut_pf) = primal_isp_follower_optimize!(
        primal_follower_instances[s][1], primal_follower_instances[s][2];
        isp_data=isp_data, α_sol=α_sol)

    println("\n  Scenario $s:")

    # --- Leader comparison ---
    obj_gap_l = abs(cut_pl[:obj_val] - cut_dl[:obj_val])
    intercept_gap_l = abs(cut_pl[:intercept] - cut_dl[:intercept])
    μhat_gap = maximum(abs.(cut_pl[:μhat] - cut_dl[:μhat]))

    println("    LEADER:")
    println("      Dual obj:      $(cut_dl[:obj_val])")
    println("      Primal obj:    $(cut_pl[:obj_val])")
    println("      Obj gap:       $obj_gap_l  $(obj_gap_l < 1e-4 ? "✓" : "✗")")
    println("      Intercept gap: $intercept_gap_l  $(intercept_gap_l < 1e-4 ? "✓" : "✗")")
    println("      Max μhat gap:  $μhat_gap  $(μhat_gap < 1e-4 ? "✓" : "✗")")

    if obj_gap_l >= 1e-4 || intercept_gap_l >= 1e-4 || μhat_gap >= 1e-4
        all_pass = false
    end

    # --- Follower comparison ---
    obj_gap_f = abs(cut_pf[:obj_val] - cut_df[:obj_val])
    intercept_gap_f = abs(cut_pf[:intercept] - cut_df[:intercept])
    μtilde_gap = maximum(abs.(cut_pf[:μtilde] - cut_df[:μtilde]))

    println("    FOLLOWER:")
    println("      Dual obj:      $(cut_df[:obj_val])")
    println("      Primal obj:    $(cut_pf[:obj_val])")
    println("      Obj gap:       $obj_gap_f  $(obj_gap_f < 1e-4 ? "✓" : "✗")")
    println("      Intercept gap: $intercept_gap_f  $(intercept_gap_f < 1e-4 ? "✓" : "✗")")
    println("      Max μtilde gap: $μtilde_gap  $(μtilde_gap < 1e-4 ? "✓" : "✗")")

    if obj_gap_f >= 1e-4 || intercept_gap_f >= 1e-4 || μtilde_gap >= 1e-4
        all_pass = false
    end
end

println("\n" * "="^80)
if all_pass
    println("ALL TESTS PASSED ✓")
else
    println("SOME TESTS FAILED ✗")
    @infiltrate
end
println("="^80)
