"""
test_duality_gap.jl — Diagnose ISP follower strong duality gap
when inner trust region is ON vs OFF.
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using LinearAlgebra
using Infiltrator
using Revise

includet("../network_generator.jl")
includet("../build_uncertainty_set.jl")
includet("../strict_benders.jl")
includet("../nested_benders_trust_region.jl")

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary

# ===== Parameters =====
S = 1
λU = 10.0
γ_ratio = 0.10
ρ_param = 0.2
v = 1.0
seed = 42
epsilon = 0.5
ϕU = 1/epsilon

# ===== Generate 4×4 Network =====
network = generate_grid_network(4, 4, seed=seed)
num_arcs = length(network.arcs) - 1
num_interdictable = sum(network.interdictable_arcs[1:num_arcs])
γ = ceil(Int, γ_ratio * num_interdictable)
println("γ = $γ")

capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
w = ρ_param * γ * c_bar
println("w = $(round(w, digits=4)), w/S = $(round(w/S, digits=4))")

capacity_scenarios_regular = capacities[1:end-1, :]
R, r_dict, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon)
uncertainty_set = Dict(:R => R, :r_dict => r_dict, :xi_bar => xi_bar, :epsilon => epsilon)

# ===== Build OMP and get initial (x,h,λ,ψ0) =====
omp_model, omp_vars = build_omp(network, ϕU, λU, γ, w; optimizer=Gurobi.Optimizer, multi_cut=true)
st, λ_sol, x_sol, h_sol, ψ0_sol = initialize_omp(omp_model, omp_vars)
println("Initial λ=$λ_sol, x=$(x_sol[1:5])..., h=$(h_sol[1:5])...")

# ===== Build IMP and ISP =====
imp_model, imp_vars = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set; mip_optimizer=Gurobi.Optimizer)
st_imp, α_sol_init = initialize_imp(imp_model, imp_vars)
println("Initial α sum = $(sum(α_sol_init)), expected w/S = $(w/S)")
println("Initial α = $α_sol_init")

# Build dual ISP instances
leader_instances, follower_instances = initialize_isp(
    network, S, ϕU, λU, γ, w, v, uncertainty_set;
    conic_optimizer=Mosek.Optimizer, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol_init)

E = ones(num_arcs, num_arcs+1)
d0 = zeros(num_arcs + 1); d0[end] = 1.0
isp_data = Dict(:E => E, :network => network, :ϕU => ϕU, :λU => λU, :γ => γ, :w => w, :v => v, :uncertainty_set => uncertainty_set, :d0 => d0, :S=>S)

# ===== Test 1: ISP follower with initial α (like inner_tr=false first iter) =====
println("\n" * "="^60)
println("TEST 1: ISP follower with INITIAL α (unconstrained)")
println("="^60)

U_s = Dict(:R => Dict(:1=>R[1]), :r_dict => Dict(:1=>r_dict[1]), :xi_bar => Dict(:1=>xi_bar[1]), :epsilon => epsilon)

# First solve to get some cuts
(status_l, cut_l) = isp_leader_optimize!(leader_instances[1][1], leader_instances[1][2];
    isp_data=isp_data, uncertainty_set=U_s, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol_init)
println("Leader status: $status_l, obj=$(cut_l[:obj_val])")

(status_f, cut_f) = isp_follower_optimize!(follower_instances[1][1], follower_instances[1][2];
    isp_data=isp_data, uncertainty_set=U_s, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol_init)
println("Follower status: $status_f, obj=$(cut_f[:obj_val])")

# ===== Test 2: Simulate trust-region-constrained α =====
println("\n" * "="^60)
println("TEST 2: ISP follower with TRUST-REGION-like α")
println("="^60)

# Simulate what happens with tight trust region:
# α_center = initial α, B_conti = 0.01 * w/S
# IMP with cuts would produce some α near the center
# Let's manually create a uniform-ish α
α_uniform = fill(w / (S * num_arcs), num_arcs)
println("Uniform α: sum=$(sum(α_uniform)), each=$(α_uniform[1])")

(status_f2, cut_f2) = isp_follower_optimize!(follower_instances[1][1], follower_instances[1][2];
    isp_data=isp_data, uncertainty_set=U_s, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_uniform)
println("Follower status: $status_f2, obj=$(cut_f2[:obj_val])")

# ===== Test 3: Various α patterns =====
println("\n" * "="^60)
println("TEST 3: Various α patterns")
println("="^60)

test_alphas = Dict(
    "sparse (all in arc 1)" => begin a = zeros(num_arcs); a[1] = w/S; a end,
    "sparse (2 arcs)" => begin a = zeros(num_arcs); a[1] = w/(2S); a[2] = w/(2S); a end,
    "uniform" => fill(w/(S*num_arcs), num_arcs),
    "half-sparse" => begin a = zeros(num_arcs); n=div(num_arcs,2); a[1:n] .= w/(S*n); a end,
)

for (name, α_test) in test_alphas
    println("\n--- α pattern: $name ---")
    println("  sum(α)=$(round(sum(α_test), digits=6)), nonzero=$(count(x->x>1e-10, α_test))/$(num_arcs)")

    model = follower_instances[1][1]
    vars = follower_instances[1][2]

    # Update and solve
    diag_x_E = Diagonal(x_sol) * E
    diag_λ_ψ = Diagonal(λ_sol*ones(num_arcs)-v.*ψ0_sol)
    true_S = isp_data[:S]
    Utilde1, Utilde3, Ztilde1_3 = vars[:Utilde1], vars[:Utilde3], vars[:Ztilde1_3]
    Ptilde1_Φ, Ptilde1_Π, Ptilde2_Φ, Ptilde2_Π = vars[:Ptilde1_Φ], vars[:Ptilde1_Π], vars[:Ptilde2_Φ], vars[:Ptilde2_Π]
    Ptilde1_Y, Ptilde1_Yts, Ptilde2_Y, Ptilde2_Yts = vars[:Ptilde1_Y], vars[:Ptilde1_Yts], vars[:Ptilde2_Y], vars[:Ptilde2_Yts]
    βtilde1_1, βtilde1_3 = vars[:βtilde1_1], vars[:βtilde1_3]

    local_S = 1
    obj_term1 = [-ϕU * sum(Utilde1[s, :, :] .* diag_x_E) for s=1:local_S]
    obj_term2 = [-ϕU * sum(Utilde3[s, :, :] .* (E-diag_x_E)) for s=1:local_S]
    obj_term4 = [sum(Ztilde1_3[s, :, :] .* (diag_λ_ψ * diagm(xi_bar[1]))) for s=1:local_S]
    obj_term5 = [(λ_sol*d0')* βtilde1_1[s,:] for s=1:local_S]
    obj_term6 = [-(h_sol + diag_λ_ψ * xi_bar[1])'* βtilde1_3[s,:] for s=1:local_S]
    obj_term_ub_tilde = [-ϕU * sum(Ptilde1_Φ[s,:,:]) - ϕU * sum(Ptilde1_Π[s,:,:]) - ϕU * sum(Ptilde1_Y[s,:,:]) - ϕU * sum(Ptilde1_Yts[s,:]) for s=1:local_S]
    obj_term_lb_tilde = [-ϕU * sum(Ptilde2_Φ[s,:,:]) - ϕU * sum(Ptilde2_Π[s,:,:]) - ϕU * sum(Ptilde2_Y[s,:,:]) - ϕU * sum(Ptilde2_Yts[s,:]) for s=1:local_S]
    @objective(model, Max, sum(obj_term1) + sum(obj_term2) + sum(obj_term4) + sum(obj_term5) + sum(obj_term6) + sum(obj_term_ub_tilde) + sum(obj_term_lb_tilde))

    coupling_cons = vec(model[:coupling_cons])
    set_normalized_rhs.(coupling_cons, α_test)

    optimize!(model)
    st_test = MOI.get(model, MOI.TerminationStatus())

    if st_test == MOI.OPTIMAL || st_test == MOI.SLOW_PROGRESS
        μtilde = shadow_price.(coupling_cons)
        ηtilde_pos = shadow_price.(vec(model[:cons_dual_constant_pos]))
        ηtilde_neg = shadow_price.(vec(model[:cons_dual_constant_neg]))
        intercept = sum((1/true_S)*(ηtilde_pos-ηtilde_neg))
        subgradient = μtilde
        dual_obj = intercept + α_test'*subgradient
        primal_obj = objective_value(model)
        gap = abs(dual_obj - primal_obj)

        println("  Status: $st_test")
        println("  Primal obj: $(round(primal_obj, digits=6))")
        println("  Dual obj:   $(round(dual_obj, digits=6))")
        println("  Gap:        $(round(gap, digits=6))  $(gap > 1e-4 ? "⚠ FAIL" : "✓ OK")")
        println("  intercept:  $(round(intercept, digits=6))")
        println("  ηtilde_pos: $(round.(ηtilde_pos, digits=6))")
        println("  ηtilde_neg: $(round.(ηtilde_neg, digits=6))")
        println("  μtilde nonzero: $(count(x->abs(x)>1e-6, μtilde))/$(length(μtilde))")

        # Mosek solution quality
        raw_status = MOI.get(model, MOI.RawStatusString())
        println("  Mosek raw status: $raw_status")
    else
        println("  Status: $st_test (not optimal)")
    end
end
