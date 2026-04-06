"""
진단: Nested Benders에서 얻은 (x*, λ*, h*, ψ0*)를 Full Model에 fix하고 풀어서
objective가 일치하는지 확인.

1) NBD solve → x*, λ*, h*, ψ0* 추출
2) Full Model (free) → obj_free
3) Full Model (x=x*, λ=λ*) → obj_fix_xλ
4) Full Model (x=x*, λ=λ*, h=h*, ψ0=ψ0*) → obj_fix_all
5) 비교

julia -t 4 debug_test/test_fix_xλ_fullmodel.jl
"""

using JuMP
using Gurobi
using Mosek, MosekTools
using HiGHS
using Hypatia
using Pajarito
using LinearAlgebra
using Printf
using Revise

const PROJECT_ROOT = joinpath(@__DIR__, "..")
includet(joinpath(PROJECT_ROOT, "network_generator.jl"))
includet(joinpath(PROJECT_ROOT, "build_uncertainty_set.jl"))
includet(joinpath(PROJECT_ROOT, "build_full_model.jl"))
includet(joinpath(PROJECT_ROOT, "parallel_utils.jl"))
includet(joinpath(PROJECT_ROOT, "strict_benders.jl"))
includet(joinpath(PROJECT_ROOT, "nested_benders_trust_region.jl"))

using .NetworkGenerator: generate_grid_network, generate_capacity_scenarios_uniform_model, print_network_summary

# ===== Setup =====
epsilon = 0.5
S = 2
seed = 42

network = generate_grid_network(3, 3, seed=seed)
print_network_summary(network)
num_arcs = length(network.arcs) - 1

ϕU_hat = 1/epsilon
ϕU_tilde = 1/epsilon
λU = ϕU_hat
γ = ceil(Int, 0.10 * sum(network.interdictable_arcs[1:num_arcs]))

capacities, F = generate_capacity_scenarios_uniform_model(length(network.arcs), S, seed=seed)
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
w = round(0.2 * γ * c_bar, digits=4)
v = 1.0

capacity_scenarios_regular = capacities[1:end-1, :]
R, r_dict_hat, r_dict_tilde, xi_bar = build_robust_counterpart_matrices(capacity_scenarios_regular, epsilon, epsilon)
uncertainty_set = Dict(
    :R => R, :r_dict_hat => r_dict_hat, :r_dict_tilde => r_dict_tilde,
    :xi_bar => xi_bar, :epsilon_hat => epsilon, :epsilon_tilde => epsilon)

source_arc_idx = findall(a -> a[1] == "s", network.arcs[1:num_arcs])
max_flow_ub = maximum(sum(capacities[source_arc_idx, s] for s in 1:S) / S)
max_cap = maximum(capacity_scenarios_regular)
πU_hat = ϕU_hat
πU_tilde = ϕU_tilde
yU = min(max_cap, ϕU_tilde)
ytsU = min(max_flow_ub, ϕU_tilde)

println("\n  ε=$epsilon, ϕU_hat=$ϕU_hat, ϕU_tilde=$ϕU_tilde, γ=$γ, w=$w")
println("  πU_hat=$πU_hat, πU_tilde=$πU_tilde, yU=$yU, ytsU=$ytsU")

# ===== 1. Nested Benders =====
println("\n" * "="^70)
println("1. NESTED BENDERS")
println("="^70)

global v_global = v
global S_global = S
global network_global = network

# tr_nested_benders_optimize! uses global v, S, network
global v = v_global
global S = S_global
global network = network_global

GC.gc()
model_b, vars_b = build_omp(network, ϕU_hat, λU, γ, w; optimizer=Gurobi.Optimizer, S=S)
result_b = tr_nested_benders_optimize!(model_b, vars_b, network, ϕU_hat, ϕU_tilde, λU, γ, w, uncertainty_set;
    mip_optimizer=Gurobi.Optimizer, conic_optimizer=Mosek.Optimizer,
    outer_tr=true, inner_tr=true,
    πU_hat=πU_hat, πU_tilde=πU_tilde, yU=yU, ytsU=ytsU,
    strengthen_cuts=:none, parallel=false, mini_benders=true, max_mini_benders_iter=3,
    ldr_mode=:both)

sol = result_b[:opt_sol]
x_star = sol[:x]
λ_star = sol[:λ]
h_star = sol[:h]
ψ0_star = sol[:ψ0]
nb_obj = minimum(result_b[:past_upper_bound])

println("  NBD obj: $(round(nb_obj, digits=6))")
println("  x*:  $x_star")
println("  λ*:  $(round(λ_star, digits=6))")
println("  h*:  $(round.(h_star, digits=4))")
println("  ψ0*: $(round.(ψ0_star, digits=4))")

# ===== 2. Full Model (free) =====
println("\n" * "="^70)
println("2. FULL MODEL — FREE (no fix)")
println("="^70)

GC.gc()
model_free, vars_free = build_full_2DRNDP_model(network, S, ϕU_hat, ϕU_tilde, λU, γ, w, v, uncertainty_set;
    mip_solver=Gurobi.Optimizer, conic_solver=Mosek.Optimizer,
    πU_hat=πU_hat, πU_tilde=πU_tilde, yU=yU, ytsU=ytsU)
add_sparsity_constraints!(model_free, vars_free, network, S)
set_optimizer_attribute(model_free, "time_limit", 300.0)
optimize!(model_free)

st_free = termination_status(model_free)
obj_free = (st_free == MOI.OPTIMAL || st_free == MOI.ALMOST_OPTIMAL) ? objective_value(model_free) : NaN
println("  Status: $st_free")
println("  Obj: $(round(obj_free, digits=6))")
if !isnan(obj_free)
    println("  x:  $(round.(value.(vars_free[:x])))")
    println("  λ:  $(round(value(vars_free[:λ]), digits=6))")
end

# ===== 3. Full Model — fix x, λ =====
println("\n" * "="^70)
println("3. FULL MODEL — FIX x*, λ*")
println("="^70)

GC.gc()
model_xλ, vars_xλ = build_full_2DRNDP_model(network, S, ϕU_hat, ϕU_tilde, λU, γ, w, v, uncertainty_set;
    conic_solver=Mosek.Optimizer,  # no mip_solver needed (x fixed → no binary)
    x_fixed=x_star, λ_fixed=λ_star,
    πU_hat=πU_hat, πU_tilde=πU_tilde, yU=yU, ytsU=ytsU)
add_sparsity_constraints!(model_xλ, vars_xλ, network, S)
optimize!(model_xλ)

st_xλ = termination_status(model_xλ)
obj_xλ = (st_xλ == MOI.OPTIMAL || st_xλ == MOI.ALMOST_OPTIMAL) ? objective_value(model_xλ) : NaN
println("  Status: $st_xλ")
println("  Obj: $(round(obj_xλ, digits=6))")

# ===== 4. Full Model — fix x, λ, h, ψ0 =====
println("\n" * "="^70)
println("4. FULL MODEL — FIX x*, λ*, h*, ψ0*")
println("="^70)

GC.gc()
model_all, vars_all = build_full_2DRNDP_model(network, S, ϕU_hat, ϕU_tilde, λU, γ, w, v, uncertainty_set;
    conic_solver=Mosek.Optimizer,
    x_fixed=x_star, λ_fixed=λ_star, h_fixed=h_star, ψ0_fixed=ψ0_star,
    πU_hat=πU_hat, πU_tilde=πU_tilde, yU=yU, ytsU=ytsU)
add_sparsity_constraints!(model_all, vars_all, network, S)
optimize!(model_all)

st_all = termination_status(model_all)
obj_all = (st_all == MOI.OPTIMAL || st_all == MOI.ALMOST_OPTIMAL) ? objective_value(model_all) : NaN
println("  Status: $st_all")
println("  Obj: $(round(obj_all, digits=6))")

# ===== 5. Summary =====
println("\n" * "="^70)
println("SUMMARY")
println("="^70)
println("┌────────────────────────┬────────────┬────────────┐")
println("│ Configuration          │  Objective │   Status   │")
println("├────────────────────────┼────────────┼────────────┤")
@printf("│ Nested Benders         │ %10.4f │ %-10s │\n", nb_obj, "OPTIMAL")
@printf("│ Full Model (free)      │ %10.4f │ %-10s │\n", isnan(obj_free) ? -999.0 : obj_free, st_free)
@printf("│ Full (fix x,λ)         │ %10.4f │ %-10s │\n", isnan(obj_xλ) ? -999.0 : obj_xλ, st_xλ)
@printf("│ Full (fix x,λ,h,ψ0)   │ %10.4f │ %-10s │\n", isnan(obj_all) ? -999.0 : obj_all, st_all)
println("└────────────────────────┴────────────┴────────────┘")

println("\nGap analysis:")
!isnan(obj_free) && println("  Free  vs NBD: $(round(abs(obj_free - nb_obj), digits=4))")
!isnan(obj_xλ)   && println("  Fix(x,λ) vs NBD: $(round(abs(obj_xλ - nb_obj), digits=4))")
!isnan(obj_all)   && println("  Fix(all) vs NBD: $(round(abs(obj_all - nb_obj), digits=4))")
