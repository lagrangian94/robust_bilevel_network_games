"""
test_alpha_step_lp.jl — α-step에서 a,d fixed일 때 LP 직접 빌드하여 feasibility 확인.

Nobel US S=20 ε=0.7에서 fix(a,d) + NonConvex=2 QCP가 infeasible 판정 받는 문제 진단용.
a,d를 변수가 아닌 파라미터로 넣어 quadratic constraint 없이 순수 LP로 빌드.
"""

using Revise
using JuMP
using Gurobi
using HiGHS
using Printf
using LinearAlgebra
using Infiltrator
if !@isdefined(NetworkGenerator)
    include("../network_generator.jl")
end
using .NetworkGenerator

include("true_dro_data.jl")
include("true_dro_build_omp.jl")
include("true_dro_build_subproblem.jl")
include("true_dro_build_isp_leader.jl")
include("true_dro_build_isp_follower.jl")
include("true_dro_benders.jl")
include("true_dro_mincut_vi.jl")

# ===== Nobel US S=20 ε=0.7 setup =====
network = generate_nobel_us_network()
print_realworld_network_summary(network)
S = 20; γ_ratio = 0.10; ρ = 0.2; seed = 42; ε = 0.7

num_arcs = length(network.arcs) - 1
capacities, _ = generate_capacity_scenarios_factor_model(length(network.arcs), S;
    interdictable_arcs=network.interdictable_arcs, seed=seed)
interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
num_interdictable = length(interdictable_idx)
γ = ceil(Int, γ_ratio * num_interdictable)
c_bar = sum(capacities[interdictable_idx, :]) / (num_interdictable * S)
w = round(ρ * γ * c_bar; digits=4)
λU = 2.0
q_hat = fill(1.0 / S, S)
td = make_true_dro_data(network, capacities, q_hat, ε, ε; w=w, lambda_U=λU, gamma=γ)

K = td.num_arcs
println("K=$K, S=$S, γ=$γ, w=$w, ε=$ε")

# ===== Step 1: 첫 iter 재현 — OMP → local bilinear → ISP-L/F =====
# OMP
omp_model, omp_vars = build_true_dro_omp(td; optimizer=Gurobi.Optimizer, silent=true)
add_phase1_mincut_vi!(omp_model, omp_vars, td)
optimize!(omp_model)
x_core = round.([value(omp_vars[:x][k]) for k in 1:K])
println("OMP x = $(round.(Int, x_core))")

# Local bilinear solve
sub_model_local, sub_vars_local = build_true_dro_subproblem(td, x_core;
    optimizer=Gurobi.Optimizer, silent=true, rho_upper_bound=10.0)
set_optimizer_attribute(sub_model_local, "NonConvex", 2)
set_optimizer_attribute(sub_model_local, "OptimalityTarget", 1)
set_optimizer_attribute(sub_model_local, "MIPGap", 1e-4)
set_time_limit_sec(sub_model_local, 30.0)
optimize!(sub_model_local)
println("Local bilinear: $(termination_status(sub_model_local)), Z₀=$(objective_value(sub_model_local))")

α_local = max.([value(sub_vars_local[:α][k]) for k in 1:K], 0.0)

# ISP-L with fixed α → get a*
isp_l_model, isp_l_vars = build_true_dro_isp_leader(td, x_core, α_local; optimizer=Gurobi.Optimizer)
optimize!(isp_l_model)
a_star = [value(isp_l_vars[:a][s]) for s in 1:S]
println("ISP-L: $(termination_status(isp_l_model)), obj=$(objective_value(isp_l_model))")
@printf("  a* = %s\n", string(round.(a_star, digits=6)))
@printf("  Σa = %.10f, TV = %.10f, 2ε̂ = %.1f\n",
    sum(a_star), sum(abs(a_star[s] - q_hat[s]) for s in 1:S), 2*ε)

# ISP-F with fixed α → get d*
isp_f_model, isp_f_vars = build_true_dro_isp_follower(td, x_core, α_local; optimizer=Gurobi.Optimizer)
optimize!(isp_f_model)
d_star = [value(isp_f_vars[:d][s]) for s in 1:S]
println("ISP-F: $(termination_status(isp_f_model)), obj=$(objective_value(isp_f_model))")
@printf("  d* = %s\n", string(round.(d_star, digits=6)))
@printf("  Σd = %.10f, TV = %.10f, 2ε̃ = %.1f\n",
    sum(d_star), sum(abs(d_star[s] - q_hat[s]) for s in 1:S), 2*ε)

# Clamp
a_min = [max(0.0, q_hat[s] - 2ε) for s in 1:S]
d_min = [max(0.0, q_hat[s] - 2ε) for s in 1:S]
a_fixed = [max(a_star[s], a_min[s]) for s in 1:S]
d_fixed = [max(d_star[s], d_min[s]) for s in 1:S]

# ===== 실제 infiltrate에서 캡처한 값 (mini-benders 2회 후) =====
# 이 값으로도 테스트. 위에서 계산한 값과 비교용.
a_captured = [0.050000000000000266, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.05, 0.05, 0.05,
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7499999999999996, 0.0, 0.0]
d_captured = [0.0, 0.39999998331479, 0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.05, 0.0,
              0.050000016685209925, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4000000166852096, 0.0, 0.0]
# Clamp captured values too
a_captured = [max(a_captured[s], a_min[s]) for s in 1:S]
d_captured = [max(d_captured[s], d_min[s]) for s in 1:S]

USE_CAPTURED = true  # true: infiltrate 값 사용, false: ISP-L/F 재계산 값 사용
if USE_CAPTURED
    a_fixed = a_captured
    d_fixed = d_captured
    println("\n*** Using captured a,d from infiltrate ***")
end

# OMP re-solve for x_alt (α-step과 동일)
optimize!(omp_model)
x_alt = round.([value(omp_vars[:x][k]) for k in 1:K])
println("\nx_alt = $(round.(Int, x_alt))")

# ===== Step 2: α-step LP 직접 빌드 (a,d를 파라미터로) =====
println("\n" * "=" ^ 70)
println("Building α-step LP with a,d as parameters (no quadratic constraints)")
println("=" ^ 70)

function build_alpha_step_lp(td, x_bar, a_fixed, d_fixed; optimizer)
    S = td.S; K = td.num_arcs; m = td.nv1
    Ny = td.Ny; Nts = td.Nts; q = td.q_hat
    ε̂ = td.eps_hat; ε̃ = td.eps_tilde
    ξ = td.xi_bar; v = td.v; w = td.w
    φ̂U = td.phi_hat_U; φ̃U = td.phi_tilde_U; λU = td.lambda_U

    a_max = [min(1.0, q[s] + 2ε̂) for s in 1:S]
    d_max = [min(1.0, q[s] + 2ε̃) for s in 1:S]

    model = Model(optimizer)
    set_silent(model)

    # α
    @variable(model, 0 <= α[1:K] <= w)
    @constraint(model, sum(α) <= w)

    # ζL[k,s] = α[k] * a_fixed[s]  →  LINEAR (a is parameter)
    @variable(model, 0 <= ζL[k=1:K, s=1:S] <= w * a_max[s])
    @constraint(model, ζL_def[k=1:K, s=1:S], ζL[k, s] == α[k] * a_fixed[s])

    # ISP-L variables
    @variable(model, σ_hat[1:S] >= 0)
    @variable(model, u_hat[1:K, 1:S] >= 0)
    @variable(model, 0 <= b[1:S] <= 2ε̂)
    @variable(model, ρ_hat_1[1:K, 1:S] >= 0)
    @variable(model, ρ_hat_2[1:K, 1:S] >= 0)
    @variable(model, ρ_hat_3[1:K, 1:S] >= 0)

    # DL-1
    @constraint(model, [j=1:m, s=1:S],
        sum(Ny[j,k] * u_hat[k,s] for k in 1:K) + Nts[j] * σ_hat[s] == 0)
    # DL-2 (a_fixed is parameter)
    @constraint(model, [k=1:K, s=1:S],
        -ξ[k,s] * a_fixed[s] - ζL[k,s] + u_hat[k,s] + ρ_hat_2[k,s] - ρ_hat_3[k,s] <= 0)
    # DL-3
    @constraint(model, [k=1:K, s=1:S],
        v[k] * ξ[k,s] * a_fixed[s] - ρ_hat_1[k,s] - ρ_hat_2[k,s] + ρ_hat_3[k,s] <= 0)
    # DL-4~7 (b만 variable, a_fixed는 constant → RHS)
    @constraint(model, [s=1:S], -b[s] <= q[s] - a_fixed[s])
    @constraint(model, [s=1:S],  b[s] >= q[s] - a_fixed[s])
    @constraint(model, sum(b) <= 2ε̂)

    # ζF[k,s] = α[k] * d_fixed[s]  →  LINEAR
    @variable(model, 0 <= ζF[k=1:K, s=1:S] <= w * d_max[s])
    @constraint(model, ζF_def[k=1:K, s=1:S], ζF[k, s] == α[k] * d_fixed[s])

    # ISP-F variables
    @variable(model, u_tilde[1:K, 1:S] >= 0)
    @variable(model, σ_tilde[1:S] >= 0)
    @variable(model, ω[1:m, 1:S])
    @variable(model, β[1:K, 1:S] >= 0)
    @variable(model, δ >= 0)
    @variable(model, 0 <= e[1:S] <= 2ε̃)
    @variable(model, ρ_tilde_1[1:K, 1:S] >= 0)
    @variable(model, ρ_tilde_2[1:K, 1:S] >= 0)
    @variable(model, ρ_tilde_3[1:K, 1:S] >= 0)
    @variable(model, ρ_psi0_1[1:K] >= 0)
    @variable(model, ρ_psi0_2[1:K] >= 0)
    @variable(model, ρ_psi0_3[1:K] >= 0)

    # DF-1~2 (d_fixed constant in e constraints)
    @constraint(model, [s=1:S], -e[s] <= q[s] - d_fixed[s])
    @constraint(model, [s=1:S],  e[s] >= q[s] - d_fixed[s])
    @constraint(model, sum(e) <= 2ε̃)

    # DF-5
    @constraint(model, [j=1:m, s=1:S],
        sum(Ny[j,k] * u_tilde[k,s] for k in 1:K) + Nts[j] * σ_tilde[s] == 0)
    # DF-6 (d_fixed parameter)
    @constraint(model, [k=1:K, s=1:S],
        -ξ[k,s] * d_fixed[s] - ζF[k,s] + u_tilde[k,s] + ρ_tilde_2[k,s] - ρ_tilde_3[k,s] <= 0)
    # DF-7
    @constraint(model, [k=1:K, s=1:S],
        v[k] * ξ[k,s] * d_fixed[s] - ρ_tilde_1[k,s] - ρ_tilde_2[k,s] + ρ_tilde_3[k,s] <= 0)
    # DF-8
    @constraint(model, [k=1:K, s=1:S],
        sum(Ny[j,k] * ω[j,s] for j in 1:m) - β[k,s] <= 0)
    # DF-9 (d_fixed constant)
    @constraint(model, [s=1:S],
        d_fixed[s] + sum(Nts[j] * ω[j,s] for j in 1:m) <= 0)
    # DF-h
    @constraint(model, [k=1:K], sum(β[k,s] for s in 1:S) <= δ)
    # DF-λ
    @constraint(model,
        sum(σ_tilde) >= sum(ξ[k,s] * β[k,s] for k in 1:K, s in 1:S)
            + w * δ + sum(ρ_psi0_2) - sum(ρ_psi0_3))
    # DF-ψ
    @constraint(model, [k=1:K],
        v[k] * sum(ξ[k,s] * β[k,s] for s in 1:S) + ρ_psi0_1[k] + ρ_psi0_2[k] >= ρ_psi0_3[k])

    # Objective
    obj_L = sum(σ_hat) -
            φ̂U * sum(x_bar[k] * ρ_hat_1[k,s] for k in 1:K, s in 1:S) -
            φ̂U * sum((1-x_bar[k]) * ρ_hat_3[k,s] for k in 1:K, s in 1:S)
    obj_F = -φ̃U * sum(x_bar[k] * ρ_tilde_1[k,s] for k in 1:K, s in 1:S) -
             φ̃U * sum((1-x_bar[k]) * ρ_tilde_3[k,s] for k in 1:K, s in 1:S) -
             λU * sum(x_bar[k] * ρ_psi0_1[k] for k in 1:K) -
             λU * sum((1-x_bar[k]) * ρ_psi0_3[k] for k in 1:K)
    @objective(model, Max, obj_L + obj_F)

    return model
end

# ===== Gurobi LP =====
lp_grb = build_alpha_step_lp(td, x_alt, a_fixed, d_fixed; optimizer=Gurobi.Optimizer)
set_optimizer_attribute(lp_grb, "Method", 2)  # barrier
optimize!(lp_grb)
st_grb = termination_status(lp_grb)
println("\nGurobi LP: $st_grb")
if st_grb == MOI.OPTIMAL
    @printf("  Z₀ = %.6f\n", objective_value(lp_grb))
elseif st_grb == MOI.INFEASIBLE
    println("  → LP도 infeasible! 모델 구조 문제.")
    compute_conflict!(lp_grb)
    n_conflict = 0
    for (F, S_) in list_of_constraint_types(lp_grb)
        for con in all_constraints(lp_grb, F, S_)
            try
                if MOI.get(lp_grb, MOI.ConstraintConflictStatus(), con.index) == MOI.IN_CONFLICT
                    n_conflict += 1
                    if n_conflict <= 20
                        println("  CONFLICT: $con")
                    end
                end
            catch; end
        end
    end
    println("  Total conflicts: $n_conflict")
end

# ===== HiGHS LP (cross-check) =====
lp_highs = build_alpha_step_lp(td, x_alt, a_fixed, d_fixed; optimizer=HiGHS.Optimizer)
optimize!(lp_highs)
st_highs = termination_status(lp_highs)
println("\nHiGHS LP: $st_highs")
if st_highs == MOI.OPTIMAL
    @printf("  Z₀ = %.6f\n", objective_value(lp_highs))
elseif st_highs == MOI.DUAL_INFEASIBLE
    println("  → Unbounded!")
end

println("\nDone.")
