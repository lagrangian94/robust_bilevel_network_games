"""
tv_verify.jl — TV-DRO 검증 스크립트.

테스트 항목:
  1. ISP 단독: 고정 (x,h,λ,ψ⁰,α)에서 ISP-L + ISP-F → OSP primal 직접 풀기와 비교
  2. Inner loop: IMP+ISP 수렴값 vs OSP 직접 풀기
  3. Full 비교: Benders Z₀ vs Full model Z₀ (3×3, S=2)
  4. ε→0: nominal 수렴 확인
"""

using JuMP
using HiGHS
using Gurobi
using Printf
using LinearAlgebra
using Random

# Load parent module
include("../network_generator.jl")
using .NetworkGenerator

# Load TV-DRO modules
include("tv_data.jl")
include("tv_build_isp_leader.jl")
include("tv_build_isp_follower.jl")
include("tv_build_imp.jl")
include("tv_build_omp.jl")
include("tv_build_full_model.jl")
include("tv_nested_benders.jl")


# ============================================================
# Helper: 네트워크 + TVData 생성
# ============================================================
function setup_instance(; m=3, n=3, S=2, seed=42,
                          eps_hat=0.1, eps_tilde=0.1,
                          gamma=2, w=nothing, lambda_U=10.0, v_val=1.0,
                          ρ=0.5)
    network = generate_grid_network(m, n; seed=seed)
    num_arcs = length(network.arcs) - 1
    num_arcs_with_dummy = length(network.arcs)

    capacities, _ = generate_capacity_scenarios_uniform_model(num_arcs_with_dummy, S; seed=seed)

    # w 자동 설정
    if w === nothing
        interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
        c_bar = sum(capacities[interdictable_idx, :]) / (length(interdictable_idx) * S)
        w = round(ρ * gamma * c_bar, digits=4)
    end

    q_hat = fill(1.0 / S, S)

    tv = make_tv_data(network, capacities, q_hat, eps_hat, eps_tilde;
                      w=w, lambda_U=lambda_U, gamma=gamma)
    return network, tv
end


# ============================================================
# Test 1: ISP 단독 테스트
# ============================================================
"""
고정 (x, h, λ, ψ⁰, α)에서:
  - ISP-L(α) + ISP-F(α) 풀기 → Z^L + Z^F
  - OSP primal (§7) 직접 풀기 → Z₀
  - 두 값 비교
"""
function test_isp_standalone(; m=3, n=3, S=2, seed=42,
                               eps_hat=0.2, eps_tilde=0.2)
    println("=" ^ 60)
    println("Test 1: ISP standalone (fixed x, h, λ, ψ⁰, α)")
    println("=" ^ 60)

    network, tv = setup_instance(; m, n, S, seed, eps_hat, eps_tilde)
    K = tv.num_arcs

    # 고정 outer solution
    x_sol = zeros(K)
    interdictable_idx = findall(tv.interdictable_arcs)
    if length(interdictable_idx) >= tv.gamma
        x_sol[interdictable_idx[1:tv.gamma]] .= 1.0
    end
    λ_sol = 1.0
    ψ0_sol = λ_sol .* x_sol
    h_sol = fill(tv.w / K, K)

    # 고정 α
    α_sol = fill(tv.w / K, K)

    # --- ISP-L ---
    isp_l_model, isp_l_vars = build_tv_isp_leader(tv, x_sol; optimizer=HiGHS.Optimizer)
    _, leader_cut = tv_isp_leader_optimize!(isp_l_model, isp_l_vars, tv, α_sol)

    # --- ISP-F ---
    isp_f_model, isp_f_vars = build_tv_isp_follower(tv, x_sol, h_sol, λ_sol, ψ0_sol;
                                                      optimizer=HiGHS.Optimizer)
    _, follower_cut = tv_isp_follower_optimize!(isp_f_model, isp_f_vars, tv, α_sol)

    Z_L = leader_cut[:obj_val]
    Z_F = follower_cut[:obj_val]
    Z_ISP = Z_L + Z_F
    @printf("  ISP-L: %.6f,  ISP-F: %.6f,  Z^L+Z^F = %.6f\n", Z_L, Z_F, Z_ISP)

    # --- OSP primal 직접 풀기 (§7): Z₀ at fixed (x,h,λ,ψ⁰) ---
    Z_OSP = solve_osp_primal(tv, x_sol, h_sol, λ_sol, ψ0_sol)
    @printf("  OSP primal (direct): %.6f\n", Z_OSP)

    # Inner loop으로 구한 값과 비교
    imp_model, imp_vars = build_tv_imp(tv; optimizer=HiGHS.Optimizer)
    isp_l2, isp_l2_vars = build_tv_isp_leader(tv, x_sol; optimizer=HiGHS.Optimizer)
    isp_f2, isp_f2_vars = build_tv_isp_follower(tv, x_sol, h_sol, λ_sol, ψ0_sol;
                                                   optimizer=HiGHS.Optimizer)

    inner_result = tv_inner_loop!(tv, imp_model, imp_vars,
                                   isp_l2, isp_l2_vars,
                                   isp_f2, isp_f2_vars;
                                   max_inner_iter=500, inner_tol=1e-6, verbose=false)
    Z_inner = inner_result[:Z0_val]
    @printf("  Inner loop: %.6f  (iters=%d)\n", Z_inner, inner_result[:inner_iters])

    gap1 = abs(Z_inner - Z_OSP) / max(abs(Z_OSP), 1e-10)
    @printf("  Inner loop vs OSP primal gap: %.2e\n", gap1)

    # ISP-L + ISP-F at optimal α from OSP primal
    Z_OSP_check = solve_osp_primal_with_alpha(tv, x_sol, h_sol, λ_sol, ψ0_sol)
    @printf("  OSP primal (α returned): Z=%.6f, α_sum=%.4f\n",
            Z_OSP_check[:obj], sum(Z_OSP_check[:α]))

    # Evaluate ISP at OSP's optimal α
    α_opt = Z_OSP_check[:α]
    isp_l3, isp_l3_vars = build_tv_isp_leader(tv, x_sol; optimizer=HiGHS.Optimizer)
    _, lc3 = tv_isp_leader_optimize!(isp_l3, isp_l3_vars, tv, α_opt)
    isp_f3, isp_f3_vars = build_tv_isp_follower(tv, x_sol, h_sol, λ_sol, ψ0_sol;
                                                   optimizer=HiGHS.Optimizer)
    _, fc3 = tv_isp_follower_optimize!(isp_f3, isp_f3_vars, tv, α_opt)
    Z_ISP_at_opt = lc3[:obj_val] + fc3[:obj_val]
    @printf("  ISP-L+F at OSP's α*: %.6f  (L=%.6f, F=%.6f)\n",
            Z_ISP_at_opt, lc3[:obj_val], fc3[:obj_val])

    gap2 = abs(Z_ISP_at_opt - Z_OSP) / max(abs(Z_OSP), 1e-10)
    @printf("  ISP@α* vs OSP gap: %.2e\n", gap2)

    pass = gap1 < 5e-3  # allow 0.5% for cutting plane
    println(pass ? "  ✓ PASS" : "  ✗ FAIL")
    return pass
end


"""
OSP primal (§7 in tv_derivation_revised.md): Z₀ at fixed (x̄, h̄, λ̄, ψ̄⁰).
McCormick linearization for ψ̂ ≈ x̄·φ̂, ψ̃ ≈ x̄·φ̃.
w·ν term included in objective.
"""
function solve_osp_primal(tv::TVData, x_sol, h_sol, λ_sol, ψ0_sol)
    S = tv.S
    K = tv.num_arcs
    m = tv.nv1
    Ny = tv.Ny
    Nts = tv.Nts
    q = tv.q_hat
    ε̂ = tv.eps_hat
    ε̃ = tv.eps_tilde
    ξ = tv.xi_bar
    v = tv.v
    w = tv.w
    φ_U = tv.phi_U

    # r_k^s = h_k + (λ - v_k ψ⁰_k) ξ̄_k^s   (from P16 RHS)
    r = zeros(K, S)
    for s in 1:S, k in 1:K
        r[k, s] = h_sol[k] + (λ_sol - v[k] * ψ0_sol[k]) * ξ[k, s]
    end

    model = Model(optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true))

    # === Variables ===
    @variable(model, ν >= 0)

    # Leader TV obj
    @variable(model, σ_Lp[1:S] >= 0)
    @variable(model, σ_Lm[1:S] >= 0)
    @variable(model, μ_L >= 0)
    @variable(model, η_L)

    # Follower TV obj
    @variable(model, σ_Fp[1:S] >= 0)
    @variable(model, σ_Fm[1:S] >= 0)
    @variable(model, μ_F >= 0)
    @variable(model, η_F)

    # Leader TV ν
    @variable(model, σ_Lνp[1:S, 1:K] >= 0)
    @variable(model, σ_Lνm[1:S, 1:K] >= 0)
    @variable(model, μ_Lν[1:K] >= 0)
    @variable(model, η_Lν[1:K])

    # Follower TV ν
    @variable(model, σ_Fνp[1:S, 1:K] >= 0)
    @variable(model, σ_Fνm[1:S, 1:K] >= 0)
    @variable(model, μ_Fν[1:K] >= 0)
    @variable(model, η_Fν[1:K])

    # Leader recourse
    @variable(model, π_hat[1:m, 1:S] >= 0)
    @variable(model, φ_hat[1:K, 1:S] >= 0)

    # Follower recourse
    @variable(model, π_tilde[1:m, 1:S] >= 0)
    @variable(model, φ_tilde[1:K, 1:S] >= 0)
    @variable(model, y_tilde[1:K, 1:S] >= 0)
    @variable(model, yts_tilde[1:S] >= 0)

    # McCormick variables: ψ̂ ≈ x̄·φ̂, ψ̃ ≈ x̄·φ̃
    @variable(model, ψ_hat_mc[1:K, 1:S] >= 0)
    @variable(model, ψ_tilde_mc[1:K, 1:S] >= 0)

    # === Constraints ===

    # McCormick for ψ̂_k^s ≈ x̄_k · φ̂_k^s (MH1-MH3)
    @constraint(model, [k=1:K, s=1:S], ψ_hat_mc[k, s] <= φ_U * x_sol[k])
    @constraint(model, [k=1:K, s=1:S], ψ_hat_mc[k, s] <= φ_hat[k, s])
    @constraint(model, [k=1:K, s=1:S], ψ_hat_mc[k, s] >= φ_hat[k, s] - φ_U * (1 - x_sol[k]))

    # McCormick for ψ̃_k^s ≈ x̄_k · φ̃_k^s (MT1-MT3)
    @constraint(model, [k=1:K, s=1:S], ψ_tilde_mc[k, s] <= φ_U * x_sol[k])
    @constraint(model, [k=1:K, s=1:S], ψ_tilde_mc[k, s] <= φ_tilde[k, s])
    @constraint(model, [k=1:K, s=1:S], ψ_tilde_mc[k, s] >= φ_tilde[k, s] - φ_U * (1 - x_sol[k]))

    # (P2): σ^{L+} - σ^{L-} + η^L ≥ Σ_k ξ̄(φ̂ - v·ψ̂_mc)
    @constraint(model, [s=1:S],
        σ_Lp[s] - σ_Lm[s] + η_L >=
        sum(ξ[k, s] * (φ_hat[k, s] - v[k] * ψ_hat_mc[k, s]) for k in 1:K))
    # (P3)
    @constraint(model, [s=1:S], μ_L - σ_Lp[s] - σ_Lm[s] >= 0)

    # (P4): σ^{F+} - σ^{F-} + η^F ≥ Σ_k ξ̄(φ̃ - v·ψ̃_mc) - ỹ_ts
    @constraint(model, [s=1:S],
        σ_Fp[s] - σ_Fm[s] + η_F >=
        sum(ξ[k, s] * (φ_tilde[k, s] - v[k] * ψ_tilde_mc[k, s]) for k in 1:K)
        - yts_tilde[s])
    # (P5)
    @constraint(model, [s=1:S], μ_F - σ_Fp[s] - σ_Fm[s] >= 0)

    # (P6): Leader dual feasibility
    @constraint(model, [k=1:K, s=1:S],
        sum(Ny[j, k] * π_hat[j, s] for j in 1:m) + φ_hat[k, s] >= 0)
    # (P7)
    @constraint(model, [s=1:S],
        sum(Nts[j] * π_hat[j, s] for j in 1:m) >= 1)

    # (P8): Follower dual feasibility
    @constraint(model, [k=1:K, s=1:S],
        sum(Ny[j, k] * π_tilde[j, s] for j in 1:m) + φ_tilde[k, s] >= 0)
    # (P9)
    @constraint(model, [s=1:S],
        sum(Nts[j] * π_tilde[j, s] for j in 1:m) >= λ_sol)

    # (P10): ν coupling
    @constraint(model, [k=1:K],
        ν - sum(q[s] * (σ_Lνp[s, k] - σ_Lνm[s, k]) for s in 1:S) - 2ε̂ * μ_Lν[k] - η_Lν[k]
          - sum(q[s] * (σ_Fνp[s, k] - σ_Fνm[s, k]) for s in 1:S) - 2ε̃ * μ_Fν[k] - η_Fν[k]
        >= 0)

    # (P11)-(P12): Leader TV-ν
    @constraint(model, [s=1:S, k=1:K],
        σ_Lνp[s, k] - σ_Lνm[s, k] + η_Lν[k] >= φ_hat[k, s])
    @constraint(model, [s=1:S, k=1:K],
        μ_Lν[k] - σ_Lνp[s, k] - σ_Lνm[s, k] >= 0)

    # (P13)-(P14): Follower TV-ν
    @constraint(model, [s=1:S, k=1:K],
        σ_Fνp[s, k] - σ_Fνm[s, k] + η_Fν[k] >= φ_tilde[k, s])
    @constraint(model, [s=1:S, k=1:K],
        μ_Fν[k] - σ_Fνp[s, k] - σ_Fνm[s, k] >= 0)

    # (P15): Follower primal feasibility
    @constraint(model, [j=1:m, s=1:S],
        -sum(Ny[j, k] * y_tilde[k, s] for k in 1:K) - Nts[j] * yts_tilde[s] >= 0)

    # (P16): ỹ_k^s ≤ r_k^s
    @constraint(model, [k=1:K, s=1:S],
        r[k, s] - y_tilde[k, s] >= 0)

    # === Objective ===
    @objective(model, Min,
        sum(q[s] * (σ_Lp[s] - σ_Lm[s]) for s in 1:S) + 2ε̂ * μ_L + η_L
        + sum(q[s] * (σ_Fp[s] - σ_Fm[s]) for s in 1:S) + 2ε̃ * μ_F + η_F
        + w * ν)

    optimize!(model)
    st = termination_status(model)
    if st != MOI.OPTIMAL
        error("OSP primal not optimal: $st")
    end
    return objective_value(model)
end


"""
OSP primal with α extraction: solve §7, return obj + dual of P10 (= α).
"""
function solve_osp_primal_with_alpha(tv::TVData, x_sol, h_sol, λ_sol, ψ0_sol)
    S = tv.S
    K = tv.num_arcs
    m = tv.nv1
    Ny = tv.Ny
    Nts = tv.Nts
    q = tv.q_hat
    ε̂ = tv.eps_hat
    ε̃ = tv.eps_tilde
    ξ = tv.xi_bar
    v = tv.v
    w = tv.w
    φ_U = tv.phi_U

    r = zeros(K, S)
    for s in 1:S, k in 1:K
        r[k, s] = h_sol[k] + (λ_sol - v[k] * ψ0_sol[k]) * ξ[k, s]
    end

    model = Model(optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true))

    @variable(model, ν >= 0)
    @variable(model, σ_Lp[1:S] >= 0)
    @variable(model, σ_Lm[1:S] >= 0)
    @variable(model, μ_L >= 0)
    @variable(model, η_L)
    @variable(model, σ_Fp[1:S] >= 0)
    @variable(model, σ_Fm[1:S] >= 0)
    @variable(model, μ_F >= 0)
    @variable(model, η_F)
    @variable(model, σ_Lνp[1:S, 1:K] >= 0)
    @variable(model, σ_Lνm[1:S, 1:K] >= 0)
    @variable(model, μ_Lν[1:K] >= 0)
    @variable(model, η_Lν[1:K])
    @variable(model, σ_Fνp[1:S, 1:K] >= 0)
    @variable(model, σ_Fνm[1:S, 1:K] >= 0)
    @variable(model, μ_Fν[1:K] >= 0)
    @variable(model, η_Fν[1:K])
    @variable(model, π_hat[1:m, 1:S] >= 0)
    @variable(model, φ_hat[1:K, 1:S] >= 0)
    @variable(model, π_tilde[1:m, 1:S] >= 0)
    @variable(model, φ_tilde[1:K, 1:S] >= 0)
    @variable(model, y_tilde[1:K, 1:S] >= 0)
    @variable(model, yts_tilde[1:S] >= 0)

    # McCormick variables
    @variable(model, ψ_hat_mc[1:K, 1:S] >= 0)
    @variable(model, ψ_tilde_mc[1:K, 1:S] >= 0)

    # McCormick constraints (MH1-MH3)
    @constraint(model, [k=1:K, s=1:S], ψ_hat_mc[k, s] <= φ_U * x_sol[k])
    @constraint(model, [k=1:K, s=1:S], ψ_hat_mc[k, s] <= φ_hat[k, s])
    @constraint(model, [k=1:K, s=1:S], ψ_hat_mc[k, s] >= φ_hat[k, s] - φ_U * (1 - x_sol[k]))

    # McCormick constraints (MT1-MT3)
    @constraint(model, [k=1:K, s=1:S], ψ_tilde_mc[k, s] <= φ_U * x_sol[k])
    @constraint(model, [k=1:K, s=1:S], ψ_tilde_mc[k, s] <= φ_tilde[k, s])
    @constraint(model, [k=1:K, s=1:S], ψ_tilde_mc[k, s] >= φ_tilde[k, s] - φ_U * (1 - x_sol[k]))

    # (P2): McCormick version
    @constraint(model, [s=1:S],
        σ_Lp[s] - σ_Lm[s] + η_L >=
        sum(ξ[k, s] * (φ_hat[k, s] - v[k] * ψ_hat_mc[k, s]) for k in 1:K))
    @constraint(model, [s=1:S], μ_L - σ_Lp[s] - σ_Lm[s] >= 0)

    # (P4): McCormick version
    @constraint(model, [s=1:S],
        σ_Fp[s] - σ_Fm[s] + η_F >=
        sum(ξ[k, s] * (φ_tilde[k, s] - v[k] * ψ_tilde_mc[k, s]) for k in 1:K)
        - yts_tilde[s])
    @constraint(model, [s=1:S], μ_F - σ_Fp[s] - σ_Fm[s] >= 0)

    @constraint(model, [k=1:K, s=1:S],
        sum(Ny[j, k] * π_hat[j, s] for j in 1:m) + φ_hat[k, s] >= 0)
    @constraint(model, [s=1:S],
        sum(Nts[j] * π_hat[j, s] for j in 1:m) >= 1)
    @constraint(model, [k=1:K, s=1:S],
        sum(Ny[j, k] * π_tilde[j, s] for j in 1:m) + φ_tilde[k, s] >= 0)
    @constraint(model, [s=1:S],
        sum(Nts[j] * π_tilde[j, s] for j in 1:m) >= λ_sol)

    # (P10): ν coupling — name it for dual extraction
    @constraint(model, P10[k=1:K],
        ν - sum(q[s] * (σ_Lνp[s, k] - σ_Lνm[s, k]) for s in 1:S) - 2ε̂ * μ_Lν[k] - η_Lν[k]
          - sum(q[s] * (σ_Fνp[s, k] - σ_Fνm[s, k]) for s in 1:S) - 2ε̃ * μ_Fν[k] - η_Fν[k]
        >= 0)

    @constraint(model, [s=1:S, k=1:K],
        σ_Lνp[s, k] - σ_Lνm[s, k] + η_Lν[k] >= φ_hat[k, s])
    @constraint(model, [s=1:S, k=1:K],
        μ_Lν[k] - σ_Lνp[s, k] - σ_Lνm[s, k] >= 0)
    @constraint(model, [s=1:S, k=1:K],
        σ_Fνp[s, k] - σ_Fνm[s, k] + η_Fν[k] >= φ_tilde[k, s])
    @constraint(model, [s=1:S, k=1:K],
        μ_Fν[k] - σ_Fνp[s, k] - σ_Fνm[s, k] >= 0)
    @constraint(model, [j=1:m, s=1:S],
        -sum(Ny[j, k] * y_tilde[k, s] for k in 1:K) - Nts[j] * yts_tilde[s] >= 0)
    @constraint(model, [k=1:K, s=1:S],
        r[k, s] - y_tilde[k, s] >= 0)

    @objective(model, Min,
        sum(q[s] * (σ_Lp[s] - σ_Lm[s]) for s in 1:S) + 2ε̂ * μ_L + η_L
        + sum(q[s] * (σ_Fp[s] - σ_Fm[s]) for s in 1:S) + 2ε̃ * μ_F + η_F
        + w * ν)

    optimize!(model)
    st = termination_status(model)
    if st != MOI.OPTIMAL
        error("OSP primal not optimal: $st")
    end

    # α = dual of P10
    α_opt = [dual(P10[k]) for k in 1:K]

    return Dict(:obj => objective_value(model), :α => α_opt, :ν => value(ν))
end


"""
OSP dual (§8 in tv_derivation_revised.md): Z₀ at fixed (x̄, h̄, λ̄, ψ̄⁰).

Includes McCormick duals ρ̂, ρ̃ for globally valid x-sensitivity.

max  Σ_s σ̂^s + λ̄ Σ_s σ̃^s
     − Σ_{s,k} [h̄_k + (λ̄ − v_k ψ̄⁰_k) ξ̄_k^s] β_k^s
     − φ^U Σ_{s,k} x̄_k (ρ̂¹ + ρ̃¹)
     − φ^U Σ_{s,k} (1−x̄_k)(ρ̂³ + ρ̃³)
"""
function solve_osp_dual(tv::TVData, x_sol, h_sol, λ_sol, ψ0_sol)
    S = tv.S
    K = tv.num_arcs
    m = tv.nv1
    Ny = tv.Ny
    Nts = tv.Nts
    q = tv.q_hat
    ε̂ = tv.eps_hat
    ε̃ = tv.eps_tilde
    ξ = tv.xi_bar
    v = tv.v
    w = tv.w
    φ_U = tv.phi_U

    r = zeros(K, S)
    for s in 1:S, k in 1:K
        r[k, s] = h_sol[k] + (λ_sol - v[k] * ψ0_sol[k]) * ξ[k, s]
    end

    model = Model(optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true))

    # === Dual variables (all ≥ 0) ===
    @variable(model, α[1:K] >= 0)
    @variable(model, û[1:K, 1:S] >= 0)
    @variable(model, σ̂[1:S] >= 0)
    @variable(model, ũ[1:K, 1:S] >= 0)
    @variable(model, σ̃[1:S] >= 0)
    @variable(model, ω[1:m, 1:S] >= 0)
    @variable(model, β[1:K, 1:S] >= 0)

    # TV duals — leader obj
    @variable(model, a[1:S] >= 0)
    @variable(model, b[1:S] >= 0)

    # TV duals — follower obj
    @variable(model, d[1:S] >= 0)
    @variable(model, e_var[1:S] >= 0)

    # TV duals — leader ν
    @variable(model, a_ν[1:S, 1:K] >= 0)
    @variable(model, b_ν[1:S, 1:K] >= 0)

    # TV duals — follower ν
    @variable(model, d_ν[1:S, 1:K] >= 0)
    @variable(model, e_ν[1:S, 1:K] >= 0)

    # McCormick duals
    @variable(model, ρ_hat_1[1:K, 1:S] >= 0)
    @variable(model, ρ_hat_2[1:K, 1:S] >= 0)
    @variable(model, ρ_hat_3[1:K, 1:S] >= 0)
    @variable(model, ρ_tilde_1[1:K, 1:S] >= 0)
    @variable(model, ρ_tilde_2[1:K, 1:S] >= 0)
    @variable(model, ρ_tilde_3[1:K, 1:S] >= 0)

    # === Constraints ===

    # (D-nu): Σ α ≤ w
    @constraint(model, sum(α[k] for k in 1:K) <= w)

    # (D-pihat): N_y û^s + N_ts σ̂^s ≤ 0,  ∀s
    @constraint(model, [j=1:m, s=1:S],
        sum(Ny[j, k] * û[k, s] for k in 1:K) + Nts[j] * σ̂[s] <= 0)

    # (D-phihat): -ξ̄_k^s a_s + û_k^s - a_{s,k}^ν + ρ̂² - ρ̂³ ≤ 0,  ∀k,s
    @constraint(model, [k=1:K, s=1:S],
        -ξ[k, s] * a[s] + û[k, s] - a_ν[s, k]
        + ρ_hat_2[k, s] - ρ_hat_3[k, s] <= 0)

    # (D-psihat): v_k ξ̄_k^s a_s - ρ̂¹ - ρ̂² + ρ̂³ ≤ 0,  ∀k,s
    @constraint(model, [k=1:K, s=1:S],
        v[k] * ξ[k, s] * a[s] - ρ_hat_1[k, s]
        - ρ_hat_2[k, s] + ρ_hat_3[k, s] <= 0)

    # (D-pitilde): N_y ũ^s + N_ts σ̃^s ≤ 0,  ∀s
    @constraint(model, [j=1:m, s=1:S],
        sum(Ny[j, k] * ũ[k, s] for k in 1:K) + Nts[j] * σ̃[s] <= 0)

    # (D-phitilde): -ξ̄_k^s d_s + ũ_k^s - d_{s,k}^ν + ρ̃² - ρ̃³ ≤ 0,  ∀k,s
    @constraint(model, [k=1:K, s=1:S],
        -ξ[k, s] * d[s] + ũ[k, s] - d_ν[s, k]
        + ρ_tilde_2[k, s] - ρ_tilde_3[k, s] <= 0)

    # (D-psitilde): v_k ξ̄_k^s d_s - ρ̃¹ - ρ̃² + ρ̃³ ≤ 0,  ∀k,s
    @constraint(model, [k=1:K, s=1:S],
        v[k] * ξ[k, s] * d[s] - ρ_tilde_1[k, s]
        - ρ_tilde_2[k, s] + ρ_tilde_3[k, s] <= 0)

    # (D-ytilde): [N_yᵀ ω^s]_k + β_k^s ≥ 0,  ∀k,s
    @constraint(model, [k=1:K, s=1:S],
        sum(Ny[j, k] * ω[j, s] for j in 1:m) + β[k, s] >= 0)

    # (D-yts): N_tsᵀ ω^s ≥ d_s,  ∀s
    @constraint(model, [s=1:S],
        sum(Nts[j] * ω[j, s] for j in 1:m) >= d[s])

    # === Leader TV duals (ε̂) ===
    @constraint(model, [s=1:S], a[s] - b[s] <= q[s])
    @constraint(model, [s=1:S], a[s] + b[s] >= q[s])
    @constraint(model, sum(b[s] for s in 1:S) <= 2ε̂)
    @constraint(model, sum(a[s] for s in 1:S) == 1)

    # === Follower TV duals (ε̃) ===
    @constraint(model, [s=1:S], d[s] - e_var[s] <= q[s])
    @constraint(model, [s=1:S], d[s] + e_var[s] >= q[s])
    @constraint(model, sum(e_var[s] for s in 1:S) <= 2ε̃)
    @constraint(model, sum(d[s] for s in 1:S) == 1)

    # === Leader TV-ν duals (ε̂), ∀k ===
    @constraint(model, [s=1:S, k=1:K], a_ν[s, k] - b_ν[s, k] <= q[s] * α[k])
    @constraint(model, [s=1:S, k=1:K], a_ν[s, k] + b_ν[s, k] >= q[s] * α[k])
    @constraint(model, [k=1:K], sum(b_ν[s, k] for s in 1:S) <= 2ε̂ * α[k])
    @constraint(model, [k=1:K], sum(a_ν[s, k] for s in 1:S) == α[k])

    # === Follower TV-ν duals (ε̃), ∀k ===
    @constraint(model, [s=1:S, k=1:K], d_ν[s, k] - e_ν[s, k] <= q[s] * α[k])
    @constraint(model, [s=1:S, k=1:K], d_ν[s, k] + e_ν[s, k] >= q[s] * α[k])
    @constraint(model, [k=1:K], sum(e_ν[s, k] for s in 1:S) <= 2ε̃ * α[k])
    @constraint(model, [k=1:K], sum(d_ν[s, k] for s in 1:S) == α[k])

    # === Objective (expanded with McCormick terms) ===
    @objective(model, Max,
        sum(σ̂[s] for s in 1:S)
        + λ_sol * sum(σ̃[s] for s in 1:S)
        - sum(r[k, s] * β[k, s] for k in 1:K, s in 1:S)
        - φ_U * sum(x_sol[k] * (ρ_hat_1[k, s] + ρ_tilde_1[k, s]) for k in 1:K, s in 1:S)
        - φ_U * sum((1 - x_sol[k]) * (ρ_hat_3[k, s] + ρ_tilde_3[k, s]) for k in 1:K, s in 1:S))

    optimize!(model)
    st = termination_status(model)
    if st != MOI.OPTIMAL
        error("OSP dual not optimal: $st")
    end
    return objective_value(model)
end


# ============================================================
# Test 2: Full model vs Benders
# ============================================================
function test_full_vs_benders(; m=3, n=3, S=2, seed=42,
                                eps_hat=0.2, eps_tilde=0.2)
    println("=" ^ 60)
    println("Test 2: Full model vs Benders ($(m)×$(n), S=$S)")
    println("=" ^ 60)

    network, tv = setup_instance(; m, n, S, seed, eps_hat, eps_tilde)
    K = tv.num_arcs

    # --- Full model ---
    println("\n  Solving full model...")
    full_model, full_vars = build_full_tv_model(tv; optimizer=Gurobi.Optimizer)
    optimize!(full_model)
    full_st = termination_status(full_model)
    if full_st != MOI.OPTIMAL
        println("  Full model not optimal: $full_st")
        return false
    end
    full_obj = objective_value(full_model)
    x_full = round.(Int, [value(full_vars[:x][k]) for k in 1:K])
    λ_full = value(full_vars[:λ])
    @printf("  Full model: obj=%.6f, λ=%.4f, x=%s\n", full_obj, λ_full, string(x_full))

    # --- Benders ---
    println("\n  Solving nested Benders...")
    result = tv_nested_benders_optimize!(tv;
        lp_optimizer=HiGHS.Optimizer,
        mip_optimizer=Gurobi.Optimizer,
        max_outer_iter=50,
        max_inner_iter=100,
        outer_tol=1e-4,
        inner_tol=1e-5,
        verbose=true)

    benders_obj = result[:Z0]
    @printf("  Benders: Z₀=%.6f, status=%s\n", benders_obj, result[:status])
    if haskey(result, :x)
        x_bd = round.(Int, result[:x])
        @printf("  Benders: λ=%.4f, x=%s\n", result[:λ], string(x_bd))
    end

    # --- Comparison ---
    gap = abs(full_obj - benders_obj) / max(abs(full_obj), 1e-10)
    @printf("\n  Full=%.6f  Benders=%.6f  gap=%.2e\n", full_obj, benders_obj, gap)
    pass = gap < 1e-3
    println(pass ? "  ✓ PASS" : "  ✗ FAIL")
    return pass
end


# ============================================================
# Test 3: ε→0 수렴 테스트
# ============================================================
function test_epsilon_convergence(; m=3, n=3, S=2, seed=42)
    println("=" ^ 60)
    println("Test 3: ε→0 convergence")
    println("=" ^ 60)

    epsilons = [0.5, 0.2, 0.1, 0.05, 0.01, 0.001]
    results = Float64[]

    for ε in epsilons
        network, tv = setup_instance(; m, n, S, seed, eps_hat=ε, eps_tilde=ε)
        full_model, full_vars = build_full_tv_model(tv; optimizer=Gurobi.Optimizer)
        optimize!(full_model)
        obj = termination_status(full_model) == MOI.OPTIMAL ? objective_value(full_model) : NaN
        push!(results, obj)
        @printf("  ε=%.4f → obj=%.6f\n", ε, obj)
    end

    # 수렴 확인: 마지막 2개 값의 차이가 작으면 pass
    if length(results) >= 2
        tail_diff = abs(results[end] - results[end-1])
        @printf("  Last two difference: %.2e\n", tail_diff)
        pass = tail_diff < 0.1 * abs(results[end])
        println(pass ? "  ✓ PASS (converging)" : "  ? CHECK (may need smaller ε)")
        return pass
    end
    return false
end


# ============================================================
# Test 4: inner cut tightness 검증
# ============================================================
"""
Inner cut tightness: 생성 시점에서 cut value = ISP obj (tight).
"""
function test_inner_cut_tightness(; m=3, n=3, S=2, seed=42,
                                    eps_hat=0.2, eps_tilde=0.2)
    println("=" ^ 60)
    println("Test 4: Inner cut tightness")
    println("=" ^ 60)

    network, tv = setup_instance(; m, n, S, seed, eps_hat, eps_tilde)
    K = tv.num_arcs

    # 고정 outer solution
    x_sol = zeros(K)
    interdictable_idx = findall(tv.interdictable_arcs)
    if length(interdictable_idx) >= tv.gamma
        x_sol[interdictable_idx[1:tv.gamma]] .= 1.0
    end
    λ_sol = 1.0
    ψ0_sol = λ_sol .* x_sol
    h_sol = fill(tv.w / K, K)

    # 여러 α 값에서 cut tightness 확인
    pass_all = true
    for trial in 1:5
        α_sol = rand(K) .* (tv.w / K)
        α_sol .*= tv.w / sum(α_sol)  # normalize to Σα = w

        # ISP-L
        isp_l, isp_l_v = build_tv_isp_leader(tv, x_sol; optimizer=HiGHS.Optimizer)
        _, l_cut = tv_isp_leader_optimize!(isp_l, isp_l_v, tv, α_sol)

        cut_val_l = l_cut[:intercept] + dot(l_cut[:subgradient], α_sol)
        tight_l = abs(cut_val_l - l_cut[:obj_val])
        @printf("  Trial %d ISP-L: obj=%.6f, cut@α=%.6f, gap=%.2e", trial, l_cut[:obj_val], cut_val_l, tight_l)
        if tight_l > 1e-5
            print(" ✗")
            pass_all = false
        else
            print(" ✓")
        end

        # ISP-F
        isp_f, isp_f_v = build_tv_isp_follower(tv, x_sol, h_sol, λ_sol, ψ0_sol;
                                                  optimizer=HiGHS.Optimizer)
        _, f_cut = tv_isp_follower_optimize!(isp_f, isp_f_v, tv, α_sol)

        cut_val_f = f_cut[:intercept] + dot(f_cut[:subgradient], α_sol)
        tight_f = abs(cut_val_f - f_cut[:obj_val])
        @printf("  ISP-F: obj=%.6f, cut@α=%.6f, gap=%.2e", f_cut[:obj_val], cut_val_f, tight_f)
        if tight_f > 1e-5
            println(" ✗")
            pass_all = false
        else
            println(" ✓")
        end
    end

    println(pass_all ? "  ✓ PASS" : "  ✗ FAIL")
    return pass_all
end


# ============================================================
# Test 5: outer cut tightness 검증
# ============================================================
"""
Outer cut tightness: 생성 시점의 (h̄,λ̄,ψ̄⁰)에서 cut value = Z₀*.
"""
function test_outer_cut_tightness(; m=3, n=3, S=2, seed=42,
                                    eps_hat=0.2, eps_tilde=0.2)
    println("=" ^ 60)
    println("Test 5: Outer cut tightness")
    println("=" ^ 60)

    network, tv = setup_instance(; m, n, S, seed, eps_hat, eps_tilde)
    K = tv.num_arcs

    # 고정 outer solution
    x_sol = zeros(K)
    interdictable_idx = findall(tv.interdictable_arcs)
    if length(interdictable_idx) >= tv.gamma
        x_sol[interdictable_idx[1:tv.gamma]] .= 1.0
    end
    λ_sol = 1.0
    ψ0_sol = λ_sol .* x_sol
    h_sol = fill(tv.w / K, K)

    # Inner loop 수렴
    imp_model, imp_vars = build_tv_imp(tv; optimizer=HiGHS.Optimizer)
    isp_l, isp_l_vars = build_tv_isp_leader(tv, x_sol; optimizer=HiGHS.Optimizer)
    isp_f, isp_f_vars = build_tv_isp_follower(tv, x_sol, h_sol, λ_sol, ψ0_sol;
                                                optimizer=HiGHS.Optimizer)

    inner_result = tv_inner_loop!(tv, imp_model, imp_vars,
                                   isp_l, isp_l_vars,
                                   isp_f, isp_f_vars;
                                   max_inner_iter=100, inner_tol=1e-6, verbose=false)

    Z0 = inner_result[:Z0_val]

    # Outer cut 계산
    outer_cut = compute_tv_outer_cut_coeffs(
        tv, inner_result[:leader_cut_info], inner_result[:follower_cut_info],
        x_sol, h_sol, λ_sol, ψ0_sol)

    # Tightness: cut(h̄, λ̄, ψ̄⁰) = Z₀*
    cut_at_bar = outer_cut[:intercept] +
        dot(outer_cut[:π_h], h_sol) +
        outer_cut[:π_λ] * λ_sol +
        dot(outer_cut[:π_ψ0], ψ0_sol) +
        dot(outer_cut[:π_x], x_sol)

    gap = abs(cut_at_bar - Z0)
    @printf("  Z₀* = %.6f\n", Z0)
    @printf("  cut(h̄,λ̄,ψ̄⁰) = %.6f\n", cut_at_bar)
    @printf("  tightness gap = %.2e\n", gap)

    pass = gap < 1e-4
    println(pass ? "  ✓ PASS" : "  ✗ FAIL")
    return pass
end


# ============================================================
# Test 6: S=3 full vs Benders
# ============================================================
function test_full_vs_benders_s3(; m=3, n=3, S=3, seed=42,
                                   eps_hat=0.15, eps_tilde=0.15)
    println("=" ^ 60)
    println("Test 6: Full vs Benders ($(m)×$(n), S=$S)")
    println("=" ^ 60)

    network, tv = setup_instance(; m, n, S, seed, eps_hat, eps_tilde)
    K = tv.num_arcs

    # --- Full model ---
    println("\n  Solving full model...")
    full_model, full_vars = build_full_tv_model(tv; optimizer=Gurobi.Optimizer)
    optimize!(full_model)
    full_st = termination_status(full_model)
    if full_st != MOI.OPTIMAL
        println("  Full model not optimal: $full_st")
        return false
    end
    full_obj = objective_value(full_model)
    x_full = round.(Int, [value(full_vars[:x][k]) for k in 1:K])
    @printf("  Full model: obj=%.6f, x=%s\n", full_obj, string(x_full))

    # --- Benders ---
    println("\n  Solving nested Benders...")
    result = tv_nested_benders_optimize!(tv;
        lp_optimizer=HiGHS.Optimizer,
        mip_optimizer=Gurobi.Optimizer,
        max_outer_iter=50,
        max_inner_iter=100,
        outer_tol=1e-4,
        inner_tol=1e-5,
        verbose=true)

    benders_obj = result[:Z0]
    @printf("  Benders: Z₀=%.6f (%d outer iters)\n", benders_obj, result[:outer_iters])

    gap = abs(full_obj - benders_obj) / max(abs(full_obj), 1e-10)
    @printf("  gap=%.2e\n", gap)
    pass = gap < 1e-3
    println(pass ? "  ✓ PASS" : "  ✗ FAIL")
    return pass
end


# ============================================================
# Run all tests
# ============================================================
function run_all_tests()
    println("╔" * "═" ^ 58 * "╗")
    println("║     TV-DRO Verification Suite                            ║")
    println("╚" * "═" ^ 58 * "╝")
    println()

    results = Dict{String, Bool}()

    results["1. ISP standalone"] = test_isp_standalone()
    println()
    results["4. Inner cut tightness"] = test_inner_cut_tightness()
    println()
    results["5. Outer cut tightness"] = test_outer_cut_tightness()
    println()
    results["2. Full vs Benders (S=2)"] = test_full_vs_benders()
    println()
    results["6. Full vs Benders (S=3)"] = test_full_vs_benders_s3()
    println()
    results["3. ε→0 convergence"] = test_epsilon_convergence()

    # Summary
    println()
    println("=" ^ 60)
    println("Summary")
    println("=" ^ 60)
    for (name, pass) in sort(collect(results))
        println("  $(pass ? "✓" : "✗")  $name")
    end
    n_pass = sum(values(results))
    n_total = length(results)
    println("\n  $n_pass / $n_total passed")
    return results
end


# ============================================================
# Entry point
# ============================================================
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_tests()
end
