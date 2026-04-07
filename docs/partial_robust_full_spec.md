# Partial Robust Nested Benders: Math + Implementation

## 0. 문제 설정

현재 코드의 2DRNDP (manuscript eq.16)는 leader(hat)와 follower(tilde) 양쪽 모두에 uncertainty set Ξₛ = {ζ: ‖ζ‖₂ ≤ ε}를 적용한다. 이로 인해 양쪽 모두 S-lemma(SDP) + robust counterpart(SOC) 구조가 생긴다.

이 문서는 **한쪽의 ε만 0으로 놓는 두 가지 special case**를 다룬다:
- **Case 1 (ε̂=0):** true distribution에 대한 robustness 없음 → leader ISP가 SDP→LP
- **Case 2 (ε̃=0):** follower distribution에 대한 robustness 없음 → follower ISP가 SDP→LP

ε=0이면 Ξₛ={0}이므로, 모든 "∀ζ∈Ξ" 제약이 ζ=0에서의 point evaluation으로 축소된다. LDR slope 변수(Φ̂_L, Π̂_L 등)는 의미가 없어지고, intercept(ϕ̂₀, π̂₀ 등)만 남는다.

**기존 코드 참조:**
- `build_primal_isp.jl`: primal ISP — LP ISP의 원형 (SDP/SOC 제거하면 LP ISP)
- `nested_benders_trust_region.jl`: 전체 알고리즘, `build_omp()`, `build_imp()`, `evaluate_master_opt_cut()`
- `network_generator.jl`, `build_uncertainty_set.jl`

---

# Part A: Mathematical Formulation

## A.1 원래 Full Model에서 어떤 제약이 변하는가

Manuscript (16)의 제약식 중 hat/tilde 각각이 갖는 구성:

| 제약 | Hat (leader) | Tilde (follower) |
|------|-------------|-----------------|
| S-lemma SDP | (16e): M̂ ≽ 0, ϑ̂ε² | (16f): M̃ ≽ 0, ϑ̃ε² |
| SOC robust counterpart | (16j): Λ̂₁R=..., (16k): Λ̂₂R=... | (16l): Λ̃₁R=..., (16m): Λ̃₂R=... |
| McCormick | (16g): full Ψ̂ matrix | (16h): full Ψ̃ matrix |
| LDR bounds | P̂₁, P̂₂ for slopes | P̃₁, P̃₂ for slopes |

ε=0으로 놓으면 해당 side의 위 4가지가 전부 linear constraint로 축소된다.

## A.2 Case 1: ε̂=0 — Leader side 축소

### Primal outer subproblem의 hat 제약 (ζ=0 evaluation):

**(16e) → linear epigraph:**
$$(\hat\phi_0^s - v \odot \hat\psi_0^s)^\top \bar\xi^s \le \hat\eta^s$$

**(16j) → point evaluation:**
$$N_y^\top \hat\pi_0^s + \hat\phi_0^s \ge 0, \quad N_{ts}^\top \hat\pi_0^s \ge 1, \quad \hat\pi_0^s \ge 0, \quad \hat\phi_0^s \ge 0$$

**(16k) → point evaluation:**
$$\hat\mu_k^s \ge \hat\phi_{0,k}^s \quad \forall k$$

**(16g) → intercept McCormick only:**
$$\hat\psi_{0,k}^s \le \phi^U x_k, \quad \hat\psi_{0,k}^s \le \hat\phi_{0,k}^s, \quad \hat\phi_{0,k}^s \le \hat\psi_{0,k}^s + \phi^U(1-x_k)$$

**삭제되는 변수:** M̂ˢ(SDP), ϑ̂ˢ, Λ̂₁ˢ, Λ̂₂ˢ(SOC), Γ̂₁ˢ, Γ̂₂ˢ, Π̂_Lˢ, Φ̂_Lˢ, Ψ̂_Lˢ(slopes), P̂(LDR bounds)
**생존 변수:** η̂ˢ, μ̂ˢ, π̂₀ˢ, ϕ̂₀ˢ, ψ̂₀ˢ (intercepts only)

Tilde side (16f,16h,16l,16m): **변경 없음** (SDP + SOC 유지).

### Leader ISP가 LP가 되는 이유

Nested Benders의 ISP는 outer subproblem을 scenario별로 decompose한 것. Hat side의 outer subproblem이 전부 linear이면, 그 subproblem (= leader ISP)도 LP.

기존 primal ISP (`build_primal_isp_leader`)에서 Mhat(SDP), ϑhat, Λhat1/2(SOC) 관련 제약만 제거하면 곧 LP leader ISP.

### Leader LP ISP (primal formulation)

기존 `build_primal_isp_leader`의 목적함수 구조를 그대로 따름:

$$Z_1^{L,s}(\alpha) = \min \quad \frac{1}{S}\hat\eta^s + \sum_k \alpha_k \hat\mu_k^s$$

subject to:
$$(\hat\phi_0^s - v \odot \hat\psi_0^s)^\top \bar\xi^s \le \hat\eta^s \tag{epigraph}$$
$$N_y^\top \hat\pi_0^s + \hat\phi_0^s \ge 0 \tag{flow dual}$$
$$N_{ts}^\top \hat\pi_0^s \ge 1 \tag{ts dual}$$
$$\hat\mu_k^s \ge \hat\phi_{0,k}^s \quad \forall k \tag{μ coupling}$$
$$\hat\psi_{0,k}^s \le \phi^U x_k, \; \hat\psi_{0,k}^s \le \hat\phi_{0,k}^s, \; \hat\phi_{0,k}^s \le \hat\psi_{0,k}^s + \phi^U(1-x_k) \tag{McCormick}$$
$$\hat\pi_0^s \ge 0, \; \hat\phi_0^s \ge 0, \; \hat\eta^s \ge 0 \tag{signs}$$

**파라미터:** α (IMP에서, inner iter마다 변경), x (OMP에서, outer iter마다 변경)

**Cut 추출:** optimal에서 value(μ̂ₖ) = subgradient w.r.t. αₖ, value(η̂) → intercept.

## A.3 Case 2: ε̃=0 — Follower side 축소

### Primal outer subproblem의 tilde 제약 (ζ=0 evaluation):

**(16f) → linear epigraph:**
$$(\tilde\phi_0^s - v \odot \tilde\psi_0^s)^\top \bar\xi^s - \tilde y_0^{ts,s} \le \tilde\eta^s$$

**(16l) → point evaluation:**
$$N_y^\top \tilde\pi_0^s + \tilde\phi_0^s \ge 0$$
$$N_{ts}^\top \tilde\pi_0^s \ge \lambda$$
$$-N_y \tilde y_0^s - N_{ts} \tilde y_0^{ts,s} \ge 0$$
$$\tilde y_{0,k}^s \le h_k + (\lambda - v_k \psi_k^0) \bar\xi_k^s \quad \forall k$$

**(16m) → point evaluation:**
$$\tilde\mu_k^s \ge \tilde\phi_{0,k}^s \quad \forall k$$

**(16h) → intercept McCormick only** (same structure as hat)

**삭제되는 변수:** M̃ˢ, ϑ̃ˢ, Λ̃₁ˢ, Λ̃₂ˢ, Γ̃₁ˢ, Γ̃₂ˢ, Π̃_Lˢ, Φ̃_Lˢ, Ψ̃_Lˢ, Ỹ_Lˢ, Ỹ_L^{ts,s}(slopes), P̃
**생존 변수:** η̃ˢ(free!), μ̃ˢ, π̃₀ˢ, ϕ̃₀ˢ, ψ̃₀ˢ, ỹ₀ˢ, ỹ₀^{ts,s} (intercepts only)

Hat side (16e,16g,16j,16k): **변경 없음**.

### Follower LP ISP (primal formulation)

$$Z_1^{F,s}(\alpha) = \min \quad \frac{1}{S}\tilde\eta^s + \sum_k \alpha_k \tilde\mu_k^s$$

subject to:
$$(\tilde\phi_0^s - v \odot \tilde\psi_0^s)^\top \bar\xi^s - \tilde y_0^{ts,s} \le \tilde\eta^s \tag{epigraph, η̃ free}$$
$$N_y^\top \tilde\pi_0^s + \tilde\phi_0^s \ge 0 \tag{flow dual}$$
$$N_{ts}^\top \tilde\pi_0^s \ge \lambda \tag{ts dual, RHS=λ}$$
$$-N_y \tilde y_0^s - N_{ts} \tilde y_0^{ts,s} \ge 0 \tag{flow conservation}$$
$$\tilde y_{0,k}^s \le h_k + (\lambda - v_k \psi_k^0) \bar\xi_k^s \quad \forall k \tag{capacity, RHS depends on h,λ,ψ⁰}$$
$$\tilde\mu_k^s \ge \tilde\phi_{0,k}^s \quad \forall k \tag{μ coupling}$$
$$\text{McCormick on } (\tilde\psi_0, \tilde\phi_0, x) \tag{intercept only}$$
$$\tilde\pi_0^s \ge 0, \; \tilde\phi_0^s \ge 0, \; \tilde y_0^s \ge 0, \; \tilde y_0^{ts,s} \ge 0 \tag{signs}$$

**주의:** η̃ˢ는 **free** (≥0 아님). Manuscript (27)에서 M̃₂₂ = 1/S가 등호인 이유.

**파라미터:** α (inner), x, h, λ, ψ⁰ (outer). Capacity constraint의 RHS에 h, λ, ψ⁰이 들어감.

**ψ⁰ 관련:** capacity에 `(λ - vₖψ⁰ₖ)ξ̄ₖ`가 나오는 이유 — manuscript (16l)의 robust counterpart를 ζ=0에서 평가하면 `diag(λ - v⊙ψ⁰)ξ̄ˢ`가 RHS에 남음. ψ⁰은 OMP의 McCormick 변수 (λ·x의 linearization).

## A.4 Decomposition 구조 요약

| | Full DRO | Case 1 (ε̂=0) | Case 2 (ε̃=0) |
|---|---|---|---|
| ISP-Leader | SDP (Mosek) | **LP** | SDP (Mosek) |
| ISP-Follower | SDP (Mosek) | SDP (Mosek) | **LP** |
| IMP | LP (unchanged) | LP (unchanged) | LP (unchanged) |
| OMP | MIP (unchanged) | MIP (unchanged) | MIP (unchanged) |
| Inner cut format | intercept + α'μ | **same** | **same** |
| Outer cut | from SDP duals | LP duals + SDP duals | SDP duals + LP duals |

---

# Part B: Implementation

## B.1 LP ISP 코드 — Leader (ε̂=0)

`build_primal_isp_leader()`에서 SDP(Mhat, ϑhat), SOC(Λhat1/2), slope 변수 제거.

```julia
function build_lp_isp_leader(network, xi_bar_s, ϕU, v_param, true_S, x_sol; optimizer)
    num_arcs = length(xi_bar_s)
    num_nodes = length(network.nodes)
    N_trunc = network.N[2:end, :]        # source row 제거
    Ny = N_trunc[:, 1:num_arcs]
    Nts = N_trunc[:, end]
    nv1 = size(Ny, 1)                    # = num_nodes - 1

    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => true))

    # --- 변수: 기존 primal ISP에서 intercept만 생존 ---
    @variable(model, ηhat >= 0)
    @variable(model, μhat[1:num_arcs] >= 0)
    @variable(model, Πhat_0[1:nv1] >= 0)              # π̂₀
    @variable(model, Φhat_0[1:num_arcs] >= 0)          # ϕ̂₀
    @variable(model, Ψhat_0[1:num_arcs] >= 0)          # ψ̂₀

    # --- (16e at ζ=0): epigraph ---
    @constraint(model,
        sum((Φhat_0[k] - v_param * Ψhat_0[k]) * xi_bar_s[k] for k in 1:num_arcs) <= ηhat)

    # --- (16j at ζ=0): flow dual feasibility ---
    @constraint(model, [k=1:num_arcs],
        sum(Ny[j,k] * Πhat_0[j] for j in 1:nv1) + Φhat_0[k] >= 0)
    @constraint(model,
        sum(Nts[j] * Πhat_0[j] for j in 1:nv1) >= 1.0)

    # --- (16k at ζ=0): μ coupling ---
    @constraint(model, [k=1:num_arcs], μhat[k] >= Φhat_0[k])

    # --- (16g intercept): McCormick ---
    @constraint(model, con_bigM1[k=1:num_arcs], Ψhat_0[k] <= ϕU * x_sol[k])
    @constraint(model, [k=1:num_arcs], Ψhat_0[k] <= Φhat_0[k])
    @constraint(model, con_bigM3[k=1:num_arcs], Φhat_0[k] <= Ψhat_0[k] + ϕU * (1 - x_sol[k]))

    # --- 목적함수: α는 μhat 계수로 들어감 (기존 primal ISP 패턴) ---
    @objective(model, Min, (1/true_S) * ηhat)
    # α는 inner iter마다 set_objective_coefficient(model, μhat[k], α_k)

    vars = Dict(:ηhat => ηhat, :μhat => μhat, :Πhat_0 => Πhat_0,
                :Φhat_0 => Φhat_0, :Ψhat_0 => Ψhat_0,
                :con_bigM1 => con_bigM1, :con_bigM3 => con_bigM3)
    return model, vars
end
```

```julia
function lp_isp_leader_optimize!(model, vars; isp_data, α_sol)
    true_S = isp_data[:S]
    num_arcs = length(α_sol)

    for k in 1:num_arcs
        set_objective_coefficient(model, vars[:μhat][k], α_sol[k])
    end
    optimize!(model)

    μhat_val = [value(vars[:μhat][k]) for k in 1:num_arcs]
    ηhat_val = value(vars[:ηhat])
    intercept = (1/true_S) * ηhat_val
    obj_val = intercept + dot(α_sol, μhat_val)

    @assert abs(obj_val - objective_value(model)) < 1e-4 "LP leader duality gap: $obj_val vs $(objective_value(model))"

    return (:OptimalityCut, Dict(:μhat => μhat_val, :intercept => intercept, :obj_val => obj_val))
end
```

```julia
function update_lp_leader_params!(model, vars; x_sol, ϕU)
    for k in 1:length(x_sol)
        set_normalized_rhs(vars[:con_bigM1][k], ϕU * x_sol[k])
        set_normalized_rhs(vars[:con_bigM3][k], ϕU * (1 - x_sol[k]))
    end
end
```

## B.2 LP ISP 코드 — Follower (ε̃=0)

`build_primal_isp_follower()`에서 SDP/SOC/slope 제거. Flow 변수 ỹ₀, ỹ₀ᵗˢ 생존.

```julia
function build_lp_isp_follower(network, xi_bar_s, ϕU, v_param, true_S,
                                x_sol, h_sol, λ_sol, ψ0_sol; optimizer)
    num_arcs = length(xi_bar_s)
    num_nodes = length(network.nodes)
    N_trunc = network.N[2:end, :]
    Ny = N_trunc[:, 1:num_arcs]
    Nts = N_trunc[:, end]
    nv1 = size(Ny, 1)

    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => true))

    # --- 변수 ---
    @variable(model, ηtilde)                              # FREE (not ≥ 0)
    @variable(model, μtilde[1:num_arcs] >= 0)
    @variable(model, Πtilde_0[1:nv1] >= 0)
    @variable(model, Φtilde_0[1:num_arcs] >= 0)
    @variable(model, Ψtilde_0[1:num_arcs] >= 0)
    @variable(model, Ytilde_0[1:num_arcs] >= 0)           # ỹ₀ (flow intercept)
    @variable(model, Ytilde_0_ts >= 0)                    # ỹ₀ᵗˢ (total flow intercept)

    # --- (16f at ζ=0): epigraph (η̃ free) ---
    @constraint(model,
        sum((Φtilde_0[k] - v_param * Ψtilde_0[k]) * xi_bar_s[k] for k in 1:num_arcs)
        - Ytilde_0_ts <= ηtilde)

    # --- (16l at ζ=0): flow dual feasibility ---
    @constraint(model, [k=1:num_arcs],
        sum(Ny[j,k] * Πtilde_0[j] for j in 1:nv1) + Φtilde_0[k] >= 0)

    # ts dual: Nₜₛᵀπ̃₀ ≥ λ
    @constraint(model, con_ts_dual,
        sum(Nts[j] * Πtilde_0[j] for j in 1:nv1) >= λ_sol)

    # flow conservation: -Ny ỹ₀ - Nₜₛ ỹ₀ᵗˢ ≥ 0
    @constraint(model, [j=1:nv1],
        -sum(Ny[j,k] * Ytilde_0[k] for k in 1:num_arcs) - Nts[j] * Ytilde_0_ts >= 0)

    # capacity at ζ=0: ỹ₀ₖ ≤ hₖ + (λ - vₖψ⁰ₖ)ξ̄ₖ
    @constraint(model, con_capacity[k=1:num_arcs],
        Ytilde_0[k] <= h_sol[k] + (λ_sol - v_param * ψ0_sol[k]) * xi_bar_s[k])

    # --- (16m at ζ=0): μ coupling ---
    @constraint(model, [k=1:num_arcs], μtilde[k] >= Φtilde_0[k])

    # --- (16h intercept): McCormick ---
    @constraint(model, con_bigM1[k=1:num_arcs], Ψtilde_0[k] <= ϕU * x_sol[k])
    @constraint(model, [k=1:num_arcs], Ψtilde_0[k] <= Φtilde_0[k])
    @constraint(model, con_bigM3[k=1:num_arcs], Φtilde_0[k] <= Ψtilde_0[k] + ϕU * (1 - x_sol[k]))

    # --- 목적함수 ---
    @objective(model, Min, (1/true_S) * ηtilde)

    vars = Dict(:ηtilde => ηtilde, :μtilde => μtilde,
                :Πtilde_0 => Πtilde_0, :Φtilde_0 => Φtilde_0, :Ψtilde_0 => Ψtilde_0,
                :Ytilde_0 => Ytilde_0, :Ytilde_0_ts => Ytilde_0_ts,
                :con_bigM1 => con_bigM1, :con_bigM3 => con_bigM3,
                :con_ts_dual => con_ts_dual, :con_capacity => con_capacity)
    return model, vars
end
```

```julia
function lp_isp_follower_optimize!(model, vars; isp_data, α_sol)
    true_S = isp_data[:S]
    num_arcs = length(α_sol)

    for k in 1:num_arcs
        set_objective_coefficient(model, vars[:μtilde][k], α_sol[k])
    end
    optimize!(model)

    μtilde_val = [value(vars[:μtilde][k]) for k in 1:num_arcs]
    ηtilde_val = value(vars[:ηtilde])
    intercept = (1/true_S) * ηtilde_val
    obj_val = intercept + dot(α_sol, μtilde_val)

    @assert abs(obj_val - objective_value(model)) < 1e-4 "LP follower duality gap"

    return (:OptimalityCut, Dict(:μtilde => μtilde_val, :intercept => intercept, :obj_val => obj_val))
end
```

```julia
function update_lp_follower_params!(model, vars; x_sol, h_sol, λ_sol, ψ0_sol, xi_bar_s, v_param, ϕU)
    num_arcs = length(x_sol)
    for k in 1:num_arcs
        set_normalized_rhs(vars[:con_bigM1][k], ϕU * x_sol[k])
        set_normalized_rhs(vars[:con_bigM3][k], ϕU * (1 - x_sol[k]))
        set_normalized_rhs(vars[:con_capacity][k],
            h_sol[k] + (λ_sol - v_param * ψ0_sol[k]) * xi_bar_s[k])
    end
    set_normalized_rhs(vars[:con_ts_dual], λ_sol)
end
```

## B.3 Outer Cut 생성

Inner loop 수렴 후 OMP에 outer Benders cut 추가.

**LP side:** primal 변수를 직접 갖고 있으므로, 제약식 RHS에 대한 shadow_price로 OMP 파라미터 sensitivity 계산.

```julia
function evaluate_lp_leader_outer_cut(model, vars; x_sol, ϕU)
    # x가 McCormick RHS에만 등장 → shadow_price(con_bigM1/3)로 sensitivity
    num_arcs = length(x_sol)
    bigM1_duals = [shadow_price(vars[:con_bigM1][k]) for k in 1:num_arcs]
    bigM3_duals = [shadow_price(vars[:con_bigM3][k]) for k in 1:num_arcs]

    coeff_x = [ϕU * bigM1_duals[k] - ϕU * bigM3_duals[k] for k in 1:num_arcs]
    coeff_h, coeff_λ, coeff_ψ0 = zeros(num_arcs), 0.0, zeros(num_arcs)
    intercept = objective_value(model) - dot(coeff_x, x_sol)

    return Dict(:intercept => intercept, :coeff_x => coeff_x,
                :coeff_h => coeff_h, :coeff_λ => coeff_λ, :coeff_ψ0 => coeff_ψ0)
end

function evaluate_lp_follower_outer_cut(model, vars; x_sol, h_sol, λ_sol, ψ0_sol, xi_bar_s, v_param, ϕU)
    num_arcs = length(x_sol)
    cap_duals  = [shadow_price(vars[:con_capacity][k]) for k in 1:num_arcs]
    bigM1_duals = [shadow_price(vars[:con_bigM1][k]) for k in 1:num_arcs]
    bigM3_duals = [shadow_price(vars[:con_bigM3][k]) for k in 1:num_arcs]
    ts_dual     = shadow_price(vars[:con_ts_dual])

    # ∂obj/∂xₖ: McCormick RHS sensitivity만 (capacity RHS에 x 없음)
    coeff_x = [ϕU * bigM1_duals[k] - ϕU * bigM3_duals[k] for k in 1:num_arcs]
    # ∂obj/∂hₖ: capacity RHS에 hₖ가 coefficient 1로 등장
    coeff_h = cap_duals
    # ∂obj/∂λ: ts_dual RHS=λ, capacity RHS에 ξ̄ₖ 만큼 등장
    coeff_λ = ts_dual + dot(xi_bar_s, cap_duals)
    # ∂obj/∂ψ⁰ₖ: capacity RHS에 -vₖξ̄ₖ 만큼 등장
    coeff_ψ0 = [-v_param * xi_bar_s[k] * cap_duals[k] for k in 1:num_arcs]

    intercept = objective_value(model) - dot(coeff_x, x_sol) - dot(coeff_h, h_sol)
                - coeff_λ * λ_sol - dot(coeff_ψ0, ψ0_sol)

    return Dict(:intercept => intercept, :coeff_x => coeff_x,
                :coeff_h => coeff_h, :coeff_λ => coeff_λ, :coeff_ψ0 => coeff_ψ0)
end
```

**SDP side:** 기존 `evaluate_master_opt_cut()` 중 해당 side(leader 또는 follower)만 추출하여 사용.

## B.4 알고리즘 Flow

`tr_nested_benders_optimize!`와 동일 구조. ★ 표시가 변경점:

```
for outer_iter:
    Solve OMP → (x*, h*, λ*, ψ⁰*)
    
    ★ LP ISP params 업데이트: update_lp_leader_params! 또는 update_lp_follower_params!
    SDP ISP params 업데이트: 기존 방식

    for inner_iter:
        Solve IMP → α*
        for s in 1:S:
            ★ LP side:  lp_isp_leader_optimize! 또는 lp_isp_follower_optimize!
              SDP side: 기존 isp_leader_optimize! 또는 isp_follower_optimize!
            Add inner cuts to IMP (동일 format: intercept + α'μ)
        Inner convergence check

    ★ Outer cut: evaluate_lp_*_outer_cut (LP side) + evaluate_master_opt_cut (SDP side)
    Add outer cut to OMP
    Outer convergence check
```

## B.5 검증

1. LP ISP 단독: `obj_val ≈ intercept + α'μ` 확인
2. ε→0 극한: SDP ISP를 ε=1e-6으로 풀었을 때 LP ISP 결과와 비교 (근사적 일치)
3. Full run: Full DRO obj ≥ Case1 obj, Full DRO obj ≥ Case2 obj
4. Outer cut tightness: cut value at current point ≈ actual subproblem value
