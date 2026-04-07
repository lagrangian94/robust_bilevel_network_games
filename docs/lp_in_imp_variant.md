# LP-in-IMP Variant: LP ISP를 IMP에 흡수

**전제:** `partial_robust_full_spec.md`를 먼저 읽을 것. 수학 모델(Part A)은 동일.

---

## 1. 아이디어

`partial_robust_full_spec.md`의 접근: LP ISP를 별도 subproblem으로 풀고 cut으로 IMP에 전달.

이 문서의 접근: LP ISP의 **제약식 자체를 IMP에 직접 넣어서** ISP 호출을 제거.

LP value function은 그 LP의 feasible region으로 exact 표현 가능하므로, cut으로 근사할 필요가 없다.

```
기존:    IMP ↔ ISP-L(LP) + ISP-F(SDP)   → inner iter마다 2S subproblem
LP흡수:  IMP(+LP) ↔ ISP-F(SDP)만         → inner iter마다 S subproblem
```

**이점:**
- ISP-L 호출 완전 제거 → 시간 절약
- LP side가 exact → inner convergence 가속
- LP cut의 shadow_price 부호 걱정 불필요

**비용:**
- IMP 크기 증가: scenario당 O(|A|) 변수/제약 추가. S ≤ 5이면 무시 가능.

---

## 2. 어떤 formulation을 넣는가

IMP는 **max** 문제다. LP ISP도 IMP에 넣으려면 max여야 자연스럽다.

`partial_robust_full_spec.md`의 LP ISP는 **primal = min** formulation:
```
Z₁ᴸˢ(α) = min (1/S)η̂ˢ + Σ αₖμ̂ₖˢ  s.t. primal constraints
```

이것의 LP dual이 **max** formulation이다. Strong duality에 의해 값이 같다.

그런데 이 LP가 간단해서 **primal을 직접 넣어도 된다**:
- IMP에 leader primal 변수(η̂, μ̂, π̂₀, ϕ̂₀, ψ̂₀)와 제약을 추가
- `t_l[s]`를 없애고, 대신 leader primal objective `(1/S)η̂ˢ + Σ αₖμ̂ₖˢ`를 IMP objective에 **빼준다**

IMP는 max이고 leader primal은 min이므로:
```
max Σₛ [ −((1/S)η̂ˢ + Σ αₖμ̂ₖˢ) + t_f[s] ] / S
```

아니, 이것보다 더 간단하게: 기존에 `t_l[s] ≤ Z₁ᴸˢ(α)` where Z₁ᴸˢ = min ... 이므로, t_l[s]를 leader primal로 **대체**한다:

```
기존:  max (1/S) Σₛ (t_l[s] + t_f[s])
       s.t. t_l[s] ≤ (cuts approximating Z₁ᴸˢ(α))

LP흡수: max (1/S) Σₛ ((1/S)η̂ˢ + Σₖ αₖμ̂ₖˢ + t_f[s])
        s.t. leader primal constraints per s
```

**잠깐 — 부호 확인.** Z₁ᴸˢ(α)는 dual outer subproblem의 leader 기여이므로, **maximize** 관점에서의 값이다. Primal ISP는 Z₁ᴸˢ = min (1/S)η̂ + Σαμ̂ 이고, 이 min 값이 곧 dual optimal = max optimal. 따라서 IMP에서 t_l[s]는 Z₁ᴸˢ 이하를 원하는 게 아니라, **Z₁ᴸˢ와 같아지는 것**을 원한다.

실제로 기존 IMP에서:
- IMP: max t_l[s] s.t. t_l[s] ≤ (cuts)
- 수렴 시: t_l[s] = Z₁ᴸˢ(α)

LP를 넣으면 t_l[s] 대신 primal objective를 직접 쓰는데, primal은 **min**이므로 IMP의 max와 반대 방향이다. Primal constraints를 만족하는 모든 (η̂, μ̂)에 대해 (1/S)η̂ + Σαμ̂ ≥ Z₁ᴸˢ(α)이고, min에서 등호가 달성된다.

그래서 IMP objective에 `(1/S)η̂ˢ + Σ αₖμ̂ₖˢ`를 더하면, IMP가 이걸 **maximize하려고** 하지만 primal constraints에 의해 **아래로 bound**된다. 이건 틀린 방향이다 — IMP가 η̂를 키우려고 하는데, primal constraints가 η̂를 키우는 걸 방해하지 않는다.

**결론: primal을 직접 넣는 것은 방향이 반대라서 안 된다.**

올바른 방법: **dual ISP (max formulation)를 IMP에 넣는다.**

---

## 3. Dual Leader LP ISP (max formulation)

`partial_robust_full_spec.md`에서 primal로 기술한 leader LP ISP의 **LP dual**:

```
Z₁ᴸˢ(α) = max  βhat1_ts − ϕU Σₖ xₖ Uhat1ₖ − ϕU Σₖ (1−xₖ) Uhat3ₖ

s.t.
  τhat ≤ 1/S                                                      (DL1)
  Ny · Zhat1_1 + Nts · βhat1_ts ≤ 0       (nv1 constraints)       (DL2)
  −ξ̄ₖ · τhat + Zhat1_1ₖ + Uhat2ₖ − Uhat3ₖ ≤ αₖ   ∀k             (DL3)
  vₖ ξ̄ₖ · τhat − Uhat1ₖ − Uhat2ₖ + Uhat3ₖ = 0   ∀k              (DL4)
  τhat, Zhat1_1, βhat1_ts, Uhat1, Uhat2, Uhat3 ≥ 0               (DL5)
```

**변수명 대응 (기존 dual ISP → LP 축소):**

| 기존 dual ISP (SDP) | LP 축소 (ε̂=0) | 역할 |
|---|---|---|
| `Mhat[s,i,j]` (matrix, SDP) | `τhat` (scalar ≥ 0) | Mhat₂₂ 만 생존 |
| `βhat1[s,1:na1]` 의 ts성분 | `βhat1_ts` (scalar) | d₀ᵀβ̂₁ˢ·¹ = βhat1_ts |
| `Zhat1[s,1:na1,:]` 의 1st block | `Zhat1_1[k]` (vector) | SOC→LP로 flow dual |
| `Uhat1, Uhat2, Uhat3` (matrices) | `Uhat1, Uhat2, Uhat3` (vectors) | McCormick duals |
| `Phat, Γhat, ϑhat` | **삭제** | LDR bounds, SOC, S-lemma 불필요 |

**핵심:** 기존 SDP ISP에서 이 변수들은 행렬이었지만, ε̂=0이면 intercept column만 남아 벡터로 축소.

---

## 4. IMP 수정 코드 (Case 1: ε̂=0)

```julia
function build_imp_with_leader_lp(network, S, ϕU, v_param, w, uncertainty_set,
                                    x_sol; mip_optimizer)
    num_arcs = length(network.arcs) - 1
    nv1 = length(network.nodes) - 1
    N_trunc = network.N[2:end, :]
    Ny = N_trunc[:, 1:num_arcs]
    Nts = N_trunc[:, end]
    xi_bar = uncertainty_set[:xi_bar]
    
    model = Model(optimizer_with_attributes(mip_optimizer, MOI.Silent() => true))
    
    # ===== 기존 IMP 변수 =====
    @variable(model, t_f[s=1:S])                      # follower epigraph (cuts)
    @variable(model, α[k=1:num_arcs] >= 0)
    @constraint(model, sum(α) == w/S)
    
    # ===== Leader LP 변수 (per scenario, inline) =====
    @variable(model, τhat[s=1:S] >= 0)
    @variable(model, Zhat1_1[s=1:S, k=1:num_arcs] >= 0)    # flow dual
    @variable(model, βhat1_ts[s=1:S] >= 0)                  # ts flow dual
    @variable(model, Uhat1[s=1:S, k=1:num_arcs] >= 0)       # McCormick 1
    @variable(model, Uhat2[s=1:S, k=1:num_arcs] >= 0)       # McCormick 2
    @variable(model, Uhat3[s=1:S, k=1:num_arcs] >= 0)       # McCormick 3
    
    for s in 1:S
        ξ̄ = xi_bar[s]
        
        # (DL1) τ̂ ≤ 1/S
        @constraint(model, τhat[s] <= 1/S)
        
        # (DL2) Ny · Zhat1_1 + Nts · βhat1_ts ≤ 0
        @constraint(model, [j=1:nv1],
            sum(Ny[j,k] * Zhat1_1[s,k] for k in 1:num_arcs) + Nts[j] * βhat1_ts[s] <= 0)
        
        # (DL3) −ξ̄ₖτ̂ + Zhat1_1ₖ + Uhat2ₖ − Uhat3ₖ ≤ αₖ
        @constraint(model, [k=1:num_arcs],
            -ξ̄[k] * τhat[s] + Zhat1_1[s,k] + Uhat2[s,k] - Uhat3[s,k] <= α[k])
        
        # (DL4) vξ̄ₖτ̂ − Uhat1ₖ − Uhat2ₖ + Uhat3ₖ = 0
        @constraint(model, [k=1:num_arcs],
            v_param * ξ̄[k] * τhat[s] - Uhat1[s,k] - Uhat2[s,k] + Uhat3[s,k] == 0)
    end
    
    # ===== McCormick (x 파라미터, outer iter마다 update) =====
    # Uhat1 ≤ ϕU·diag(x) 구조. LP축소로 이건:
    # dual ISP에서 Ψ̂₀ₖ ≤ ϕUxₖ의 dual → Uhat1ₖ의 objective coeff = −ϕUxₖ
    # → IMP에서는 objective coefficient로 처리 (아래 참조)
    
    # ===== 목적함수 =====
    # Leader dual obj: βhat1_ts − ϕU xₖ Uhat1ₖ − ϕU(1−xₖ) Uhat3ₖ
    # Follower: t_f[s] (cuts)
    @objective(model, Max,
        (1/S) * sum(
            βhat1_ts[s]
            - ϕU * sum(x_sol[k] * Uhat1[s,k] for k in 1:num_arcs)
            - ϕU * sum((1-x_sol[k]) * Uhat3[s,k] for k in 1:num_arcs)
            + t_f[s]
            for s in 1:S))
    
    vars = Dict(
        :t_f => t_f, :α => α,
        :τhat => τhat, :Zhat1_1 => Zhat1_1, :βhat1_ts => βhat1_ts,
        :Uhat1 => Uhat1, :Uhat2 => Uhat2, :Uhat3 => Uhat3)
    
    return model, vars
end
```

### Outer iteration마다 x 업데이트

x가 바뀌면 IMP **목적함수** 계수가 변한다:
```julia
function update_imp_leader_lp!(model, vars; x_sol, ϕU, S)
    for s in 1:S, k in 1:length(x_sol)
        set_objective_coefficient(model, vars[:Uhat1][s,k], -(1/S) * ϕU * x_sol[k])
        set_objective_coefficient(model, vars[:Uhat3][s,k], -(1/S) * ϕU * (1 - x_sol[k]))
    end
end
```

---

## 5. IMP 수정 코드 (Case 2: ε̃=0)

Follower LP를 IMP에 넣는다. Leader는 SDP ISP 그대로.

### Dual Follower LP ISP (max formulation)

```
Z₁ᶠˢ(α) = max  λ·βtilde1_ts − Σₖ Cₖ·Ztilde_cap_k − ϕU x'Utilde1 − ϕU(1−x)'Utilde3

where Cₖ = hₖ + (λ − vₖψ⁰ₖ)ξ̄ₖ

s.t.
  τtilde = 1/S                                                    (DF1) equality
  Ny · Ztilde1_1 + Nts · βtilde1_ts ≤ 0                           (DF2)
  −ξ̄ₖ · τtilde + Ztilde1_1ₖ + Utilde2ₖ − Utilde3ₖ ≤ αₖ           (DF3)
  vₖ ξ̄ₖ · τtilde − Utilde1ₖ − Utilde2ₖ + Utilde3ₖ = 0             (DF4)
  Nyᵀ · Ztilde_cy + Ztilde_cap ≥ 0                                 (DF5)
  τtilde + Ntsᵀ · Ztilde_cy ≥ 0                                    (DF6)
  all ≥ 0                                                           (DF7)
```

**변수명 대응:**

| 기존 dual ISP (SDP) | LP 축소 | 역할 |
|---|---|---|
| `Mtilde₂₂` | `τtilde = 1/S` (고정) | η̃ free → 등호 |
| `βtilde1` ts성분 | `βtilde1_ts` | λ·flow benefit |
| `Ztilde1` 1st block | `Ztilde1_1` | flow dual |
| `Ztilde1` 3rd block | `Ztilde_cap` | capacity dual (∂/∂h) |
| `Ztilde1` 2nd block | `Ztilde_cy` | flow conservation dual |
| `Utilde1,2,3` | vectors | McCormick |

```julia
function build_imp_with_follower_lp(network, S, ϕU, v_param, w, uncertainty_set,
                                      x_sol, h_sol, λ_sol, ψ0_sol; mip_optimizer)
    # ... network setup ...
    
    model = Model(...)
    
    # 기존 IMP
    @variable(model, t_l[s=1:S])                     # leader epigraph (SDP cuts)
    @variable(model, α[k=1:num_arcs] >= 0)
    @constraint(model, sum(α) == w/S)
    
    # Follower LP 변수 (per s)
    @variable(model, τtilde[s=1:S] >= 0)
    @variable(model, Ztilde1_1[s=1:S, k=1:num_arcs] >= 0)
    @variable(model, βtilde1_ts[s=1:S] >= 0)
    @variable(model, Ztilde_cy[s=1:S, j=1:nv1] >= 0)
    @variable(model, Ztilde_cap[s=1:S, k=1:num_arcs] >= 0)
    @variable(model, Utilde1[s=1:S, k=1:num_arcs] >= 0)
    @variable(model, Utilde2[s=1:S, k=1:num_arcs] >= 0)
    @variable(model, Utilde3[s=1:S, k=1:num_arcs] >= 0)
    
    for s in 1:S
        ξ̄ = xi_bar[s]
        
        # (DF1) τ̃ = 1/S
        @constraint(model, τtilde[s] == 1/S)
        
        # (DF2) flow
        @constraint(model, [j=1:nv1],
            sum(Ny[j,k] * Ztilde1_1[s,k] for k in 1:num_arcs) + Nts[j] * βtilde1_ts[s] <= 0)
        
        # (DF3) capacity + α coupling
        @constraint(model, [k=1:num_arcs],
            -ξ̄[k] * τtilde[s] + Ztilde1_1[s,k] + Utilde2[s,k] - Utilde3[s,k] <= α[k])
        
        # (DF4) McCormick balance
        @constraint(model, [k=1:num_arcs],
            v_param * ξ̄[k] * τtilde[s] - Utilde1[s,k] - Utilde2[s,k] + Utilde3[s,k] == 0)
        
        # (DF5) Nyᵀ Ztilde_cy + Ztilde_cap ≥ 0
        @constraint(model, [k=1:num_arcs],
            sum(Ny[j,k] * Ztilde_cy[s,j] for j in 1:nv1) + Ztilde_cap[s,k] >= 0)
        
        # (DF6) τ + Ntsᵀ Ztilde_cy ≥ 0
        @constraint(model, τtilde[s] + sum(Nts[j] * Ztilde_cy[s,j] for j in 1:nv1) >= 0)
    end
    
    # 목적함수: x, h, λ, ψ⁰ 의존 (outer iter마다 update)
    @objective(model, Max,
        (1/S) * sum(
            t_l[s]
            + λ_sol * βtilde1_ts[s]
            - sum((h_sol[k] + (λ_sol - v_param * ψ0_sol[k]) * xi_bar[s][k]) * Ztilde_cap[s,k]
                  for k in 1:num_arcs)
            - ϕU * sum(x_sol[k] * Utilde1[s,k] for k in 1:num_arcs)
            - ϕU * sum((1-x_sol[k]) * Utilde3[s,k] for k in 1:num_arcs)
            for s in 1:S))
    
    vars = Dict(
        :t_l => t_l, :α => α,
        :τtilde => τtilde, :Ztilde1_1 => Ztilde1_1, :βtilde1_ts => βtilde1_ts,
        :Ztilde_cy => Ztilde_cy, :Ztilde_cap => Ztilde_cap,
        :Utilde1 => Utilde1, :Utilde2 => Utilde2, :Utilde3 => Utilde3)
    
    return model, vars
end
```

Outer iteration마다 x, h, λ, ψ⁰ 업데이트 → `set_objective_coefficient` for each variable.

---

## 6. Outer Cut 생성 — 단순화

LP를 IMP에 넣으면 **LP side의 outer cut 추출이 불필요**해진다. 이유:

기존에 outer cut은 "inner loop 수렴 후 dual outer subproblem의 최적해에서 OMP 파라미터에 대한 sensitivity"를 계산했다. LP side와 SDP side를 합쳐서.

LP가 IMP 안에 있으면, IMP의 최적해에 LP 변수(Uhat1, βhat1_ts 등)가 직접 포함되어 있다. Outer cut 생성 시 이 변수 값을 읽어서 LP side 기여를 바로 계산할 수 있다.

구체적으로:

**Case 1 (leader LP in IMP):**
```julia
# IMP 최적해에서 leader LP 변수 읽기:
for s in 1:S
    Uhat1_val[s,:] = value.(vars[:Uhat1][s,:])
    Uhat3_val[s,:] = value.(vars[:Uhat3][s,:])
    βhat1_ts_val[s] = value(vars[:βhat1_ts][s])
end

# Leader side outer cut coefficients:
leader_coeff_x[k] = (1/S) * Σₛ ϕU * (Uhat3_val[s,k] - Uhat1_val[s,k])
leader_coeff_h = 0, leader_coeff_λ = 0, leader_coeff_ψ0 = 0

# Follower side: 기존 evaluate_master_opt_cut의 follower 부분 사용
```

이렇게 하면 `evaluate_lp_leader_outer_cut`을 별도 함수로 만들 필요 없이, IMP solution에서 직접 읽는다.

---

## 7. 알고리즘 Flow 비교

### 기존 (LP separate ISP)
```
outer loop:
  OMP → (x,h,λ,ψ⁰)
  inner loop:
    IMP → α
    ISP-L(LP): solve, add cut to IMP     ← S calls
    ISP-F(SDP): solve, add cut to IMP    ← S calls
  outer cut: LP extraction + SDP extraction
```

### LP-in-IMP
```
outer loop:
  OMP → (x,h,λ,ψ⁰)
  update IMP objective (x,h,λ,ψ⁰ in LP coefficients)
  inner loop:
    IMP(+LP) → α, LP vars               ← LP side exact
    ISP-F(SDP): solve, add cut to IMP    ← S calls only
  outer cut: read LP vars from IMP + SDP extraction
```

Inner iteration당 subproblem 호출이 **절반**으로 줄어든다.

---

## 8. 주의사항

1. **IMP 크기:** S·(6·num_arcs + 2 + nv1) 변수 추가 (Case 1). S=5, num_arcs=40이면 ~1200 변수. LP solver에게 trivial.

2. **IMP rebuild vs update:** x가 바뀔 때 목적함수 계수만 바뀌므로 rebuild 불필요, `set_objective_coefficient`로 warm-start 유지.

3. **Follower LP in IMP (Case 2):** 목적함수에 h, λ, ψ⁰이 들어가므로 outer iteration마다 더 많은 coefficient를 업데이트해야 한다. S·num_arcs개의 `set_objective_coefficient` 호출.

4. **LP-in-IMP vs LP-separate:** 수학적으로 동치이므로 같은 optimal에 도달. 차이는 computational efficiency뿐.

5. **OMP에 LP를 넣지 않는 이유:** μ̂와 μ̃가 ν를 통해 coupling되어 있으므로, LP를 OMP에 올려도 tilde subproblem의 scenario decomposability를 위해 inner Benders가 여전히 필요. IMP에 넣는 게 자연스럽다.
