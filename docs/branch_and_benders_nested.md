# Branch-and-Benders (Single Tree) Implementation Guide
## Based on `nested_benders_trust_region.jl`

## 1. 기존 Nested Benders 구조

```
tr_nested_benders_optimize!:
    build OMP (x, h, λ, ψ0, t_0)
    build IMP (α, t_1_l, t_1_f)
    build ISP instances (leader × S, follower × S)
    
    while not converged:                          ← Outer loop
        optimize!(omp_model)  → χ* = (x, h, λ, ψ0)
        tr_imp_optimize!(χ*)                      ← Inner loop (IMP ↔ ISP)
            while not converged:
                optimize!(imp_model) → α*
                for s in 1:S:
                    isp_leader_optimize!(α*, χ*)   → cut coefficients
                    isp_follower_optimize!(α*, χ*) → cut coefficients
                add inner cuts to IMP
            return cut_info (α_sol, Uhat1, Utilde1, ..., intercept_l, intercept_f, obj_val)
        
        add_optimality_cuts!(omp, cut_info)        ← t_0 ≥ f(x,h,λ,ψ0)
        evaluate_mw_opt_cut(...)                   ← MW strengthened cut (optional)
        alpha_fixed_benders_phase!(...)             ← mini-Benders (optional)
        check gap
```

**핵심**: OMP의 cut은 **`t_0 ≥ Σ_s [leader_s(x) + follower_s(x,h,λ,ψ0)]`** 형태의
aggregated single cut. 시나리오별 multi-cut이 아님.

---

## 2. Branch-and-Benders 구조

```
branch_and_benders_optimize!:
    build OMP (동일한 build_omp)
    build IMP + ISP instances (동일)
    
    [Phase 1: LP warming - optional]
    relax x → continuous
    for k in 1:N_warmup:
        solve OMP LP → χ_lp
        tr_imp_optimize!(χ_lp) → cut_info
        add_optimality_cuts!(omp, cut_info)    ← 일반 constraint
    restore x → binary

    [Phase 2: single tree]
    set LazyConstraints = 1
    set callback = benders_lazy_callback
    optimize!(omp_model)                       ← 한 번만 호출. solver가 B&B 수행.

    benders_lazy_callback(cb_data):
        if not integer-feasible: return
        χ_k = callback_value(cb_data, ...)
        t_0_k = callback_value(cb_data, t_0)
        
        tr_imp_optimize!(χ_k) → cut_info       ← 기존 코드 그대로 호출
        Q_k = cut_info[:obj_val]
        
        if t_0_k < Q_k * S - ε:               ← violation check
            build lazy cut from cut_info
            MOI.submit(LazyConstraint, cut)
            
            [optional: MW cut]
            [optional: α-fixed mini-Benders cuts]
```

**C&CG 대비 차이점**:
- C&CG outer loop (vertex 관리) 없음
- Per-vertex, per-scenario epigraph 변수 없음
- **단일 t_0**에 대한 aggregated cut만 사용
- IMP가 α를 최적화하는 구조 그대로 유지

---

## 3. 기존 코드와의 매핑 (변경 최소화)

| 기존 코드 | 역할 | Branch-and-Benders에서 | 변경? |
|---|---|---|---|
| `build_omp()` | OMP 구축 | **그대로 사용** | 없음 |
| `initialize_omp()` | 초기해 추출 | Phase 1에서 사용 | 없음 |
| `build_imp()` | IMP 구축 | **그대로 사용** | 없음 |
| `initialize_imp()` | IMP 초기화 | **그대로 사용** | 없음 |
| `initialize_isp()` | ISP leader/follower 구축 | **그대로 사용** | 없음 |
| `tr_imp_optimize!()` | inner Benders (IMP↔ISP) | **callback 내에서 호출** | 없음 |
| `evaluate_master_opt_cut()` | cut coefficient 추출 | **그대로 사용** | 없음 |
| `add_optimality_cuts!()` | `@constraint(t_0 >= ...)` | **두 가지로 분리** | **변경** |
| `evaluate_mw_opt_cut()` | MW strengthening | **callback 내에서 호출** | 없음 |
| `alpha_fixed_benders_phase!()` | mini-Benders | ISP-only 부분을 α-history로 변형 사용 | **변형** |
| `generate_core_points()` | MW core point 생성 | **그대로 사용** | 없음 |
| outer while loop + gap check | 수렴 판정 | **제거** (solver 위임) | **제거** |
| trust region logic | regularization | **사용 불가** (§6 참고) | **제거** |

**변경이 필요한 유일한 함수**: `add_optimality_cuts!`를 두 가지 버전으로 분리.

---

## 4. `add_optimality_cuts!` 분리

기존 `add_optimality_cuts!` (strict_benders.jl ~L180):
```julia
function add_optimality_cuts!(omp_model, omp_vars, cut_info, 
    diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ_var, h, S, iter; ...)
    opt_cut = AffExpr(0.0)
    for s in 1:S
        add_to_expression!(opt_cut, -ϕU, dot(cut_info[:Uhat1][s,:,:], diag_x_E))
        add_to_expression!(opt_cut, -ϕU, dot(cut_info[:Uhat3][s,:,:], E - diag_x_E))
        add_to_expression!(opt_cut, cut_info[:intercept_l][s])
        # ... follower terms ...
    end
    c = @constraint(omp_model, omp_vars[:t_0] >= opt_cut)  # ← 이 줄만 다름
    return opt_cut
end
```

### 4.1 Callback용: `build_optimality_cut_expr`

Cut의 JuMP AffExpr만 구성하고, constraint 추가는 하지 않는 버전.
기존 `add_optimality_cuts!`에서 마지막 `@constraint` 줄만 제거한 것:

```julia
function build_optimality_cut_expr(omp_vars, cut_info,
    diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ_var, h_var, S)
    """
    기존 add_optimality_cuts!의 opt_cut 구성 로직 그대로.
    @constraint 대신 AffExpr만 반환.
    """
    opt_cut = AffExpr(0.0)
    for s in 1:S
        add_to_expression!(opt_cut, -ϕU, dot(cut_info[:Uhat1][s,:,:], diag_x_E))
        add_to_expression!(opt_cut, -ϕU, dot(cut_info[:Uhat3][s,:,:], E - diag_x_E))
        add_to_expression!(opt_cut, cut_info[:intercept_l][s])
        add_to_expression!(opt_cut, -ϕU, dot(cut_info[:Utilde1][s,:,:], diag_x_E))
        add_to_expression!(opt_cut, -ϕU, dot(cut_info[:Utilde3][s,:,:], E - diag_x_E))
        add_to_expression!(opt_cut, dot(cut_info[:Ztilde1_3][s,:,:], diag_λ_ψ * diagm(xi_bar[s])))
        add_to_expression!(opt_cut, dot(d0, cut_info[:βtilde1_1][s,:]), λ_var)
        add_to_expression!(opt_cut, -1.0, dot(cut_info[:βtilde1_3][s,:], h_var + diag_λ_ψ * xi_bar[s]))
        add_to_expression!(opt_cut, cut_info[:intercept_f][s])
    end
    return opt_cut
end
```

### 4.2 Callback에서의 violation check

```julia
function evaluate_affexpr_at_callback(expr::AffExpr, cb_data)
    result = expr.constant
    for (var, coef) in expr.terms
        result += coef * callback_value(cb_data, var)
    end
    return result
end
```

### 4.3 Lazy Constraint 추가

```julia
# Callback 내부:
opt_cut_expr = build_optimality_cut_expr(omp_vars, cut_info, ...)
cut_val = evaluate_affexpr_at_callback(opt_cut_expr, cb_data)
t_0_k = callback_value(cb_data, omp_vars[:t_0])

if t_0_k < cut_val - 1e-5
    con = @build_constraint(t_0 >= opt_cut_expr)
    MOI.submit(omp_model, MOI.LazyConstraint(cb_data), con)
end
```

---

## 5. 메인 함수 구현

```julia
function branch_and_benders_optimize!(
    omp_model::Model, omp_vars::Dict,
    network, ϕU, λU, γ, w, v, uncertainty_set;
    mip_optimizer=Gurobi.Optimizer,
    conic_optimizer=Mosek.Optimizer,
    inner_tr=true,
    strengthen_cuts=:none,     # :none, :mw, :sherali
    use_alpha_history=false,   # α-fixed cheap cuts from previous callbacks
    max_alpha_history=3,       # 최근 N개 α만 유지
    lp_warmup_iters=5,         # Phase 1 LP warming iterations
    tol=1e-4,
    πU=ϕU, yU=ϕU, ytsU=ϕU,
    parallel=false)

    S = length(uncertainty_set[:xi_bar])
    num_arcs = length(network.arcs) - 1
    E = ones(num_arcs, num_arcs + 1)
    d0 = zeros(num_arcs + 1); d0[end] = 1.0
    xi_bar = uncertainty_set[:xi_bar]

    x, h, λ_var, ψ0 = omp_vars[:x], omp_vars[:h], omp_vars[:λ], omp_vars[:ψ0]
    t_0 = omp_vars[:t_0]
    diag_x_E = Diagonal(x) * E
    diag_λ_ψ = Diagonal(λ_var * ones(num_arcs) - v .* ψ0)

    isp_data = Dict(:E => E, :network => network, :ϕU => ϕU,
        :πU => πU, :yU => yU, :ytsU => ytsU,
        :λU => λU, :γ => γ, :w => w, :v => v,
        :uncertainty_set => uncertainty_set, :d0 => d0, :S => S)

    # ====== IMP + ISP 초기화 (기존 코드 그대로) ======
    st, λ_sol, x_sol, h_sol, ψ0_sol = initialize_omp(omp_model, omp_vars)

    imp_model, imp_vars = build_imp(network, S, ϕU, λU, γ, w, v, uncertainty_set;
        mip_optimizer=mip_optimizer)
    _, α_sol = initialize_imp(imp_model, imp_vars)

    leader_instances, follower_instances = initialize_isp(
        network, S, ϕU, λU, γ, w, v, uncertainty_set;
        conic_optimizer=conic_optimizer,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol, α_sol=α_sol,
        πU=πU, yU=yU, ytsU=ytsU)

    imp_cuts = Dict{Symbol, Any}(:old_tr_constraints => nothing)

    # MW core points (pre-compute)
    core_points = nothing
    if strengthen_cuts != :none
        interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
        core_points = generate_core_points(network, γ, λU, w, v;
            interdictable_idx=interdictable_idx, strategy=:interior)
    end

    result = Dict(:cuts => Dict(), :inner_iter => Int[], :lazy_cut_count => 0)

    # ====================================================================
    # Phase 1: LP Warming (initial cut seeding)
    # ====================================================================
    # x를 continuous [0,1]로 완화하고 cutting-plane loop으로 initial cuts 축적.
    # Bonami et al. (2020): root LP에서 in-out stabilization으로 cut 생성.
    # McDaniel & Devine (1977): LP relaxation master에서 valid cut 생성.

    if lp_warmup_iters > 0
        @info "Phase 1: LP warming ($lp_warmup_iters iterations)"
        relax_integrality(omp_model)

        for k in 1:lp_warmup_iters
            optimize!(omp_model)
            st_lp = MOI.get(omp_model, MOI.TerminationStatus())
            if st_lp ∉ (MOI.OPTIMAL, MOI.DUAL_INFEASIBLE)
                @warn "LP warmup: OMP status $st_lp at iter $k"
                break
            end
            x_lp = value.(x)  # fractional
            h_lp, λ_lp, ψ0_lp = value.(h), value(λ_var), value.(ψ0)

            status_k, cut_info_k = tr_imp_optimize!(
                imp_model, imp_vars, leader_instances, follower_instances;
                isp_data=isp_data,
                λ_sol=λ_lp, x_sol=x_lp, h_sol=h_lp, ψ0_sol=ψ0_lp,
                outer_iter=k, imp_cuts=imp_cuts, inner_tr=inner_tr,
                parallel=parallel)

            if status_k == :OptimalityCut
                add_optimality_cuts!(omp_model, omp_vars, cut_info_k,
                    diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ_var, h, S, k;
                    prefix="warmup", result_cuts=result[:cuts])
                @info "  Warmup iter $k: cut added (obj=$(round(cut_info_k[:obj_val], digits=6)))"
            end
        end
        # x를 다시 binary로 복원
        for i in 1:num_arcs
            set_binary(x[i])
        end
        @info "Phase 1 complete: $(length(result[:cuts])) initial cuts seeded"
    end

    # ====================================================================
    # Phase 2: Branch-and-Benders (single tree)
    # ====================================================================
    @info "Phase 2: Branch-and-Benders (single tree)"
    cut_count = Ref(0)
    ub_tracker = Ref(Inf)
    α_history = Vector{Vector{Float64}}()  # α-fixed cheap cut용

    function benders_lazy_callback(cb_data)
        status = callback_node_status(cb_data, omp_model)
        if status != MOI.CALLBACK_NODE_STATUS_INTEGER
            return
        end

        # 현재 해 추출
        x_k = round.([callback_value(cb_data, x[i]) for i in 1:num_arcs])
        h_k = [callback_value(cb_data, h[i]) for i in 1:num_arcs]
        λ_k = callback_value(cb_data, λ_var)
        ψ0_k = [callback_value(cb_data, ψ0[i]) for i in 1:num_arcs]
        t_0_k = callback_value(cb_data, t_0)

        # ── α-history cheap cuts (IMP 없이 ISP만 evaluate) ──
        # 이전 callback에서 찾은 α들로 cheap cut 생성.
        # Full IMP보다 훨씬 저렴 (ISP 2S conic solves only, no MIP).
        # 서로 다른 α에서 생성된 cuts → t_0 epigraph를 다방향으로 tighten.
        if use_alpha_history && !isempty(α_history)
            recent_αs = α_history[max(1, end-max_alpha_history+1):end]
            for α_old in recent_αs
                # ISP evaluate at (α_old, χ_k) — 기존 alpha_fixed_benders_phase!의
                # ISP evaluate 로직 재사용 (OMP re-solve 부분 제외)
                _, _ = solve_scenarios(S; parallel=parallel) do s
                    U_s = Dict(:R => Dict(:1=>uncertainty_set[:R][s]),
                               :r_dict => Dict(:1=>uncertainty_set[:r_dict][s]),
                               :xi_bar => Dict(:1=>uncertainty_set[:xi_bar][s]),
                               :epsilon => uncertainty_set[:epsilon])
                    isp_leader_optimize!(leader_instances[s][1], leader_instances[s][2];
                        isp_data=isp_data, uncertainty_set=U_s,
                        λ_sol=λ_k, x_sol=x_k, h_sol=h_k, ψ0_sol=ψ0_k, α_sol=α_old)
                    isp_follower_optimize!(follower_instances[s][1], follower_instances[s][2];
                        isp_data=isp_data, uncertainty_set=U_s,
                        λ_sol=λ_k, x_sol=x_k, h_sol=h_k, ψ0_sol=ψ0_k, α_sol=α_old)
                    return (true, nothing)
                end
                # Extract cut coefficients (기존 evaluate_master_opt_cut 패턴)
                α_fixed_cut_info = extract_cut_coefficients_from_isp(
                    leader_instances, follower_instances, S)
                α_fixed_cut_info[:α_sol] = α_old

                α_expr = build_optimality_cut_expr(
                    omp_vars, α_fixed_cut_info,
                    diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ_var, h, S)
                α_con = @build_constraint(t_0 >= α_expr)
                MOI.submit(omp_model, MOI.LazyConstraint(cb_data), α_con)
                cut_count[] += 1
            end
        end

        # ── Full IMP solve (기존 코드 그대로) ──
        status_imp, cut_info = tr_imp_optimize!(
            imp_model, imp_vars, leader_instances, follower_instances;
            isp_data=isp_data,
            λ_sol=λ_k, x_sol=x_k, h_sol=h_k, ψ0_sol=ψ0_k,
            outer_iter=cut_count[] + 1,
            imp_cuts=imp_cuts, inner_tr=inner_tr,
            parallel=parallel)

        if status_imp != :OptimalityCut
            @warn "IMP did not converge in callback"
            return
        end

        Q_k = cut_info[:obj_val]   # /S averaged
        ub_tracker[] = min(ub_tracker[], Q_k)
        push!(result[:inner_iter], cut_info[:iter])

        # α_history 업데이트 (다음 callback의 cheap cut용)
        if use_alpha_history && haskey(cut_info, :α_sol)
            push!(α_history, copy(cut_info[:α_sol]))
            # 최근 N개만 유지
            while length(α_history) > max_alpha_history
                popfirst!(α_history)
            end
        end

        # Violation check: t_0은 raw sum, Q_k는 /S average
        if t_0_k < Q_k * S - tol * S
            # Direct optimality cut
            opt_expr = build_optimality_cut_expr(
                omp_vars, cut_info,
                diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ_var, h, S)
            con = @build_constraint(t_0 >= opt_expr)
            MOI.submit(omp_model, MOI.LazyConstraint(cb_data), con)
            cut_count[] += 1

            # MW strengthening (optional)
            if strengthen_cuts != :none && core_points !== nothing
                for cp in core_points
                    if strengthen_cuts == :mw
                        mw_info = evaluate_mw_opt_cut(
                            leader_instances, follower_instances,
                            isp_data, cut_info, cut_count[];
                            x_sol=x_k, λ_sol=λ_k, h_sol=h_k, ψ0_sol=ψ0_k,
                            x_core=cp.x, λ_core=cp.λ,
                            h_core=cp.h, ψ0_core=cp.ψ0,
                            parallel=parallel)
                    elseif strengthen_cuts == :sherali
                        mw_info = evaluate_sherali_opt_cut(
                            leader_instances, follower_instances,
                            isp_data, cut_info, cut_count[];
                            x_sol=x_k, λ_sol=λ_k, h_sol=h_k, ψ0_sol=ψ0_k,
                            x_core=cp.x, λ_core=cp.λ,
                            h_core=cp.h, ψ0_core=cp.ψ0,
                            parallel=parallel)
                    end
                    mw_expr = build_optimality_cut_expr(
                        omp_vars, mw_info,
                        diag_x_E, E, diag_λ_ψ, xi_bar, d0, ϕU, λ_var, h, S)
                    mw_con = @build_constraint(t_0 >= mw_expr)
                    MOI.submit(omp_model, MOI.LazyConstraint(cb_data), mw_con)
                    cut_count[] += 1
                end
            end
        end
    end

    # Gurobi 설정
    set_attribute(omp_model, "LazyConstraints", 1)
    set_attribute(omp_model, "Threads", 1)
    set_attribute(omp_model, "PreCrush", 1)
    MOI.set(omp_model, MOI.LazyConstraintCallback(), benders_lazy_callback)

    optimize!(omp_model)

    # 결과 추출
    result[:lazy_cut_count] = cut_count[]
    result[:upper_bound] = ub_tracker[]
    st_final = MOI.get(omp_model, MOI.TerminationStatus())
    if st_final == MOI.OPTIMAL
        result[:opt_sol] = Dict(
            :x => round.(value.(x)), :h => value.(h),
            :λ => value(λ_var), :ψ0 => value.(ψ0))
        result[:obj_val] = value(t_0) / S
    end
    @info "Branch-and-Benders: $(cut_count[]) lazy cuts, status=$st_final"
    return result
end
```

---

## 6. 기존 전략들과의 양립성

### 6.1 Outer Trust Region → **사용 불가**

기존 `outer_tr=true`에서는 ‖x - x̂‖_∞ ≤ B_bin 제약을 동적으로 추가/제거하고,
serious/null step 판정으로 center를 업데이트한다.
Single tree에서는 solver가 B&B tree를 관리하므로 OMP의 feasible region을
동적으로 변경하면 이전 pruning 정보가 무효화됨. **제거해야 함**.

대체 효과: **LP warming (Phase 1)**이 trust region의 초기 수렴 가속을 부분적으로 대체.

### 6.2 Inner Trust Region → **유지 가능**

`tr_imp_optimize!`의 `inner_tr=true`는 IMP 내부 α에 대한 trust region.
OMP와 독립적인 모델이므로 callback 내에서도 **그대로 사용 가능**.

### 6.3 MW Strengthening → **유지 가능** (비용 주의)

`evaluate_mw_opt_cut`은 ISP를 2회 추가 solve하여 Pareto-optimal cut 생성.
Callback 내에서 호출 가능하나, per-callback 비용 증가:
- Direct cut만: 2S conic solves (leader + follower)
- MW 추가 시: +2S conic solves per core point

**권장**: 먼저 direct cut만으로 구현 후, MW의 marginal benefit 측정하여 결정.

### 6.4 α-fixed Mini-Benders → **OMP re-solve는 불가, ISP-only evaluate는 가능**

`alpha_fixed_benders_phase!`는 두 부분으로 구성됨:
1. OMP re-solve → 새 χ 생성 → **callback에서 불가** (OMP를 re-solve할 수 없음)
2. 고정된 α에서 ISP evaluate → cut coefficient 추출 → **callback에서 가능**

**α-history 전략**: 이전 callback에서 `tr_imp_optimize!`가 찾은 α를 저장해두고,
다음 callback에서 full IMP를 풀기 전에 저장된 α로 ISP만 evaluate하여 cheap cut을 생성.

```julia
# α_history: 이전 callback들의 converged α를 저장하는 Vector (closure로 공유)
# 각 callback에서:
#   1. α_history의 최근 α들로 ISP-only evaluate → lazy cuts (IMP 없이, cheap)
#   2. full tr_imp_optimize! → α_new 찾음 → lazy cut
#   3. push!(α_history, α_new)
```

**비용**: ISP 2S conic solves per α_old (IMP MIP solve 없음). Full IMP 대비 훨씬 저렴.

**이점**: 한 callback에서 서로 다른 α에서 생성된 cuts가 동시에 모델에 들어가므로,
t_0의 epigraph를 서로 다른 방향에서 동시에 tighten. 특히 수렴 근처에서 α가
callback 간에 크게 변하지 않으면, cheap cut이 이미 상당히 tight함.

**구현**: `use_alpha_history` flag로 on/off. §5의 callback 코드 참고.
α_history 길이는 최근 N개만 유지 (`max_alpha_history` 파라미터).

---

## 7. IMP 상태 관리 (Callback 간)

### 7.1 IMP Cuts 재사용

기존 nested Benders에서는 outer iteration마다 IMP inner cuts를 삭제하고 재생성.
이유: χ가 바뀌면 ISP의 RHS가 바뀌므로 old inner cuts가 valid하지 않을 수 있음.

**기존 코드도 `imp_cuts[:old_cuts]`로 이전 cuts를 일부 재활용하고 있음.**
Branch-and-Benders에서도 동일하게 `imp_cuts` dict를 callback 간에 유지.
`tr_imp_optimize!` 내부가 알아서 old cuts를 삭제하고 새로 생성함.

### 7.2 ISP Instance 재사용

ISP model은 `set_normalized_rhs`로 χ 파라미터를 업데이트하고 re-solve.
Callback 내에서도 동일. ISP는 Mosek instance로 OMP (Gurobi)와 별도이므로 충돌 없음.

---

## 8. Gurobi Callback 주의사항

### 8.1 `callback_value` 사용
```julia
# ✓ Correct (callback 내부)
x_k = callback_value(cb_data, x[i])
# ✗ Wrong (callback 내부에서 금지)
x_k = value(x[i])
```

### 8.2 Lazy vs. User Cut

`MIPNODE`에서 Benders cut을 넣으려면 반드시 `MOI.LazyConstraint` 사용.
`MOI.UserCut`으로 넣으면 solver가 purge할 수 있어서 incorrect solution 위험.
(Gurobi Help Center: Ruthmair case 참고)

### 8.3 Thread Safety

`Threads=1` 권장: callback 내에서 Mosek ISP + Gurobi IMP 동시 호출 시 안전.

### 8.4 `PreCrush=1`

Gurobi presolve가 변수를 eliminate하면 callback에서 해당 변수에 접근 불가.
`PreCrush=1`로 이를 방지.

---

## 9. 실험 설계

| Variant | 설명 |
|---|---|
| `iterative` | 기존 `tr_nested_benders_optimize!` (outer_tr=false) |
| `iterative_tr` | 기존 trust region 버전 (outer_tr=true) |
| `B&B_direct` | Branch-and-Benders, MW 없음, LP warming 없음 |
| `B&B_warm` | Branch-and-Benders + LP warming (N=5) |
| `B&B_warm_MW` | Branch-and-Benders + LP warming + MW strengthening |
| `B&B_warm_αhist` | Branch-and-Benders + LP warming + α-history (N=3) |
| `B&B_warm_MW_αhist` | Branch-and-Benders + LP warming + MW + α-history |

보고 항목: total solve time, #outer cuts (= #callbacks with violation),
#inner iterations (total across all callbacks), #B&B nodes, final gap.

---

## 10. 파일 구조

```
branch_and_benders.jl                      # 새 파일
├── build_optimality_cut_expr()            # cut AffExpr만 구성
├── evaluate_affexpr_at_callback()         # AffExpr → scalar (callback 내)
├── extract_cut_coefficients_from_isp()    # ISP solved 상태에서 cut coeff 추출
├── branch_and_benders_optimize!()         # 메인 (Phase 1 + Phase 2)
└── benders_lazy_callback()                # lazy callback (closure)

# 기존 파일 (수정 없음):
strict_benders.jl                          # build_omp, add_optimality_cuts!
nested_benders_trust_region.jl             # tr_imp_optimize!, evaluate_mw_opt_cut,
                                           # generate_core_points, initialize_isp,
                                           # isp_leader_optimize!, isp_follower_optimize!,
                                           # build_imp, initialize_imp
```

**새로 작성할 코드량**: `build_optimality_cut_expr` (~20줄, 기존 복사) +
`evaluate_affexpr_at_callback` (~8줄) +
`extract_cut_coefficients_from_isp` (~20줄, 기존 evaluate_master_opt_cut에서 추출) +
메인 함수 (~150줄, α_history 포함).
**나머지는 전부 기존 함수 import.**
