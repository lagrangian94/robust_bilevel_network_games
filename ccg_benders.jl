"""
Column-and-Constraint Generation (C&CG) Benders decomposition.

## 핵심 아이디어
최적 α*는 simplex Δ = {α ≥ 0 : Σα = w/S}의 vertex에 존재.
즉 α* = (w/S)·e_j for some arc j. 이를 이용해 IMP를 제거하고,
active vertex set J를 관리하는 C&CG outer loop + 각 vertex별 Benders inner loop.

## 구조
- OMP: vertex-scenario epigraph 변수 t_{j,s}와 linking constraint t_0 ≥ Σ_s t_{j,s}
- Benders phase: J 내 모든 vertex에 대해 ISP 풀고 per-scenario cut 추가
- Pricing phase: 현재 χ*에서 worst-case vertex 탐색 (IMP ↔ ISP inner Benders)

## 의존성
- network_generator.jl, build_uncertainty_set.jl (네트워크/불확실성)
- nested_benders_trust_region.jl (ISP builders, IMP, inner Benders loop)
- strict_benders.jl (OMP 구조 참고, cut 조립 로직)
"""

using JuMP
using LinearAlgebra
using Infiltrator

# ──────────────────────────────────────────────────────────────────
# Data structure for a vertex α^j = (w/S) · e_j
# ──────────────────────────────────────────────────────────────────
struct VertexData
    j::Int                                              # arc index
    α::Vector{Float64}                                  # α vector
    leader_instances::Dict{Int, Tuple{Model, Dict}}     # s => (model, vars)
    follower_instances::Dict{Int, Tuple{Model, Dict}}   # s => (model, vars)
end


# ──────────────────────────────────────────────────────────────────
# 1. OMP construction (vertex-scenario epigraph)
# ──────────────────────────────────────────────────────────────────
"""
    build_omp_ccg(network, ϕU, λU, γ, w; optimizer)

Strict Benders의 build_omp와 동일한 1단계 변수 구조.
vertex-scenario epigraph 변수 t_{j,s}는 add_vertex_to_omp!로 동적 추가.
"""
function build_omp_ccg(network, ϕU, λU, γ, w; optimizer=nothing)
    num_arcs = length(network.arcs) - 1
    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => true))

    # First-stage variables
    @variable(model, λ, lower_bound=0.0, upper_bound=λU)
    @variable(model, x[1:num_arcs], Bin)
    @variable(model, h[1:num_arcs] >= 0)
    @variable(model, ψ0[1:num_arcs] >= 0)
    @variable(model, t_0)  # overall epigraph

    # Constraints (same as build_omp in strict_benders.jl)
    @constraint(model, λ >= 0.001)
    @constraint(model, resource_budget, sum(h) <= λ * w)
    @constraint(model, sum(x) <= γ)
    for i in 1:num_arcs
        if !network.interdictable_arcs[i]
            @constraint(model, x[i] == 0)
        end
    end
    # McCormick for ψ0 = λ·x
    for k in 1:num_arcs
        @constraint(model, ψ0[k] <= λU * x[k])
        @constraint(model, ψ0[k] <= λ)
        @constraint(model, ψ0[k] >= λ - λU * (1 - x[k]))
        @constraint(model, ψ0[k] >= 0)
    end

    @objective(model, Min, t_0)

    vars = Dict{Symbol, Any}(
        :t_0 => t_0, :λ => λ, :x => x, :h => h, :ψ0 => ψ0,
        :vertex_vars => Dict{Int, Vector{VariableRef}}(),       # j => [t_{j,1}, ..., t_{j,S}]
        :vertex_constraints => Dict{Int, ConstraintRef}(),      # j => linking constraint
    )
    return model, vars
end


# ──────────────────────────────────────────────────────────────────
# 2. Adding a vertex to OMP
# ──────────────────────────────────────────────────────────────────
"""
    add_vertex_to_omp!(omp_model, omp_vars, j, S)

Vertex j에 대해 시나리오별 epigraph 변수 t_{j,s}와 linking constraint t_0 ≥ Σ_s t_{j,s} / S 추가.
"""
function add_vertex_to_omp!(omp_model, omp_vars, j::Int, S::Int)
    t_js = @variable(omp_model, [s=1:S], base_name="t_$(j)_s")
    # Linking: t_0 ≥ Σ_s t_{j,s}  (OMP objective는 Min t_0 / S이므로 linking도 /S 반영)
    con = @constraint(omp_model, omp_vars[:t_0] >= sum(t_js))
    omp_vars[:vertex_vars][j] = t_js
    omp_vars[:vertex_constraints][j] = con
    return t_js
end


# ──────────────────────────────────────────────────────────────────
# 3. Build ISP instances for a vertex
# ──────────────────────────────────────────────────────────────────
"""
    build_vertex_isps(j, network, S, ϕU, λU, γ, w, v, uncertainty_set;
                      conic_optimizer, λ_sol, x_sol, h_sol, ψ0_sol, πU, yU, ytsU)

Vertex j에 대해 α_j = (w/S)·e_j로 고정된 ISP leader/follower 인스턴스를 모든 시나리오에 대해 생성.
"""
function build_vertex_isps(j::Int, network, S, ϕU, λU, γ, w, v_param, uncertainty_set;
                           conic_optimizer=nothing, λ_sol=nothing, x_sol=nothing,
                           h_sol=nothing, ψ0_sol=nothing, πU=ϕU, yU=ϕU, ytsU=ϕU)
    num_arcs = length(network.arcs) - 1
    α_j = zeros(num_arcs)
    α_j[j] = w / S

    leader_instances, follower_instances = initialize_isp(
        network, S, ϕU, λU, γ, w, v_param, uncertainty_set;
        conic_optimizer=conic_optimizer,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
        α_sol=α_j, πU=πU, yU=yU, ytsU=ytsU)

    return VertexData(j, α_j, leader_instances, follower_instances)
end


# ──────────────────────────────────────────────────────────────────
# 4. Evaluate a vertex at current χ — returns obj and cut coefficients
# ──────────────────────────────────────────────────────────────────
"""
    evaluate_vertex!(vdata, isp_data; λ_sol, x_sol, h_sol, ψ0_sol)

Vertex j의 ISP를 현재 (x, h, λ, ψ0)에서 풀고, 전체 objective와 per-scenario cut 계수 반환.
α는 vertex에 의해 fix되어 있으므로 coupling_cons RHS만 set_normalized_rhs로 갱신.
"""
function evaluate_vertex!(vdata::VertexData, isp_data::Dict;
                          λ_sol, x_sol, h_sol, ψ0_sol)
    S = isp_data[:S]
    uncertainty_set = isp_data[:uncertainty_set]
    R = uncertainty_set[:R]
    r_dict = uncertainty_set[:r_dict]
    xi_bar = uncertainty_set[:xi_bar]
    epsilon = uncertainty_set[:epsilon]

    total_obj = 0.0
    cut_coeff_per_s = Dict{Int, Dict}()

    for s in 1:S
        U_s = Dict(:R => Dict(:1=>R[s]), :r_dict => Dict(:1=>r_dict[s]),
                   :xi_bar => Dict(:1=>xi_bar[s]), :epsilon => epsilon)

        # Solve ISP leader
        (status_l, ci_l) = isp_leader_optimize!(
            vdata.leader_instances[s][1], vdata.leader_instances[s][2];
            isp_data=isp_data, uncertainty_set=U_s,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
            α_sol=vdata.α)

        # Solve ISP follower
        (status_f, ci_f) = isp_follower_optimize!(
            vdata.follower_instances[s][1], vdata.follower_instances[s][2];
            isp_data=isp_data, uncertainty_set=U_s,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
            α_sol=vdata.α)

        if status_l != :OptimalityCut || status_f != :OptimalityCut
            error("ISP not optimal for vertex $(vdata.j), scenario $s: leader=$status_l, follower=$status_f")
        end

        s_obj = ci_l[:obj_val] + ci_f[:obj_val]
        total_obj += s_obj

        # Extract cut coefficients from ISP primal variables (= outer Benders cut 계수)
        leader_model = vdata.leader_instances[s][1]
        leader_vars = vdata.leader_instances[s][2]
        follower_model = vdata.follower_instances[s][1]
        follower_vars = vdata.follower_instances[s][2]

        cut_coeff_per_s[s] = Dict(
            :Uhat1 => value.(leader_vars[:Uhat1]),
            :Uhat3 => value.(leader_vars[:Uhat3]),
            :Utilde1 => value.(follower_vars[:Utilde1]),
            :Utilde3 => value.(follower_vars[:Utilde3]),
            :Ztilde1_3 => value.(follower_vars[:Ztilde1_3]),
            :βtilde1_1 => value.(follower_vars[:βtilde1_1]),
            :βtilde1_3 => value.(follower_vars[:βtilde1_3]),
            :intercept_l => value.(leader_vars[:intercept]),
            :intercept_f => value.(follower_vars[:intercept]),
            :obj_val => s_obj,
        )
    end

    total_obj /= S  # average over scenarios
    return total_obj, cut_coeff_per_s
end


# ──────────────────────────────────────────────────────────────────
# 5. Add per-scenario Benders cuts to OMP
# ──────────────────────────────────────────────────────────────────
"""
    add_scenario_cuts_to_omp!(omp_model, omp_vars, vdata, cut_coeff_per_s, isp_data, iter)

Vertex j의 각 시나리오 s에 대해 optimality cut을 t_{j,s}에 추가.
cut 구조: t_{j,s} ≥ leader_s(χ) + follower_s(χ)
strict_benders.jl의 add_optimality_cuts!와 동일한 cut 계수 사용, 단 per-scenario 분리.
"""
function add_scenario_cuts_to_omp!(omp_model, omp_vars, vdata::VertexData,
                                    cut_coeff_per_s::Dict, isp_data::Dict, iter::Int)
    j = vdata.j
    t_js = omp_vars[:vertex_vars][j]
    S = isp_data[:S]
    x, h, λ_var, ψ0 = omp_vars[:x], omp_vars[:h], omp_vars[:λ], omp_vars[:ψ0]
    num_arcs = length(x)
    E = isp_data[:E]
    ϕU = isp_data[:ϕU]
    πU = get(isp_data, :πU, ϕU)
    yU = get(isp_data, :yU, ϕU)
    ytsU = get(isp_data, :ytsU, ϕU)
    v_param = isp_data[:v]
    d0 = isp_data[:d0]
    xi_bar = isp_data[:uncertainty_set][:xi_bar]

    diag_x_E = Diagonal(x) * E
    diag_λ_ψ = Diagonal(λ_var * ones(num_arcs) - v_param .* ψ0)

    for s in 1:S
        cc = cut_coeff_per_s[s]
        # ISP는 S=1로 빌드되므로 인덱스는 항상 s=1
        s_idx = 1

        # Leader term: -ϕU·(Û₁·diag(x)E + Û₃·(E-diag(x)E)) + intercept_l
        leader_expr = -ϕU * sum(cc[:Uhat1][s_idx, :, :] .* diag_x_E) +
                      -ϕU * sum(cc[:Uhat3][s_idx, :, :] .* (E - diag_x_E)) +
                      cc[:intercept_l]

        # Follower term: -ϕU·(Ũ₁·diag(x)E + Ũ₃·(E-diag(x)E)) + Z̃·(diag(λ-vψ₀)·ξ̄) + λ·d₀'β̃₁ - (h+diag(λ-vψ₀)ξ̄)'β̃₃ + intercept_f
        follower_expr = -ϕU * sum(cc[:Utilde1][s_idx, :, :] .* diag_x_E) +
                        -ϕU * sum(cc[:Utilde3][s_idx, :, :] .* (E - diag_x_E)) +
                        sum(cc[:Ztilde1_3][s_idx, :, :] .* (diag_λ_ψ * diagm(xi_bar[s]))) +
                        (d0' * cc[:βtilde1_1][s_idx, :]) * λ_var +
                        -(h + diag_λ_ψ * xi_bar[s])' * cc[:βtilde1_3][s_idx, :] +
                        cc[:intercept_f]

        c = @constraint(omp_model, t_js[s] >= leader_expr + follower_expr)
        set_name(c, "ccg_cut_v$(j)_s$(s)_iter$(iter)")
    end
end


# ──────────────────────────────────────────────────────────────────
# 6. Pricing phase: find worst-case vertex
# ──────────────────────────────────────────────────────────────────
"""
    pricing_solve!(network, S, ϕU, λU, γ, w, v, uncertainty_set, isp_data;
                   mip_optimizer, conic_optimizer,
                   λ_sol, x_sol, h_sol, ψ0_sol, πU, yU, ytsU,
                   inner_tr, tol)

현재 χ*에서 max_{α ∈ Δ} Q₁(α, χ*)를 풀어 worst-case vertex 탐색.
기존 build_imp + tr_imp_optimize! (inner Benders loop) 재사용.
Returns: (j_star, obj_val, α_sol)
"""
function pricing_solve!(network, S, ϕU, λU, γ, w, v_param, uncertainty_set, isp_data;
                        mip_optimizer=nothing, conic_optimizer=nothing,
                        λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing,
                        πU=ϕU, yU=ϕU, ytsU=ϕU,
                        inner_tr=true, tol=1e-4)
    num_arcs = length(network.arcs) - 1

    # Build fresh IMP + ISP instances
    imp_model, imp_vars = build_imp(network, S, ϕU, λU, γ, w, v_param, uncertainty_set;
                                     mip_optimizer=mip_optimizer)
    _, α_init = initialize_imp(imp_model, imp_vars)

    leader_instances, follower_instances = initialize_isp(
        network, S, ϕU, λU, γ, w, v_param, uncertainty_set;
        conic_optimizer=conic_optimizer,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
        α_sol=α_init, πU=πU, yU=yU, ytsU=ytsU)

    # Run inner Benders loop
    imp_cuts = Dict{Symbol, Any}(:old_tr_constraints => nothing)
    status, result = tr_imp_optimize!(
        imp_model, imp_vars, leader_instances, follower_instances;
        isp_data=isp_data,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
        outer_iter=1, imp_cuts=imp_cuts,
        inner_tr=inner_tr, tol=tol)

    if status != :OptimalityCut
        error("Pricing IMP did not converge: status=$status")
    end

    α_sol = result[:α_sol]
    obj_val = result[:obj_val]

    # Vertex optimality: α*는 vertex에 집중. argmax로 식별.
    j_star = argmax(α_sol)

    @info "  [Pricing] α_sol max=$(round(maximum(α_sol), digits=6)), j*=$j_star, " *
          "α[j*]=$(round(α_sol[j_star], digits=6)), w/S=$(round(w/S, digits=6))"

    return j_star, obj_val, α_sol
end


# ──────────────────────────────────────────────────────────────────
# 7. Main C&CG algorithm
# ──────────────────────────────────────────────────────────────────
"""
    ccg_benders_optimize!(network, ϕU, λU, γ, w, v, uncertainty_set;
                          mip_optimizer, conic_optimizer,
                          ε_benders, ε_pricing,
                          max_ccg_iter, max_benders_iter,
                          πU, yU, ytsU, inner_tr, tol)

C&CG + Benders 메인 알고리즘.

1. OMP 초기화 → pricing으로 초기 vertex 탐색
2. Benders phase: J 내 모든 vertex에 대해 ISP 평가 + per-scenario cut 추가 → 수렴까지
3. Pricing phase: worst-case vertex 탐색 → J에 없으면 추가 → Benders phase 반복
"""
function ccg_benders_optimize!(network, ϕU, λU, γ, w, v_param, uncertainty_set;
                                mip_optimizer=nothing, conic_optimizer=nothing,
                                ε_benders=1e-4, ε_pricing=1e-4,
                                max_ccg_iter=50, max_benders_iter=200,
                                πU=ϕU, yU=ϕU, ytsU=ϕU,
                                inner_tr=true, tol=1e-4)
    S = length(uncertainty_set[:xi_bar])
    num_arcs = length(network.arcs) - 1
    E = ones(num_arcs, num_arcs + 1)
    d0 = zeros(num_arcs + 1); d0[end] = 1.0
    xi_bar = uncertainty_set[:xi_bar]

    isp_data = Dict(
        :E => E, :network => network, :ϕU => ϕU, :πU => πU,
        :yU => yU, :ytsU => ytsU, :λU => λU, :γ => γ,
        :w => w, :v => v_param, :uncertainty_set => uncertainty_set,
        :d0 => d0, :S => S,
    )

    # ========== Initialization ==========
    omp_model, omp_vars = build_omp_ccg(network, ϕU, λU, γ, w; optimizer=mip_optimizer)

    # Initial OMP solve (trivial: no cuts → t_0 unbounded below → DUAL_INFEASIBLE)
    optimize!(omp_model)
    st = MOI.get(omp_model, MOI.TerminationStatus())
    # 초기해: t_0 bound 없이는 DUAL_INFEASIBLE → feasible한 χ 추출 위해 임시 bound
    if st == MOI.DUAL_INFEASIBLE || st == MOI.INFEASIBLE_OR_UNBOUNDED
        # t_0에 임시 하한 추가해서 feasible solution 추출
        @constraint(omp_model, omp_vars[:t_0] >= -1e6)
        optimize!(omp_model)
    end
    λ_sol = value(omp_vars[:λ])
    x_sol = round.(value.(omp_vars[:x]))
    h_sol = value.(omp_vars[:h])
    ψ0_sol = value.(omp_vars[:ψ0])

    # Initial pricing: 첫 번째 vertex 탐색
    @info "===== C&CG Initialization: Pricing for initial vertex ====="
    j_init, _, _ = pricing_solve!(network, S, ϕU, λU, γ, w, v_param, uncertainty_set, isp_data;
        mip_optimizer=mip_optimizer, conic_optimizer=conic_optimizer,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
        πU=πU, yU=yU, ytsU=ytsU, inner_tr=inner_tr, tol=tol)

    # Active vertex set
    J = Set{Int}()
    vertex_data = Dict{Int, VertexData}()

    push!(J, j_init)
    add_vertex_to_omp!(omp_model, omp_vars, j_init, S)
    vertex_data[j_init] = build_vertex_isps(j_init, network, S, ϕU, λU, γ, w, v_param, uncertainty_set;
        conic_optimizer=conic_optimizer, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
        πU=πU, yU=yU, ytsU=ytsU)

    # ========== History ==========
    history = Dict(
        :ccg_iter => Int[],
        :benders_iter => Int[],
        :lower_bound => Float64[],
        :upper_bound => Float64[],
        :J_size => Int[],
        :vertices_added => Int[],
    )
    time_start = time()

    # ========== Main C&CG Loop ==========
    upper_bound = Inf
    t_0_sol = -Inf

    for ccg_iter in 1:max_ccg_iter
        @info "=" ^ 60
        @info "===== C&CG Iteration $ccg_iter, |J| = $(length(J)), J = $J ====="

        # -------- Benders Phase --------
        benders_converged = false
        benders_iters = 0

        for benders_iter in 1:max_benders_iter
            benders_iters = benders_iter

            # 1. Solve OMP
            optimize!(omp_model)
            st = MOI.get(omp_model, MOI.TerminationStatus())
            if st != MOI.OPTIMAL
                @warn "OMP status: $st"
                if st == MOI.DUAL_INFEASIBLE || st == MOI.INFEASIBLE_OR_UNBOUNDED
                    # cuts가 아직 부족 → 계속 진행
                    x_sol = round.(value.(omp_vars[:x]))
                    h_sol = value.(omp_vars[:h])
                    λ_sol = value(omp_vars[:λ])
                    ψ0_sol = value.(omp_vars[:ψ0])
                    t_0_sol = -Inf
                else
                    error("OMP unexpected status: $st")
                end
            else
                x_sol = round.(value.(omp_vars[:x]))
                h_sol = value.(omp_vars[:h])
                λ_sol = value(omp_vars[:λ])
                ψ0_sol = value.(omp_vars[:ψ0])
                t_0_sol = value(omp_vars[:t_0]) / S  # /S for average
            end

            # 2. Evaluate ALL vertices in J
            max_Q_j = -Inf
            for j in J
                obj_j, cut_coeff_j = evaluate_vertex!(vertex_data[j], isp_data;
                    λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)

                max_Q_j = max(max_Q_j, obj_j)

                # 3. Add per-scenario cuts
                add_scenario_cuts_to_omp!(omp_model, omp_vars, vertex_data[j],
                    cut_coeff_j, isp_data, benders_iter + (ccg_iter - 1) * max_benders_iter)
            end

            # 4. Update upper bound
            upper_bound = min(upper_bound, max_Q_j)
            gap = (t_0_sol > -Inf) ? abs(upper_bound - t_0_sol) / max(abs(upper_bound), 1e-10) : Inf
            abs_converged = (t_0_sol > -Inf) && (t_0_sol >= upper_bound - 1e-4)

            @info "  [Benders] Iter $benders_iter: LB=$(round(t_0_sol, digits=6)), UB=$(round(upper_bound, digits=6)), gap=$(round(gap, digits=8))"

            if gap <= ε_benders || abs_converged
                @info "  Benders phase converged. (gap=$gap, abs_converged=$abs_converged)"
                benders_converged = true
                break
            end
        end

        push!(history[:ccg_iter], ccg_iter)
        push!(history[:benders_iter], benders_iters)
        push!(history[:lower_bound], t_0_sol)
        push!(history[:upper_bound], upper_bound)
        push!(history[:J_size], length(J))

        if !benders_converged
            @warn "  Benders phase did not converge within $max_benders_iter iterations."
        end

        # -------- Pricing Phase --------
        @info "  [Pricing] Searching for worst-case vertex..."
        j_new, Q_new, α_new = pricing_solve!(network, S, ϕU, λU, γ, w, v_param, uncertainty_set, isp_data;
            mip_optimizer=mip_optimizer, conic_optimizer=conic_optimizer,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
            πU=πU, yU=yU, ytsU=ytsU, inner_tr=inner_tr, tol=tol)

        @info "  [Pricing] Best vertex: j=$j_new, Q=$(round(Q_new, digits=6)), current LB=$(round(t_0_sol, digits=6))"

        if j_new in J
            @info "  Worst-case vertex j=$j_new already in J. OPTIMAL."
            push!(history[:vertices_added], 0)
            break
        elseif t_0_sol > -Inf && Q_new <= t_0_sol + ε_pricing * max(abs(t_0_sol), 1.0)
            @info "  No improving vertex found (Q_new=$Q_new ≤ LB+ε=$(t_0_sol + ε_pricing)). ε-OPTIMAL."
            push!(history[:vertices_added], 0)
            break
        else
            @info "  Adding vertex j=$j_new to J."
            push!(J, j_new)
            add_vertex_to_omp!(omp_model, omp_vars, j_new, S)
            vertex_data[j_new] = build_vertex_isps(j_new, network, S, ϕU, λU, γ, w, v_param, uncertainty_set;
                conic_optimizer=conic_optimizer, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                πU=πU, yU=yU, ytsU=ytsU)
            push!(history[:vertices_added], j_new)
            # New vertex 추가 시 UB 갱신 불필요 (다음 Benders phase에서 자연스럽게 반영)
        end
    end

    time_end = time()
    @info "=" ^ 60
    @info "C&CG completed in $(round(time_end - time_start, digits=1))s"
    @info "  Final: LB=$(round(t_0_sol, digits=6)), UB=$(round(upper_bound, digits=6))"
    @info "  |J| = $(length(J)), J = $J"

    return Dict(
        :opt_sol => Dict(:x => x_sol, :h => h_sol, :λ => λ_sol, :ψ0 => ψ0_sol),
        :obj_val => upper_bound,
        :lower_bound => t_0_sol,
        :active_vertices => J,
        :vertex_data => vertex_data,
        :history => history,
        :solution_time => time_end - time_start,
    )
end
