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
# 3. Build ISP instances for a vertex (legacy, vertex property 불성립으로 미사용)
# ──────────────────────────────────────────────────────────────────
# """
#     build_vertex_isps(j, network, S, ϕU, λU, γ, w, v, uncertainty_set; ...)
# Vertex j에 대해 α_j = (w/S)·e_j로 고정된 ISP leader/follower 인스턴스를 모든 시나리오에 대해 생성.
# """
# function build_vertex_isps(j::Int, network, S, ϕU, λU, γ, w, v_param, uncertainty_set;
#                            conic_optimizer=nothing, λ_sol=nothing, x_sol=nothing,
#                            h_sol=nothing, ψ0_sol=nothing, πU=ϕU, yU=ϕU, ytsU=ϕU)
#     num_arcs = length(network.arcs) - 1
#     α_j = zeros(num_arcs)
#     α_j[j] = w / S
#     leader_instances, follower_instances = initialize_isp(
#         network, S, ϕU, λU, γ, w, v_param, uncertainty_set;
#         conic_optimizer=conic_optimizer,
#         λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
#         α_sol=α_j, πU=πU, yU=yU, ytsU=ytsU)
#     return VertexData(j, α_j, leader_instances, follower_instances)
# end

# ──────────────────────────────────────────────────────────────────
# 3a. Build ISP instances for arbitrary α_sol
# ──────────────────────────────────────────────────────────────────
"""
    build_alpha_isps(id, α_sol, network, S, ϕU_hat, ϕU_tilde, λU, γ, w, v, uncertainty_set; ...)

임의 α_sol에 대해 ISP leader/follower 인스턴스를 모든 시나리오에 대해 생성.
VertexData struct 재활용 (j → id로 사용).
"""
function build_alpha_isps(id::Int, α_sol::Vector{Float64}, network, S, ϕU_hat, ϕU_tilde, λU, γ, w, v_param, uncertainty_set;
                          conic_optimizer=nothing, λ_sol=nothing, x_sol=nothing,
                          h_sol=nothing, ψ0_sol=nothing, πU_hat=ϕU_hat, πU_tilde=ϕU_tilde, yU=ϕU_tilde, ytsU=ϕU_tilde)
    leader_instances, follower_instances = initialize_isp(
        network, S, ϕU_hat, ϕU_tilde, λU, γ, w, v_param, uncertainty_set;
        conic_optimizer=conic_optimizer,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
        α_sol=α_sol, πU_hat=πU_hat, πU_tilde=πU_tilde, yU=yU, ytsU=ytsU)

    return VertexData(id, α_sol, leader_instances, follower_instances)
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
    r_dict_hat = uncertainty_set[:r_dict_hat]
    r_dict_tilde = uncertainty_set[:r_dict_tilde]
    xi_bar = uncertainty_set[:xi_bar]
    epsilon_hat = uncertainty_set[:epsilon_hat]
    epsilon_tilde = uncertainty_set[:epsilon_tilde]

    total_obj = 0.0
    cut_coeff_per_s = Dict{Int, Dict}()

    for s in 1:S
        U_s_hat = Dict(:R => Dict(:1=>R[s]), :r_dict => Dict(:1=>r_dict_hat[s]),
                       :xi_bar => Dict(:1=>xi_bar[s]), :epsilon => epsilon_hat)
        U_s_tilde = Dict(:R => Dict(:1=>R[s]), :r_dict => Dict(:1=>r_dict_tilde[s]),
                         :xi_bar => Dict(:1=>xi_bar[s]), :epsilon => epsilon_tilde)

        # Solve ISP leader
        (status_l, ci_l) = isp_leader_optimize!(
            vdata.leader_instances[s][1], vdata.leader_instances[s][2];
            isp_data=isp_data, uncertainty_set=U_s_hat,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
            α_sol=vdata.α)

        # Solve ISP follower
        (status_f, ci_f) = isp_follower_optimize!(
            vdata.follower_instances[s][1], vdata.follower_instances[s][2];
            isp_data=isp_data, uncertainty_set=U_s_tilde,
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
    ϕU_hat = isp_data[:ϕU_hat]
    ϕU_tilde = isp_data[:ϕU_tilde]
    v_param = isp_data[:v]
    d0 = isp_data[:d0]
    xi_bar = isp_data[:uncertainty_set][:xi_bar]

    diag_x_E = Diagonal(x) * E
    diag_λ_ψ = Diagonal(λ_var * ones(num_arcs) - v_param .* ψ0)

    for s in 1:S
        cc = cut_coeff_per_s[s]
        # ISP는 S=1로 빌드되므로 인덱스는 항상 s=1
        s_idx = 1

        # Leader term: -ϕU_hat·(Û₁·diag(x)E + Û₃·(E-diag(x)E)) + intercept_l
        leader_expr = -ϕU_hat * sum(cc[:Uhat1][s_idx, :, :] .* diag_x_E) +
                      -ϕU_hat * sum(cc[:Uhat3][s_idx, :, :] .* (E - diag_x_E)) +
                      cc[:intercept_l]

        # Follower term: -ϕU_tilde·(Ũ₁·diag(x)E + Ũ₃·(E-diag(x)E)) + Z̃·(diag(λ-vψ₀)·ξ̄) + λ·d₀'β̃₁ - (h+diag(λ-vψ₀)ξ̄)'β̃₃ + intercept_f
        follower_expr = -ϕU_tilde * sum(cc[:Utilde1][s_idx, :, :] .* diag_x_E) +
                        -ϕU_tilde * sum(cc[:Utilde3][s_idx, :, :] .* (E - diag_x_E)) +
                        sum(cc[:Ztilde1_3][s_idx, :, :] .* (diag_λ_ψ * diagm(xi_bar[s]))) +
                        (d0' * cc[:βtilde1_1][s_idx, :]) * λ_var +
                        -(h + diag_λ_ψ * xi_bar[s])' * cc[:βtilde1_3][s_idx, :] +
                        cc[:intercept_f]

        c = @constraint(omp_model, t_js[s] >= leader_expr + follower_expr)
        set_name(c, "ccg_cut_v$(j)_s$(s)_iter$(iter)")
    end
end


# ──────────────────────────────────────────────────────────────────
# 5a-2. Convert evaluate_master_opt_cut format → per-scenario dict
# ──────────────────────────────────────────────────────────────────
"""
    convert_to_per_scenario_cuts(master_cut_info, S)

evaluate_master_opt_cut 반환값 (concatenated [S x ...]) → per-scenario Dict{Int, Dict} 변환.
add_scenario_cuts_to_omp!에서 사용 가능한 형식으로 변환.
"""
function convert_to_per_scenario_cuts(info::Dict, S::Int)
    per_s = Dict{Int, Dict}()
    for s in 1:S
        per_s[s] = Dict(
            :Uhat1 => info[:Uhat1][s:s, :, :],
            :Uhat3 => info[:Uhat3][s:s, :, :],
            :Utilde1 => info[:Utilde1][s:s, :, :],
            :Utilde3 => info[:Utilde3][s:s, :, :],
            :Ztilde1_3 => info[:Ztilde1_3][s:s, :, :],
            :βtilde1_1 => info[:βtilde1_1][s:s, :],
            :βtilde1_3 => info[:βtilde1_3][s:s, :],
            :intercept_l => info[:intercept_l][s],
            :intercept_f => info[:intercept_f][s],
        )
    end
    return per_s
end


# ──────────────────────────────────────────────────────────────────
# 5b. MW cut strengthening for a vertex (per-scenario)
# ──────────────────────────────────────────────────────────────────
"""
    add_mw_cuts_for_vertex!(omp_model, omp_vars, vdata, isp_data, iter;
                            λ_sol, x_sol, h_sol, ψ0_sol, network, γ, λU, w, v_param)

Vertex j의 ISP가 이미 solved 상태에서, MW cut strengthening으로 Pareto-optimal per-scenario cuts 추가.
각 시나리오의 vertex ISP (S=1)에 대해 evaluate_mw_opt_cut을 호출.
"""
function add_mw_cuts_for_vertex!(omp_model, omp_vars, vdata::VertexData, isp_data::Dict, iter::Int;
                                  λ_sol, x_sol, h_sol, ψ0_sol,
                                  network, γ, λU, w, v_param)
    S = isp_data[:S]
    num_arcs = length(x_sol)
    uncertainty_set = isp_data[:uncertainty_set]

    interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
    core_points = generate_core_points(network, γ, λU, w, v_param;
        interdictable_idx=interdictable_idx, strategy=:interior)

    for (cp_idx, cp) in enumerate(core_points)
        mw_cut_coeff_per_s = Dict{Int, Dict}()
        for s in 1:S
            # vertex ISP는 S=1로 빌드 → isp_data_s1으로 호출
            isp_data_s1 = copy(isp_data)
            isp_data_s1[:S] = 1
            isp_data_s1[:uncertainty_set] = Dict(
                :R => Dict(:1 => uncertainty_set[:R][s]),
                :r_dict_hat => Dict(:1 => uncertainty_set[:r_dict_hat][s]),
                :r_dict_tilde => Dict(:1 => uncertainty_set[:r_dict_tilde][s]),
                :xi_bar => Dict(:1 => uncertainty_set[:xi_bar][s]),
                :epsilon_hat => uncertainty_set[:epsilon_hat],
                :epsilon_tilde => uncertainty_set[:epsilon_tilde])
            cut_info_s = Dict(:α_sol => vdata.α)
            leader_insts_s = Dict(1 => vdata.leader_instances[s])
            follower_insts_s = Dict(1 => vdata.follower_instances[s])
            mw_info = evaluate_mw_opt_cut(
                leader_insts_s, follower_insts_s, isp_data_s1, cut_info_s, iter;
                x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                x_core=cp.x, λ_core=cp.λ, h_core=cp.h, ψ0_core=cp.ψ0)
            mw_cut_coeff_per_s[s] = Dict(
                :Uhat1 => mw_info[:Uhat1],
                :Uhat3 => mw_info[:Uhat3],
                :Utilde1 => mw_info[:Utilde1],
                :Utilde3 => mw_info[:Utilde3],
                :Ztilde1_3 => mw_info[:Ztilde1_3],
                :βtilde1_1 => mw_info[:βtilde1_1],
                :βtilde1_3 => mw_info[:βtilde1_3],
                :intercept_l => mw_info[:intercept_l][1],
                :intercept_f => mw_info[:intercept_f][1],
            )
        end
        add_scenario_cuts_to_omp!(omp_model, omp_vars, vdata,
            mw_cut_coeff_per_s, isp_data, iter)
    end
end


# ──────────────────────────────────────────────────────────────────
# 6. Pricing phase: find worst-case vertex
# ──────────────────────────────────────────────────────────────────
"""
    pricing_solve!(network, S, ϕU_hat, ϕU_tilde, λU, γ, w, v, uncertainty_set, isp_data;
                   mip_optimizer, conic_optimizer,
                   λ_sol, x_sol, h_sol, ψ0_sol, πU_hat, πU_tilde, yU, ytsU,
                   inner_tr, tol)

현재 χ*에서 max_{α ∈ Δ} Q₁(α, χ*)를 풀어 worst-case vertex 탐색.
기존 build_imp + tr_imp_optimize! (inner Benders loop) 재사용.
Returns: (j_star, obj_val, α_sol)
"""
function pricing_solve!(network, S, ϕU_hat, ϕU_tilde, λU, γ, w, v_param, uncertainty_set, isp_data;
                        mip_optimizer=nothing, conic_optimizer=nothing,
                        λ_sol=nothing, x_sol=nothing, h_sol=nothing, ψ0_sol=nothing,
                        πU_hat=ϕU_hat, πU_tilde=ϕU_tilde, yU=ϕU_tilde, ytsU=ϕU_tilde,
                        inner_tr=true, tol=1e-4,
                        warm_start_cuts=false)
    num_arcs = length(network.arcs) - 1

    # Build fresh IMP + ISP instances
    imp_model, imp_vars = build_imp(network, S, ϕU_hat, λU, γ, w, v_param, uncertainty_set;
                                     mip_optimizer=mip_optimizer)
    _, α_init = initialize_imp(imp_model, imp_vars)

    leader_instances, follower_instances = initialize_isp(
        network, S, ϕU_hat, ϕU_tilde, λU, γ, w, v_param, uncertainty_set;
        conic_optimizer=conic_optimizer,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
        α_sol=α_init, πU_hat=πU_hat, πU_tilde=πU_tilde, yU=yU, ytsU=ytsU)

    # Run inner Benders loop
    imp_cuts = Dict{Symbol, Any}(:old_tr_constraints => nothing)
    status, result = tr_imp_optimize!(
        imp_model, imp_vars, leader_instances, follower_instances;
        isp_data=isp_data,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
        outer_iter=1, imp_cuts=imp_cuts,
        inner_tr=inner_tr)

    if status != :OptimalityCut
        error("Pricing IMP did not converge: status=$status")
    end

    α_sol = result[:α_sol]
    obj_val = result[:obj_val]

    # Vertex optimality: α*는 vertex에 집중. argmax로 식별.
    j_star = argmax(α_sol)

    # if α_sol[j_star] < (w / S) - 1e-4
    #     @warn "α not concentrated on vertex: α[j*]=$(α_sol[j_star]), w/S=$(w/S). Re-solving with vertex enforcement."

    #     # Vertex-enforcing constraints: z[k] binary, α[k] ≤ z[k]·(w/S), Σz = 1
    #     α_imp = imp_vars[:α]
    #     z_vertex = @variable(imp_model, [k=1:num_arcs], Bin, base_name="z_vertex")
    #     vertex_link_cons = @constraint(imp_model, [k=1:num_arcs], α_imp[k] <= z_vertex[k] * (w / S))
    #     vertex_sum_con = @constraint(imp_model, sum(z_vertex) == 1)

    #     # 모델 수정 후 optimize! 호출하여 valid 상태로 만듦 (tr_imp_optimize! 진입 시 value query 가능하도록)
    #     optimize!(imp_model)

    #     # Re-solve with vertex enforcement — outer_iter=1 + fresh imp_cuts로 기존 cuts 유지
    #     imp_cuts_vertex = Dict{Symbol, Any}(:old_tr_constraints => nothing)
    #     status2, result2 = tr_imp_optimize!(
    #         imp_model, imp_vars, leader_instances, follower_instances;
    #         isp_data=isp_data,
    #         λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
    #         outer_iter=1, imp_cuts=imp_cuts_vertex,
    #         inner_tr=inner_tr)

    #     if status2 != :OptimalityCut
    #         error("Pricing IMP (vertex-enforced) did not converge: status=$status2")
    #     end

    #     α_sol = result2[:α_sol]
    #     obj_val = result2[:obj_val]
    #     j_star = argmax(α_sol)
    #     @info "  [Pricing] Vertex-enforced: j*=$j_star, α[j*]=$(round(α_sol[j_star], digits=6)), obj=$(round(obj_val, digits=6))"

    #     # Clean up vertex-enforcing constraints (다음 pricing에서 fresh build하지만 안전하게 정리)
    #     delete.(imp_model, vertex_link_cons)
    #     delete(imp_model, vertex_sum_con)
    #     delete.(imp_model, z_vertex)
    # end
    
    @info "  [Pricing] α_sol max=$(round(maximum(α_sol), digits=6)), j*=$j_star, " *
          "α[j*]=$(round(α_sol[j_star], digits=6)), w/S=$(round(w/S, digits=6))"

    # Warm-start cuts: evaluate_master_opt_cut + MW strengthening
    pricing_cuts = nothing
    pricing_mw_cuts = nothing
    if warm_start_cuts
        # α_vertex 대신 α_sol 사용 (vertex property 불성립)
        # α_vertex = zeros(num_arcs)
        # α_vertex[j_star] = w / S
        cut_info_for_eval = Dict(:α_sol => α_sol, :obj_val => obj_val)
        pricing_cuts = evaluate_master_opt_cut(
            leader_instances, follower_instances, isp_data, cut_info_for_eval, 0)

        # MW strengthening (ISP가 이미 solved 상태)
        interdictable_idx = findall(network.interdictable_arcs[1:num_arcs])
        core_points = generate_core_points(network, γ, λU, w, v_param;
            interdictable_idx=interdictable_idx, strategy=:interior)
        pricing_mw_cuts = []
        for (cp_idx, cp) in enumerate(core_points)
            mw_info = evaluate_mw_opt_cut(
                leader_instances, follower_instances, isp_data, cut_info_for_eval, 0;
                x_sol=x_sol, λ_sol=λ_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                x_core=cp.x, λ_core=cp.λ, h_core=cp.h, ψ0_core=cp.ψ0)
            push!(pricing_mw_cuts, mw_info)
        end
        @info "  [Pricing] Warm-start cuts + $(length(pricing_mw_cuts)) MW cuts extracted (α_sol)"
    end

    return j_star, obj_val, α_sol, pricing_cuts, pricing_mw_cuts
end


# ──────────────────────────────────────────────────────────────────
# 7. Main C&CG algorithm
# ──────────────────────────────────────────────────────────────────
"""
    ccg_benders_optimize!(network, ϕU_hat, ϕU_tilde, λU, γ, w, v, uncertainty_set;
                          mip_optimizer, conic_optimizer,
                          ε_benders, ε_pricing,
                          max_ccg_iter, max_benders_iter,
                          πU_hat, πU_tilde, yU, ytsU, inner_tr, tol)

C&CG + Benders 메인 알고리즘.

1. OMP 초기화 → pricing으로 초기 vertex 탐색
2. Benders phase: J 내 모든 vertex에 대해 ISP 평가 + per-scenario cut 추가 → 수렴까지
3. Pricing phase: worst-case vertex 탐색 → J에 없으면 추가 → Benders phase 반복
"""
function ccg_benders_optimize!(network, ϕU_hat, ϕU_tilde, λU, γ, w, v_param, uncertainty_set;
                                mip_optimizer=nothing, conic_optimizer=nothing,
                                ε_benders=1e-4, ε_pricing=1e-4,
                                max_ccg_iter=50, max_benders_iter=10000,
                                πU_hat=ϕU_hat, πU_tilde=ϕU_tilde, yU=ϕU_tilde, ytsU=ϕU_tilde,
                                inner_tr=true, tol=1e-4,
                                strengthen_cuts=:none,
                                warm_start_cuts=false)
    S = length(uncertainty_set[:xi_bar])
    num_arcs = length(network.arcs) - 1
    E = ones(num_arcs, num_arcs + 1)
    d0 = zeros(num_arcs + 1); d0[end] = 1.0
    xi_bar = uncertainty_set[:xi_bar]

    isp_data = Dict(
        :E => E, :network => network, :ϕU_hat => ϕU_hat, :ϕU_tilde => ϕU_tilde,
        :πU_hat => πU_hat, :πU_tilde => πU_tilde,
        :yU => yU, :ytsU => ytsU, :λU => λU, :γ => γ,
        :w => w, :v => v_param, :uncertainty_set => uncertainty_set,
        :d0 => d0, :S => S,
    )

    # ========== Initialization ==========
    omp_model, omp_vars = build_omp_ccg(network, ϕU_hat, λU, γ, w; optimizer=mip_optimizer)

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

    # Initial pricing: 첫 번째 α 탐색
    @info "===== C&CG Initialization: Pricing for initial α ====="
    _, Q_init, α_init, init_pricing_cuts, init_mw_cuts = pricing_solve!(network, S, ϕU_hat, ϕU_tilde, λU, γ, w, v_param, uncertainty_set, isp_data;
        mip_optimizer=mip_optimizer, conic_optimizer=conic_optimizer,
        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
        πU_hat=πU_hat, πU_tilde=πU_tilde, yU=yU, ytsU=ytsU, inner_tr=inner_tr, tol=tol,
        warm_start_cuts=warm_start_cuts)

    # Active α set (sequential ID로 관리)
    alpha_id_counter = 1
    active_ids = Set{Int}()
    alpha_data = Dict{Int, VertexData}()  # VertexData 재활용 (j → id)

    push!(active_ids, alpha_id_counter)
    add_vertex_to_omp!(omp_model, omp_vars, alpha_id_counter, S)
    alpha_data[alpha_id_counter] = build_alpha_isps(alpha_id_counter, α_init, network, S, ϕU_hat, ϕU_tilde, λU, γ, w, v_param, uncertainty_set;
        conic_optimizer=conic_optimizer, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
        πU_hat=πU_hat, πU_tilde=πU_tilde, yU=yU, ytsU=ytsU)

    # Warm-start: pricing에서 추출한 cuts + MW cuts를 OMP에 즉시 추가
    if warm_start_cuts && init_pricing_cuts !== nothing
        init_per_s = convert_to_per_scenario_cuts(init_pricing_cuts, S)
        add_scenario_cuts_to_omp!(omp_model, omp_vars, alpha_data[alpha_id_counter],
            init_per_s, isp_data, 0)
        if init_mw_cuts !== nothing
            for mw_info in init_mw_cuts
                mw_per_s = convert_to_per_scenario_cuts(mw_info, S)
                add_scenario_cuts_to_omp!(omp_model, omp_vars, alpha_data[alpha_id_counter],
                    mw_per_s, isp_data, 0)
            end
        end
        @info "  [Warm-start] Initial pricing cuts + MW added for α #$alpha_id_counter"
    end

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
    upper_bound = Q_init
    t_0_sol = -Inf

    for ccg_iter in 1:max_ccg_iter
        @info "=" ^ 60
        @info "===== C&CG Iteration $ccg_iter, |α set| = $(length(active_ids)) ====="

        # -------- Benders Phase --------
        benders_converged = false
        benders_iters = 0
        benders_ub = Inf  # Benders phase 내 local UB (CCG iteration마다 초기화)

        for benders_iter in 1:max_benders_iter
            benders_iters = benders_iter

            # 1. Solve OMP
            optimize!(omp_model)
            st = MOI.get(omp_model, MOI.TerminationStatus())
            if st != MOI.OPTIMAL
                @warn "OMP status: $st"
                if st == MOI.DUAL_INFEASIBLE || st == MOI.INFEASIBLE_OR_UNBOUNDED
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

            # 2. Evaluate ALL α in active set
            max_Q = -Inf
            for id in active_ids
                obj_id, cut_coeff_id = evaluate_vertex!(alpha_data[id], isp_data;
                    λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol)

                max_Q = max(max_Q, obj_id)

                # 3. Add per-scenario cuts
                iter_label = benders_iter + (ccg_iter - 1) * max_benders_iter
                add_scenario_cuts_to_omp!(omp_model, omp_vars, alpha_data[id],
                    cut_coeff_id, isp_data, iter_label)

                # 4. MW cut strengthening
                if strengthen_cuts == :mw
                    add_mw_cuts_for_vertex!(omp_model, omp_vars, alpha_data[id], isp_data, iter_label;
                        λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                        network=network, γ=γ, λU=λU, w=w, v_param=v_param)
                end
            end

            # 4. Benders phase convergence (local UB for this phase)
            benders_ub = min(benders_ub, max_Q)
            gap = (t_0_sol > -Inf) ? abs(benders_ub - t_0_sol) / max(abs(benders_ub), 1e-10) : Inf
            abs_converged = (t_0_sol > -Inf) && (t_0_sol >= benders_ub - 1e-4)

            global_gap = (t_0_sol > -Inf && upper_bound < Inf) ? abs(upper_bound - t_0_sol) / max(abs(upper_bound), 1e-10) : Inf
            @info "  [Benders] Iter $benders_iter: LB=$(round(t_0_sol, digits=6)), UB=$(round(benders_ub, digits=6)), gap=$(round(gap, digits=8)) (global LB=$(round(t_0_sol, digits=6)), UB=$(round(upper_bound, digits=6)), gap=$(round(global_gap, digits=8)))"

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
        push!(history[:J_size], length(active_ids))

        if !benders_converged
            @warn "  Benders phase did not converge within $max_benders_iter iterations."
        end

        # -------- Pricing Phase --------
        @info "  [Pricing] Searching for worst-case α..."
        _, Q_new, α_new, new_pricing_cuts, new_mw_cuts = pricing_solve!(network, S, ϕU_hat, ϕU_tilde, λU, γ, w, v_param, uncertainty_set, isp_data;
            mip_optimizer=mip_optimizer, conic_optimizer=conic_optimizer,
            λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
            πU_hat=πU_hat, πU_tilde=πU_tilde, yU=yU, ytsU=ytsU, inner_tr=inner_tr, tol=tol,
            warm_start_cuts=warm_start_cuts)

        upper_bound = min(upper_bound, Q_new)
        @info "  [Pricing] Q=$(round(Q_new, digits=6)), globalUB=$(round(upper_bound, digits=6)), LB=$(round(t_0_sol, digits=6))"

        if t_0_sol > -Inf && Q_new <= t_0_sol + ε_pricing * max(abs(t_0_sol), 1.0)
            @info "  No improving α found (Q_new=$Q_new ≤ LB+ε=$(t_0_sol + ε_pricing)). ε-OPTIMAL."
            push!(history[:vertices_added], 0)
            break
        else
            alpha_id_counter += 1
            @info "  Adding α #$alpha_id_counter to active set."
            push!(active_ids, alpha_id_counter)
            add_vertex_to_omp!(omp_model, omp_vars, alpha_id_counter, S)
            alpha_data[alpha_id_counter] = build_alpha_isps(alpha_id_counter, α_new, network, S, ϕU_hat, ϕU_tilde, λU, γ, w, v_param, uncertainty_set;
                conic_optimizer=conic_optimizer, λ_sol=λ_sol, x_sol=x_sol, h_sol=h_sol, ψ0_sol=ψ0_sol,
                πU_hat=πU_hat, πU_tilde=πU_tilde, yU=yU, ytsU=ytsU)
            # Warm-start: pricing에서 추출한 cuts + MW cuts를 새 α의 OMP에 즉시 추가
            if warm_start_cuts && new_pricing_cuts !== nothing
                new_per_s = convert_to_per_scenario_cuts(new_pricing_cuts, S)
                add_scenario_cuts_to_omp!(omp_model, omp_vars, alpha_data[alpha_id_counter],
                    new_per_s, isp_data, 0)
                if new_mw_cuts !== nothing
                    for mw_info in new_mw_cuts
                        mw_per_s = convert_to_per_scenario_cuts(mw_info, S)
                        add_scenario_cuts_to_omp!(omp_model, omp_vars, alpha_data[alpha_id_counter],
                            mw_per_s, isp_data, 0)
                    end
                end
                @info "  [Warm-start] Pricing cuts + MW added for α #$alpha_id_counter"
            end
            push!(history[:vertices_added], alpha_id_counter)
        end
    end

    time_end = time()
    @info "=" ^ 60
    @info "C&CG completed in $(round(time_end - time_start, digits=1))s"
    @info "  Final: LB=$(round(t_0_sol, digits=6)), UB=$(round(upper_bound, digits=6))"
    @info "  |α set| = $(length(active_ids))"

    return Dict(
        :opt_sol => Dict(:x => x_sol, :h => h_sol, :λ => λ_sol, :ψ0 => ψ0_sol),
        :obj_val => upper_bound,
        :lower_bound => t_0_sol,
        :active_alphas => active_ids,
        :alpha_data => alpha_data,
        :history => history,
        :solution_time => time_end - time_start,
    )
end
