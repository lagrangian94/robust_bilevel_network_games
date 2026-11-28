"""
true_dro_mincut_vi.jl — Min-Cut Valid Inequalities for True-DRO OMP.

Phase 1 (1회): h=0 min-cut bound, all S scenarios.
Phase 2B (매 iter): comp-min scenario + α* recovery, single scenario.

See docs/omp_valid_inequality.md for proofs (Propositions 1, 4).
"""

using JuMP


"""
    extract_arc_topology(Ny, nv1) → (tail, head)

Ny convention: Ny[i,k]=+1 → tail(k)=i, Ny[i,k]=-1 → head(k)=i.
Source (removed from Ny) → tail/head = 0.  Sink 의 행은 `_mincut_sink_row(td)` (마지막 행이 아닐 수 있음).
Returns vectors of length K (num arcs).
"""
function extract_arc_topology(Ny::Matrix{Float64}, nv1::Int)
    K = size(Ny, 2)
    tail = zeros(Int, K)
    head = zeros(Int, K)

    for k in 1:K
        for i in 1:nv1
            if Ny[i, k] > 0.5
                tail[k] = i
            elseif Ny[i, k] < -0.5
                head[k] = i
            end
        end
        # tail[k]==0 means source is tail, head[k]==0 means source is head
    end

    return (tail=tail, head=head)
end


"""
    _mincut_sink_row(td) → Ny 에서 sink t 의 행 번호

더미 호 (t → s) 의 열 Nts 에서 읽는다. s 행은 Ny 에서 빠져 있으므로 Nts 의 비영 원소는 t 행의 +1 하나다.
2026-11-28 수정: 예전 판은 sink = nv1 (Ny 의 마지막 행) 으로 가정했다. grid5x5·Sioux Falls 는 맞지만
Polska·Abilene·Nobel-US 는 t 가 9 행 (nv1 = 11, 11, 13) 이라 VI 가 t 가 아닌 마디까지의 cut 을 하한으로 넣었다
(Polska s4 x = {7,18}: 우변 28.07, 실제 h = 0 최대흐름 5.01). docs/nz_ccg_bnp_2026-11-23.md 11.28 절.
"""
function _mincut_sink_row(td::TrueDROData)
    nzr = findall(!iszero, td.Nts)
    (length(nzr) == 1 && td.Nts[nzr[1]] == 1.0) ||
        error("_mincut_sink_row: Nts 의 비영 원소가 t 행의 +1 하나가 아님 (행 $(nzr), 값 $(td.Nts[nzr]))")
    return nzr[1]
end


"""
    add_phase1_mincut_vi!(omp_model, omp_vars, td)

Phase 1: h=0, all S scenarios, added once.
Adds variables δ[k,j], τ[i,j], w[k,j] ∈ [0,1] and DC/MC/link constraints.

(DC-link): t₀ ≥ Σ_j q̂_j Σ_k ξ̄^j_k (δ^j_k - v_k w^j_k)
"""
function add_phase1_mincut_vi!(omp_model, omp_vars, td::TrueDROData)
    K = td.num_arcs
    S = td.S
    nv1 = td.nv1

    arc_topo = extract_arc_topology(td.Ny, nv1)
    tsk = _mincut_sink_row(td)

    x = omp_vars[:x]
    t_0 = omp_vars[:t_0]

    # Variables
    δ = @variable(omp_model, [1:K, 1:S], lower_bound=0, upper_bound=1,
                  base_name="mc_p1_δ")
    τ = @variable(omp_model, [1:nv1, 1:S], lower_bound=0, upper_bound=1,
                  base_name="mc_p1_τ")
    w = @variable(omp_model, [1:K, 1:S], lower_bound=0, upper_bound=1,
                  base_name="mc_p1_w")

    for j in 1:S
        # DC-2: τ_sink = 1
        @constraint(omp_model, τ[tsk, j] == 1)

        for k in 1:K
            # DC-1: δ_k ≥ τ_{head(k)} - τ_{tail(k)}
            τ_head = arc_topo.head[k] == 0 ? 0.0 : τ[arc_topo.head[k], j]
            τ_tail = arc_topo.tail[k] == 0 ? 0.0 : τ[arc_topo.tail[k], j]
            @constraint(omp_model, δ[k, j] >= τ_head - τ_tail)

            # MC-1: w_k ≤ x_k
            @constraint(omp_model, w[k, j] <= x[k])
            # MC-2: w_k ≤ δ_k
            @constraint(omp_model, w[k, j] <= δ[k, j])
            # MC-3: w_k ≥ δ_k - (1 - x_k)
            @constraint(omp_model, w[k, j] >= δ[k, j] - (1 - x[k]))
        end
    end

    # DC-link: t₀ ≥ Σ_j q̂_j Σ_k ξ̄^j_k (δ^j_k - v_k^j w^j_k)
    link_expr = sum(
        td.q_hat[j] * sum(td.xi_bar[k, j] * (δ[k, j] - td.v[k, j] * w[k, j]) for k in 1:K)
        for j in 1:S
    )
    @constraint(omp_model, t_0 >= link_expr)

    return nothing
end


"""
    add_phase2B_mincut_vi!(omp_model, omp_vars, td, α_val, iter; arc_topology)

Phase 2 Variant B: comp-min scenario + α* recovery, added each iteration.
Single min-cut with ξ̄^min_k = min_j ξ̄^j_k.

(SB-link): t₀ ≥ Σ_k [ξ̄^min_k (δ_k - v_k w_k) + α_k δ_k]
"""
function add_phase2B_mincut_vi!(omp_model, omp_vars, td::TrueDROData,
                                 α_val::Vector{Float64}, iter::Int;
                                 arc_topology)
    K = td.num_arcs
    nv1 = td.nv1

    x = omp_vars[:x]
    t_0 = omp_vars[:t_0]

    # Componentwise-min capacity
    ξ_min = vec(minimum(td.xi_bar, dims=2))  # length K

    tag = "mc_p2B_$(iter)"

    # Variables
    δ = @variable(omp_model, [1:K], lower_bound=0, upper_bound=1,
                  base_name="$(tag)_δ")
    τ = @variable(omp_model, [1:nv1], lower_bound=0, upper_bound=1,
                  base_name="$(tag)_τ")
    w = @variable(omp_model, [1:K], lower_bound=0, upper_bound=1,
                  base_name="$(tag)_w")

    # DC-2: τ_sink = 1
    @constraint(omp_model, τ[_mincut_sink_row(td)] == 1)

    for k in 1:K
        # DC-1
        τ_head = arc_topology.head[k] == 0 ? 0.0 : τ[arc_topology.head[k]]
        τ_tail = arc_topology.tail[k] == 0 ? 0.0 : τ[arc_topology.tail[k]]
        @constraint(omp_model, δ[k] >= τ_head - τ_tail)

        # MC-1~3
        @constraint(omp_model, w[k] <= x[k])
        @constraint(omp_model, w[k] <= δ[k])
        @constraint(omp_model, w[k] >= δ[k] - (1 - x[k]))
    end

    # SB-link: t₀ ≥ Σ_k [ξ̄^min_k (δ_k - max_s(v_k^s) w_k) + α_k δ_k]
    v_max = vec(maximum(td.v, dims=2))  # worst-case v per arc (valid lower bound)
    link_expr = sum(
        ξ_min[k] * (δ[k] - v_max[k] * w[k]) + α_val[k] * δ[k]
        for k in 1:K
    )
    @constraint(omp_model, t_0 >= link_expr)

    return nothing
end
