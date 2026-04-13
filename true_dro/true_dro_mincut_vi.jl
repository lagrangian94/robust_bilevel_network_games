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
Source (removed from Ny) → tail/head = 0.  Sink = nv1 (last row).
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
        @constraint(omp_model, τ[nv1, j] == 1)

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

    # DC-link: t₀ ≥ Σ_j q̂_j Σ_k ξ̄^j_k (δ^j_k - v_k w^j_k)
    link_expr = sum(
        td.q_hat[j] * sum(td.xi_bar[k, j] * (δ[k, j] - td.v[k] * w[k, j]) for k in 1:K)
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
    @constraint(omp_model, τ[nv1] == 1)

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

    # SB-link: t₀ ≥ Σ_k [ξ̄^min_k (δ_k - v_k w_k) + α_k δ_k]
    link_expr = sum(
        ξ_min[k] * (δ[k] - td.v[k] * w[k]) + α_val[k] * δ[k]
        for k in 1:K
    )
    @constraint(omp_model, t_0 >= link_expr)

    return nothing
end
