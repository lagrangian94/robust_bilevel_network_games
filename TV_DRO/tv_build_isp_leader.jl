"""
tv_build_isp_leader.jl — Leader Inner Subproblem (ISP-L) for TV-DRO.

All-scenario pooled LP. Constraints L1-L11 from tv_derivation.md §9.3.

Given α_k from IMP, c_k^s from OMP:
  Z^L = max  Σ_s σ̂^s
  s.t. L1-L11

Sensitivity ∂Z^L/∂α_k comes from dual values of L7-L10 (α appears in RHS).
"""

using JuMP
using LinearAlgebra


"""
    build_tv_isp_leader(tv::TVData, c::Matrix{Float64}; optimizer)

Build leader ISP LP (L1-L11). All scenarios pooled.

# Arguments
- `tv`: TVData struct
- `c`: |A| × S matrix, c_k^s = ξ̄_k^s(1 - v_k x_k)
- `optimizer`: LP solver (e.g., HiGHS.Optimizer)

# Returns
- `(model, vars)` where vars contains all variables and named constraints
"""
function build_tv_isp_leader(tv::TVData, c::Matrix{Float64}; optimizer)
    S = tv.S
    K = tv.num_arcs
    m = tv.nv1
    Ny = tv.Ny
    Nts = tv.Nts
    q = tv.q_hat
    ε = tv.eps_hat

    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => true))

    # --- Variables (L11) ---
    @variable(model, σ_hat[1:S] >= 0)
    @variable(model, u_hat[1:K, 1:S] >= 0)
    @variable(model, a[1:S] >= 0)
    @variable(model, b[1:S] >= 0)
    @variable(model, a_nu[1:S, 1:K] >= 0)
    @variable(model, b_nu[1:S, 1:K] >= 0)

    # --- (L1): N_y û^s + N_ts σ̂^s ≤ 0,  ∀s ---
    @constraint(model, L1[j=1:m, s=1:S],
        sum(Ny[j, k] * u_hat[k, s] for k in 1:K) + Nts[j] * σ_hat[s] <= 0)

    # --- (L2): -c_k^s a_s + û_k^s - a_{s,k}^ν ≤ 0,  ∀k,s ---
    @constraint(model, L2[k=1:K, s=1:S],
        -c[k, s] * a[s] + u_hat[k, s] - a_nu[s, k] <= 0)

    # --- (L3): a_s - b_s ≤ q̂_s,  ∀s ---
    @constraint(model, L3[s=1:S], a[s] - b[s] <= q[s])

    # --- (L4): a_s + b_s ≥ q̂_s,  ∀s ---
    @constraint(model, L4[s=1:S], a[s] + b[s] >= q[s])

    # --- (L5): Σ_s b_s ≤ 2ε̂ ---
    @constraint(model, L5, sum(b[s] for s in 1:S) <= 2 * ε)

    # --- (L6): Σ_s a_s = 1 ---
    @constraint(model, L6, sum(a[s] for s in 1:S) == 1)

    # --- (L7): a_{s,k}^ν - b_{s,k}^ν ≤ q̂_s α_k,  ∀s,k ---
    # RHS = q̂_s · α_k, initially α=0 → RHS=0
    @constraint(model, L7[s=1:S, k=1:K], a_nu[s, k] - b_nu[s, k] <= 0)

    # --- (L8): a_{s,k}^ν + b_{s,k}^ν ≥ q̂_s α_k,  ∀s,k ---
    @constraint(model, L8[s=1:S, k=1:K], a_nu[s, k] + b_nu[s, k] >= 0)

    # --- (L9): Σ_s b_{s,k}^ν ≤ 2ε̂ α_k,  ∀k ---
    @constraint(model, L9[k=1:K], sum(b_nu[s, k] for s in 1:S) <= 0)

    # --- (L10): Σ_s a_{s,k}^ν = α_k,  ∀k ---
    @constraint(model, L10[k=1:K], sum(a_nu[s, k] for s in 1:S) == 0)

    # --- Objective: max Σ_s σ̂^s ---
    @objective(model, Max, sum(σ_hat[s] for s in 1:S))

    vars = Dict(
        :σ_hat => σ_hat, :u_hat => u_hat,
        :a => a, :b => b, :a_nu => a_nu, :b_nu => b_nu,
        :L7 => L7, :L8 => L8, :L9 => L9, :L10 => L10,
        :L2 => L2,
    )
    return model, vars
end


"""
    update_tv_isp_leader_alpha!(model, vars, tv::TVData, α_sol::Vector{Float64})

Update RHS of L7-L10 for new α from IMP.
"""
function update_tv_isp_leader_alpha!(model, vars, tv::TVData, α_sol::Vector{Float64})
    S = tv.S
    K = tv.num_arcs
    q = tv.q_hat
    ε = tv.eps_hat

    for s in 1:S, k in 1:K
        set_normalized_rhs(vars[:L7][s, k], q[s] * α_sol[k])
        set_normalized_rhs(vars[:L8][s, k], q[s] * α_sol[k])
    end
    for k in 1:K
        set_normalized_rhs(vars[:L9][k], 2 * ε * α_sol[k])
        set_normalized_rhs(vars[:L10][k], α_sol[k])
    end
end


"""
    update_tv_isp_leader_outer!(model, vars, tv::TVData, c_new::Matrix{Float64})

Update L2 coefficients when c_k^s changes (new x from OMP).
Rebuild approach: delete old L2, add new. (LP → cheap)
"""
function update_tv_isp_leader_outer!(model, vars, tv::TVData, c_new::Matrix{Float64})
    S = tv.S
    K = tv.num_arcs

    # Delete old L2 constraints
    for k in 1:K, s in 1:S
        delete(model, vars[:L2][k, s])
    end

    # Add new L2 with updated c
    a = vars[:a]
    u_hat = vars[:u_hat]
    a_nu = vars[:a_nu]
    L2_new = @constraint(model, [k=1:K, s=1:S],
        -c_new[k, s] * a[s] + u_hat[k, s] - a_nu[s, k] <= 0)
    vars[:L2] = L2_new
end


"""
    tv_isp_leader_optimize!(model, vars, tv::TVData, α_sol::Vector{Float64})

Update α, solve ISP-L, extract inner cut info.

Returns (status, cut_info) where cut_info contains:
- `:obj_val`: Z^L*
- `:subgradient`: ∂Z^L/∂α (from dual of L7-L10)
- `:intercept`: Z^L* - subgradient' * α_sol
"""
function tv_isp_leader_optimize!(model, vars, tv::TVData, α_sol::Vector{Float64})
    S = tv.S
    K = tv.num_arcs
    q = tv.q_hat
    ε = tv.eps_hat

    # Update α in RHS
    update_tv_isp_leader_alpha!(model, vars, tv, α_sol)

    optimize!(model)
    st = termination_status(model)
    if st != MOI.OPTIMAL
        error("TV ISP-L not optimal: $st")
    end

    obj_val = objective_value(model)

    # Extract subgradient ∂Z^L/∂α_k from dual values of L7-L10
    # L7: a_nu - b_nu ≤ q_s α_k  →  dual ≥ 0 (≤ constraint in max)
    # L8: a_nu + b_nu ≥ q_s α_k  →  dual ≥ 0 (≥ constraint in max)
    # L9: Σ_s b_nu ≤ 2ε α_k      →  dual ≥ 0
    # L10: Σ_s a_nu = α_k         →  dual free
    #
    # ∂Z^L/∂α_k = Σ_s q_s · shadow(L7[s,k]) + Σ_s q_s · shadow(L8[s,k])
    #            + 2ε · shadow(L9[k]) + shadow(L10[k])
    #
    # Note: JuMP shadow_price convention: for max problem,
    #   ≤ constraint → shadow_price ≤ 0 (∂obj/∂RHS ≤ 0 wouldn't make sense)
    #   Actually shadow_price = ∂obj/∂RHS for the appropriate sign convention.
    #   For max with ≤: shadow_price = dual value (≥ 0 for max)
    #   For max with ≥: shadow_price = dual value (≤ 0... hmm)
    #
    # Use dual() instead to get raw dual values, then multiply by RHS coefficient of α.
    # shadow_price in JuMP: returns the change in obj per unit increase in RHS.

    subgradient = zeros(K)
    for k in 1:K
        sg_k = 0.0
        for s in 1:S
            sg_k += q[s] * shadow_price(vars[:L7][s, k])  # RHS coeff of α_k in L7
            sg_k += q[s] * shadow_price(vars[:L8][s, k])  # RHS coeff of α_k in L8
        end
        sg_k += 2 * ε * shadow_price(vars[:L9][k])        # RHS coeff of α_k in L9
        sg_k += 1.0 * shadow_price(vars[:L10][k])          # RHS coeff of α_k in L10
        subgradient[k] = sg_k
    end

    intercept = obj_val - dot(subgradient, α_sol)

    cut_info = Dict(
        :obj_val => obj_val,
        :subgradient => subgradient,
        :intercept => intercept,
    )
    return :OptimalityCut, cut_info
end
