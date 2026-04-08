"""
tv_build_imp.jl — Inner Master Problem (IMP) for TV-DRO.

IMP coordinates α_k between leader/follower ISPs.

Wasserstein과 차이: per-scenario t_1_l[s], t_1_f[s] 대신 단일 θ^L, θ^F.
TV ISP가 전 시나리오를 통합 처리하므로 single cut variable로 충분.

  max  θ^L + θ^F
  s.t. Σ_k α_k ≤ w,  α_k ≥ 0
       θ^L ≤ (leader Benders cuts)
       θ^F ≤ (follower Benders cuts)
"""

using JuMP
using LinearAlgebra


"""
    build_tv_imp(tv::TVData; optimizer)

Build IMP LP.

# Returns
- `(model, vars)` with vars[:α], vars[:θ_L], vars[:θ_F]
"""
function build_tv_imp(tv::TVData; optimizer)
    K = tv.num_arcs
    w = tv.w

    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => true))

    @variable(model, α[1:K] >= 0)
    @variable(model, θ_L)  # free (leader value approximation)
    @variable(model, θ_F)  # free (follower value approximation)

    # Budget constraint: Σ α ≤ w, individual bound α_k ≤ w
    @constraint(model, sum(α[k] for k in 1:K) <= w)
    @constraint(model, [k=1:K], α[k] <= w)

    # Objective: max θ^L + θ^F
    @objective(model, Max, θ_L + θ_F)

    vars = Dict(:α => α, :θ_L => θ_L, :θ_F => θ_F)
    return model, vars
end


"""
    add_tv_inner_cut_leader!(imp_model, imp_vars, cut_info, iter)

Add leader Benders cut to IMP:
  θ^L ≤ intercept + subgradient' * α
"""
function add_tv_inner_cut_leader!(imp_model, imp_vars, cut_info, iter)
    α = imp_vars[:α]
    K = length(α)
    intercept = cut_info[:intercept]
    sg = cut_info[:subgradient]

    c = @constraint(imp_model,
        imp_vars[:θ_L] <= intercept + sum(sg[k] * α[k] for k in 1:K))
    set_name(c, "isp_l_cut_$iter")
    return c
end


"""
    add_tv_inner_cut_follower!(imp_model, imp_vars, cut_info, iter)

Add follower Benders cut to IMP:
  θ^F ≤ intercept + subgradient' * α
"""
function add_tv_inner_cut_follower!(imp_model, imp_vars, cut_info, iter)
    α = imp_vars[:α]
    K = length(α)
    intercept = cut_info[:intercept]
    sg = cut_info[:subgradient]

    c = @constraint(imp_model,
        imp_vars[:θ_F] <= intercept + sum(sg[k] * α[k] for k in 1:K))
    set_name(c, "isp_f_cut_$iter")
    return c
end
