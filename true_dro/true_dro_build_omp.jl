"""
true_dro_build_omp.jl — Outer Master Problem (OMP) for True-DRO-Exact.

True-DRO에서 OMP 변수는 (x, t₀) 만. h, λ, ψ⁰는 모두 ISP-F primal에 흡수됨
(true_dro_v5.md §9.1).

  min  t₀
  s.t. x ∈ X = {x ∈ {0,1}^|A| : 1ᵀx ≤ γ}
       t₀ ≥ Z₀(x̄_iter) + Σ_k π_{x_k}(x_k - x̄_k_iter)   (optimality cuts)

Cut slope (true_dro_v5.md §9.3):
  π_{x_k} = -φ̂^U Σ_s ρ̂¹_{k,s} + φ̂^U Σ_s ρ̂³_{k,s}
            -φ̃^U Σ_s ρ̃¹_{k,s} + φ̃^U Σ_s ρ̃³_{k,s}
            -λ^U ρ⁰¹_k + λ^U ρ⁰³_k
"""

using JuMP
using LinearAlgebra


"""
    build_true_dro_omp(td::TrueDROData; optimizer)

Build OMP MILP. Returns (model, vars).
"""
function build_true_dro_omp(td::TrueDROData; optimizer)
    K = td.num_arcs
    γ = td.gamma

    model = Model(optimizer_with_attributes(optimizer, MOI.Silent() => true))

    @variable(model, t_0 >= -1e+4)
    @variable(model, x[1:K], Bin)

    # Interdiction budget
    @constraint(model, sum(x[k] for k in 1:K) <= γ)

    # Fix non-interdictable arcs to 0
    for k in 1:K
        if !td.interdictable_arcs[k]
            @constraint(model, x[k] == 0)
        end
    end

    @objective(model, Min, t_0)

    vars = Dict(:t_0 => t_0, :x => x)
    return model, vars
end


"""
    compute_true_dro_outer_cut(td::TrueDROData, sub_info, x_sol)

Compute outer cut coefficients from converged subproblem.

# Inputs
- `sub_info`: Dict from `solve_true_dro_subproblem!` containing
  - :Z0_val
  - :rho_hat_1_val, :rho_hat_3_val   (K × S)
  - :rho_tilde_1_val, :rho_tilde_3_val  (K × S)
  - :rho_psi0_1_val, :rho_psi0_3_val (K)
- `x_sol`: current OMP x̄

Returns Dict with :intercept, :π_x, :Z0_val.
"""
function compute_true_dro_outer_cut(td::TrueDROData, sub_info, x_sol)
    S = td.S
    K = td.num_arcs
    φ̂U = td.phi_hat_U
    φ̃U = td.phi_tilde_U
    λU = td.lambda_U

    ρ̂1 = sub_info[:rho_hat_1_val]
    ρ̂3 = sub_info[:rho_hat_3_val]
    ρ̃1 = sub_info[:rho_tilde_1_val]
    ρ̃3 = sub_info[:rho_tilde_3_val]
    ρ01 = sub_info[:rho_psi0_1_val]
    ρ03 = sub_info[:rho_psi0_3_val]
    Z0 = sub_info[:Z0_val]

    # π_{x_k} = -φ̂U·Σ_s ρ̂¹ + φ̂U·Σ_s ρ̂³  -φ̃U·Σ_s ρ̃¹ + φ̃U·Σ_s ρ̃³  -λU·ρ⁰¹ + λU·ρ⁰³
    π_x = [(-φ̂U * sum(ρ̂1[k, s] for s in 1:S) + φ̂U * sum(ρ̂3[k, s] for s in 1:S)
            -φ̃U * sum(ρ̃1[k, s] for s in 1:S) + φ̃U * sum(ρ̃3[k, s] for s in 1:S)
            -λU * ρ01[k] + λU * ρ03[k]) for k in 1:K]

    intercept = Z0 - dot(π_x, x_sol)

    return Dict(:intercept => intercept, :π_x => π_x, :Z0_val => Z0)
end


"""
    add_true_dro_optimality_cut!(omp_model, omp_vars, outer_cut, iter)

Add `t₀ ≥ intercept + π_x'·x` to OMP.
"""
function add_true_dro_optimality_cut!(omp_model, omp_vars, outer_cut, iter)
    K = length(omp_vars[:x])
    x = omp_vars[:x]
    t_0 = omp_vars[:t_0]
    intercept = outer_cut[:intercept]
    π_x = outer_cut[:π_x]

    cut_expr = intercept + sum(π_x[k] * x[k] for k in 1:K)
    c = @constraint(omp_model, t_0 >= cut_expr)
    set_name(c, "true_dro_opt_cut_$iter")
    return c
end
