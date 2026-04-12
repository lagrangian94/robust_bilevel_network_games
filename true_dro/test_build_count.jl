"""
test_build_count.jl — Build subproblem and verify constraint counts.
"""

using Revise
using JuMP
using Gurobi

if !@isdefined(NetworkGenerator)
    include("../network_generator.jl")
end
using .NetworkGenerator

includet("true_dro_data.jl")
includet("true_dro_build_subproblem.jl")

# Small instance
m, n, S = 2, 2, 2
network = generate_grid_network(m, n; seed=42)
num_arcs_with_dummy = length(network.arcs)
scenarios, _ = generate_capacity_scenarios_uniform_model(num_arcs_with_dummy, S; seed=42)
q_hat = fill(1.0/S, S)

td = make_true_dro_data(network, scenarios, q_hat, 0.1, 0.1;
                        w=1.0, lambda_U=10.0, gamma=2)

K = td.num_arcs
mm = td.nv1
println("Network: $(m)x$(n), |A|=$K, |V|-1=$mm, S=$S")

# Build
x_init = zeros(K)
sub_model, sub_vars = build_true_dro_subproblem(td, x_init; optimizer=Gurobi.Optimizer)

# Counts
n_var = num_variables(sub_model)
n_quad_eq = num_constraints(sub_model, GenericQuadExpr{Float64,VariableRef}, MOI.EqualTo{Float64})
n_quad_le = num_constraints(sub_model, GenericQuadExpr{Float64,VariableRef}, MOI.LessThan{Float64})
n_quad_ge = num_constraints(sub_model, GenericQuadExpr{Float64,VariableRef}, MOI.GreaterThan{Float64})
n_aff_le = num_constraints(sub_model, GenericAffExpr{Float64,VariableRef}, MOI.LessThan{Float64})
n_aff_ge = num_constraints(sub_model, GenericAffExpr{Float64,VariableRef}, MOI.GreaterThan{Float64})
n_aff_eq = num_constraints(sub_model, GenericAffExpr{Float64,VariableRef}, MOI.EqualTo{Float64})

println("\n=== Variable / constraint counts ===")
println("Total variables:        $n_var")
println("Quadratic ==:           $n_quad_eq    (expected 2*K*S = $(2*K*S))")
println("Quadratic ≤:            $n_quad_le")
println("Quadratic ≥:            $n_quad_ge")
println("Affine ≤:               $n_aff_le")
println("Affine ≥:               $n_aff_ge")
println("Affine ==:              $n_aff_eq")

# Expected breakdowns:
# Variables:
#   α[K] + ζL[K*S] + ζF[K*S]
#   ISP-L: σ_hat[S] + u_hat[K*S] + a[S] + b[S] + ρ̂1/2/3[K*S each] = S+K*S+S+S+3*K*S
#   ISP-F: d[S] + e[S] + ũ[K*S] + σ̃[S] + ω[m*S] + β[K*S] + δ + ρ̃1/2/3[K*S each] + ρ⁰1/2/3[K each]
#         = S+S+K*S+S+m*S+K*S+1+3*K*S+3*K
expected_var = (K + 2*K*S) +                                  # α, ζL, ζF
               (S + K*S + 2*S + 3*K*S) +                       # ISP-L
               (2*S + K*S + S + mm*S + K*S + 1 + 3*K*S + 3*K)  # ISP-F
println("Expected variables:     $expected_var")

# Constraints (affine, excluding the Σα ≤ w):
# ISP-L:
#   DL1: m*S (==)
#   DL2: K*S (≤)
#   DL3: K*S (≤)
#   DL4: S (≤)
#   DL5: S (≥)
#   DL6: 1 (≤)
#   DL7: 1 (==)
# ISP-F:
#   DF1: S (≤)
#   DF2: S (≥)
#   DF3: 1 (≤)
#   DF4: 1 (==)
#   DF5: m*S (==)
#   DF6: K*S (≤)
#   DF7: K*S (≤)
#   DF8: K*S (≤)
#   DF9: S (≤)
#   DFh: K (≤)
#   DFlam: 1 (≥)
#   DFpsi: K (≥)
# α budget: 1 (≤)
exp_aff_eq = 2*mm*S + 1 + 1
exp_aff_le = K*S + K*S + S + 1 + S + 1 + K*S + K*S + K*S + S + K + 1
exp_aff_ge = S + S + 1 + K
println("\nExpected affine ==:     $exp_aff_eq  (DL1+DL7+DF4+DF5)")
println("Expected affine ≤:      $exp_aff_le")
println("Expected affine ≥:      $exp_aff_ge  (DL5+DF2+DFlam+DFpsi)")

# Quadratic eq: ζL_def (K*S) + ζF_def (K*S) = 2*K*S
println("\nMatch quadratic eq:     ", n_quad_eq == 2*K*S ? "✓" : "✗")
println("Match affine ==:        ", n_aff_eq == exp_aff_eq ? "✓" : "✗")
println("Match affine ≤:         ", n_aff_le == exp_aff_le ? "✓" : "✗")
println("Match affine ≥:         ", n_aff_ge == exp_aff_ge ? "✓" : "✗")
println("Match variables:        ", n_var == expected_var ? "✓" : "✗")
