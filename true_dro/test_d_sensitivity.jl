"""
test_d_sensitivity.jl — d 고정 실험.
grid3x3, S=3, ε̂=ε̃=0.2, β=0.95, x*=[3,8].

(a) d = q̂ (center) → α₁*, Z₁
(b) d = TV boundary point → α₂*, Z₂
+ (b)에서 α₁*를 강제했을 때 obj_F 확인
"""

using JuMP, Gurobi, Printf, LinearAlgebra, Random

include("../network_generator.jl")
NG = NetworkGenerator
include("true_dro_data.jl")
include("true_dro_build_omp.jl")
include("true_dro_build_subproblem.jl")

net = NG.generate_grid_network(3, 3; seed=42)
num_arcs = length(net.arcs) - 1
S = 3; γ = 2
caps, _ = NG.generate_capacity_scenarios_uniform_model(length(net.arcs), S;
    interdictable_arcs=net.interdictable_arcs, seed=42)
intd_idx = findall(net.interdictable_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
q_hat = fill(1.0/S, S)

# x 고정 (이전 optimal)
x_bar = zeros(num_arcs)
x_bar[3] = 1.0; x_bar[8] = 1.0

# ── λU sweep ──
for λU in [2.0, 10.0, 50.0]

td = make_true_dro_data(net, caps, q_hat, 0.2, 0.2; w=w, lambda_U=λU, gamma=γ, beta=0.95)

println("\n" * "#"^70)
@printf("λU=%.1f, φ̂U=%.1f, φ̃U=%.1f\n", λU, td.phi_hat_U, td.phi_tilde_U)
@printf("grid3x3: K=%d, S=%d, w=%.4f, β=%.2f, ε̂=%.2f, ε̃=%.2f\n",
    num_arcs, S, w, td.beta, td.eps_hat, td.eps_tilde)
println("x = $(findall(x_bar .> 0.5))")
println("q̂ = $q_hat")

# ── Helper: subproblem에서 d를 고정하고 풀기, obj_L/obj_F 분리 ──
function solve_with_fixed_d(td, x_bar, d_fixed; fix_alpha=nothing)
    sub_model, sub_vars = build_true_dro_subproblem(td, x_bar;
        optimizer=Gurobi.Optimizer, silent=true)
    set_optimizer_attribute(sub_model, "NonConvex", 2)

    # d 고정
    for s in 1:td.S
        JuMP.fix(sub_vars[:d][s], d_fixed[s]; force=true)
    end

    # α 고정 (옵션)
    if fix_alpha !== nothing
        for k in 1:td.num_arcs
            JuMP.fix(sub_vars[:α][k], fix_alpha[k]; force=true)
        end
    end

    JuMP.optimize!(sub_model)
    st = termination_status(sub_model)
    if st != MOI.OPTIMAL
        @printf("  ⚠ status = %s\n", st)
        return nothing
    end

    K = td.num_arcs; S = td.S
    φ̂U = td.phi_hat_U; φ̃U = td.phi_tilde_U; λU = td.lambda_U

    σ̂ = [value(sub_vars[:σ_hat][s]) for s in 1:S]
    ρ̂1 = [value(sub_vars[:ρ_hat_1][k,s]) for k in 1:K, s in 1:S]
    ρ̂3 = [value(sub_vars[:ρ_hat_3][k,s]) for k in 1:K, s in 1:S]
    ρ̃1 = [value(sub_vars[:ρ_tilde_1][k,s]) for k in 1:K, s in 1:S]
    ρ̃3 = [value(sub_vars[:ρ_tilde_3][k,s]) for k in 1:K, s in 1:S]
    ρ⁰1 = [value(sub_vars[:ρ_psi0_1][k]) for k in 1:K]
    ρ⁰3 = [value(sub_vars[:ρ_psi0_3][k]) for k in 1:K]
    α_val = [value(sub_vars[:α][k]) for k in 1:K]
    a_val = [value(sub_vars[:a][s]) for s in 1:S]

    obj_L = sum(σ̂) -
            φ̂U * sum(x_bar[k] * ρ̂1[k,s] for k in 1:K, s in 1:S) -
            φ̂U * sum((1-x_bar[k]) * ρ̂3[k,s] for k in 1:K, s in 1:S)

    obj_F = -φ̃U * sum(x_bar[k] * ρ̃1[k,s] for k in 1:K, s in 1:S) -
             φ̃U * sum((1-x_bar[k]) * ρ̃3[k,s] for k in 1:K, s in 1:S) -
             λU * sum(x_bar[k] * ρ⁰1[k] for k in 1:K) -
             λU * sum((1-x_bar[k]) * ρ⁰3[k] for k in 1:K)

    Z0 = objective_value(sub_model)
    return (Z0=Z0, obj_L=obj_L, obj_F=obj_F, α=α_val, a=a_val)
end

# ── (a) d = q̂ (center) ──
println("\n" * "="^70)
println("(a) d = q̂ (center)")
println("="^70)
d_center = copy(q_hat)
res_a = solve_with_fixed_d(td, x_bar, d_center)
@printf("  Z₁=%.6f  obj_L=%.6f  obj_F=%.6f\n", res_a.Z0, res_a.obj_L, res_a.obj_F)
α1 = res_a.α
nz1 = findall(abs.(α1) .> 1e-6)
@printf("  α₁* nonzero: %s\n", ["k$(k)=$(round(α1[k];digits=4))" for k in nz1])
@printf("  a = %s\n", [round(v;digits=4) for v in res_a.a])

# ── (b) d = TV boundary point ──
# worst scenario 방향: scenario 1에 mass를 최대한 이동
println("\n" * "="^70)
println("(b) d = TV boundary (mass shift to scenario 1)")
println("="^70)
ε̃ = td.eps_tilde
d_boundary = copy(q_hat)
# scenario 1에 +ε̃, 나머지에서 -ε̃/2씩 (Σd=1, ||d-q̂||₁ = 2ε̃)
d_boundary[1] += ε̃
d_boundary[2] -= ε̃/2
d_boundary[3] -= ε̃/2
@printf("  d_boundary = %s  (||d-q̂||₁=%.4f)\n",
    [round(v;digits=4) for v in d_boundary], sum(abs.(d_boundary .- q_hat)))

# (b-1) α free
println("\n--- (b-1) α FREE ---")
res_b1 = solve_with_fixed_d(td, x_bar, d_boundary)
@printf("  Z₂=%.6f  obj_L=%.6f  obj_F=%.6f\n", res_b1.Z0, res_b1.obj_L, res_b1.obj_F)
α2 = res_b1.α
nz2 = findall(abs.(α2) .> 1e-6)
@printf("  α₂* nonzero: %s\n", ["k$(k)=$(round(α2[k];digits=4))" for k in nz2])
@printf("  a = %s\n", [round(v;digits=4) for v in res_b1.a])

# (b-2) α₁* 강제
println("\n--- (b-2) α = α₁* FORCED ---")
res_b2 = solve_with_fixed_d(td, x_bar, d_boundary; fix_alpha=α1)
@printf("  Z=%.6f  obj_L=%.6f  obj_F=%.6f\n", res_b2.Z0, res_b2.obj_L, res_b2.obj_F)

# ── Summary ──
println("\n" * "="^70)
println("SUMMARY")
println("="^70)
α_diff = sum(abs.(α1 .- α2))
@printf("  Z₁ (d=q̂)       = %.6f\n", res_a.Z0)
@printf("  Z₂ (d=boundary) = %.6f\n", res_b1.Z0)
@printf("  Z gap            = %.6f\n", res_b1.Z0 - res_a.Z0)
@printf("  ||α₁-α₂||₁      = %.6f  → %s\n", α_diff, α_diff < 1e-4 ? "α SAME" : "α DIFFERENT")
@printf("  obj_F(b-1, α free)   = %.6f\n", res_b1.obj_F)
@printf("  obj_F(b-2, α=α₁*)   = %.6f  → %s\n", res_b2.obj_F,
    abs(res_b2.obj_F) < 1e-4 ? "α₁* admissible at d_boundary" : "α₁* NOT admissible at d_boundary")

if res_b1.Z0 > res_a.Z0 + 1e-4
    println("\n  ⚠ Z₂ > Z₁ → free solver가 d*=q̂로 stuck했을 가능성 (코드 버그 의심)")
elseif α_diff < 1e-4
    println("\n  → d-invariant (saturation). Instance/β 변경 필요.")
else
    println("\n  → d-sensitive! α가 d에 따라 변함.")
end

end  # λU sweep
