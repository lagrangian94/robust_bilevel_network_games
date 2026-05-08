"""
test_objF_zero.jl — obj_F = 0 조건 검증.
α를 optimal이 아닌 임의 feasible 값으로 고정했을 때 obj_F가 여전히 0인지 확인.
grid3x3, S=3, ε̂=ε̃=0.2, β=0.95.
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

td = make_true_dro_data(net, caps, q_hat, 0.2, 0.2; w=w, lambda_U=2.0, gamma=γ, beta=0.95)

# x 고정
x_bar = zeros(num_arcs)
x_bar[3] = 1.0; x_bar[8] = 1.0  # 임의 interdiction

@printf("grid3x3: K=%d, S=%d, w=%.4f, β=%.2f\n", num_arcs, S, w, td.beta)
println("x = $(findall(x_bar .> 0.5))")

# ── Helper: subproblem 풀고 obj_L, obj_F 분리 반환 ──
function solve_and_decompose(td, x_bar; fix_alpha=nothing)
    sub_model, sub_vars = build_true_dro_subproblem(td, x_bar;
        optimizer=Gurobi.Optimizer, silent=true)
    set_optimizer_attribute(sub_model, "NonConvex", 2)

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

    K = td.num_arcs
    S = td.S
    φ̂U = td.phi_hat_U
    φ̃U = td.phi_tilde_U
    λU = td.lambda_U

    σ̂ = [value(sub_vars[:σ_hat][s]) for s in 1:S]
    ρ̂1 = [value(sub_vars[:ρ_hat_1][k,s]) for k in 1:K, s in 1:S]
    ρ̂3 = [value(sub_vars[:ρ_hat_3][k,s]) for k in 1:K, s in 1:S]
    ρ̃1 = [value(sub_vars[:ρ_tilde_1][k,s]) for k in 1:K, s in 1:S]
    ρ̃3 = [value(sub_vars[:ρ_tilde_3][k,s]) for k in 1:K, s in 1:S]
    ρ⁰1 = [value(sub_vars[:ρ_psi0_1][k]) for k in 1:K]
    ρ⁰3 = [value(sub_vars[:ρ_psi0_3][k]) for k in 1:K]
    α_val = [value(sub_vars[:α][k]) for k in 1:K]
    a_val = [value(sub_vars[:a][s]) for s in 1:S]
    d_val = [value(sub_vars[:d][s]) for s in 1:S]

    obj_L = sum(σ̂) -
            φ̂U * sum(x_bar[k] * ρ̂1[k,s] for k in 1:K, s in 1:S) -
            φ̂U * sum((1-x_bar[k]) * ρ̂3[k,s] for k in 1:K, s in 1:S)

    obj_F = -φ̃U * sum(x_bar[k] * ρ̃1[k,s] for k in 1:K, s in 1:S) -
             φ̃U * sum((1-x_bar[k]) * ρ̃3[k,s] for k in 1:K, s in 1:S) -
             λU * sum(x_bar[k] * ρ⁰1[k] for k in 1:K) -
             λU * sum((1-x_bar[k]) * ρ⁰3[k] for k in 1:K)

    Z0 = objective_value(sub_model)
    return (Z0=Z0, obj_L=obj_L, obj_F=obj_F, α=α_val, a=a_val, d=d_val)
end

# ── Case 1: α free (optimal) ──
println("\n" * "="^60)
println("Case 1: α FREE (optimal)")
println("="^60)
res_opt = solve_and_decompose(td, x_bar)
if res_opt !== nothing
    @printf("  Z₀=%.6f  obj_L=%.6f  obj_F=%.6f\n", res_opt.Z0, res_opt.obj_L, res_opt.obj_F)
    nz = findall(abs.(res_opt.α) .> 1e-6)
    @printf("  α nonzero: %s\n", ["k$(k)=$(round(res_opt.α[k];digits=4))" for k in nz])
end

# ── Case 2~5: α를 임의 feasible 값으로 고정 ──
Random.seed!(42)
for trial in 1:4
    # 랜덤 α: K개 중 1~3개에 랜덤 값, 합 ≤ w
    α_rand = zeros(num_arcs)
    n_nonzero = rand(1:3)
    chosen = sort(shuffle(1:num_arcs)[1:n_nonzero])
    raw = rand(n_nonzero)
    raw .*= (w * 0.8) / sum(raw)  # 합 = 0.8w (feasible, but suboptimal)
    for (i, k) in enumerate(chosen)
        α_rand[k] = raw[i]
    end

    println("\n" * "="^60)
    @printf("Case %d: α FIXED (random)\n", trial+1)
    println("="^60)
    nz = findall(abs.(α_rand) .> 1e-6)
    @printf("  fixed α: %s  (sum=%.4f, w=%.4f)\n",
        ["k$(k)=$(round(α_rand[k];digits=4))" for k in nz], sum(α_rand), w)

    res = solve_and_decompose(td, x_bar; fix_alpha=α_rand)
    if res !== nothing
        @printf("  Z₀=%.6f  obj_L=%.6f  obj_F=%.6f\n", res.Z0, res.obj_L, res.obj_F)
        @printf("  obj_F == 0?  %s  (|obj_F|=%.2e)\n",
            abs(res.obj_F) < 1e-4 ? "YES" : "NO", abs(res.obj_F))
    end
end
