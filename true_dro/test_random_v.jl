# test_random_v.jl — v_k^s ~ Bernoulli(0.75) per arc per scenario.
# Source-cut saturation 깨지는지 확인.
# grid3x3, S=3, λU=50, β=0.95, ε̂=ε̃=0.2.

using JuMP, Gurobi, Printf, LinearAlgebra, Random
include("../network_generator.jl"); NG = NetworkGenerator
include("true_dro_data.jl"); include("true_dro_build_omp.jl"); include("true_dro_build_subproblem.jl")

net = NG.generate_grid_network(3, 3; seed=42)
num_arcs = length(net.arcs) - 1; S = 7
caps, _ = NG.generate_capacity_scenarios_uniform_model(length(net.arcs), S;
    interdictable_arcs=net.interdictable_arcs, seed=42)
intd_idx = findall(net.interdictable_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
q_hat = fill(1.0/S, S)
K = num_arcs; λU = 50.0

# ── 모든 arc interdictable로 변경 ──
intd_all = fill(true, length(net.arcs))
net = NG.GridNetworkData(net.m, net.n, net.nodes, net.arcs, net.N,
    intd_all, net.arc_directions, net.arc_adjacency, net.node_arc_incidence)

# ── v_k^s ~ Bernoulli(0.75), 모든 arc ──
Random.seed!(77)
v_rand = zeros(K, S)
for k in 1:K
    for s in 1:S
        v_rand[k, s] = rand() < 0.75 ? 1.0 : 0.0
    end
end

println("--- v_k^s (interdictable arcs only) ---")
@printf("%5s", "k")
for s in 1:S; @printf("  s=%d", s); end
@printf("  arc\n")
for k in 1:K
    if net.interdictable_arcs[k]
        @printf("%5d", k)
        for s in 1:S; @printf("  %.0f  ", v_rand[k,s]); end
        @printf("  %s→%s\n", net.arcs[k][1], net.arcs[k][2])
    end
end

# γ sweep용 x 후보 (고정 seed로 선택)
Random.seed!(99)
x_pool = shuffle(1:K)  # arc 순서 랜덤
println("interdictable arcs: $(findall(net.interdictable_arcs[1:K]))")
println("x_pool order: $x_pool")

# (effective capacity 출력 생략 — x_bar는 γ sweep 내에서 정의)

function solve_gap(td, x_bar)
    # (1) free solve
    m1, v1 = build_true_dro_subproblem(td, x_bar; optimizer=Gurobi.Optimizer, silent=false)
    set_optimizer_attribute(m1, "NonConvex", 2); JuMP.optimize!(m1)
    Z_free = objective_value(m1)
    a_f = [value(v1[:a][s]) for s in 1:td.S]
    d_f = [value(v1[:d][s]) for s in 1:td.S]
    r_f = [value(v1[:r][s]) for s in 1:td.S]
    α_f = [value(v1[:α][k]) for k in 1:K]

    # # (2) α=α*, d=a* 고정 → obj_F 계산 (tied 대용) — 주석처리
    # m2, v2 = build_true_dro_subproblem(td, x_bar; optimizer=Gurobi.Optimizer, silent=true)
    # set_optimizer_attribute(m2, "NonConvex", 2)
    # for k in 1:K; JuMP.fix(v2[:α][k], α_f[k]; force=true); end
    # for s in 1:td.S; JuMP.fix(v2[:d][s], a_f[s]; force=true); end
    # JuMP.optimize!(m2)
    # Z_forced = objective_value(m2)
    #
    # S = td.S; K2 = td.num_arcs
    # φ̂U = td.phi_hat_U; φ̃U = td.phi_tilde_U; λU2 = td.lambda_U
    # ρ̃1 = [value(v2[:ρ_tilde_1][k,s]) for k in 1:K2, s in 1:S]
    # ρ̃3 = [value(v2[:ρ_tilde_3][k,s]) for k in 1:K2, s in 1:S]
    # ρ⁰1 = [value(v2[:ρ_psi0_1][k]) for k in 1:K2]
    # ρ⁰3 = [value(v2[:ρ_psi0_3][k]) for k in 1:K2]
    # obj_F_forced = -φ̃U*sum(x_bar[k]*ρ̃1[k,s] for k in 1:K2, s in 1:S) -
    #                 φ̃U*sum((1-x_bar[k])*ρ̃3[k,s] for k in 1:K2, s in 1:S) -
    #                 λU2*sum(x_bar[k]*ρ⁰1[k] for k in 1:K2) -
    #                 λU2*sum((1-x_bar[k])*ρ⁰3[k] for k in 1:K2)

    # (2) a=d 제약식 추가 (tied QCP)
    m2, v2 = build_true_dro_subproblem(td, x_bar; optimizer=Gurobi.Optimizer, silent=false)
    set_optimizer_attribute(m2, "NonConvex", 2)
    @constraint(m2, tied[s=1:td.S], v2[:a][s] == v2[:d][s])
    JuMP.optimize!(m2)
    Z_tied = objective_value(m2)
    a_t = [value(v2[:a][s]) for s in 1:td.S]
    d_t = [value(v2[:d][s]) for s in 1:td.S]
    α_t = [value(v2[:α][k]) for k in 1:K]

    return (Z_free=Z_free, Z_tied=Z_tied,
            a_f=a_f, d_f=d_f, r_f=r_f, α_f=α_f,
            a_t=a_t, d_t=d_t, α_t=α_t)
end

# ── γ=2 v-heterogeneity test ──
# (A) x=[1,11]: source + v-homogeneous (v=1 all s)
# (B) x=[1,9]:  source + v-heterogeneous (k9: v=[0,0,1,1,1,1,1])
# (C) x=[1,9,10,11,14]: γ=5 reference
for (label, x_list, γ_val) in [
    ("A: x=[1,11] v-homo", [1,11], 2),
    ("B: x=[1,9] v-hetero", [1,9], 2),
    ("C: x=[1,9,10,11,14] γ=5 ref", [1,9,10,11,14], 5),
]
γ = γ_val
x_bar = zeros(K)
for k in x_list; x_bar[k] = 1.0; end
x_arcs = findall(x_bar .> 0.5)

td = make_true_dro_data(net, caps, q_hat, 0.2, 0.5;
    w=w, lambda_U=λU, gamma=γ, beta=0.95, v_scenarios=v_rand)

println("\n" * "="^80)
@printf("γ=%d, x=%s, w=%.1f, λU=%.1f, β=0.95, S=%d\n", γ, x_arcs, w, λU, S)
println("="^80)

# (1) free solve
println("\n--- FREE (a≠d) ---")
m1, v1 = build_true_dro_subproblem(td, x_bar; optimizer=Gurobi.Optimizer, silent=false)
set_optimizer_attribute(m1, "NonConvex", 2)
JuMP.optimize!(m1)
Z_free = objective_value(m1)
a_f = [value(v1[:a][s]) for s in 1:S]
d_f = [value(v1[:d][s]) for s in 1:S]
α_f = [value(v1[:α][k]) for k in 1:K]
@printf("  Z_free = %.6f\n", Z_free)
@printf("  a = %s\n", [round(v; digits=4) for v in a_f])
@printf("  d = %s\n", [round(v; digits=4) for v in d_f])
@printf("  ||a-d||₁ = %.6f\n", sum(abs.(a_f .- d_f)))
α_nz = findall(abs.(α_f) .> 1e-6)
α_str = join(["k$(k)=$(round(α_f[k];digits=2))" for k in α_nz], ", ")
@printf("  α: [%s]  (sum=%.4f)\n", α_str, sum(α_f))

# (2) tied (a=d) — MIPGap=0.008
println("\n--- TIED (a=d), MIPGap=0.8% ---")
m2, v2 = build_true_dro_subproblem(td, x_bar; optimizer=Gurobi.Optimizer, silent=false)
set_optimizer_attribute(m2, "NonConvex", 2)
set_optimizer_attribute(m2, "MIPGap", 0.008)
@constraint(m2, tied[s=1:S], v2[:a][s] == v2[:d][s])
JuMP.optimize!(m2)
Z_tied = objective_value(m2)
best_bd = objective_bound(m2)
@printf("  Z_tied (incumbent) = %.6f\n", Z_tied)
@printf("  best bound         = %.6f\n", best_bd)
@printf("  MIP gap            = %.4f%%\n", 100.0 * abs(Z_tied - best_bd) / max(abs(Z_tied), 1e-10))
a_t = [value(v2[:a][s]) for s in 1:S]
α_t = [value(v2[:α][k]) for k in 1:K]
@printf("  a=d = %s\n", [round(v; digits=4) for v in a_t])
α_nz_t = findall(abs.(α_t) .> 1e-6)
α_str_t = join(["k$(k)=$(round(α_t[k];digits=2))" for k in α_nz_t], ", ")
@printf("  α: [%s]  (sum=%.4f)\n", α_str_t, sum(α_t))

# (3) comparison
println("\n--- COMPARISON ---")
@printf("  Z_free       = %.6f\n", Z_free)
@printf("  Z_tied (inc) = %.6f\n", Z_tied)
@printf("  Z_tied (bd)  = %.6f\n", best_bd)
@printf("  gap (free - tied_inc) = %.6f\n", Z_free - Z_tied)
if best_bd < Z_free - 0.1
    println("  → best_bound < Z_free → SATURATION BROKEN (a≠d matters)")
else
    println("  → best_bound ≈ Z_free → saturation holds")
end

# (4) v pattern + α on interdicted arcs
intd = findall(x_bar .> 0.5)
println("\n--- v pattern & α on interdicted arcs ---")
@printf("%5s %15s", "k", "arc")
for s in 1:S; @printf(" s%d", s); end
@printf("  α_free  α_tied\n")
for k in intd
    @printf("%5d %15s", k, "$(net.arcs[k][1])→$(net.arcs[k][2])")
    for s in 1:S; @printf("  %d", Int(v_rand[k,s])); end
    @printf("  %5.2f   %5.2f\n", α_f[k], α_t[k])
end

# (5) a, d, r
r_f = [value(v1[:r][s]) for s in 1:S]
@printf("\n  a_free = %s\n", [round(v; digits=3) for v in a_f])
@printf("  d_free = %s\n", [round(v; digits=3) for v in d_f])
@printf("  r_free = %s\n", [round(v; digits=3) for v in r_f])
@printf("  a_tied = %s\n", [round(v; digits=3) for v in a_t])

end  # for loop
