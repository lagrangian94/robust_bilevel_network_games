using JuMP, Gurobi, Printf, LinearAlgebra
include("../network_generator.jl"); NG = NetworkGenerator
include("true_dro_data.jl"); include("true_dro_build_omp.jl"); include("true_dro_build_subproblem.jl")

net = NG.generate_grid_network(3, 3; seed=42)
num_arcs = length(net.arcs) - 1; S = 3; γ = 2
caps, _ = NG.generate_capacity_scenarios_uniform_model(length(net.arcs), S;
    interdictable_arcs=net.interdictable_arcs, seed=42)
intd_idx = findall(net.interdictable_arcs[1:num_arcs])
w = round(maximum(caps[intd_idx, :]); digits=4)
q_hat = fill(1.0/S, S)
λU = 50.0

td = make_true_dro_data(net, caps, q_hat, 0.2, 0.2; w=w, lambda_U=λU, gamma=γ, beta=0.95)
x_bar = zeros(num_arcs); x_bar[3] = 1.0; x_bar[8] = 1.0
K = num_arcs

function solve_and_extract(td, x_bar; force_a_eq_d=false)
    m, v = build_true_dro_subproblem(td, x_bar; optimizer=Gurobi.Optimizer, silent=true)
    set_optimizer_attribute(m, "NonConvex", 2)
    if force_a_eq_d
        for s in 1:td.S; @constraint(m, v[:a][s] == v[:d][s]); end
    end
    JuMP.optimize!(m)
    
    K = td.num_arcs; S = td.S
    res = Dict{Symbol, Any}()
    res[:Z0] = objective_value(m)
    res[:a] = [value(v[:a][s]) for s in 1:S]
    res[:d] = [value(v[:d][s]) for s in 1:S]
    res[:r] = [value(v[:r][s]) for s in 1:S]
    res[:α] = [value(v[:α][k]) for k in 1:K]
    res[:σ_hat] = [value(v[:σ_hat][s]) for s in 1:S]
    res[:σ_tilde] = [value(v[:σ_tilde][s]) for s in 1:S]
    res[:ζL] = [value(v[:ζL][k,s]) for k in 1:K, s in 1:S]
    res[:ζF] = [value(v[:ζF][k,s]) for k in 1:K, s in 1:S]
    res[:ρ_hat_1] = [value(v[:ρ_hat_1][k,s]) for k in 1:K, s in 1:S]
    res[:ρ_hat_2] = [value(v[:ρ_hat_2][k,s]) for k in 1:K, s in 1:S]
    res[:ρ_hat_3] = [value(v[:ρ_hat_3][k,s]) for k in 1:K, s in 1:S]
    res[:ρ_tilde_1] = [value(v[:ρ_tilde_1][k,s]) for k in 1:K, s in 1:S]
    res[:ρ_tilde_2] = [value(v[:ρ_tilde_2][k,s]) for k in 1:K, s in 1:S]
    res[:ρ_tilde_3] = [value(v[:ρ_tilde_3][k,s]) for k in 1:K, s in 1:S]
    res[:ρ_psi0_1] = [value(v[:ρ_psi0_1][k]) for k in 1:K]
    res[:ρ_psi0_2] = [value(v[:ρ_psi0_2][k]) for k in 1:K]
    res[:ρ_psi0_3] = [value(v[:ρ_psi0_3][k]) for k in 1:K]
    res[:u_hat] = [value(v[:u_hat][k,s]) for k in 1:K, s in 1:S]
    res[:u_tilde] = [value(v[:u_tilde][k,s]) for k in 1:K, s in 1:S]
    res[:b] = [value(v[:b][s]) for s in 1:S]
    res[:e] = [value(v[:e][s]) for s in 1:S]
    return res
end

free = solve_and_extract(td, x_bar; force_a_eq_d=false)
tied = solve_and_extract(td, x_bar; force_a_eq_d=true)

φ̂U = td.phi_hat_U; φ̃U = td.phi_tilde_U

println("="^80)
@printf("λU=%.0f, φ̂U=%.1f, φ̃U=%.1f, β=%.2f\n", λU, φ̂U, φ̃U, td.beta)
@printf("Z₀(free)=%.6f  Z₀(tied)=%.6f\n", free[:Z0], tied[:Z0])
println("="^80)

# ── 분포 비교 ──
println("\n--- Distributions ---")
@printf("%5s  %10s %10s %10s | %10s %10s %10s\n", "s", "a(free)", "d(free)", "r(free)", "a(tied)", "d(tied)", "r(tied)")
for s in 1:S
    @printf("%5d  %10.6f %10.6f %10.6f | %10.6f %10.6f %10.6f\n",
        s, free[:a][s], free[:d][s], free[:r][s], tied[:a][s], tied[:d][s], tied[:r][s])
end
@printf("%5s  %10.6f %10.6f %10.6f | %10.6f %10.6f %10.6f\n",
    "Σ", sum(free[:a]), sum(free[:d]), sum(free[:r]), sum(tied[:a]), sum(tied[:d]), sum(tied[:r]))
@printf("||a-d||₁ free=%.6f  tied=%.6f\n", sum(abs.(free[:a].-free[:d])), sum(abs.(tied[:a].-tied[:d])))

# ── α 비교 (nonzero만) ──
println("\n--- α (nonzero) ---")
@printf("%5s  %10s %10s\n", "k", "α(free)", "α(tied)")
for k in 1:K
    if abs(free[:α][k]) > 1e-6 || abs(tied[:α][k]) > 1e-6
        @printf("%5d  %10.4f %10.4f\n", k, free[:α][k], tied[:α][k])
    end
end
@printf("%5s  %10.4f %10.4f\n", "Σ", sum(free[:α]), sum(tied[:α]))

# ── per-scenario obj 분해 ──
println("\n--- Per-scenario obj_L contribution (Σ_k terms) ---")
@printf("%5s  %12s %12s | %12s %12s\n", "s", "σ̂(free)", "σ̂(tied)", "Σρ̂₁x(free)", "Σρ̂₁x(tied)")
for s in 1:S
    rho1x_f = sum(x_bar[k] * free[:ρ_hat_1][k,s] for k in 1:K)
    rho3x_f = sum((1-x_bar[k]) * free[:ρ_hat_3][k,s] for k in 1:K)
    rho1x_t = sum(x_bar[k] * tied[:ρ_hat_1][k,s] for k in 1:K)
    rho3x_t = sum((1-x_bar[k]) * tied[:ρ_hat_3][k,s] for k in 1:K)
    @printf("%5d  %12.6f %12.6f | %12.6f %12.6f\n", s,
        free[:σ_hat][s], tied[:σ_hat][s], rho1x_f+rho3x_f, rho1x_t+rho3x_t)
end

println("\n--- Per-scenario obj_F contribution (Σ_k terms) ---")
@printf("%5s  %12s %12s\n", "s", "ρ̃ terms(free)", "ρ̃ terms(tied)")
for s in 1:S
    rho_f = φ̃U * sum(x_bar[k]*free[:ρ_tilde_1][k,s] + (1-x_bar[k])*free[:ρ_tilde_3][k,s] for k in 1:K)
    rho_t = φ̃U * sum(x_bar[k]*tied[:ρ_tilde_1][k,s] + (1-x_bar[k])*tied[:ρ_tilde_3][k,s] for k in 1:K)
    @printf("%5d  %12.6f %12.6f\n", s, -rho_f, -rho_t)
end
rho0_f = λU * sum(x_bar[k]*free[:ρ_psi0_1][k] + (1-x_bar[k])*free[:ρ_psi0_3][k] for k in 1:K)
rho0_t = λU * sum(x_bar[k]*tied[:ρ_psi0_1][k] + (1-x_bar[k])*tied[:ρ_psi0_3][k] for k in 1:K)
@printf("%5s  %12.6f %12.6f\n", "ρ⁰", -rho0_f, -rho0_t)

# ── ζL, ζF 비교 (bilinear terms) ──
println("\n--- ζL = α·r (leader bilinear) per-scenario sum ---")
@printf("%5s  %12s %12s\n", "s", "Σ_k ζL(free)", "Σ_k ζL(tied)")
for s in 1:S
    @printf("%5d  %12.6f %12.6f\n", s, sum(free[:ζL][k,s] for k in 1:K), sum(tied[:ζL][k,s] for k in 1:K))
end

println("\n--- ζF = α·d (follower bilinear) per-scenario sum ---")
@printf("%5s  %12s %12s\n", "s", "Σ_k ζF(free)", "Σ_k ζF(tied)")
for s in 1:S
    @printf("%5d  %12.6f %12.6f\n", s, sum(free[:ζF][k,s] for k in 1:K), sum(tied[:ζF][k,s] for k in 1:K))
end

# ── obj_L, obj_F 총합 ──
obj_L_f = sum(free[:σ_hat]) - φ̂U*sum(x_bar[k]*free[:ρ_hat_1][k,s] for k in 1:K, s in 1:S) -
          φ̂U*sum((1-x_bar[k])*free[:ρ_hat_3][k,s] for k in 1:K, s in 1:S)
obj_F_f = -φ̃U*sum(x_bar[k]*free[:ρ_tilde_1][k,s] for k in 1:K, s in 1:S) -
           φ̃U*sum((1-x_bar[k])*free[:ρ_tilde_3][k,s] for k in 1:K, s in 1:S) -
           λU*sum(x_bar[k]*free[:ρ_psi0_1][k] for k in 1:K) -
           λU*sum((1-x_bar[k])*free[:ρ_psi0_3][k] for k in 1:K)
obj_L_t = sum(tied[:σ_hat]) - φ̂U*sum(x_bar[k]*tied[:ρ_hat_1][k,s] for k in 1:K, s in 1:S) -
          φ̂U*sum((1-x_bar[k])*tied[:ρ_hat_3][k,s] for k in 1:K, s in 1:S)
obj_F_t = -φ̃U*sum(x_bar[k]*tied[:ρ_tilde_1][k,s] for k in 1:K, s in 1:S) -
           φ̃U*sum((1-x_bar[k])*tied[:ρ_tilde_3][k,s] for k in 1:K, s in 1:S) -
           λU*sum(x_bar[k]*tied[:ρ_psi0_1][k] for k in 1:K) -
           λU*sum((1-x_bar[k])*tied[:ρ_psi0_3][k] for k in 1:K)

println("\n--- TOTAL ---")
@printf("  FREE: obj_L=%.6f  obj_F=%.6f  Z₀=%.6f\n", obj_L_f, obj_F_f, obj_L_f+obj_F_f)
@printf("  TIED: obj_L=%.6f  obj_F=%.6f  Z₀=%.6f\n", obj_L_t, obj_F_t, obj_L_t+obj_F_t)
