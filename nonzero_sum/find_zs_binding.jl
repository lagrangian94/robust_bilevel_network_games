#= find_zs_binding.jl — 결합이 binding 하는 작은 zero-sum 인스턴스를 KKT 로 빠르게 찾기 (x 몇 개만).
   원고 생성 방식 grid, 반경 (ε̂, ε̃), β, δ 조합에서 V*_rect(x) − V*_δ(x) > 0 인지. =#
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, Random
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
include(joinpath(root, "network_generator.jl")); using .NetworkGenerator
include(joinpath(root, "true_dro", "true_dro_data.jl")); include(joinpath(root, "nonzero_sum", "nz_paper_instances.jl"))
kv(nd, x) = (r = try nz_kkt_value(nd, x; optimizer=GRB, time_limit=60.0) catch; nothing end;
             r === nothing || r[:status] != MOI.OPTIMAL ? NaN : r[:value])
found = false
for (g, seed) in [("grid3x3", 1), ("grid3x3", 2), ("grid3x4", 1), ("grid3x3", 3), ("grid3x4", 2)], β in (0.7, 0.4),
    (ε̂, ε̃, δ) in ((0.2, 0.2, 0.05), (0.1, 0.3, 0.05))
    nd, _ = make_paper_maxflow(g; S=5, seed=seed, beta=β, eps_hat=ε̂, eps_tilde=ε̃)
    intd = findall(nd.x_allowed); rng = MersenneTwister(seed)
    xs = [zeros(nd.nx)]
    for _ in 1:3; x = zeros(nd.nx); x[rand(rng, intd, 2)] .= 1; push!(xs, x); end
    for x in xs
        R = kv(nd, x); V = kv(nz_with_delta(nd, δ), x)
        gap = (R - V) / max(1, abs(R))
        @printf("%s seed=%d β=%.1f ε̂=%.2f ε̃=%.2f δ=%.2f x=%-8s rect=%.4f δ=%.4f rel gap=%.2e%s\n", g, seed, β, ε̂, ε̃, δ,
                string(findall(x .> 0.5)), R, V, gap, gap > 1e-4 ? "  ← binding" : "")
        flush(stdout)
        gap > 1e-4 && (global found = true)
    end
    found && break
end
