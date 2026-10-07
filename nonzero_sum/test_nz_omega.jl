"""
test_nz_omega.jl — 비제로섬 Ω 검증 1단계.
  (1) zero-sum 교차검증: 원고 max-flow (grid 3×3) 를 일반 recourse 로 옮겨 ℓ=c, θ=0 일 때
      기존 build_true_dro_subproblem 과 Ω(x) 가 같은지.
  (2) location (Goyal 기본 사례 규모): 모든 x ∈ X 에서 Ω(x) (θᵁ, λᵁ, πᵁ 사용) 와
      big-M 없는 KKT 평가 V*(x) 비교, θ 진단.
실행: julia nonzero_sum/test_nz_omega.jl
"""

root = dirname(@__DIR__)
using JuMP, Gurobi, HiGHS, Printf, LinearAlgebra, Random
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)

include(joinpath(root, "nonzero_sum", "nz_data.jl"))
include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))

all_x(nd) = [Float64.(collect(bits)) for bits in Iterators.product(fill(0:1, nd.nx)...)
             if sum(bits) <= nd.gamma && all(nd.x_allowed[i] || bits[i] == 0 for i in 1:nd.nx)]

# ---------------------------------------------------------------- (1)
if get(ENV, "NZ_SKIP_ZS", "0") != "1"
    include(joinpath(root, "network_generator.jl"))
    using .NetworkGenerator
    include(joinpath(root, "true_dro", "true_dro_data.jl"))
    include(joinpath(root, "true_dro", "true_dro_build_subproblem.jl"))
    println("="^70, "\n(1) zero-sum 교차검증: max-flow grid 3×3, S=3, β=0.4\n", "="^70)
    net = generate_grid_network(3, 3; seed=42)
    caps, _ = generate_capacity_scenarios_uniform_model(length(net.arcs), 3; seed=42)
    td = make_true_dro_data(net, caps, fill(1/3, 3), 0.2, 0.2; w=1.0, lambda_U=10.0, gamma=2, beta=0.4)
    ndz = nz_from_true_dro(td)
    K = td.num_arcs
    old_m, old_v = build_true_dro_subproblem(td, zeros(K); optimizer=GRB, silent=true)
    set_optimizer_attribute(old_m, "NonConvex", 2); set_optimizer_attribute(old_m, "MIPGap", 1e-6)
    O = build_nz_omega(ndz; optimizer=GRB)
    rng = MersenneTwister(0)
    intd = findall(td.interdictable_arcs[1:K])
    xs = [zeros(K)]
    for _ in 1:4
        x = zeros(K); x[rand(rng, intd, 2)] .= 1; push!(xs, x)
    end
    for x in xs
        info = solve_true_dro_subproblem!(old_m, old_v, td, x)
        res = nz_solve!(O, ndz, x)
        @printf("  x=%-12s  기존 Ω=%.6f   일반 Ω=%.6f   diff=%.2e\n", string(findall(x .> 0.5)),
                info[:Z0_val], res[:Fval], res[:Fval] - info[:Z0_val])
    end
end

# ---------------------------------------------------------------- (2)
println("="^70, "\n(2) location: Ω(x) vs KKT V*(x)\n", "="^70)
nd = make_location_instance(; S=parse(Int, get(ENV, "NZ_S", "4")), seed=parse(Int, get(ENV, "NZ_SEED", "1")),
                            lambdaU=(haskey(ENV, "NZ_LAMBDA") ? parse(Float64, ENV["NZ_LAMBDA"]) : nothing),
                            reservation=Symbol(get(ENV, "NZ_RES", "pooled")),
                            quota=parse(Float64, get(ENV, "NZ_QUOTA", "100")), wres=parse(Float64, get(ENV, "NZ_WRES", "300")))
@printf("%s: θᵁ=%.1f  max π̂ᵁ=%.1f  λᵁ=%.1f  cmax=%.2f\n", nd.name, nd.thetaU, maximum(nd.piLU), nd.lambdaU, nd.meta[:cmax])
println("  거리 (A행 먼저):"); display(nd.meta[:dist])
println("  수요 ξ (고객 × S):"); display(nd.meta[:xi])

O = build_nz_omega(nd; optimizer=GRB)
for x in all_x(nd)
    t1 = @elapsed (res = nz_solve!(O, nd, x; time_limit=600))
    bel = nz_read_belief(O, nd)
    t2 = @elapsed (kk = nz_kkt_value(nd, x; optimizer=GRB, time_limit=600))
    # θ 진단: Ω 해의 α 에서
    dg = [nz_theta_diag(nd, x, bel[:α], s; lp_optimizer=GRB) for s in 1:nd.S]
    @printf("  x=%-10s Ω=%11.4f [%s %.1fs]  KKT=%11.4f (bd %.4f, %.1fs)  diff=%+.2e | θneed=%.2f  pen_excess=%.1e  π̂/π̂ᵁ=%.3f\n",
            string(findall(x .> 0.5)), res[:Fval], res[:status], t1, kk[:value], kk[:bound], t2,
            res[:Fval] - kk[:value], maximum(d[:theta_need] for d in dg),
            maximum(d[:penalty_excess] for d in dg), maximum(d[:pi_ratio] for d in dg))
    flush(stdout)
end
