# zs_bd_orcl_check.jl (2026-11-28 아침): zero-sum 고정 x̄ 를 Benders 쪽 oracle (true_dro Ω 의 global_bilinear_solve, belief_menu_benders 가 쓰는 것) 로 평가.
#   B&P (nz 변환 + nz_alpha_bnb) 와 Benders (true_dro Ω) 가 같은 x̄ 에서 같은 값을 내는지 본다. Polska s4 에서 B&P 해 {7,18} 21.3786 < Benders 하한 21.5071 이라 확인용.
#   인스턴스는 true_dro/factor_5/run_nig_coupled_scaling.jl 과 같은 규칙 (zs_orcl_check.jl 과 같음).
#   julia -t 12,1 zs_bd_orcl_check.jl   환경 변수 NET polska, ZSEED 4, XS "7,18;22,33", LAMU 10, TLIM 600
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Statistics
import MathOptInterface as MOI
include(joinpath(root, "network_generator.jl")); using .NetworkGenerator; const NG = NetworkGenerator
include(joinpath(root, "true_dro", "true_dro_data.jl"))
include(joinpath(root, "true_dro", "true_dro_build_subproblem.jl"))
include(joinpath(root, "true_dro", "global_bilinear_solver.jl"))
const NET = get(ENV, "NET", "polska"); const ZSEED = parse(Int, get(ENV, "ZSEED", "4"))
const ZEPS = 0.3; const ZBETA = 0.4; const ZDELTA = 0.1; const S = 10
LAMU = parse(Float64, get(ENV, "LAMU", "10")); TLIM = parse(Float64, get(ENV, "TLIM", "600"))
if NET == "grid5x5"
    net = NG.generate_grid_network(5, 5; seed=42); intd = net.interdictable_arcs
else
    gen = Dict("polska" => NG.generate_polska_network, "abilene" => NG.generate_abilene_network,
               "nobel_us" => NG.generate_nobel_us_network, "sioux_falls" => NG.generate_sioux_falls_network)
    net0 = gen[NET](); intd = fill(true, length(net0.arcs))
    net = NG.RealWorldNetworkData(net0.name, net0.original_node_names, net0.nodes, net0.arcs, net0.N, intd, net0.arc_adjacency, net0.node_arc_incidence)
end
γ = 2; Kz = length(net.arcs) - 1
caps, _ = NG.generate_capacity_scenarios_factor_additive(length(net.arcs), S; interdictable_arcs=intd, seed=ZSEED, num_factors=5)
w = round(0.5 * γ * median(caps[findall(intd[1:Kz]), :]); digits=4)
Random.seed!(ZSEED); v = zeros(Kz, S)
for k in 1:Kz, s in 1:S; v[k, s] = intd[k] ? (rand() < 0.75 ? 1.0 : 0.0) : 0.0; end
td = make_true_dro_data(net, caps, fill(1.0 / S, S), ZEPS, ZEPS; w=w, lambda_U=LAMU, gamma=γ, beta=ZBETA, v_scenarios=v)
nw = Threads.nthreads() - 2
@printf("Benders oracle (global_bilinear_solve): net %s seed %d λᵁ %.0e δ %.2f, worker %d, w %.4f\n", NET, ZSEED, LAMU, ZDELTA, nw, w); flush(stdout)
for xs in split(get(ENV, "XS", "7,18;22,33"), ";")
    x = zeros(td.num_arcs); for i in parse.(Int, split(xs, ",")); x[i] = 1.0; end
    t = @elapsed (r = global_bilinear_solve(td, x; nworkers=nw, time_limit=TLIM, rel_gap=1e-4, verbose=false, delta_couple=ZDELTA))
    @printf("x {%s}  Ω in [%.6f, %.6f], %.1f s\n", xs, r[:LB], r[:UB], t); flush(stdout)
end
