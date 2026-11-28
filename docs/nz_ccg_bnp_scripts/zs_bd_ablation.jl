# zs_bd_ablation.jl (2026-11-28 아침): zero-sum Benders (belief_menu_benders_optimize!, menu 변형) 의 틀린 하한 원인 가리기.
#   Polska s4 에서 Benders 가 x = {22,33}, LB 21.5071 로 "Optimal" 을 냈지만 세 oracle 모두 V({7,18}) = 21.3786 (< LB).
#   저장소 코드는 그대로, 인자만 바꿔 (valid_inequality :mincut | :none) × (menu_mw true | false) 네 조합을 돌린다.
#   또 min-cut VI 의 우변 (h = 0, q̂ 기대 최소 cut) 을 {7,18} 에서 직접 계산해 21.3786 과 비교한다.
#   julia -t 12,1 zs_bd_ablation.jl   환경 변수 NET polska, ZSEED 4, CONF "mincut:1,none:1,mincut:0,none:0", WALL 900, XCHK "7,18", VIFIX 0 (1 = zs_vi_sinkfix.jl 의 VI)
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, Dates, Serialization, LinearAlgebra, Random, Statistics
import MathOptInterface as MOI
include(joinpath(root, "network_generator.jl")); NG = NetworkGenerator
for f in ("true_dro_data.jl", "true_dro_build_omp.jl", "true_dro_build_subproblem.jl", "true_dro_build_isp_leader.jl",
          "true_dro_build_isp_follower.jl", "true_dro_benders.jl", "true_dro_mincut_vi.jl", "belief_menu_benders.jl")
    include(joinpath(root, "true_dro", f))
end
get(ENV, "VIFIX", "0") == "1" && include(joinpath(root, "docs", "nz_ccg_bnp_scripts", "zs_vi_sinkfix.jl"))   # sink 행 수정판 VI
const NET = get(ENV, "NET", "polska"); const ZSEED = parse(Int, get(ENV, "ZSEED", "4")); const S = 10
WALL = parse(Float64, get(ENV, "WALL", "900"))
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
td = make_true_dro_data(net, caps, fill(1.0 / S, S), 0.3, 0.3; w=w, lambda_U=10.0, gamma=γ, beta=0.4, v_scenarios=v)
ss = nothing
if NET != "grid5x5"
    src = [i for i in 1:Kz if net.arcs[i][1] == "s"]; snk = [i for i in 1:Kz if net.arcs[i][2] == "t"]
    ss = Dict{Symbol,Vector{Int}}(); length(src) >= 2 && (ss[:source_arcs] = src); length(snk) >= 2 && (ss[:sink_arcs] = snk)
    isempty(ss) && (ss = nothing)
end
# min-cut VI 우변을 {XCHK} 에서: Phase 1 VI 와 같은 식 (q̂ 기대, h = 0, 용량 ξ̄(1 − v x)) 의 최소 cut LP
xc = zeros(td.num_arcs); for i in parse.(Int, split(get(ENV, "XCHK", "7,18"), ",")); xc[i] = 1.0; end
let m = Model(Gurobi.Optimizer); set_silent(m)
    @variable(m, t0); @objective(m, Min, t0)
    vars = Dict(:x => xc, :t_0 => t0)
    # add_phase1_mincut_vi! 는 x 를 변수로 받지만 상수 벡터도 식으로 들어간다
    add_phase1_mincut_vi!(m, vars, td); optimize!(m)
    @printf("min-cut VI 우변 at x {%s}: %.6f (%s)\n", get(ENV, "XCHK", "7,18"), objective_value(m), string(termination_status(m)))
end
flush(stdout)
for c in split(get(ENV, "CONF", "mincut:1,none:1,mincut:0,none:0"), ",")
    vi, mw = split(c, ":"); vi = Symbol(vi); mw = mw == "1"
    t = @elapsed (res = belief_menu_benders_optimize!(td; mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer, lp_optimizer=Gurobi.Optimizer,
                        nworkers=Threads.nthreads() - 2, max_iter=1000, tol=5e-3, verbose=false, valid_inequality=vi, menu_mw=mw,
                        source_sink_cut=ss, wall_time_limit=WALL, delta_couple=0.1, belief_cuts=true))
    xs = findall(round.(Int, res[:x]) .> 0)
    @printf("CONF vi=%s mw=%s: status %s LB %.6f UB %.6f x %s iters %d, %.1f s\n", vi, mw, string(res[:status]), res[:lower_bound], res[:upper_bound],
            string(xs), res[:iters], t); flush(stdout)
end
