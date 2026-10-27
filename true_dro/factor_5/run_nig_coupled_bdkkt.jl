#=
run_nig_coupled_bdkkt.jl — §6.2 (zero-sum 네트워크 차단, 결합 집합) 의 벤치마크 BD-KKT / BD-G (bd_kkt_benders.jl, NS_METHOD = kkt | omega).
인스턴스·시간 한도·허용오차·δ·β·ε*(네트워크) 는 run_nig_coupled_scaling.jl 의 menu 실행과 같다 (그 스크립트의 make_network·
make_ss_cut·인스턴스 생성을 그대로 옮김). 결정 (2026-10-21 사용자): 먼저 S = 10 만, 잘 안 풀리면 S ≥ 20 은 "--".
KKT 모형의 Gurobi 스레드 = NS_NWORKERS (menu 의 α-B&B worker 수와 같게).

환경변수: NS_NETS = grid5x5,polska,abilene,nobel_us,sioux_falls   NS_S = 10   NS_DELTA = 0.1   NS_BETA = 0.4
          NS_NWORKERS = 12   NS_WALL = 3600   NS_TOL = 5e-3   NS_LOGROOT = logs/nig_coupled
로그: <NS_LOGROOT>/S_<S>/<net>_bdkkt.log (RESULT 줄이 있으면 건너뜀)
실행: julia -t 2 run_nig_coupled_bdkkt.jl
=#
using JuMP, Gurobi, Printf, Dates, LinearAlgebra, Random, Statistics

include("../../network_generator.jl")
NG = NetworkGenerator
include("../true_dro_data.jl")
include("../true_dro_build_omp.jl")
include("../true_dro_build_subproblem.jl")
include("../true_dro_build_isp_leader.jl")
include("../true_dro_build_isp_follower.jl")
include("../true_dro_benders.jl")
include("../true_dro_mincut_vi.jl")
include("../belief_menu_benders.jl")
include("../true_dro_structured_eval.jl")
include("../bd_kkt_benders.jl")

function make_network(name)
    if name == "grid5x5"
        net = NG.generate_grid_network(5, 5; seed=42)
        return net, 2, net.interdictable_arcs
    end
    gen = Dict("polska" => NG.generate_polska_network, "abilene" => NG.generate_abilene_network,
               "nobel_us" => NG.generate_nobel_us_network, "sioux_falls" => NG.generate_sioux_falls_network)
    net = gen[name]()
    intd_arcs = fill(true, length(net.arcs))
    net = NG.RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
        net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)
    return net, 2, intd_arcs
end
function make_ss_cut(net, num_arcs, network_name)
    network_name == "grid5x5" && return nothing
    source_arcs = [i for i in 1:num_arcs if net.arcs[i][1] == "s"]
    sink_arcs   = [i for i in 1:num_arcs if net.arcs[i][2] == "t"]
    ss = Dict{Symbol, Vector{Int}}()
    length(source_arcs) >= 2 && (ss[:source_arcs] = source_arcs)
    length(sink_arcs) >= 2   && (ss[:sink_arcs]   = sink_arcs)
    return isempty(ss) ? nothing : ss
end

envlist(k, d) = [strip(s) for s in split(get(ENV, k, d), ",") if !isempty(strip(s))]
log_done(p) = isfile(p) && any(startswith(l, "RESULT ") for l in eachline(p))
const NETWORK_EPS = Dict("grid5x5" => 0.1, "polska" => 0.1, "sioux_falls" => 0.1, "abilene" => 0.2, "nobel_us" => 0.2)
networks = String.(envlist("NS_NETS", "grid5x5,polska,abilene,nobel_us,sioux_falls"))
S_list   = parse.(Int, envlist("NS_S", "10"))
δ        = parse(Float64, get(ENV, "NS_DELTA", "0.1"))
β_risk   = parse(Float64, get(ENV, "NS_BETA", "0.4"))
nworkers = parse(Int, get(ENV, "NS_NWORKERS", "12"))
wall     = parse(Float64, get(ENV, "NS_WALL", "3600"))
tol      = parse(Float64, get(ENV, "NS_TOL", "5e-3"))
meth     = Symbol(get(ENV, "NS_METHOD", "kkt")); meth in (:kkt, :omega) || error("NS_METHOD = kkt | omega")
vname    = meth == :kkt ? "bdkkt" : "bdg"
log_root = joinpath(@__DIR__, get(ENV, "NS_LOGROOT", joinpath("logs", "nig_coupled")))
λU = 10.0
const GRB_ENV = Gurobi.Env()
GRBK() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), nworkers); o)

for S in S_list, network_name in networks
    log_dir = joinpath(log_root, "S_$(S)"); mkpath(log_dir)
    log_path = joinpath(log_dir, "$(network_name)_$(vname).log")
    log_done(log_path) && (@printf("  [skip] %s\n", log_path); continue)
    eps = NETWORK_EPS[network_name]
    net, γ, intd_arcs = make_network(network_name)
    num_arcs = length(net.arcs) - 1
    caps, _ = NG.generate_capacity_scenarios_factor_additive(length(net.arcs), S;
        interdictable_arcs=intd_arcs, seed=42, num_factors=5)
    intd_idx = findall(intd_arcs[1:num_arcs])
    w = round(0.5 * γ * median(caps[intd_idx, :]); digits=4)
    q_hat = fill(1.0/S, S)
    ss_cut = make_ss_cut(net, num_arcs, network_name)
    Random.seed!(42)
    v_rand = zeros(num_arcs, S)
    for k in 1:num_arcs, s in 1:S
        v_rand[k, s] = intd_arcs[k] ? (rand() < 0.75 ? 1.0 : 0.0) : 0.0
    end
    open(log_path, "w") do io
        redirect_stdout(io) do
            println("=" ^ 60)
            @printf("%s — %s: S=%d, ε̂=ε̃=%.2f, β=%.2f, δ=%.3f (bd_kkt_benders_optimize!, KKT 스레드 %d, VI=mincut)\n",
                    network_name, vname, S, eps, β_risk, δ, nworkers)
            @printf("arcs=%d, intd=%d, γ=%d, w=%.4f, started %s\n", num_arcs, length(intd_idx), γ, w, string(now()))
            println("=" ^ 60); flush(stdout)
            td = make_true_dro_data(net, caps, q_hat, eps, eps; w=w, lambda_U=λU, gamma=γ, beta=β_risk, v_scenarios=v_rand)
            t0 = time()
            res = bd_kkt_benders_optimize!(td; mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer,
                                           lp_optimizer=Gurobi.Optimizer, kkt_optimizer=GRBK, max_iter=1000, tol=tol, verbose=true,
                                           valid_inequality=:mincut, source_sink_cut=ss_cut, wall_time_limit=wall, delta_couple=δ, oracle=meth)
            wt = time() - t0
            xs = findall(round.(Int, res[:x]) .> 0)
            @printf("RESULT net=%s variant=%s S=%d beta=%.2f eps=%.2f delta=%.3f status=%s Z0=%.8f LB=%.8f UB=%.8f iters=%d time=%.2f x=%s\n",
                    network_name, vname, S, β_risk, eps, δ, string(res[:status]), res[:Z0], res[:lower_bound], res[:upper_bound],
                    res[:iters], wt, join(xs, ";"))
            flush(stdout)
        end
    end
    @printf("  [done] %s: %s\n", log_path, filter(l -> startswith(l, "RESULT"), readlines(log_path)))
    flush(stdout)
end
