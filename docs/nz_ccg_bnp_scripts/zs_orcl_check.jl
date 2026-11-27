# zs_orcl_check.jl (2026-11-28 새벽): zero-sum 고정 x̄ 에서 α-B&B oracle 설정별 상한·하한 비교. B&P 의 상한이 독립 하한 (zsi.jl 17.9315, Polska s42 {6,33}) 보다 낮게 나온 원인 진단.
#   julia -t 12,1 zs_orcl_check.jl   환경 변수 NET polska, ZSEED 42, XS "6,33", LAMU 1e4, TLIM 300, MODES "dr,dr_mip,plain"
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Statistics
import MathOptInterface as MOI
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
isdefined(Main, :nz_alpha_bnb_dr) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
include(joinpath(root, "network_generator.jl")); using .NetworkGenerator; const NG = NetworkGenerator
include(joinpath(root, "true_dro", "true_dro_data.jl"))
const NET = get(ENV, "NET", "polska"); const ZSEED = parse(Int, get(ENV, "ZSEED", "42"))
const ZEPS = 0.3; const ZBETA = 0.4; const ZDELTA = 0.1; const S = 10
LAMU = parse(Float64, get(ENV, "LAMU", "1e4")); TLIM = parse(Float64, get(ENV, "TLIM", "300"))
function zs_instance(S, lamU)   # gen_zs.py 와 같은 규칙
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
    td = make_true_dro_data(net, caps, fill(1.0 / S, S), ZEPS, ZEPS; w=w, lambda_U=lamU, gamma=γ, beta=ZBETA, v_scenarios=v)
    return nz_with_delta(nz_from_true_dro(td), ZDELTA)
end
nd = zs_instance(S, LAMU)
xb = zeros(nd.nx); for i in parse.(Int, split(get(ENV, "XS", "6,33"), ",")); xb[i] = 1.0; end
@printf("net %s seed %d x %s λᵁ %.0e, 스레드 %d\n", NET, ZSEED, get(ENV, "XS", "6,33"), LAMU, Threads.nthreads()); flush(stdout)
for mode in split(get(ENV, "MODES", "dr,dr_mip,plain"), ",")
    t = @elapsed (r = mode == "dr"     ? nz_alpha_bnb_dr(nd, xb; nworkers=Threads.nthreads() - 1, time_limit=TLIM, rel_gap=1e-4, verbose=false, resp=true) :
                      mode == "dr_mip" ? nz_alpha_bnb_dr(nd, xb; nworkers=Threads.nthreads() - 1, time_limit=TLIM, rel_gap=1e-4, verbose=false, resp=true, range_lp=false) :
                      mode == "dr_mipn" ? nz_alpha_bnb_dr(nd, xb; nworkers=Threads.nthreads() - 1, time_limit=TLIM, rel_gap=1e-4, verbose=false, resp=false, range_lp=false) :
                                         nz_alpha_bnb(nd, xb; nworkers=Threads.nthreads() - 2, time_limit=TLIM, rel_gap=1e-4))
    @printf("%-8s V in [%.6f, %.6f] status %s, %.1f s\n", mode, r[:LB], r[:UB], string(get(r, :status, "?")), t); flush(stdout)
end
