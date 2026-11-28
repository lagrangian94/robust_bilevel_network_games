# zs_sdcut_check.jl (2026-11-28, 사용자 "dd kkt-mip 강쌍대컷 추가해서 시드넓혀보자"): zero-sum 고정 x̄ 에서 강쌍대 cut (abb_dd_patch.jl 의 dd_sdcut) 이
#   oracle 을 빠르게 하는지. 강쌍대 cut 은 nz_alpha_bnb_dr 의 KKT-MIP (범위 MIP·DD MIP) 에 붙는다. zero-sum 은 꼭짓점 LP 범위 (range_lp = true) 가 틀리므로
#   (corner premise, md 11.29) 범위 MIP 판 (range_lp = false) 에만 붙일 수 있다. 지금 zero-sum B&P 의 oracle 은 nz_alpha_bnb (plain) 다.
#   시험점 기본 = Abilene 시드 42 γ = 2 의 x* {5;11} (B&P oracle 5 번 1,814 s, Benders α-B&B 3,620 s 로 못 닫은 점).
#   julia -t 16,1 zs_sdcut_check.jl   환경 변수 ROOT, NET abilene, ZSEED 42, ZGAMMA 2, XS "5,11", LAMU 1e4, TLIM 300, MODES "plain,mip,mip_sd,mipdd_sd"
#   ABB_SD_CHECK·ABB_SD_TIGHT (abb_dd_patch.jl 검증용, 프로세스 시작 때 읽음)
root = get(ENV, "ROOT", "C:/Users/user/robust_bilevel_network_games")
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Statistics
import MathOptInterface as MOI
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
isdefined(Main, :nz_alpha_bnb_dr) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
include(joinpath(root, "docs", "nz_ccg_bnp_scripts", "abb_dd_patch.jl"))
include(joinpath(root, "network_generator.jl")); using .NetworkGenerator; const NG = NetworkGenerator
include(joinpath(root, "true_dro", "true_dro_data.jl"))
const NET = get(ENV, "NET", "abilene"); const ZSEED = parse(Int, get(ENV, "ZSEED", "42")); const ZGAMMA = parse(Int, get(ENV, "ZGAMMA", "2"))
const ZEPS = 0.3; const ZBETA = 0.4; const ZDELTA = 0.1; const S = 10
LAMU = parse(Float64, get(ENV, "LAMU", "1e4")); TLIM = parse(Float64, get(ENV, "TLIM", "300"))
function zs_instance(S, lamU)   # gen_zs.py 와 같은 규칙 (γ = ZGAMMA, w = 0.5 γ median)
    if NET == "grid5x5"
        net = NG.generate_grid_network(5, 5; seed=42); intd = net.interdictable_arcs
    else
        gen = Dict("polska" => NG.generate_polska_network, "abilene" => NG.generate_abilene_network,
                   "nobel_us" => NG.generate_nobel_us_network, "sioux_falls" => NG.generate_sioux_falls_network)
        net0 = gen[NET](); intd = fill(true, length(net0.arcs))
        net = NG.RealWorldNetworkData(net0.name, net0.original_node_names, net0.nodes, net0.arcs, net0.N, intd, net0.arc_adjacency, net0.node_arc_incidence)
    end
    γ = ZGAMMA; Kz = length(net.arcs) - 1
    caps, _ = NG.generate_capacity_scenarios_factor_additive(length(net.arcs), S; interdictable_arcs=intd, seed=ZSEED, num_factors=5)
    w = round(0.5 * γ * median(caps[findall(intd[1:Kz]), :]); digits=4)
    Random.seed!(ZSEED); v = zeros(Kz, S)
    for k in 1:Kz, s in 1:S; v[k, s] = intd[k] ? (rand() < 0.75 ? 1.0 : 0.0) : 0.0; end
    td = make_true_dro_data(net, caps, fill(1.0 / S, S), ZEPS, ZEPS; w=w, lambda_U=lamU, gamma=γ, beta=ZBETA, v_scenarios=v)
    return nz_with_delta(nz_from_true_dro(td), ZDELTA)
end
nd = zs_instance(S, LAMU)
xb = zeros(nd.nx); for i in parse.(Int, split(get(ENV, "XS", "5,11"), ",")); xb[i] = 1.0; end
@printf("net %s seed %d γ %d x %s λᵁ %.0e, 스레드 %d, TLIM %.0f s, ABB_SD_CHECK %d, ABB_SD_TIGHT %s\n", NET, ZSEED, ZGAMMA, get(ENV, "XS", "5,11"), LAMU,
        Threads.nthreads(), TLIM, _ABB_SD_CHECK, string(_ABB_SD_TIGHT)); flush(stdout)
nw = Threads.nthreads() - 1
for mode in split(get(ENV, "MODES", "plain,mip,mip_sd,mipdd_sd"), ",")
    t = @elapsed (r = mode == "plain"    ? nz_alpha_bnb(nd, xb; nworkers=nw - 1, time_limit=TLIM, rel_gap=1e-4) :
                      mode == "mip"      ? nz_alpha_bnb_dr(nd, xb; nworkers=nw, time_limit=TLIM, rel_gap=1e-4, verbose=false, resp=false, range_lp=false) :
                      mode == "mip_sd"   ? nz_alpha_bnb_dr(nd, xb; nworkers=nw, time_limit=TLIM, rel_gap=1e-4, verbose=false, resp=false, range_lp=false, dd_sdcut=true) :
                      mode == "mipdd"    ? nz_alpha_bnb_dr(nd, xb; nworkers=nw, time_limit=TLIM, rel_gap=1e-4, verbose=false, resp=false, range_lp=false, dd=true) :
                      mode == "mipdd_sd" ? nz_alpha_bnb_dr(nd, xb; nworkers=nw, time_limit=TLIM, rel_gap=1e-4, verbose=false, resp=false, range_lp=false, dd=true, dd_sdcut=true) :
                                           error("MODES: plain | mip | mip_sd | mipdd | mipdd_sd"))
    gap = 100 * (r[:UB] - r[:LB]) / max(1.0, abs(r[:LB]))
    @printf("%-9s V in [%.6f, %.6f] gap %.3f%%, status %s, 노드 %s, %.1f s\n", mode, r[:LB], r[:UB], gap, string(get(r, :status, "?")),
            string(get(r, :nodes, "?")), t)
    haskey(r, :dd_screen) && println("          dd_screen ", r[:dd_screen])
    for k in (:t_range, :range_mips, :t_relax)
        haskey(r, k) && print("          ", k, " = ", r[k])
    end
    println(); flush(stdout)
end
