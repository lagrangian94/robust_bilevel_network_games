# repo_fix_check.jl (2026-11-28, 사용자 "다 고쳐"): 저장소 세 곳 수정 뒤 확인.
#   (A) true_dro_mincut_vi.jl 의 sink 행 수정: 다섯 망 (grid5x5·Polska·Abilene·Nobel-US·Sioux Falls) 에서 무작위 x 몇 개를 고정하고
#       Phase 1 VI 만 있는 모형의 min t₀ 가 q̂ 기대 h = 0 최대흐름과 같은지, Phase 2B (α = 0) 의 min t₀ 가 (ξ̄ 최소, v 최대) 용량의 최대흐름과 같은지.
#   (B) nz_alpha_bnb_dr.jl 의 range_lp 오류: zero-sum (meta[:kind] = :maxflow) 에서 기본 호출은 오류, range_lp = false 는 돎.
#   julia -t 6,1 repo_fix_check.jl   환경 변수 ZSEED 4, NX 5 (망마다 x 개수), GUARD 1
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Statistics
import MathOptInterface as MOI
include(joinpath(root, "network_generator.jl")); using .NetworkGenerator; const NG = NetworkGenerator
include(joinpath(root, "true_dro", "true_dro_data.jl")); include(joinpath(root, "true_dro", "true_dro_mincut_vi.jl"))
const ZSEED = parse(Int, get(ENV, "ZSEED", "4")); const S = 10; const NX = parse(Int, get(ENV, "NX", "5"))
const GENV = Gurobi.Env()
function zs_td(netname)   # gen_zs.py·run_nig_coupled_scaling.jl 과 같은 인스턴스 규칙
    if netname == "grid5x5"
        net = NG.generate_grid_network(5, 5; seed=42); intd = net.interdictable_arcs
    else
        gen = Dict("polska" => NG.generate_polska_network, "abilene" => NG.generate_abilene_network,
                   "nobel_us" => NG.generate_nobel_us_network, "sioux_falls" => NG.generate_sioux_falls_network)
        net0 = gen[netname](); intd = fill(true, length(net0.arcs))
        net = NG.RealWorldNetworkData(net0.name, net0.original_node_names, net0.nodes, net0.arcs, net0.N, intd, net0.arc_adjacency, net0.node_arc_incidence)
    end
    γ = 2; Kz = length(net.arcs) - 1
    caps, _ = NG.generate_capacity_scenarios_factor_additive(length(net.arcs), S; interdictable_arcs=intd, seed=ZSEED, num_factors=5)
    w = round(0.5 * γ * median(caps[findall(intd[1:Kz]), :]); digits=4)
    Random.seed!(ZSEED); v = zeros(Kz, S)
    for k in 1:Kz, s in 1:S; v[k, s] = intd[k] ? (rand() < 0.75 ? 1.0 : 0.0) : 0.0; end
    return make_true_dro_data(net, caps, fill(1.0 / S, S), 0.3, 0.3; w=w, lambda_U=10.0, gamma=γ, beta=0.4, v_scenarios=v)
end
function maxflow(td, cap)
    m = Model(() -> Gurobi.Optimizer(GENV)); set_silent(m)
    @variable(m, 0 <= y[k=1:td.num_arcs] <= cap[k]); @variable(m, yts >= 0)
    @constraint(m, td.Ny * y .+ td.Nts .* yts .== 0); @objective(m, Max, yts); optimize!(m)
    termination_status(m) == MOI.OPTIMAL || error("maxflow: $(termination_status(m))")
    return objective_value(m)
end
function vi_min(td, xc, phase)
    m = Model(() -> Gurobi.Optimizer(GENV)); set_silent(m)
    @variable(m, x[1:td.num_arcs]); fix.(x, xc); @variable(m, t_0)
    vars = Dict(:x => x, :t_0 => t_0)
    phase == 1 ? add_phase1_mincut_vi!(m, vars, td) :
                 add_phase2B_mincut_vi!(m, vars, td, zeros(td.num_arcs), 1; arc_topology=extract_arc_topology(td.Ny, td.nv1))
    @objective(m, Min, t_0); optimize!(m)
    termination_status(m) == MOI.OPTIMAL || error("vi_min: $(termination_status(m))")
    return objective_value(m)
end
println("(A) min-cut VI 우변 = 최대흐름 (시드 $(ZSEED), S $(S), 망마다 x $(NX) 개, 첫 x 는 0)")
@printf("%-12s %4s %6s %6s %12s %12s\n", "망", "nv1", "t 행", "x 수", "P1 최대 차", "P2B 최대 차")
worst = 0.0
for netname in ("grid5x5", "polska", "abilene", "nobel_us", "sioux_falls")
    td = zs_td(netname); K = td.num_arcs; rng = MersenneTwister(ZSEED)
    al = findall(td.interdictable_arcs[1:K])
    d1 = 0.0; d2 = 0.0
    for t in 1:NX
        xc = zeros(K); t > 1 && (xc[shuffle(rng, al)[1:2]] .= 1.0)
        f1 = mean(maxflow(td, [td.xi_bar[k, j] * (1 - td.v[k, j] * xc[k]) for k in 1:K]) for j in 1:S)
        ξmin = vec(minimum(td.xi_bar, dims=2)); vmax = vec(maximum(td.v, dims=2))
        f2 = maxflow(td, [ξmin[k] * (1 - vmax[k] * xc[k]) for k in 1:K])
        d1 = max(d1, abs(vi_min(td, xc, 1) - f1)); d2 = max(d2, abs(vi_min(td, xc, 2) - f2))
    end
    global worst = max(worst, d1, d2)
    @printf("%-12s %4d %6d %6d %12.2e %12.2e\n", netname, td.nv1, _mincut_sink_row(td), NX, d1, d2)
end
worst <= 1e-6 || error("(A) 실패: VI 우변과 최대흐름 차이 $(worst)")
println("(A) 통과: 최대 차 $(worst)")

if get(ENV, "GUARD", "1") == "1"
    include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
    isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
    isdefined(Main, :nz_alpha_bnb_dr) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
    nd =nz_with_delta(nz_from_true_dro(zs_td("polska")), 0.1); xb = zeros(nd.nx); xb[[7, 18]] .= 1.0
    msg = try
        nz_alpha_bnb_dr(nd, xb; nworkers=max(1, Threads.nthreads() - 1), time_limit=10.0, verbose=false); "오류 없음"
    catch e
        sprint(showerror, e)
    end
    println("(B) zero-sum 기본 호출: ", first(msg, 160))
    occursin("submodularity", msg) || error("(B) 실패: range_lp 오류가 안 남")
    r = nz_alpha_bnb_dr(nd, xb; nworkers=max(1, Threads.nthreads() - 1), time_limit=20.0, verbose=false, resp=false, range_lp=false)
    @printf("(B) range_lp = false: V in [%.6f, %.6f] (20 s 제한, 참값 21.3786)\n", r[:LB], r[:UB])
    println("(B) 통과")
end
