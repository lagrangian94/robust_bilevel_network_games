"""
run_nig_coupled_scaling.jl — 원고 §6.2 (zero-sum 네트워크 차단) 계산 성능을 결합 ambiguity set D_δ 로 다시 (2026-10-20 사용자 결정).

결정: δ = 0.1 (입지 실험과 같음), (β, ε) 는 한 조합으로 고정 — β = 0.4, ε̂ = ε̃ = ε*(네트워크) (out-of-sample 보정값:
grid5x5·polska·sioux_falls 0.1, abilene·nobel_us 0.2, generate_oos_summary_table.jl 의 NETWORK_EPS). S = 10, 20, 50.
방법 (원고 tab:computational 의 세 변형, 기존 run_naive_batch.jl / run_baseline_batch.jl 과 같은 Benders 인자):
  naive   = Plain BD          (mini_benders = false, valid_inequality = :none)
  mb_only = Enhanced BD       (mini_benders = true,  valid_inequality = :none)
  full    = Enhanced BD + VIs (mini_benders = true,  valid_inequality = :mincut)
  menu    = Algorithm 2 의 zero-sum 판 (belief_menu_benders_optimize!: α-B&B oracle, local cut, belief menu + MW, min-cut VI) — 2026-10-20 추가
S = 10 은 네 변형 모두, S = 20·50 은 menu 만 (비교는 작은 S 에서, 확장성은 우리 방법 = Algorithm 2 로).
인스턴스 생성은 run_coupling_batch.jl 과 같음 (factor_additive k=5, seed 42, w = 0.5γ·median, v ~ Bernoulli(0.75), λᵁ = 10, γ = 2).
boost (전역 subproblem) 는 true_dro_benders_optimize! 의 기본값 = zero-sum α-B&B (global_bilinear_solver.jl, worker NS_NWORKERS).
시간 한도는 입지 실험과 같은 3,600 s (2026-10-20 사용자 결정, 기존 tab:computational 은 7,200 s).

환경변수 (쉼표 구분)
  NS_NETS = grid5x5,polska,abilene,nobel_us,sioux_falls   NS_S = 10,20,50   NS_DELTA = 0.1   NS_BETA = 0.4
  NS_VARIANTS_S10 = naive,mb_only,full,menu   NS_VARIANTS_LARGE = menu   NS_NWORKERS = 12   NS_WALL = 3600   NS_TOL = 5e-3
  NS_LOGROOT = logs/nig_coupled   (factor_5 기준)
로그: <NS_LOGROOT>/S_<S>/<net>_<variant>.log, 끝까지 간 로그 (RESULT 줄) 는 건너뜀.
실행: julia -t 14 run_nig_coupled_scaling.jl
"""

using JuMP, Gurobi, Printf, Dates, Serialization, LinearAlgebra, Random

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

# ── tee helper (run_baseline_batch.jl 과 같음) ──
function run_with_tee(f, log_path)
    log_io = open(log_path, "w")
    original_stdout = stdout
    rd, wr = redirect_stdout()
    reader_task = @async begin
        try
            while !eof(rd)
                line = readline(rd; keep=false)
                println(original_stdout, line)
                println(log_io, line)
                flush(original_stdout)
                flush(log_io)
            end
        catch e
            e isa InterruptException || e isa Base.IOError || rethrow()
        end
    end
    try
        f()
    finally
        redirect_stdout(original_stdout)
        close(wr)
        wait(reader_task)
        close(rd)
        close(log_io)
    end
end

# ── network definitions (run_baseline_batch.jl 과 같음) ──
function make_network(name)
    if name == "grid5x5"
        net = NG.generate_grid_network(5, 5; seed=42)
        return net, 2, net.interdictable_arcs
    end
    gen = Dict(
        "polska"      => NG.generate_polska_network,
        "abilene"     => NG.generate_abilene_network,
        "nobel_us"    => NG.generate_nobel_us_network,
        "sioux_falls" => NG.generate_sioux_falls_network,
    )
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

fmt_val(v) = replace(@sprintf("%.2g", v), "." => "p")
envlist(k, d) = [strip(s) for s in split(get(ENV, k, d), ",") if !isempty(strip(s))]
log_done(p) = isfile(p) && any(startswith(l, "RESULT ") for l in eachline(p))

const NETWORK_EPS = Dict("grid5x5" => 0.1, "polska" => 0.1, "sioux_falls" => 0.1, "abilene" => 0.2, "nobel_us" => 0.2)
# (mini_benders, valid_inequality). menu 는 belief_menu_benders_optimize! (mini-Benders 대신 belief menu, min-cut VI)
const VARIANTS = Dict("naive" => (false, :none), "mb_only" => (true, :none), "full" => (true, :mincut), "menu" => (false, :mincut))

networks   = String.(envlist("NS_NETS", "grid5x5,polska,abilene,nobel_us,sioux_falls"))
S_list     = parse.(Int, envlist("NS_S", "10,20,50"))
δ          = parse(Float64, get(ENV, "NS_DELTA", "0.1"))
β_risk     = parse(Float64, get(ENV, "NS_BETA", "0.4"))
var_s10    = String.(envlist("NS_VARIANTS_S10", "naive,mb_only,full,menu"))
var_large  = String.(envlist("NS_VARIANTS_LARGE", "menu"))
nworkers   = parse(Int, get(ENV, "NS_NWORKERS", "12"))
wall       = parse(Float64, get(ENV, "NS_WALL", "3600"))
tol        = parse(Float64, get(ENV, "NS_TOL", "5e-3"))
log_root   = joinpath(@__DIR__, get(ENV, "NS_LOGROOT", joinpath("logs", "nig_coupled")))
λU = 10.0
all(v -> haskey(VARIANTS, v), vcat(var_s10, var_large)) || error("알 수 없는 변형: $(vcat(var_s10, var_large))")
all(n -> haskey(NETWORK_EPS, n), networks) || error("ε* 가 정해지지 않은 네트워크: $networks")
Threads.nthreads() >= nworkers + 2 ||
    error("α-B&B boost: Julia 스레드 $(Threads.nthreads()) < NS_NWORKERS+2 = $(nworkers + 2) (julia -t $(nworkers + 2) 로 실행)")

println("=" ^ 70)
@printf("NIG coupled scaling: δ=%.3f, β=%.2f, ε̂=ε̃=ε*(net), λU=%.1f, S ∈ %s, wall=%.0fs, tol=%.0e, boost workers=%d\n",
        δ, β_risk, λU, string(S_list), wall, tol, nworkers)
@printf("Networks: %s | S=10 변형: %s | S>10 변형: %s\n", join(networks, ", "), join(var_s10, ","), join(var_large, ","))
println("=" ^ 70)
flush(stdout)

for S in S_list
    log_dir = joinpath(log_root, "S_$(S)")
    mkpath(log_dir)
    variants = S == 10 ? var_s10 : var_large
    for network_name in networks
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
        for variant in variants
            mb, vi = VARIANTS[variant]
            log_path = joinpath(log_dir, "$(network_name)_$(variant).log")
            if log_done(log_path)
                @printf("  [skip] %s\n", log_path); flush(stdout); continue
            end
            run_with_tee(log_path) do
                println("=" ^ 60)
                @printf("%s — %s: S=%d, ε̂=ε̃=%.2f, β=%.2f, δ=%.3f (%s)\n", network_name, variant, S, eps, β_risk, δ,
                        variant == "menu" ? "belief_menu_benders_optimize!, oracle α-B&B, local_first, menu MW, VI=mincut" :
                                            "true_dro_benders_optimize!, mini_benders=$(mb), VI=$(vi), boost α-B&B")
                @printf("arcs=%d, intd=%d, γ=%d, w=%.4f, started %s\n", num_arcs, length(intd_idx), γ, w, string(now()))
                println("=" ^ 60)
                flush(stdout)
                td = make_true_dro_data(net, caps, q_hat, eps, eps;
                    w=w, lambda_U=λU, gamma=γ, beta=β_risk, v_scenarios=v_rand)
                t0 = time()
                res = if variant == "menu"
                    belief_menu_benders_optimize!(td;
                        mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer, lp_optimizer=Gurobi.Optimizer,
                        nworkers=nworkers, max_iter=1000, tol=tol, verbose=true,
                        valid_inequality=:mincut, source_sink_cut=ss_cut, wall_time_limit=wall, delta_couple=δ)
                else
                    true_dro_benders_optimize!(td;
                        mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer, lp_optimizer=Gurobi.Optimizer,
                        max_iter=1000, tol=tol, verbose=true, sub_time_limit=15.0,
                        mini_benders=mb, max_mini_benders_iter=5,
                        strengthen_cuts=:mw, valid_inequality=vi,
                        inexact=true, nonconvex_attr=("NonConvex" => 2),
                        source_sink_cut=ss_cut, wall_time_limit=wall,
                        boost_nworkers=nworkers, delta_couple=δ)
                end
                wt = time() - t0
                xs = findall(round.(Int, res[:x]) .> 0)
                @printf("\nResult (%s, S=%d): Z₀=%.6f, iters=%d, time=%.1fs\n", variant, S, res[:Z0], res[:iters], wt)
                println("x arcs = $(xs)")
                @printf("RESULT net=%s variant=%s S=%d beta=%.2f eps=%.2f delta=%.3f status=%s Z0=%.8f LB=%.8f UB=%.8f iters=%d time=%.2f x=%s\n",
                        network_name, variant, S, β_risk, eps, δ, string(res[:status]), res[:Z0],
                        res[:lower_bound], res[:upper_bound], res[:iters], wt, join(xs, ";"))
                flush(stdout)
            end
            @printf("  [done] %s\n", log_path); flush(stdout)
        end
    end
end
println("\n" * "=" ^ 70)
println("NIG coupled scaling done")
println("=" ^ 70)
