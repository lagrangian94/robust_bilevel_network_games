"""
run_coupling_batch.jl — 종속 ambiguity set (belief 결합 δ) zero-sum 실험 배치.
run_baseline_batch.jl 의 "double" (ε̂ = ε̃ = ε) 과 같은 인스턴스·같은 Benders 구성 (Enhanced BD + VIs) 에 δ 차원을 추가.

  δ = ρ · (ε̂ + ε̃) = 2ρε.  ρ = 1 → δ = ε̂ + ε̃ = 기존 rectangular (Lemma geom), ρ = 0 → p̃ = p.
  ε̂ = ε̃ 이므로 유효 반경은 ρ 와 무관 (ε̃' = min(ε̃, ε̂ + δ) = ε) → 값 차이는 순수 결합 효과.
  β = 0 (위험중립) 은 saturation 이 모든 δ 에서 성립해 결합 효과가 없으므로 기본에서 뺌 (CPL_BETAS 로 추가 가능).

환경변수 (모두 쉼표 구분)
  CPL_NETS   = grid5x5,polska,abilene,nobel_us,sioux_falls
  CPL_BETAS  = 0.05,0.4,0.7
  CPL_EPS    = 0.1,0.2,0.5
  CPL_RHOS   = 1,0.5,0.25,0      (ρ = 1 은 같은 머신에서 rect 를 다시 풀어 시간 비교를 짝지음)
  CPL_NWORKERS = 12              (α-B&B boost worker 수, 기본은 run_baseline_batch.jl 과 같음. 학술 WLS PC 는 1)
  CPL_WALL   = 7200
  CPL_TOL    = 5e-3             (원고 baseline. 값 효과가 이보다 작으면 구별 안 됨 → 값 표용으로 줄여 돌릴 때는 CPL_LOGROOT 를 따로)
  CPL_LOGROOT = logs/coupling    (factor_5 기준)
로그: <CPL_LOGROOT>/eps_<ε>_beta_<β>/<net>_rho_<ρ>.log. 끝까지 간 로그 (마지막 "RESULT" 줄) 가 있으면 건너뜀.
요약: summarize_coupling_batch.jl

Usage (스레드 ≥ CPL_NWORKERS + 2):
  julia -t 14 run_coupling_batch.jl
  CPL_NETS=grid5x5 CPL_BETAS=0.4 CPL_EPS=0.1 CPL_RHOS=0 CPL_NWORKERS=1 julia -t 3 run_coupling_batch.jl   (스모크)
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

# ── settings ──
networks  = String.(envlist("CPL_NETS", "grid5x5,polska,abilene,nobel_us,sioux_falls"))
β_list    = parse.(Float64, envlist("CPL_BETAS", "0.05,0.4,0.7"))
eps_list  = parse.(Float64, envlist("CPL_EPS", "0.1,0.2,0.5"))
ρ_list    = parse.(Float64, envlist("CPL_RHOS", "1,0.5,0.25,0"))
nworkers  = parse(Int, get(ENV, "CPL_NWORKERS", "12"))
wall      = parse(Float64, get(ENV, "CPL_WALL", "7200"))
tol       = parse(Float64, get(ENV, "CPL_TOL", "5e-3"))
log_root  = joinpath(@__DIR__, get(ENV, "CPL_LOGROOT", joinpath("logs", "coupling")))
S  = 10
λU = 10.0
Threads.nthreads() >= nworkers + 2 ||
    error("α-B&B boost: Julia 스레드 $(Threads.nthreads()) < CPL_NWORKERS+2 = $(nworkers + 2) (julia -t $(nworkers + 2) 로 실행)")

println("=" ^ 70)
@printf("Coupling batch: S=%d, λU=%.1f, q̂=uniform, scenario=factor_additive(k=5), ε̂=ε̃=ε, δ=ρ(ε̂+ε̃)\n", S, λU)
@printf("Networks: %s\nβ: %s\nε: %s\nρ: %s\nboost workers=%d, wall=%.0fs, tol=%.0e\n", join(networks, ", "),
        join(β_list, ", "), join(eps_list, ", "), join(ρ_list, ", "), nworkers, wall, tol)
println("=" ^ 70)
flush(stdout)

for β_risk in β_list, eps in eps_list
    log_dir = joinpath(log_root, "eps_$(fmt_val(eps))_beta_$(fmt_val(β_risk))")
    mkpath(log_dir)
    for network_name in networks
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

        for ρ in ρ_list
            δ = ρ >= 1 ? Inf : ρ * 2eps
            log_path = joinpath(log_dir, "$(network_name)_rho_$(fmt_val(ρ)).log")
            if log_done(log_path)
                @printf("  [skip] %s\n", log_path)
                continue
            end
            run_with_tee(log_path) do
                println("=" ^ 60)
                @printf("%s — double: ε̂=ε̃=%.2f, β=%.2f, ρ=%.2f, δ=%s\n", network_name, eps, β_risk, ρ, string(δ))
                @printf("arcs=%d, intd=%d, γ=%d, w=%.4f, started %s\n", num_arcs, length(intd_idx), γ, w, string(now()))
                println("=" ^ 60)
                flush(stdout)
                td = make_true_dro_data(net, caps, q_hat, eps, eps;
                    w=w, lambda_U=λU, gamma=γ, beta=β_risk, v_scenarios=v_rand)
                t0 = time()
                res = true_dro_benders_optimize!(td;
                    mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer, lp_optimizer=Gurobi.Optimizer,
                    max_iter=1000, tol=tol, verbose=true, sub_time_limit=15.0,
                    mini_benders=true, max_mini_benders_iter=5,
                    strengthen_cuts=:mw, valid_inequality=:mincut,
                    inexact=true, nonconvex_attr=("NonConvex" => 2),
                    source_sink_cut=ss_cut, wall_time_limit=wall,
                    boost_nworkers=nworkers, delta_couple=δ)
                wt = time() - t0
                xs = findall(round.(Int, res[:x]) .> 0)
                @printf("\nResult (double, ρ=%.2f): Z₀=%.6f, iters=%d, time=%.1fs\n", ρ, res[:Z0], res[:iters], wt)
                println("x arcs = $(xs)")
                # 요약 스크립트가 읽는 한 줄
                @printf("RESULT net=%s beta=%.2f eps=%.2f rho=%.2f delta=%s status=%s Z0=%.8f LB=%.8f UB=%.8f iters=%d time=%.2f x=%s\n",
                        network_name, β_risk, eps, ρ, string(δ), string(res[:status]), res[:Z0],
                        res[:lower_bound], res[:upper_bound], res[:iters], wt, join(xs, ";"))
                flush(stdout)
            end
            @printf("  [done] %s\n", log_path)
            flush(stdout)
        end
    end
end

println("\n" * "=" ^ 70)
println("All coupling batch done!")
println("=" ^ 70)
