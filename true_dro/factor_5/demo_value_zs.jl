#=
demo_value_zs.jl — 학회 발표용 zero-sum (최대 흐름 네트워크 차단) 예시: 모델의 가치 (2026-10-20).

메시지: 리더는 진짜 분포 p 와 follower belief p̃ 를 모두 모른다. belief 의 모호성을 무시하면 (Belief-known) 꼬리 위험이
커지고, 둘을 따로 막으면 (Product) 지나치게 보수적이며, 결합 (Coupled, d_TV(p, p̃) ≤ δ) 이 그 사이에 있다.

1. 풀기 (우리 알고리즘 = belief_menu_benders_optimize!, 같은 네트워크·S·β):
   nominal  ε̂ = ε̃ = 0
   partial  ε̂ = ε, ε̃ = 0           (belief 를 안다고 가정, Sadana·Lei 류)
   product  ε̂ = ε̃ = ε, δ = ∞      (현재 원고의 Full DR)
   coupled  ε̂ = ε̃ = ε, δ ∈ DV_DELTAS (우리)
2. 평가 (같은 난수를 모든 모델에 씀): belief 오차 τ 마다 N 쌍
   p ~ Dir(DV_CP · S · q̂)                       (진실, q̂ 중심)
   p̃ = (1 − τ) p + τ u,  u ~ Dir(1)              (belief, d_TV(p, p̃) ≤ τ)
   h = follower 최적 회복 (belief p̃ 아래 기대 흐름 최대, solve_follower_weighted — LP 꼭짓점 해)
   실현 손실 = CVaR_β^p[ 시나리오별 max-flow(x, h) ]
   τ 별로 평균·중앙값·q95·최댓값을 CSV 로.
결과: logs/demo_value/<net>/ (풀이 로그, x_<model>.txt, eval_draws.csv, eval_summary.csv). 그림은 demo_value_plot.jl.

환경변수: DV_NET = polska   DV_S = 10   DV_BETA = 0.4   DV_EPS = (네트워크 ε*)   DV_DELTAS = 0.05,0.1,0.2
          DV_TAUS = 0,0.05,0.1,0.15,0.2,0.3,0.4   DV_N = 300   DV_CP = 5   DV_SEED = 2026
          DV_NWORKERS = 12   DV_WALL = 3600   DV_TOL = 5e-3   DV_STAGE = all | solve | eval
실행: julia -t 14,1 demo_value_zs.jl   (α-B&B 에 interactive 스레드 필요)
=#

using JuMP, Gurobi, Printf, Dates, Serialization, LinearAlgebra, Random, Statistics, Distributions

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
include("../../oos_evaluation.jl")
include("../oos_evaluate.jl")

function run_with_tee(f, log_path)
    log_io = open(log_path, "w")
    original_stdout = stdout
    rd, wr = redirect_stdout()
    reader_task = @async begin
        try
            while !eof(rd)
                line = readline(rd; keep=false)
                println(original_stdout, line); println(log_io, line)
                flush(original_stdout); flush(log_io)
            end
        catch e
            e isa InterruptException || e isa Base.IOError || rethrow()
        end
    end
    try
        f()
    finally
        redirect_stdout(original_stdout); close(wr); wait(reader_task); close(rd); close(log_io)
    end
end

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

"CVaR_β^p[Z] (손실의 위쪽 꼬리) = max{νᵀZ : 0 ≤ ν ≤ p/(1−β), 1ᵀν = 1}"
function cvar_upper(Z::AbstractVector, p::AbstractVector, β::Float64)
    ord = sortperm(Z; rev=true); cap = p ./ (1 - β); rest = 1.0; val = 0.0
    for s in ord
        m = min(cap[s], rest); val += m * Z[s]; rest -= m
        rest <= 1e-12 && break
    end
    return val
end

envlist(k, d) = [strip(s) for s in split(get(ENV, k, d), ",") if !isempty(strip(s))]
const NETWORK_EPS = Dict("grid5x5" => 0.1, "polska" => 0.1, "sioux_falls" => 0.1, "abilene" => 0.2, "nobel_us" => 0.2)

net_name = get(ENV, "DV_NET", "polska")
S        = parse(Int, get(ENV, "DV_S", "10"))
β_risk   = parse(Float64, get(ENV, "DV_BETA", "0.4"))
eps      = parse(Float64, get(ENV, "DV_EPS", string(NETWORK_EPS[net_name])))
deltas   = parse.(Float64, envlist("DV_DELTAS", "0.05,0.1,0.2"))
taus     = parse.(Float64, envlist("DV_TAUS", "0,0.05,0.1,0.15,0.2,0.3,0.4"))
Ndraw    = parse(Int, get(ENV, "DV_N", "300"))
cp       = parse(Float64, get(ENV, "DV_CP", "5"))
seed     = parse(Int, get(ENV, "DV_SEED", "2026"))
nworkers = parse(Int, get(ENV, "DV_NWORKERS", "12"))
wall     = parse(Float64, get(ENV, "DV_WALL", "3600"))
tol      = parse(Float64, get(ENV, "DV_TOL", "5e-3"))
stage    = get(ENV, "DV_STAGE", "all")
outdir   = joinpath(@__DIR__, "logs", "demo_value", net_name)
mkpath(outdir)
λU = 10.0

net, γ, intd_arcs = make_network(net_name)
num_arcs = length(net.arcs) - 1
caps, _ = NG.generate_capacity_scenarios_factor_additive(length(net.arcs), S; interdictable_arcs=intd_arcs, seed=42, num_factors=5)
intd_idx = findall(intd_arcs[1:num_arcs])
w = round(0.5 * γ * median(caps[intd_idx, :]); digits=4)
q_hat = fill(1.0 / S, S)
ss_cut = make_ss_cut(net, num_arcs, net_name)
Random.seed!(42)
v_rand = zeros(num_arcs, S)
for k in 1:num_arcs, s in 1:S
    v_rand[k, s] = intd_arcs[k] ? (rand() < 0.75 ? 1.0 : 0.0) : 0.0
end

# (이름, ε̂, ε̃, δ)
models = [("nominal", 0.0, 0.0, Inf), ("partial", eps, 0.0, Inf), ("product", eps, eps, Inf)]
for δ in deltas
    push!(models, ("coupled_d$(replace(string(δ), "." => "p"))", eps, eps, δ))
end
@printf("demo_value_zs: net=%s S=%d β=%.2f ε=%.2f δ=%s | τ=%s N=%d cp=%.1f seed=%d | stage=%s\n",
        net_name, S, β_risk, eps, string(deltas), string(taus), Ndraw, cp, seed, stage)
flush(stdout)

xfile(name) = joinpath(outdir, "x_$(name).txt")

if stage in ("all", "solve")
    Threads.nthreads() >= nworkers + 2 || error("julia -t $(nworkers + 2) 이상 필요")
    for (name, eh, et, δ) in models
        if isfile(xfile(name))
            @printf("  [skip] %s (x 있음)\n", name); flush(stdout); continue
        end
        run_with_tee(joinpath(outdir, "solve_$(name).log")) do
            @printf("%s: ε̂=%.2f ε̃=%.2f δ=%s β=%.2f S=%d (belief_menu_benders_optimize!, α-B&B oracle)\n",
                    name, eh, et, string(δ), β_risk, S)
            td = make_true_dro_data(net, caps, q_hat, eh, et; w=w, lambda_U=λU, gamma=γ, beta=β_risk, v_scenarios=v_rand)
            t0 = time()
            res = belief_menu_benders_optimize!(td; mip_optimizer=Gurobi.Optimizer, nlp_optimizer=Gurobi.Optimizer,
                lp_optimizer=Gurobi.Optimizer, nworkers=nworkers, max_iter=1000, tol=tol, verbose=true,
                valid_inequality=:mincut, source_sink_cut=ss_cut, wall_time_limit=wall, delta_couple=δ)
            xs = findall(round.(Int, res[:x]) .> 0)
            @printf("RESULT model=%s status=%s Z0=%.8f LB=%.8f UB=%.8f iters=%d time=%.1f x=%s\n", name,
                    string(res[:status]), res[:Z0], res[:lower_bound], res[:upper_bound], res[:iters], time() - t0, join(xs, ";"))
            open(xfile(name), "w") do io; println(io, join(xs, ",")); end
        end
    end
end

if stage in ("all", "eval")
    xs_of = Dict{String,Vector{Float64}}()
    for (name, _, _, _) in models
        isfile(xfile(name)) || error("x 파일 없음: $(xfile(name)) (DV_STAGE=solve 먼저)")
        idx = [parse(Int, t) for t in split(strip(read(xfile(name), String)), ",") if !isempty(t)]
        x = zeros(num_arcs); x[idx] .= 1.0; xs_of[name] = x
        @printf("  x[%s] = %s\n", name, string(idx))
    end
    rng = MersenneTwister(seed)
    draws = open(joinpath(outdir, "eval_draws.csv"), "w")
    println(draws, "tau,draw,tv,model,cvar")
    summ = Dict{Tuple{Float64,String},Vector{Float64}}()
    for τ in taus, n in 1:Ndraw
        p = rand(rng, Dirichlet(cp .* S .* q_hat))
        u = rand(rng, Dirichlet(ones(S)))
        pt = (1 - τ) .* p .+ τ .* u
        tv = 0.5 * sum(abs.(p .- pt))
        for (name, _, _, _) in models
            x = xs_of[name]
            h = solve_follower_weighted(net, x, v_rand, w, caps, pt)
            flows = compute_maxflow_per_scenario(net, x, h, v_rand, caps)
            c = cvar_upper(flows, p, β_risk)
            @printf(draws, "%.3f,%d,%.5f,%s,%.6f\n", τ, n, tv, name, c)
            push!(get!(summ, (τ, name), Float64[]), c)
        end
        n % 100 == 0 && (@printf("  τ=%.2f: %d/%d\n", τ, n, Ndraw); flush(stdout); flush(draws))
    end
    close(draws)
    open(joinpath(outdir, "eval_summary.csv"), "w") do io
        println(io, "tau,model,mean,median,q95,max")
        for τ in taus, (name, _, _, _) in models
            v = summ[(τ, name)]
            @printf(io, "%.3f,%s,%.6f,%.6f,%.6f,%.6f\n", τ, name, mean(v), quantile(v, 0.5), quantile(v, 0.95), maximum(v))
        end
    end
    println("EVAL_DONE $(outdir)")
end
