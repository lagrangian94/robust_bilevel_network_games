#=
oos_coupled_eval.jl — zero-sum 네트워크 차단의 out-of-sample, belief 를 진짜 분포에 묶어 추출 (2026-11-05, docs/robustness_value_todo.md 1 절).

모형 4 개의 x* 를 run_nig_coupled_scaling.jl 로그 (NS_EPS = OC_EPS, NS_MODEL = nominal | datadr | product | coupled, menu 변형) 의
RESULT 줄에서 읽는다. 상대 gap (UB − LB)/max(1, |UB|) 가 OC_GAPTOL (기본 5e-3 = 계산 실험의 허용오차) 을 넘거나 ε·δ 가 다르면 오류.
인스턴스: 네트워크 × seed (OC_SEEDS), 생성은 run_nig_coupled_scaling.jl 과 같음 (factor_additive k=5, w = 0.5γ·median, v ~ Bernoulli(0.75)).
두 게임 공통 OOS 설계 (2026-11-05, nonzero_sum/loc_oos.jl 과 같음):
  진짜 분포 p ~ Dir(γ·S·q̂), γ ∈ OC_GAMMAS (q̂ 균등이라 Dir(γ·1)). γ = 5 는 p 가 ε 공 안 (E d_TV(p, q̂) ≈ 0.17), γ = 0.1 은 멀리 (≈ 0.7).
  belief p̃ = (1 − τ) p + τ u, u ~ Dir(1), τ ~ U[0,1]. d_TV(p, p̃) 로 세 구간, (γ, 구간) 마다 OC_N 쌍:
    Close        d_TV ≤ δ          (Coupled 의 가정이 성립)
    Moderate     δ < d_TV ≤ 2ε     (product 만 덮음, 2ε = ε̂ + ε̃)
    Misspecified d_TV > 2ε
  follower 는 p̃ 아래 최적 복구 h (LP 꼭짓점 해). draw 의 값 = 리더 목적 = p 아래 시나리오별 max-flow 의 CVaR_β (낮을수록 좋음).
  요약 = draw 들의 평균과 가장 나쁜 OC_TAIL (5%) draw 의 평균 (인스턴스마다, γ 별과 γ 전체).
모든 모형에 같은 (p, p̃) 쌍. 같은 x 는 한 번만 계산.

환경변수: OC_NETS = grid5x5,polska,sioux_falls,abilene,nobel_us   OC_SEEDS = 42,1,2,3,4   OC_EPS = 0.3   OC_DELTA = 0.1   OC_BETA = 0.4
          OC_GAMMAS = 0.1,0.3,1,5   OC_N = 200   OC_TAIL = 0.05   OC_GAPTOL = 5e-3   OC_OOS_SEED = 2026
          OC_XLOG = logs/value_eps0p3/S_10   OC_OUT = logs/oos_common   (factor_5 기준)
출력: <OC_OUT>/<net>_draws.csv (쌍마다 한 줄), <net>_summary.csv (seed × γ (all = γ 전체) × 구간 × 모형 의 n·평균·가장 나쁜 5% 평균),
      표준출력에 구간 × 모형 의 Nominal 대비 % (seed 평균, γ 전체)
실행: julia oos_coupled_eval.jl   (네트워크마다 따로 띄우면 병렬)
=#
using JuMP, Gurobi, Printf, Statistics, Random, Distributions, LinearAlgebra

include("../../network_generator.jl")
using .NetworkGenerator
include("../../oos_evaluation.jl")
include("../oos_evaluate.jl")          # compute_maxflow_per_scenario, compute_cvar

const GRB_ENV = Gurobi.Env()
envlist(k, d) = [strip(s) for s in split(get(ENV, k, d), ",") if !isempty(strip(s))]
networks = String.(envlist("OC_NETS", "grid5x5,polska,sioux_falls,abilene,nobel_us"))
seeds    = parse.(Int, envlist("OC_SEEDS", "42,1,2,3,4"))
ε        = parse(Float64, get(ENV, "OC_EPS", "0.3"))
δ        = parse(Float64, get(ENV, "OC_DELTA", "0.1"))
β_risk   = parse(Float64, get(ENV, "OC_BETA", "0.4"))
gammas   = parse.(Float64, envlist("OC_GAMMAS", "0.1,0.3,1,5"))
Npl      = parse(Int, get(ENV, "OC_N", "200"))
tail     = parse(Float64, get(ENV, "OC_TAIL", "0.05"))
gaptol   = parse(Float64, get(ENV, "OC_GAPTOL", "5e-3"))
oos_seed = parse(Int, get(ENV, "OC_OOS_SEED", "2026"))
xlog     = joinpath(@__DIR__, get(ENV, "OC_XLOG", joinpath("logs", "value_eps0p3", "S_10")))
outdir   = joinpath(@__DIR__, get(ENV, "OC_OUT", joinpath("logs", "oos_common"))); mkpath(outdir)
δ < 2ε || error("δ = $δ ≥ 2ε = $(2ε) 이면 Moderate 구간이 비어 있음")

const MODELS = ["Nominal", "Data-DR", "Product", "Coupled"]
const MODEL_TAG = Dict("Nominal" => "_nominal", "Data-DR" => "_datadr", "Product" => "_product", "Coupled" => "")
const LEVELS = ["Close", "Moderate", "Misspecified"]
level(tv) = tv <= δ ? 1 : tv <= 2ε ? 2 : 3

# ── 인스턴스 (run_nig_coupled_scaling.jl 의 make_network·생성과 같음) ──
function make_network(name)
    if name == "grid5x5"
        net = generate_grid_network(5, 5; seed=42)
        return net, 2, net.interdictable_arcs
    end
    gen = Dict("polska" => generate_polska_network, "abilene" => generate_abilene_network,
               "nobel_us" => generate_nobel_us_network, "sioux_falls" => generate_sioux_falls_network)
    net = gen[name]()
    intd_arcs = fill(true, length(net.arcs))
    net = RealWorldNetworkData(net.name, net.original_node_names, net.nodes, net.arcs,
        net.N, intd_arcs, net.arc_adjacency, net.node_arc_incidence)
    return net, 2, intd_arcs
end

function make_instance(name, seed; S=10)
    net, γ, intd_arcs = make_network(name)
    num_arcs = length(net.arcs) - 1
    caps, _ = generate_capacity_scenarios_factor_additive(length(net.arcs), S;
        interdictable_arcs=intd_arcs, seed=seed, num_factors=5)
    intd_idx = findall(intd_arcs[1:num_arcs])
    w = round(0.5 * γ * median(caps[intd_idx, :]); digits=4)
    Random.seed!(seed)
    v = zeros(num_arcs, S)
    for k in 1:num_arcs, s in 1:S
        v[k, s] = intd_arcs[k] ? (rand() < 0.75 ? 1.0 : 0.0) : 0.0
    end
    return net, caps, v, w
end

# ── x* (RESULT 줄) ──
function read_x(name, model, seed, num_arcs)
    path = joinpath(xlog, "$(name)_menu$(MODEL_TAG[model])" * (seed == 42 ? "" : "_s$(seed)") * ".log")
    isfile(path) || error("x* 로그 없음: $path")
    line = filter(l -> startswith(l, "RESULT "), readlines(path))
    length(line) == 1 || error("RESULT 줄이 $(length(line)) 개: $path")
    kv = Dict(split(t, "="; limit=2)[1] => split(t, "="; limit=2)[2] for t in split(line[1])[2:end] if occursin("=", t))
    gapx = (parse(Float64, kv["UB"]) - parse(Float64, kv["LB"])) / max(1.0, abs(parse(Float64, kv["UB"])))
    gapx <= gaptol || error("x* 가 허용오차 안에서 인증되지 않음 ($(kv["status"]), gap $gapx > $gaptol): $path")
    eh = parse(Float64, kv["eps"]); et = haskey(kv, "eps_tilde") ? parse(Float64, kv["eps_tilde"]) : eh
    want = model == "Nominal" ? (0.0, 0.0) : model == "Data-DR" ? (ε, 0.0) : (ε, ε)
    isapprox(eh, want[1]) && isapprox(et, want[2]) || error("ε̂·ε̃ = ($eh, $et), 기대 $want: $path")
    model == "Coupled" && !isapprox(parse(Float64, kv["delta"]), δ) && error("δ = $(kv["delta"]) ≠ $δ: $path")
    x = zeros(num_arcs)
    for a in split(kv["x"], ";"; keepempty=false); x[parse(Int, a)] = 1.0; end
    return x, kv["x"]
end

# ── follower: p̃ 아래 최적 복구 (oos_evaluate.jl 의 solve_follower_weighted 와 같은 LP, OPTIMAL 이 아니면 오류) ──
function follower_h(net, x, v, w, caps, pt)
    na = length(net.arcs); num_arcs = na - 1; K = size(caps, 2)
    dummy = findfirst(a -> a == ("t", "s"), net.arcs)
    m = Model(() -> Gurobi.Optimizer(GRB_ENV)); set_silent(m); set_optimizer_attribute(m, "Threads", 1)
    @variable(m, h[1:num_arcs] >= 0); @variable(m, y[1:na, 1:K] >= 0)
    @objective(m, Max, sum(pt[k] * y[dummy, k] for k in 1:K))
    @constraint(m, sum(h) <= w)
    for k in 1:K
        @constraint(m, net.N * y[:, k] .== 0.0)
        for a in 1:num_arcs
            @constraint(m, y[a, k] <= caps[a, k] * (1.0 - v[a, k] * x[a]) + h[a])
        end
    end
    optimize!(m)
    termination_status(m) == MOI.OPTIMAL || error("follower LP: $(termination_status(m))")
    return value.(h)
end

# ── 추출: γ 마다 구간별 Npl 쌍 (γ = 5 의 Misspecified 는 드물어 S = 10 에서 약 3,000 번에 1 번) ──
function draw_pairs(rng, S, γ, q̂)
    pairs = [Tuple{Vector{Float64},Vector{Float64},Float64}[] for _ in 1:3]
    ndraw = 0
    while any(length.(pairs) .< Npl)
        ndraw += 1
        ndraw > 20_000 * Npl && error("구간을 채우지 못함: $(length.(pairs)) (γ=$γ)")
        p = rand(rng, Dirichlet(γ .* S .* q̂)); u = rand(rng, Dirichlet(ones(S))); τ = rand(rng)
        pt = (1 - τ) .* p .+ τ .* u; tv = 0.5 * sum(abs.(p .- pt)); l = level(tv)
        length(pairs[l]) < Npl && push!(pairs[l], (p, pt, tv))
    end
    return pairs, ndraw
end

"가장 나쁜 tail 비율 draw 의 평균 (비용이라 큰 쪽)"
worst(c) = (k = max(1, ceil(Int, tail * length(c))); mean(sort(c; rev=true)[1:k]))

for name in networks
    dio = open(joinpath(outdir, "$(name)_draws.csv"), "w")
    println(dio, "net,seed,gamma,level,draw,tv,tv_qhat,model,x,cost")
    srows = String[]                                                  # summary.csv 의 줄
    rel = Dict{Tuple{Int,String},Vector{NTuple{2,Float64}}}()         # (구간, 모형) → seed 마다 Nominal 대비 % (평균, 가장 나쁜 5%)
    for seed in seeds
        net, caps, v, w = make_instance(name, seed)
        num_arcs = length(net.arcs) - 1; S = size(caps, 2); q̂ = fill(1 / S, S)
        xs = Dict(mdl => read_x(name, mdl, seed, num_arcs) for mdl in MODELS)
        ux = unique(last(xs[mdl]) for mdl in MODELS)
        @printf("%s seed=%d w=%.4f: %s\n", name, seed, w, join(["$mdl={$(last(xs[mdl]))}" for mdl in MODELS], "  "))
        flush(stdout)
        allc = Dict((l, k) => Float64[] for l in 1:3, k in ux)       # γ 전체를 합친 비용
        for (gi, γ) in enumerate(gammas)
            rng = MersenneTwister(hash((oos_seed, name, seed, gi)))
            pairs, ndraw = draw_pairs(rng, S, γ, q̂)
            for l in 1:3
                cost = Dict(k => Vector{Float64}(undef, Npl) for k in ux)
                for (n, (p, pt, tv)) in enumerate(pairs[l]), k in ux
                    x = first(xs[MODELS[findfirst(mdl -> last(xs[mdl]) == k, MODELS)]])
                    h = follower_h(net, x, v, w, caps, pt)
                    cost[k][n] = compute_cvar(p, compute_maxflow_per_scenario(net, x, h, v, caps), β_risk)
                end
                for k in ux; append!(allc[(l, k)], cost[k]); end
                for mdl in MODELS
                    c = cost[last(xs[mdl])]
                    for (n, (p, _, tv)) in enumerate(pairs[l])
                        @printf(dio, "%s,%d,%.2f,%s,%d,%.5f,%.5f,%s,%s,%.6f\n", name, seed, γ, LEVELS[l], n, tv,
                                0.5 * sum(abs.(p .- q̂)), mdl, last(xs[mdl]), c[n])
                    end
                    push!(srows, @sprintf("%s,%d,%.2f,%s,%s,%s,%d,%.6f,%.6f", name, seed, γ, LEVELS[l], mdl, last(xs[mdl]),
                                          length(c), mean(c), worst(c)))
                end
            end
            @printf("  γ=%.1f (추출 %d 번)\n", γ, ndraw); flush(stdout)
        end
        for l in 1:3
            cN = allc[(l, last(xs["Nominal"]))]
            for mdl in MODELS
                c = allc[(l, last(xs[mdl]))]
                push!(srows, @sprintf("%s,%d,all,%s,%s,%s,%d,%.6f,%.6f", name, seed, LEVELS[l], mdl, last(xs[mdl]),
                                      length(c), mean(c), worst(c)))
                push!(get!(rel, (l, mdl), NTuple{2,Float64}[]), (100 * (mean(c) / mean(cN) - 1), 100 * (worst(c) / worst(cN) - 1)))
            end
        end
    end
    close(dio)
    open(joinpath(outdir, "$(name)_summary.csv"), "w") do f
        println(f, "net,seed,gamma,level,model,x,n,mean,worst$(tail)")
        foreach(r -> println(f, r), srows)
    end
    @printf("\n%s (ε=%.2f δ=%.2f β=%.2f, γ = %s, seed %d 개): Nominal 대비 %% (seed 평균, 비용이라 음수가 좋음)\n%-13s %-8s %8s %8s\n",
            name, ε, δ, β_risk, join(gammas, ","), length(seeds), "level", "model", "평균", "나쁜$(round(Int, 100tail))%")
    for l in 1:3, mdl in MODELS
        r = rel[(l, mdl)]
        @printf("%-13s %-8s %+8.1f %+8.1f\n", LEVELS[l], mdl, mean(first.(r)), mean(last.(r)))
    end
    println("OOS_COUPLED_DONE $name"); flush(stdout)
end
