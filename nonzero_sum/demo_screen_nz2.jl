#=
demo_screen_nz2.jl — 발표용 비제로섬 예시 인스턴스 찾기, 원고 OOS 기준 (2026-10-21 사용자 요청).
찾는 것: 원고 §6.2 그림처럼 Nominal → Partial DR (ε̃ = 0) → 우리 모형 (ε̂ = ε̃ = ε, δ = DS_DELTA) 순서로
         OOS 실현 비용의 q95 가 좋아지는 인스턴스.
1. (S, 한도, seed) 마다 nominal 을 한 번 풀고, ε 마다 partial 과 coupled 를 푼다 (demo_value_nz.jl 과 같은 Algorithm 2 인자).
2. partial ≠ coupled (follower 쪽 모호성이 결정을 바꿈) 인 경우만 OOS 평가: demo_value_nz.jl DN_STAGE=phaseab 와 같은 추출
   (γ ∈ DS_GAMMAS, ζ ∈ {0, 0.5, 1, 2, 5}, 칸마다 DS_M 개, seed 42). 칸별 q95 를 ζ 에 대해 평균.
3. 판정 (γ 마다): mono = q95(ours) < q95(partial) < q95(nominal), ours_best = q95(ours) < 나머지 둘.
결과: logs/demo_value/screen2_sgb128.csv (이어서 돌릴 수 있음), 끝에 후보 정렬 (BEST 줄).
환경변수: DS_S = 10,5   DS_EPS = 0.05,0.1,0.15,0.2,0.25,0.3,0.4   DS_QUOTA = 100,150,200,300   DS_SEEDS = 1:10
          DS_BETA = 0.4   DS_DELTA = 0.1   DS_LAMBDA = 100   DS_TL = 600   DS_NWORKERS = 12   DS_GAMMAS = 0.1,1.0   DS_M = 300
실행: julia -t 14,1 nonzero_sum/demo_screen_nz2.jl
=#
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Statistics, Distributions
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
isdefined(Main, :nz_alpha_bnb_dr) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
include(joinpath(root, "nonzero_sum", "nz_root_tighten.jl"))
include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))

envf(k, d) = get(ENV, k, d)
envlist(k, d) = [strip(s) for s in split(envf(k, d), ",") if !isempty(strip(s))]
function intlist(k, d)
    s = envf(k, d)
    occursin(":", s) && (a = parse.(Int, split(s, ":")); return collect(a[1]:a[2]))
    return parse.(Int, envlist(k, d))
end
Ss = intlist("DS_S", "10,5"); epss = parse.(Float64, envlist("DS_EPS", "0.05,0.1,0.15,0.2,0.25,0.3,0.4"))
quotas = parse.(Float64, envlist("DS_QUOTA", "100,150,200,300")); seeds = intlist("DS_SEEDS", "1:10")
β = parse(Float64, envf("DS_BETA", "0.4")); δ = parse(Float64, envf("DS_DELTA", "0.1")); lam = parse(Float64, envf("DS_LAMBDA", "100"))
TL = parse(Float64, envf("DS_TL", "600")); nw = parse(Int, envf("DS_NWORKERS", "12"))
gammas = parse.(Float64, envlist("DS_GAMMAS", "0.1,1.0")); M = parse(Int, envf("DS_M", "300")); zetas = [0.0, 0.5, 1.0, 2.0, 5.0]
Threads.nthreads() >= nw + 2 || error("julia -t $(nw + 2),1 이상 필요")
outdir = joinpath(root, "nonzero_sum", "logs", "demo_value"); mkpath(outdir)
csv = joinpath(outdir, "screen2_sgb128.csv")
xstr(x) = "{" * join(findall(x .> 0.5), ";") * "}"

# demo_value_nz.jl 과 같은 함수 (CVaR 위쪽 꼬리, retailer 최적 예약)
function cvar_upper(Z::AbstractVector, p::AbstractVector, β::Float64)
    ord = sortperm(Z; rev=true); cap = p ./ (1 - β); rest = 1.0; val = 0.0
    for s in ord
        m = min(cap[s], rest); val += m * Z[s]; rest -= m
        rest <= 1e-12 && break
    end
    return val
end
function follower_response(nd::NZData, x̄, pt)
    nh, ny, S, m = nz_nh(nd), nz_ny(nd), nd.S, nz_m(nd)
    fm = Model(GRB1); set_optimizer_attribute(fm, "Method", 0)
    @variable(fm, 0 <= hf[i = 1:nh] <= nd.hU[i]); @variable(fm, yf[1:ny, 1:S] >= 0)
    @constraint(fm, nd.W * hf .<= nd.wvec)
    for s in 1:S
        r0 = nz_rhs(nd, x̄, zeros(nh), s)
        for k in 1:m
            @constraint(fm, sum(nd.A[k, j] * yf[j, s] for j in 1:ny if nd.A[k, j] != 0) <=
                            r0[k] + (nd.hrow[k] > 0 ? nd.hcoef[k] * hf[nd.hrow[k]] : 0.0))
        end
    end
    @objective(fm, Max, dot(nd.c0, hf) + sum(pt[s] * nd.c[j] * yf[j, s] for j in 1:ny, s in 1:S if nd.c[j] != 0))
    optimize!(fm)
    termination_status(fm) == MOI.OPTIMAL || error("follower LP: $(termination_status(fm))")
    return clamp.(value.(hf), 0.0, nd.hU)
end

function mk(S, seed, quota, eh, et, d)
    nd = make_location_instance(; coords=:sgb128, S=S, seed=seed, reservation=:pooled, pool_limit=:store, quota=quota,
                                eps_hat=eh, eps_tilde=et, beta=β)
    isfinite(d) && (nd = nz_with_delta(nd, d))
    return nz_with_bounds(nd; thetaU=nz_theta_circuit_exact(nd)[1], lambdaU=lam)
end
function solve(nd)
    r = nz_belief_menu_benders(nd; optimizer=GRB, tol=5e-3, oracle=:alpha_bnb, bnb=:dr, oracle_gap=1e-3, nworkers=nw,
                               time_limit=TL, local_first=true, local_time=60.0, bnb_kw=(ipopt_time=120.0,), verbose=false)
    return r[:x], r[:UB], string(r[:status])
end

"OOS: 결정마다 γ 별 (ζ 평균 q95, 전체 평균). 추출은 demo_value_nz.jl DN_STAGE=phaseab 와 같음"
function oos(nd, xs::Vector{Vector{Float64}})
    base = nd.S .* nd.q_hat; cache = Dict{Tuple{Int,Vector{Float64}},Vector{Float64}}()
    q95 = zeros(length(gammas), length(xs)); mn = zeros(length(gammas), length(xs))
    for (gi, γ) in enumerate(gammas)
        for ζ in zetas
            rng = MersenneTwister(42); c = [zeros(M) for _ in xs]
            for n in 1:M
                if ζ == 0
                    p = rand(rng, Dirichlet(γ .* base)); pt = rand(rng, Dirichlet(γ .* base))
                else
                    p = rand(rng, Dirichlet(max.(γ .* base .* (1 .+ ζ .* randn(rng, nd.S)), 0.01)))
                    pt = rand(rng, Dirichlet(max.(γ .* base .* (1 .+ ζ .* randn(rng, nd.S)), 0.01)))
                end
                for (i, x) in enumerate(xs)
                    h = round.(follower_response(nd, x, pt); digits=6)
                    φ = get!(() -> nz_recourse_values(nd, x, h)[2], cache, (i, h))
                    c[i][n] = dot(nd.qx, x) + cvar_upper(φ, p, β)
                end
            end
            for i in eachindex(xs)
                q95[gi, i] += quantile(c[i], 0.95) / length(zetas); mn[gi, i] += mean(c[i]) / length(zetas)
            end
        end
    end
    return q95, mn
end

hdr = "S,eps,quota,seed,x_nominal,x_partial,x_ours,v_nominal,v_partial,v_ours,status," *
      join(["q95_g$(g)_nom,q95_g$(g)_par,q95_g$(g)_our,mean_g$(g)_nom,mean_g$(g)_par,mean_g$(g)_our,mono_g$(g),best_g$(g)" for g in gammas], ",")
done = Set{String}()
if isfile(csv)
    for l in readlines(csv)[2:end]
        push!(done, join(split(l, ",")[1:4], ","))
    end
else
    open(csv, "w") do io; println(io, hdr); end
end
@printf("demo_screen_nz2: S=%s ε=%s quota=%s seeds=%s β=%.2f δ=%.2f λᵁ=%.0f | OOS γ=%s M=%d\n", string(Ss), string(epss),
        string(quotas), string(seeds), β, δ, lam, string(gammas), M)
flush(stdout)
for S in Ss, quota in quotas, seed in seeds
    keys_ = [@sprintf("%d,%.2f,%.0f,%d", S, ε, quota, seed) for ε in epss]
    all(k -> k in done, keys_) && continue
    t0 = time()
    xn, vn, sn = solve(mk(S, seed, quota, 0.0, 0.0, Inf))
    for (ε, key) in zip(epss, keys_)
        key in done && continue
        xp, vp, sp = solve(mk(S, seed, quota, ε, 0.0, Inf))
        xo, vo, so = solve(mk(S, seed, quota, ε, ε, δ))
        row = @sprintf("%s,%s,%s,%s,%.4f,%.4f,%.4f,%s", key, xstr(xn), xstr(xp), xstr(xo), vn, vp, vo, join([sn, sp, so], ";"))
        tag = ""
        if xstr(xp) != xstr(xo)
            q, m = oos(mk(S, seed, quota, 0.0, 0.0, Inf), [xn, xp, xo])
            for gi in eachindex(gammas)
                mono = q[gi, 3] < q[gi, 2] < q[gi, 1]; best = q[gi, 3] < min(q[gi, 1], q[gi, 2])
                row *= @sprintf(",%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d", q[gi, 1], q[gi, 2], q[gi, 3], m[gi, 1], m[gi, 2], m[gi, 3], mono, best)
                tag *= @sprintf(" | γ=%.1f q95 %.0f / %.0f / %.0f%s", gammas[gi], q[gi, 1], q[gi, 2], q[gi, 3],
                                mono ? " MONO" : best ? " BEST" : "")
            end
        else
            row *= repeat(",,,,,,,,", length(gammas))
        end
        open(csv, "a") do io; println(io, row); end
        @printf("  %s: x = %s | %s | %s%s\n", key, xstr(xn), xstr(xp), xstr(xo), tag); flush(stdout)
    end
    @printf("  [S=%d 한도 %.0f seed %d] %.0fs\n", S, quota, seed, time() - t0); flush(stdout)
end

# 후보: γ 마다 mono 개수, best 개수 순
rows = [split(l, ",") for l in readlines(csv)[2:end]]
hd = split(hdr, ","); ci(n) = findfirst(==(n), hd)
sc(r) = (sum(r[ci("mono_g$(g)")] == "1" for g in gammas), sum(r[ci("best_g$(g)")] == "1" for g in gammas),
         length(unique(r[5:7])))
cands = sort([r for r in rows if r[ci("mono_g$(gammas[1])")] != ""]; by=r -> sc(r), rev=true)
@printf("\nOOS 평가한 인스턴스 %d 개 (전체 %d):\n", length(cands), length(rows))
for r in cands[1:min(15, end)]
    @printf("  (mono %d, best %d, 결정 %d 종) S=%s ε=%s 한도 %s seed %s | x %s %s %s | %s\n", sc(r)..., r[1], r[2], r[3], r[4],
            r[5], r[6], r[7], join(["γ=$(g): " * join(r[ci("q95_g$(g)_nom"):ci("q95_g$(g)_our")], "/") for g in gammas], "  "))
end
isempty(cands) ? println("BEST none") : (b = cands[1]; @printf("BEST S=%s eps=%s quota=%s seed=%s mono=%d best=%d\n", b[1], b[2], b[3], b[4], sc(b)[1:2]...))
