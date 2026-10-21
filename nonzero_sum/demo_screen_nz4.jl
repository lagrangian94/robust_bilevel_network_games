#=
demo_screen_nz4.jl — 비제로섬 예시 탐색, 지역 요인 수요 (2026-10-21 사용자 요청).
진단 (iid 대 factor, seed 1–6): 요인 수요에서 follower 쪽 벌점이 리더 쪽 대비 0.19 → 0.83 으로 커지고 결정이 바뀌는 경우가 생김.
1. (S, 한도, seed) 마다 nominal 한 번, ε 마다 partial (ε̃ = 0), ours (ε̂ = ε̃ = ε, δ = DS_DELTA), product (δ = ∞) 를 푼다
   (demo_value_nz.jl 과 같은 Algorithm 2 인자). 인스턴스 생성 make_location_instance(...; demand = DS_DEMAND).
2. 네 결정 중 하나라도 다르면 OOS 평가. 진짜 분포 p ~ Dir(γ·S·q̂), follower 믿음 p̃ 를 세 방식으로:
   indep : p̃ ~ Dir(γ·S·q̂) 독립 (원고 Phase A)
   data  : p̃ = Multinomial(N_F, p) / N_F                 (N_F ∈ DS_NF)
   mix   : p̃ = (1 − τ) p + τ u, u ~ Dir(1)               (τ ∈ DS_TAUS)
   실현 비용 = q_Bᵀx + CVaR_β^p[φ(x, h(p̃))] (낮을수록 좋음). 같은 (S, 한도, seed) 는 결정별 비용을 ε 사이에 재사용.
결과: logs/demo_value/screen4_<demand>_dec.csv (결정), screen4_<demand>_oos.csv (OOS 칸), 끝에 요약
      요약 = partial ≠ ours 인 인스턴스와, 그중 결합 가정이 맞는 칸 (실현 d_TV 평균 ≤ δ) 에서 우리 모형이 q95 단독 최선인 칸 수.
환경변수: DS_DEMAND = factor   DS_S = 10   DS_EPS = 0.1,0.2,0.3   DS_QUOTA = 100,150,200   DS_SEEDS = 1:10   DS_BETA = 0.4
          DS_DELTA = 0.1   DS_LAMBDA = 100   DS_TL = 600   DS_NWORKERS = 12   DS_GAMMAS = 0.1,1,5   DS_NF = 10,30,100,300
          DS_TAUS = 0,0.05,0.1,0.2,0.4   DS_M = 300   DS_SEED = 42
실행: julia -t 14,1 nonzero_sum/demo_screen_nz4.jl
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
demand = Symbol(envf("DS_DEMAND", "factor"))
Ss = intlist("DS_S", "10"); epss = parse.(Float64, envlist("DS_EPS", "0.1,0.2,0.3"))
quotas = parse.(Float64, envlist("DS_QUOTA", "100,150,200")); seeds = intlist("DS_SEEDS", "1:10")
β = parse(Float64, envf("DS_BETA", "0.4")); δ = parse(Float64, envf("DS_DELTA", "0.1")); lam = parse(Float64, envf("DS_LAMBDA", "100"))
TL = parse(Float64, envf("DS_TL", "600")); nw = parse(Int, envf("DS_NWORKERS", "12"))
gammas = parse.(Float64, envlist("DS_GAMMAS", "0.1,1,5")); NFs = parse.(Int, envlist("DS_NF", "10,30,100,300"))
taus = parse.(Float64, envlist("DS_TAUS", "0,0.05,0.1,0.2,0.4")); M = parse(Int, envf("DS_M", "300")); oseed = parse(Int, envf("DS_SEED", "42"))
Threads.nthreads() >= nw + 2 || error("julia -t $(nw + 2),1 이상 필요")
outdir = joinpath(root, "nonzero_sum", "logs", "demo_value"); mkpath(outdir)
dcsv = joinpath(outdir, "screen4_$(demand)_dec.csv"); ocsv = joinpath(outdir, "screen4_$(demand)_oos.csv")
xstr(x) = "{" * join(findall(x .> 0.5), ";") * "}"
function xparse(s, nx)
    x = zeros(nx); body = strip(s, ['{', '}'])
    isempty(body) || (x[parse.(Int, split(body, ";"))] .= 1.0)
    return x
end

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
                                eps_hat=eh, eps_tilde=et, beta=β, demand=demand)
    isfinite(d) && (nd = nz_with_delta(nd, d))
    return nz_with_bounds(nd; thetaU=nz_theta_circuit_exact(nd)[1], lambdaU=lam)
end
function solve(nd)
    r = nz_belief_menu_benders(nd; optimizer=GRB, tol=5e-3, oracle=:alpha_bnb, bnb=:dr, oracle_gap=1e-3, nworkers=nw,
                               time_limit=TL, local_first=true, local_time=60.0, bnb_kw=(ipopt_time=120.0,), verbose=false)
    return xstr(r[:x]), r[:UB], string(r[:status])
end

"추출 칸 하나: 결정 x 마다 비용 M 개와 실현 d_TV 평균"
function cell(nd, xs, design, level, γ, cache)
    base = nd.S .* nd.q_hat; rng = MersenneTwister(oseed)
    c = [zeros(M) for _ in xs]; tv = 0.0
    for n in 1:M
        p = rand(rng, Dirichlet(γ .* base)); p ./= sum(p)
        pt = design == "indep" ? (q = rand(rng, Dirichlet(γ .* base)); q ./ sum(q)) :
             design == "data" ? rand(rng, Multinomial(Int(level), p)) ./ level :
             (u = rand(rng, Dirichlet(ones(nd.S))); (1 - level) .* p .+ level .* u)
        tv += 0.5 * sum(abs.(p .- pt)) / M
        for (i, x) in enumerate(xs)
            h = round.(follower_response(nd, x, pt); digits=6)
            φ = get!(() -> nz_recourse_values(nd, x, h)[2], cache, (xstr(x), h))
            c[i][n] = dot(nd.qx, x) + cvar_upper(φ, p, β)
        end
    end
    return c, tv
end

names_ = ["nom", "par", "our", "prd"]
done = Set{String}()
if isfile(dcsv)
    for l in readlines(dcsv)[2:end]; push!(done, join(split(l, ",")[1:4], ",")); end
else
    open(dcsv, "w") do io; println(io, "S,eps,quota,seed,x_nom,x_par,x_our,x_prd,v_nom,v_par,v_our,v_prd,status"); end
    open(ocsv, "w") do io
        println(io, "S,eps,quota,seed,design,level,gamma,tv_mean,x_nom,x_par,x_our,x_prd," *
                    join(["q95_$n" for n in names_], ",") * "," * join(["mean_$n" for n in names_], ","))
    end
end
cells = vcat([("indep", 0.0)], [("data", Float64(n)) for n in NFs], [("mix", t) for t in taus])
@printf("demo_screen_nz4: 수요 %s | S=%s ε=%s 한도=%s seeds=%s β=%.2f δ=%.2f λᵁ=%.0f | OOS γ=%s N_F=%s τ=%s M=%d\n", demand,
        string(Ss), string(epss), string(quotas), string(seeds), β, δ, lam, string(gammas), string(NFs), string(taus), M)
flush(stdout)
for S in Ss, quota in quotas, seed in seeds
    keys_ = [@sprintf("%d,%.2f,%.0f,%d", S, ε, quota, seed) for ε in epss]
    all(k -> k in done, keys_) && continue
    t0 = time()
    xn, vn, sn = solve(mk(S, seed, quota, 0.0, 0.0, Inf))
    dec = Dict{String,Vector{String}}()
    for (ε, key) in zip(epss, keys_)
        key in done && continue
        xp, vp, sp = solve(mk(S, seed, quota, ε, 0.0, Inf))
        xo, vo, so = solve(mk(S, seed, quota, ε, ε, δ))
        xr, vr, sr = solve(mk(S, seed, quota, ε, ε, Inf))
        dec[key] = [xn, xp, xo, xr]
        open(dcsv, "a") do io
            @printf(io, "%s,%s,%s,%s,%s,%.4f,%.4f,%.4f,%.4f,%s\n", key, xn, xp, xo, xr, vn, vp, vo, vr, join([sn, sp, so, sr], ";"))
        end
        @printf("  %s: nom %s | par %s | our %s | prd %s%s\n", key, xn, xp, xo, xr, xp != xo ? "  ← follower 쪽이 결정을 바꿈" : "")
        flush(stdout)
    end
    evk = [k for k in keys(dec) if length(unique(dec[k])) > 1]
    if !isempty(evk)
        nd = mk(S, seed, quota, 0.0, 0.0, Inf)
        ux = sort(unique(vcat([dec[k] for k in evk]...))); xs = [xparse(s, nd.nx) for s in ux]
        cache = Dict{Tuple{String,Vector{Float64}},Vector{Float64}}()
        res = Dict((dz, lv, γ) => cell(nd, xs, dz, lv, γ, cache) for (dz, lv) in cells, γ in gammas)
        open(ocsv, "a") do io
            for k in sort(evk)
                d = dec[k]; idx = [findfirst(==(s), ux) for s in d]
                for (dz, lv) in cells, γ in gammas
                    c, tv = res[(dz, lv, γ)]
                    @printf(io, "%s,%s,%g,%g,%.4f,%s,%s,%s,%s,%s,%s\n", k, dz, lv, γ, tv, d...,
                            join([@sprintf("%.2f", quantile(c[i], 0.95)) for i in idx], ","), join([@sprintf("%.2f", mean(c[i])) for i in idx], ","))
                end
            end
        end
    end
    @printf("  [S=%d 한도 %.0f seed %d] OOS 평가 %d 개, %.0fs\n", S, quota, seed, length(evk), time() - t0); flush(stdout)
end

# ---- 요약 ----
D = [split(l, ",") for l in readlines(dcsv)[2:end] if !isempty(l)]
@printf("\n결정 (%d 개): nominal≠partial %d, partial≠ours %d, ours≠product %d, nominal≠ours %d\n", length(D),
        count(r -> r[5] != r[6], D), count(r -> r[6] != r[7], D), count(r -> r[7] != r[8], D), count(r -> r[5] != r[7], D))
L = [split(l, ",") for l in readlines(ocsv)[2:end] if !isempty(l)]
hd = split(readlines(ocsv)[1], ","); ci(n) = findfirst(==(n), hd)
q(r, n) = parse(Float64, r[ci("q95_$n")])
function ourwin(r)
    others = [n for n in ("nom", "par", "prd") if r[ci("x_$n")] != r[ci("x_our")]]
    isempty(others) && return false
    return all(q(r, "our") < q(r, n) - 1e-6 for n in others)
end
summ = []
for k in unique([join(r[1:4], ",") for r in L])
    rs = [r for r in L if join(r[1:4], ",") == k]; r1 = rs[1]
    r1[ci("x_par")] == r1[ci("x_our")] && continue                       # follower 쪽이 결정을 바꾼 것만
    ok = [r for r in rs if r[ci("design")] != "indep" && parse(Float64, r[ci("tv_mean")]) <= δ]
    ind = [r for r in rs if r[ci("design")] == "indep"]
    push!(summ, (k, count(ourwin, ok), length(ok), count(ourwin, ind), length(ind),
                 r1[ci("x_nom")], r1[ci("x_par")], r1[ci("x_our")], r1[ci("x_prd")]))
end
sort!(summ; by=s -> (s[2] / max(s[3], 1), s[4] / max(s[5], 1)), rev=true)
@printf("partial ≠ ours 인 인스턴스 %d 개: 우리 모형 q95 단독 최선 칸 — 결합 가정 맞는 칸 (d_TV ≤ %.2f) / 독립 추출 칸\n", length(summ), δ)
for s in summ
    @printf("  %s | 맞는 칸 %d/%d, 독립 %d/%d | x nom %s par %s our %s prd %s\n", s...)
end
isempty(summ) ? println("BEST none") : @printf("BEST %s\n", summ[1][1])
