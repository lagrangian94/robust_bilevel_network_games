#=
demo_screen_nz3.jl — 비제로섬 발표 예시 재탐색, follower 믿음을 진짜 분포에 묶은 OOS (2026-10-21 사용자 요청).
원고 OOS (p, p̃ 독립) 는 d_TV(p, p̃) 중앙값이 0.47~0.90 이라 결합 가정 (≤ δ) 이 거의 항상 깨진다 → 두 추출을 추가.
  진짜 분포   p ~ Dir(γ·S·q̂)  (γ ∈ DS_GAMMAS)
  ③ data : follower 가 p 에서 N_F 개 관측 → p̃ = Multinomial(N_F, p) / N_F      (N_F ∈ DS_NF)
  ① mix  : p̃ = (1 − τ) p + τ u,  u ~ Dir(1)  →  d_TV(p, p̃) ≤ τ                   (τ ∈ DS_TAUS)
  실현 비용 = q_Bᵀx + CVaR_β^p[φ(x, h(p̃))]  (낮을수록 좋음, demo_value_nz.jl 과 같은 평가)
대상: screen2_sgb128.csv (demo_screen_nz2.jl, 560 개) 에서 nominal / partial / ours (δ = DS_DELTA) 중 결정이 갈린 (S, 한도, seed).
  그 묶음의 모든 ε 에 대해 product (δ = ∞) 를 새로 풀고 (screen3_product.csv, 이어서 가능), 결정 4 종을 평가.
  같은 (S, 한도, seed) 는 데이터가 같고 난수도 같으므로 결정 x 별 비용을 한 번만 계산해 ε 사이에 재사용.
결과: logs/demo_value/screen3_sgb128.csv — 한 줄 = (인스턴스, 추출, 수준, γ): 모형 4 개의 q95·평균, 실현 d_TV 평균
      끝에 요약: 결합 가정이 맞는 칸 (실현 d_TV 평균 ≤ δ) 에서 우리 모형이 q95 단독 최선인 칸 수로 정렬.
환경변수: DS_GAMMAS = 0.1,1.0   DS_NF = 10,30,100,300   DS_TAUS = 0,0.05,0.1,0.2,0.4,1   DS_M = 300   DS_DELTA = 0.1
          DS_BETA = 0.4   DS_LAMBDA = 100   DS_TL = 600   DS_NWORKERS = 12   DS_SEED = 42
실행: julia -t 14,1 nonzero_sum/demo_screen_nz3.jl
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
gammas = parse.(Float64, envlist("DS_GAMMAS", "0.1,1.0")); NFs = parse.(Int, envlist("DS_NF", "10,30,100,300"))
taus = parse.(Float64, envlist("DS_TAUS", "0,0.05,0.1,0.2,0.4,1")); M = parse(Int, envf("DS_M", "300"))
δ = parse(Float64, envf("DS_DELTA", "0.1")); β = parse(Float64, envf("DS_BETA", "0.4")); lam = parse(Float64, envf("DS_LAMBDA", "100"))
TL = parse(Float64, envf("DS_TL", "600")); nw = parse(Int, envf("DS_NWORKERS", "12")); oseed = parse(Int, envf("DS_SEED", "42"))
Threads.nthreads() >= nw + 2 || error("julia -t $(nw + 2),1 이상 필요")
outdir = joinpath(root, "nonzero_sum", "logs", "demo_value")
src = joinpath(outdir, "screen2_sgb128.csv"); pcsv = joinpath(outdir, "screen3_product.csv"); csv = joinpath(outdir, "screen3_sgb128.csv")
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
                                eps_hat=eh, eps_tilde=et, beta=β)
    isfinite(d) && (nd = nz_with_delta(nd, d))
    return nz_with_bounds(nd; thetaU=nz_theta_circuit_exact(nd)[1], lambdaU=lam)
end
function solve(nd)
    r = nz_belief_menu_benders(nd; optimizer=GRB, tol=5e-3, oracle=:alpha_bnb, bnb=:dr, oracle_gap=1e-3, nworkers=nw,
                               time_limit=TL, local_first=true, local_time=60.0, bnb_kw=(ipopt_time=120.0,), verbose=false)
    return r[:x], r[:UB], string(r[:status])
end

"추출 칸 (design, level, γ) 하나: 결정 x 마다 비용 벡터 M 개와 실현 d_TV 평균"
function cell(nd, xs::Vector{Vector{Float64}}, design, level, γ, cache)
    base = nd.S .* nd.q_hat; rng = MersenneTwister(oseed)
    c = [zeros(M) for _ in xs]; tv = 0.0
    for n in 1:M
        p = rand(rng, Dirichlet(γ .* base)); p ./= sum(p)
        if design == "data"
            pt = rand(rng, Multinomial(Int(level), p)) ./ level
        else
            u = rand(rng, Dirichlet(ones(nd.S))); pt = (1 - level) .* p .+ level .* u
        end
        tv += 0.5 * sum(abs.(p .- pt)) / M
        for (i, x) in enumerate(xs)
            h = round.(follower_response(nd, x, pt); digits=6)
            φ = get!(() -> nz_recourse_values(nd, x, h)[2], cache, (xstr(x), h))
            c[i][n] = dot(nd.qx, x) + cvar_upper(φ, p, β)
        end
    end
    return c, tv
end

# ---- 대상 묶음 ----
rows = [split(l, ",") for l in readlines(src)[2:end] if !isempty(l)]
gkey(r) = (parse(Int, r[1]), parse(Float64, r[3]), parse(Int, r[4]))
groups = sort(unique([gkey(r) for r in rows if !(r[5] == r[6] == r[7])]))
@printf("demo_screen_nz3: 대상 묶음 %d 개 (S, 한도, seed) | γ=%s N_F=%s τ=%s M=%d δ=%.2f\n", length(groups), string(gammas),
        string(NFs), string(taus), M, δ)
flush(stdout)

# ---- product 결정 (없으면 풂) ----
prodx = Dict{String,String}()
isfile(pcsv) || open(pcsv, "w") do io; println(io, "S,eps,quota,seed,x_product,v_product,status"); end
for l in readlines(pcsv)[2:end]
    f = split(l, ","); prodx[join(f[1:4], ",")] = f[5]
end
for g in groups, r in rows
    gkey(r) == g || continue
    k = join(r[1:4], ",")
    haskey(prodx, k) && continue
    S, quota, seed = g; ε = parse(Float64, r[2])
    x, v, st = solve(mk(S, seed, quota, ε, ε, Inf))
    prodx[k] = xstr(x)
    open(pcsv, "a") do io; @printf(io, "%s,%s,%.4f,%s\n", k, xstr(x), v, st); end
    @printf("  product %s: %s\n", k, xstr(x)); flush(stdout)
end

# ---- 평가 ----
names_ = ["nom", "par", "our", "prd"]
done = Set{String}()
if isfile(csv)
    for l in readlines(csv)[2:end]
        push!(done, join(split(l, ",")[1:4], ","))
    end
else
    open(csv, "w") do io
        println(io, "S,eps,quota,seed,design,level,gamma,tv_mean,x_nom,x_par,x_our,x_prd," *
                    join(["q95_$n" for n in names_], ",") * "," * join(["mean_$n" for n in names_], ","))
    end
end
cells = vcat([("data", Float64(n)) for n in NFs], [("mix", t) for t in taus])
for g in groups
    S, quota, seed = g
    grows = [r for r in rows if gkey(r) == g && !(join(r[1:4], ",") in done)]
    isempty(grows) && continue
    t0 = time()
    nd = mk(S, seed, quota, 0.0, 0.0, Inf)
    dec = Dict(join(r[1:4], ",") => [r[5], r[6], r[7], prodx[join(r[1:4], ",")]] for r in grows)
    ux = sort(unique(vcat(values(dec)...)))                       # 이 묶음에 나온 모든 결정
    xs = [xparse(s, nd.nx) for s in ux]; cache = Dict{Tuple{String,Vector{Float64}},Vector{Float64}}()
    res = Dict{Tuple{String,Float64,Float64},Tuple{Vector{Vector{Float64}},Float64}}()
    for (design, level) in cells, γ in gammas
        res[(design, level, γ)] = cell(nd, xs, design, level, γ, cache)
    end
    open(csv, "a") do io
        for r in grows
            k = join(r[1:4], ","); d = dec[k]; idx = [findfirst(==(s), ux) for s in d]
            for (design, level) in cells, γ in gammas
                c, tv = res[(design, level, γ)]
                @printf(io, "%s,%s,%g,%.2f,%.4f,%s,%s,%s,%s,%s,%s\n", k, design, level, γ, tv, d...,
                        join([@sprintf("%.2f", quantile(c[i], 0.95)) for i in idx], ","), join([@sprintf("%.2f", mean(c[i])) for i in idx], ","))
            end
        end
    end
    @printf("  [S=%d 한도 %.0f seed %d] 결정 %s, %d 개 ε, %.0fs\n", S, quota, seed, join(ux, " "), length(grows), time() - t0)
    flush(stdout)
end

# ---- 요약 ----
L = [split(l, ",") for l in readlines(csv)[2:end] if !isempty(l)]
hd = split(readlines(csv)[1], ","); ci(n) = findfirst(==(n), hd)
q(r, n) = parse(Float64, r[ci("q95_$n")]); mn(r, n) = parse(Float64, r[ci("mean_$n")])
"우리 모형이 q95 단독 최선: 결정이 다른 모든 모형보다 q95 가 낮음 (같은 결정은 비교에서 뺌), 다른 결정이 하나 이상"
function ourwin(r)
    others = [n for n in ("nom", "par", "prd") if r[ci("x_$n")] != r[ci("x_our")]]
    isempty(others) && return false
    return all(q(r, "our") < q(r, n) - 1e-6 for n in others)
end
inst = unique([join(r[1:4], ",") for r in L])
summ = []
for k in inst
    rs = [r for r in L if join(r[1:4], ",") == k]
    ok = [r for r in rs if parse(Float64, r[ci("tv_mean")]) <= δ]          # 결합 가정이 맞는 칸
    bad = [r for r in rs if parse(Float64, r[ci("tv_mean")]) > δ]
    r1 = rs[1]
    push!(summ, (k, count(ourwin, ok), length(ok), count(ourwin, bad), length(bad),
                 r1[ci("x_nom")], r1[ci("x_par")], r1[ci("x_our")], r1[ci("x_prd")]))
end
sort!(summ; by=s -> (s[2] / max(s[3], 1), s[4] / max(s[5], 1)), rev=true)
@printf("\n요약 (인스턴스 %d 개): 우리 모형이 q95 단독 최선인 칸 수 — 결합 가정 맞는 칸 (실현 d_TV ≤ %.2f) / 틀린 칸\n", length(summ), δ)
for s in summ[1:min(20, end)]
    @printf("  %s | 맞는 칸 %d/%d, 틀린 칸 %d/%d | x nom %s par %s our %s prd %s\n", s...)
end
isempty(summ) ? println("BEST none") : @printf("BEST %s\n", summ[1][1])
