#=
loc_oos.jl — 원고 6.1 의 Table loc-oos (기본 인스턴스의 out-of-sample, 2026-11-03).

결정: loc_insample.jl (LI_RES = LO_RES) 의 CSV 에서 모형마다 (quota = LO_QUOTA) 최적 x 를 읽는다 (KKT 전수 열거의 argmin).
  Nominal (0,0) / Data-DR (ε,0) / Coupled (ε,ε,δ) 각 δ / Coupled (product) (ε,ε,Inf)
  LO_RES = sgb15 (2026-11-05) 이면 sgb15_insample.jl 의 로그 (LO_XLOG/<model>_s<NZ_SEED>.log) 의 RESULT 줄에서 네 모형의 x* 를 읽는다.
  상대 gap 이 LO_GAPTOL (기본 5e-3 = 계산 실험의 허용오차) 을 넘거나 ε̂·ε̃·δ·β·S·seed·한도·거리 해상도가 다르면 오류.
평가 (모든 모형에 같은 난수, 두 게임 공통 OOS 설계 2026-11-05, true_dro/factor_5/oos_coupled_eval.jl 과 같음):
  진짜 분포 p ~ Dir(γ · S · q̂), γ ∈ LO_GAMMAS. belief p̃ = (1 − τ) p + τ u, u ~ Dir(1), τ ~ U[0, 1].
  d_TV(p, p̃) 로 세 수준에 나눔 ((γ, 수준) 마다 LO_N 쌍이 찰 때까지 추출):
    Close        d_TV ≤ δ₀ (LO_DELTA0 = 0.1, 결합 반경)
    Moderate     δ₀ < d_TV ≤ 2ε (product 만 덮음, 2ε = ε̂ + ε̃. 2026-11-05: 예전 경계 ε̃ 는 ε = δ₀ 이면 구간이 비어 바꿈)
    Misspecified d_TV > 2ε
  retailer (follower) 는 p̃ 아래 최적 예약 h (2 단계 LP 의 꼭짓점 해, demo_value_nz.jl 의 follower_response 와 같음),
  recourse 는 그 h 에서 pessimistic (nz_recourse_values 의 φ: follower recourse 최적해 중 B 에 가장 불리한 값).
  draw 의 값 = 리더 목적 q_Bᵀx + CVaR_β^p[φ] (낮을수록 좋음). 참고로 B 의 기대 이익 −(q_Bᵀx + Σ_s p_s φ_s) 도 draws.csv 에 기록.
요약 (γ (all = γ 전체) × 수준 × 모형): 리더 목적의 평균, 가장 나쁜 LO_TAIL (5%) draw 의 평균.

환경변수: LO_RES = pooled_store | pair (2026-11-04, 원고 6.1 모형 가치 표 = pair) | sgb15   LO_CSV (기본 logs/loc_insample_sgb128_<LO_RES>_S3_seed1_eps0.30_beta0.40.csv)
          LO_QUOTA = 150 (sgb15 은 300)   LO_EPS = 0.3   LO_DELTA0 = 0.1   LO_BETA = 0.4   LO_GAMMAS = 0.1,0.3,1,5   LO_N = 1000   LO_SEED = 2026   LO_TAIL = 0.05
          NZ_S = 3 (sgb15 은 10)   NZ_SEED = 1   NZ_DEMAND = iid | factor   (loc_insample.jl 과 같은 인스턴스)
          sgb15 전용: LO_XLOG (기본 logs/sgb15/S<NZ_S>_res<LO_DISTRES>)   LO_DISTRES = 10   LO_GAPTOL = 5e-3
출력: logs/loc_oos_common/<CSV 이름 | sgb15_S<S>_seed<seed>_res<해상도>>/draws.csv, summary.csv, 표준출력에 표
실행: julia nonzero_sum/loc_oos.jl
=#
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Statistics, Distributions
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))

envf(k, d) = get(ENV, k, d); pf(k, d) = parse(Float64, envf(k, d)); pint(k, d) = parse(Int, envf(k, d))
RES = envf("LO_RES", "pooled_store"); RES in ("pair", "pooled_store", "sgb15") || error("LO_RES = pair | pooled_store | sgb15")
SG = RES == "sgb15"
S = pint("NZ_S", SG ? "10" : "3"); seed_inst = pint("NZ_SEED", "1")
DEM = Symbol(envf("NZ_DEMAND", "iid")); DEM in (:iid, :factor) || error("NZ_DEMAND = iid | factor")
SG && DEM != :iid && error("LO_RES = sgb15 은 NZ_DEMAND = iid 전용 (Yanıkoğlu–Kuhn 의 독립 균등 수요)")
ε = pf("LO_EPS", "0.3"); δ0 = pf("LO_DELTA0", "0.1"); β = pf("LO_BETA", "0.4"); quota = pf("LO_QUOTA", SG ? "300" : "150")
gammas = parse.(Float64, split(envf("LO_GAMMAS", "0.1,0.3,1,5"), ",")); Npl = pint("LO_N", "1000"); seed = pint("LO_SEED", "2026"); tail = pf("LO_TAIL", "0.05")
distres = pf("LO_DISTRES", "10"); gaptol = pf("LO_GAPTOL", "5e-3")
if SG
    xlog = envf("LO_XLOG", joinpath(@__DIR__, "logs", "sgb15", @sprintf("S%d_res%d", S, round(Int, distres))))
    isdir(xlog) || error("sgb15 x* 로그 폴더 없음: $xlog (sgb15_insample.jl 먼저)")
    outdir = joinpath(@__DIR__, "logs", "loc_oos_common", @sprintf("sgb15_S%d_seed%d_res%d", S, seed_inst, round(Int, distres)))
else
    csvin = envf("LO_CSV", joinpath(@__DIR__, "logs", @sprintf("loc_insample_sgb128_%s%s_S%d_seed%d_eps%.2f_beta%.2f.csv", RES, DEM == :factor ? "_factor" : "", S, seed_inst, ε, β)))
    isfile(csvin) || error("loc_insample CSV 없음: $csvin (LI_RES=$RES 로 loc_insample.jl 먼저)")
    outdir = joinpath(@__DIR__, "logs", "loc_oos_common", splitext(basename(csvin))[1])
end
mkpath(outdir)

"CVaR_β^p[Z] (손실의 위쪽 꼬리) = max{νᵀZ : 0 ≤ ν ≤ p/(1−β), 1ᵀν = 1} (demo_value_nz.jl 과 같음)"
function cvar_upper(Z::AbstractVector, p::AbstractVector, β::Float64)
    ord = sortperm(Z; rev=true); cap = p ./ (1 - β); rest = 1.0; val = 0.0
    for s in ord
        m = min(cap[s], rest); val += m * Z[s]; rest -= m
        rest <= 1e-12 && break
    end
    return val
end

"retailer 최적 예약: belief p̃ 아래 max c0ᵀh + Σ_s p̃_s cᵀy_s (꼭짓점 해, demo_value_nz.jl 의 follower_response 와 같음)"
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

# 평가는 θ·λ·ε 와 무관 (정확한 LP). 데이터 (수요, 비용, 용량, 예약 한도) 만 씀 — loc_insample.jl 의 mk 와 같은 생성
nd = SG ?
    make_sgb15_instance(; S=S, seed=seed_inst, dist=:euclid, dist_res=distres, quota=quota, eps_hat=0.0, eps_tilde=0.0, beta=β) :
    RES == "pair" ?
    make_location_instance(; coords=:sgb128, reservation=:pair, quota=quota, wres=300.0,
                           S=S, seed=seed_inst, eps_hat=0.0, eps_tilde=0.0, beta=β, demand=DEM) :
    make_location_instance(; coords=:sgb128, reservation=:pooled, pool_limit=:store, quota=quota,
                           S=S, seed=seed_inst, eps_hat=0.0, eps_tilde=0.0, beta=β, demand=DEM)
xstr(x) = "{" * join(findall(x .> 0.5), ",") * "}"
parsex(s) = (x = zeros(nd.nx); for t in split(strip(s, ['{', '}', '"']), ","; keepempty=false); x[parse(Int, t)] = 1.0; end; x)

models = Tuple{String,Vector{Float64}}[]
if SG
    # ---- 모형별 x* (sgb15_insample.jl 의 RESULT 줄) ----
    for (tag, name, want) in [("nominal", "Nominal", (0.0, 0.0, Inf)), ("datadr", "Data-DR", (ε, 0.0, Inf)),
                              ("product", "Coupled (product)", (ε, ε, Inf)), ("coupled", "Coupled δ=$(δ0)", (ε, ε, δ0))]
        path = joinpath(xlog, "$(tag)_s$(seed_inst).log")
        isfile(path) || error("x* 로그 없음: $path")
        line = filter(l -> startswith(l, "RESULT sgb15 "), readlines(path))
        length(line) == 1 || error("RESULT 줄이 $(length(line)) 개: $path")
        kv = Dict(split(t, "="; limit=2)[1] => split(t, "="; limit=2)[2] for t in split(line[1])[3:end] if occursin("=", t))
        UB, LB = parse(Float64, kv["UB"]), parse(Float64, kv["LB"]); gapx = (UB - LB) / max(1.0, abs(UB))
        gapx <= gaptol || error("x* 가 허용오차 안에서 인증되지 않음 ($(kv["status"]), gap $gapx > $gaptol): $path")
        got = (parse(Float64, kv["epsh"]), parse(Float64, kv["epst"]), parse(Float64, kv["delta"]))
        all(isapprox.(got, want)) || error("ε̂·ε̃·δ = $got, 기대 $want: $path")
        (parse(Int, kv["S"]) == S && parse(Int, kv["seed"]) == seed_inst && isapprox(parse(Float64, kv["beta"]), β) &&
         isapprox(parse(Float64, kv["quota"]), quota) && isapprox(parse(Float64, kv["distres"]), distres) && kv["dist"] == "euclid") ||
            error("인스턴스 인자가 다름 (S·seed·β·한도·거리): $path")
        push!(models, (name, parsex(kv["x"])))
        @printf("  %-20s (ε̂=%.2f ε̃=%.2f δ=%s): x* = %s (%s), UB %.2f, gap %.1e\n", name, got[1], got[2], kv["delta"], kv["x"],
                kv["cities"], UB, gapx)
    end
else
    # ---- 모형별 최적 x (CSV 의 argmin) ----
    rows = Dict{Tuple{Float64,Float64,String},Vector{Tuple{String,Float64}}}()
    for (i, line) in enumerate(eachline(csvin))
        i == 1 && continue
        m = match(r"^([^,]+),([^,]+),([^,]+),([^,]+),\"([^\"]*)\",([^,]+),([^,]+),", line)
        m === nothing && error("CSV 형식: $line")
        parse(Float64, m[1]) == quota || continue
        st = m[7]; st == "OPT" || error("CSV 의 KKT 가 OPTIMAL 이 아님 ($st): $line")
        push!(get!(rows, (parse(Float64, m[2]), parse(Float64, m[3]), m[4]), Tuple{String,Float64}[]), (m[5], parse(Float64, m[6])))
    end
    for (key, name) in sort([(k, k[1] == 0 && k[2] == 0 ? "Nominal" : k[2] == 0 ? "Data-DR" :
                              (k[3] == "Inf" ? "Coupled (product)" : "Coupled δ=$(k[3])") * (k[2] == ε ? "" : " ε̃=$(k[2])"))
                             for k in keys(rows)]; by=last)
        r = rows[key]; length(r) == length(unique(first.(r))) || error("CSV 에 같은 x 가 두 번: $key")
        i = argmin(last.(r)); push!(models, (name, parsex(r[i][1])))
        @printf("  %-20s (ε̂=%.2f ε̃=%.2f δ=%s): x* = %s, 값 %.2f (x %d 개 중)\n", name, key..., r[i][1], r[i][2], length(r))
    end
    isempty(models) && error("quota=$quota 인 행이 CSV 에 없음")
end

# ---- 추출 (모든 모형에 같은 쌍, γ 마다 따로 난수) ----
levels = ["Close", "Moderate", "Misspecified"]
δ0 < 2ε || error("LO_DELTA0 = $δ0 ≥ 2ε = $(2ε) 이면 Moderate 구간이 비어 있음")
lev(tv) = tv <= δ0 ? 1 : tv <= 2ε ? 2 : 3
function draw_pairs(rng, γ)
    pairs = [Tuple{Vector{Float64},Vector{Float64},Float64}[] for _ in 1:3]
    ndraw = 0
    while any(length.(pairs) .< Npl)
        ndraw += 1
        ndraw > 20_000 * Npl && error("수준을 채우지 못함: $(length.(pairs)) (γ=$γ, ε·δ₀ 확인)")
        p = rand(rng, Dirichlet(γ .* S .* nd.q_hat)); u = rand(rng, Dirichlet(ones(S))); τ = rand(rng)
        pt = (1 - τ) .* p .+ τ .* u; tv = 0.5 * sum(abs.(p .- pt)); l = lev(tv)
        length(pairs[l]) < Npl && push!(pairs[l], (p, pt, tv))
    end
    return pairs, ndraw
end
@printf("loc_oos: SGB128 %s quota=%.0f S=%d seed=%d, ε̃=%.2f δ₀=%.2f, Dir(γ·S·q̂) γ ∈ %s, (γ, 수준) 마다 %d 쌍, β=%.2f\n",
        RES, quota, S, seed_inst, ε, δ0, join(gammas, ","), Npl, β)
flush(stdout)

ux = unique(xstr(x) for (_, x) in models); xu = Dict(xstr(x) => x for (_, x) in models)
φcache = Dict{Tuple{String,Vector{Float64}},Vector{Float64}}()       # (x, 예약 h) → φ (꼭짓점 h 는 자주 반복)
io = open(joinpath(outdir, "draws.csv"), "w")
println(io, "gamma,level,draw,tv,tv_qhat,model,x,profit,objective")
obj = Dict{Tuple{Float64,Int,String},Vector{Float64}}()              # (γ, 수준, 모형) → 리더 목적
for (gi, γ) in enumerate(gammas)
    pairs, ndraw = draw_pairs(MersenneTwister(hash((seed, gi))), γ)
    for l in 1:3, (n, (p, pt, tv)) in enumerate(pairs[l])
        res = Dict{String,Tuple{Float64,Float64}}()
        for k in ux
            x = xu[k]; h = round.(follower_response(nd, x, pt); digits=6)
            φ = get!(() -> nz_recourse_values(nd, x, h)[2], φcache, (k, h))
            res[k] = (-(dot(nd.qx, x) + dot(p, φ)), dot(nd.qx, x) + cvar_upper(φ, p, β))
        end
        for (name, x) in models
            pr, ob = res[xstr(x)]
            @printf(io, "%.2f,%s,%d,%.5f,%.5f,%s,%s,%.6f,%.6f\n", γ, levels[l], n, tv, 0.5 * sum(abs.(p .- nd.q_hat)),
                    name, replace(xstr(x), "," => ";"), pr, ob)
            push!(get!(obj, (γ, l, name), Float64[]), ob)
        end
    end
    @printf("  γ=%.1f (추출 %d 번)\n", γ, ndraw); flush(stdout)
end
close(io)

"가장 나쁜 tail 비율 draw 의 평균 (리더 목적은 비용이라 큰 쪽)"
worst(v) = (k = max(1, ceil(Int, tail * length(v))); mean(sort(v; rev=true)[1:k]))
open(joinpath(outdir, "summary.csv"), "w") do f
    println(f, "gamma,level,model,x,n,mean,worst$(tail)")
    @printf("\n리더 목적 (낮을수록 좋음), γ 전체: 평균 / 가장 나쁜 %d%%\n%-20s %-10s", round(Int, 100tail), "model", "x*")
    for L in levels; @printf(" | %-12s %9s %9s", L, "mean", "worst"); end; println()
    for (name, x) in models
        @printf("%-20s %-10s", name, xstr(x))
        for l in 1:3
            for γ in gammas
                v = obj[(γ, l, name)]
                @printf(f, "%.2f,%s,%s,%s,%d,%.6f,%.6f\n", γ, levels[l], name, replace(xstr(x), "," => ";"), length(v), mean(v), worst(v))
            end
            v = reduce(vcat, (obj[(γ, l, name)] for γ in gammas))
            @printf(f, "all,%s,%s,%s,%d,%.6f,%.6f\n", levels[l], name, replace(xstr(x), "," => ";"), length(v), mean(v), worst(v))
            @printf(" | %-12s %9.2f %9.2f", "", mean(v), worst(v))
        end
        println()
    end
end
println("LOC_OOS_DONE $(outdir) (φ 캐시 $(length(φcache)))")
