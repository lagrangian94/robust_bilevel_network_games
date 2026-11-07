#=
사본 (2026-11-07): 지역 요인 수요 (fac_builder.jl 의 make_sgb15_fac) 와 logs/sgb15_fac/fi15/ 의 in-sample 결정 (sgb15_fac_insample.jl) 으로 바꿈.
  tiltm 에 지역 방향 (서·중·동 평균 수요 ±, draw = ±(시장 수 + 1 + R)) 을 더함. 나머지는 nz_oos_designs.jl 그대로.
nz_oos_designs.jl —SGB15 OOS 설계 비교 (2026-11-05 밤). 저장소 코드는 고치지 않음 (nonzero_sum/loc_oos.jl 의 sgb15 경로 사본).
설계 (zs_oos_designs.jl 과 같은 규칙): NO_DESIGN = common (loc_oos.jl 과 같은 난수) | tvr (TV 거리 정확히 r, Q ~ Dir(κ·1)) | stress (a = (1−t)q̂ + t e_i, belief learn·fixed)
  | tilt (p_θ ∝ q̂ exp(θ z), z = 시나리오 총수요의 표준화에 − 를 붙인 것 → r > 0 이면 총수요가 작은 쪽, d_TV = |r|)
  | tiltm (2026-11-06: 시장별 방향. 시장 j 마다 z = ±(시장 j 수요의 표준화), 총수요 ± 도 (j = 시장 수 + 1). r = NO_R (0.1~0.5), belief learn·fixed.
           출력 param = r, level = learn|fixed, draw = ±j (+ = 그 시장 호황 쪽)).
draw 값 = 리더 목적 q_Bᵀx + CVaR_β^p[φ] (낮을수록 좋음), retailer 는 p̃ 아래 최적 예약 (꼭짓점), recourse 는 pessimistic φ (loc_oos.jl 과 같음).
모형 NO_MODELS (쉼표): 이름=폴더:태그:ε̂:ε̃:δ (폴더는 nonzero_sum/logs/sgb15 아래, 태그 = nominal|datadr|product|coupled). 기본
  N=S10_res10:nominal:0:0:Inf, D3=S10_res10:datadr:0.3:0:Inf, P3=S10_res10:product:0.3:0.3:Inf, C3=S10_res10:coupled:0.3:0.3:0.1
  로그가 없거나 gap > NO_GAPTOL (5e-3) 인 모형은 빼고 "제외" 줄 (N 이 빠지면 오류).
출력: NO_OUT/<design>/sgb15_s<seed>.csv (seed,design,param,level,draw,tv,tv_qhat,model,x,objective), 끝 줄 NO_DONE
=#
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Random, Statistics, Distributions
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
GRB1() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_exact_eval.jl"))
const SPD = joinpath(@__DIR__, "logs", "sgb15_fac")
include(joinpath(@__DIR__, "fac_builder.jl"))

seed_inst = parse(Int, get(ENV, "NO_SEED", "1")); S = 10
design = get(ENV, "NO_DESIGN", "common"); design in ("common", "tvr", "stress", "tilt", "tiltm") || error("NO_DESIGN = common | tvr | stress | tilt | tiltm")
ε, δ0, β, quota, distres = 0.3, 0.1, 0.4, 300.0, 10.0
Npl = parse(Int, get(ENV, "NO_N", design == "common" ? "1000" : "300")); κ = parse(Float64, get(ENV, "NO_KAPPA", "0.5"))
gaptol = parse(Float64, get(ENV, "NO_GAPTOL", "5e-3"))
outdir = joinpath(get(ENV, "NO_OUT", joinpath(SPD, "oosd_fac")), design)
mkpath(outdir)
mspec = split(get(ENV, "NO_MODELS", "N=fi15:nominal:0:0:Inf,D3=fi15:datadr:0.3:0:Inf,C3=fi15:coupled:0.3:0.3:0.1"), ",")

"CVaR_β^p[Z] (위쪽 꼬리), loc_oos.jl 과 같음"
function cvar_upper(Z::AbstractVector, p::AbstractVector, β::Float64)
    ord = sortperm(Z; rev=true); cap = p ./ (1 - β); rest = 1.0; val = 0.0
    for s in ord
        m = min(cap[s], rest); val += m * Z[s]; rest -= m
        rest <= 1e-12 && break
    end
    return val
end
"retailer 최적 예약 (loc_oos.jl 과 같음)"
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

nd = make_sgb15_fac(; S=S, seed=seed_inst, dem=:region, eps_hat=0.0, eps_tilde=0.0, beta=β)
xstr(x) = "{" * join(findall(x .> 0.5), ",") * "}"
parsex(s) = (x = zeros(nd.nx); for t in split(strip(s, ['{', '}', '"']), ","; keepempty=false); x[parse(Int, t)] = 1.0; end; x)
models = Tuple{String,Vector{Float64}}[]
for sp in mspec
    nm, rest = split(sp, "="); dir, tag, eh, et, dl = split(rest, ":")
    want = (parse(Float64, eh), parse(Float64, et), dl == "Inf" ? Inf : parse(Float64, dl))
    path = joinpath(SPD, dir, tag == "nominal" ? "n_s$(seed_inst).log" : "dc_s$(seed_inst).log")
    line = isfile(path) ? filter(l -> startswith(l, "RESULT sgb15fac dem=region ") && occursin(" model=$(tag) ", l), readlines(path)) : String[]
    if length(line) != 1
        nm == "N" && error("Nominal x* 없음: $path")
        @printf("제외 s%d %s: RESULT 줄 %d 개 (%s)\n", seed_inst, nm, length(line), path); continue
    end
    kv = Dict(split(t, "="; limit=2)[1] => split(t, "="; limit=2)[2] for t in split(line[1])[3:end] if occursin("=", t))
    UB, LB = parse(Float64, kv["UB"]), parse(Float64, kv["LB"]); gx = (UB - LB) / max(1.0, abs(UB))
    got = (parse(Float64, kv["epsh"]), parse(Float64, kv["epst"]), parse(Float64, kv["delta"]))
    all(isapprox.(got, want)) || error("ε̂·ε̃·δ = $got, 기대 $want: $path")
    (parse(Int, kv["S"]) == S && parse(Int, kv["seed"]) == seed_inst && isapprox(parse(Float64, kv["beta"]), β) &&
     isapprox(parse(Float64, kv["quota"]), quota) && isapprox(parse(Float64, kv["distres"]), distres) && kv["dist"] == "euclid") ||
        error("인스턴스 인자가 다름: $path")
    if gx > gaptol
        nm == "N" && error("Nominal x* 미인증 (gap $gx): $path")
        @printf("제외 s%d %s: x=%s gap %.2e > %.0e\n", seed_inst, nm, kv["x"], gx, gaptol); continue
    end
    push!(models, (String(nm), parsex(kv["x"])))
end
ux = unique(xstr(x) for (_, x) in models); xu = Dict(xstr(x) => x for (_, x) in models)
@printf("NO sgb15 s%d design=%s: %s\n", seed_inst, design, join(["$m=$(xstr(x))" for (m, x) in models], " ")); flush(stdout)

φcache = Dict{Tuple{String,Vector{Float64}},Vector{Float64}}()
function objs(p, pt)
    res = Dict{String,Float64}()
    for k in ux
        x = xu[k]; h = round.(follower_response(nd, x, pt); digits=6)
        φ = get!(() -> nz_recourse_values(nd, x, h)[2], φcache, (k, h))
        res[k] = dot(nd.qx, x) + cvar_upper(φ, p, β)
    end
    return res
end
io = open(joinpath(outdir, "sgb15_s$(seed_inst).csv"), "w")
println(io, "seed,design,param,level,draw,tv,tv_qhat,model,x,objective")
emit(param, lv, n, p, pt, r) = for (m, x) in models
    @printf(io, "%d,%s,%s,%s,%d,%.5f,%.5f,%s,%s,%.6f\n", seed_inst, design, param, lv, n, 0.5 * sum(abs.(p .- pt)),
            0.5 * sum(abs.(p .- nd.q_hat)), m, replace(xstr(x), "," => ";"), r[xstr(x)])
end
levels = ["Close", "Moderate", "Misspecified"]; lev(tv) = tv <= δ0 ? 1 : tv <= 2ε ? 2 : 3
function draw_pairs(rng, sampler)
    pairs = [Tuple{Vector{Float64},Vector{Float64}}[] for _ in 1:3]; ndr = 0
    while any(length.(pairs) .< Npl)
        ndr += 1; ndr > 20_000 * Npl && error("수준을 채우지 못함: $(length.(pairs))")
        p = sampler(rng); u = rand(rng, Dirichlet(ones(S))); τ = rand(rng)
        pt = (1 - τ) .* p .+ τ .* u; l = lev(0.5 * sum(abs.(p .- pt)))
        length(pairs[l]) < Npl && push!(pairs[l], (p, pt))
    end
    return pairs, ndr
end
function tv_sphere(rng, r)
    q̂ = nd.q_hat
    for _ in 1:1_000_000
        Q = rand(rng, Dirichlet(κ .* ones(S))); d = 0.5 * sum(abs.(Q .- q̂)); d < 1e-9 && continue
        p = q̂ .+ (r / d) .* (Q .- q̂)
        minimum(p) >= 0 && return p ./ sum(p)
    end
    error("TV 구면 추출 실패 (r=$r κ=$κ)")
end
t0 = time()
if design == "common" || design == "tvr"
    params = design == "common" ? [0.1, 0.3, 1.0, 5.0] : parse.(Float64, split(get(ENV, "NO_R", "0.1,0.2,0.3,0.4,0.5"), ","))
    for (gi, prm) in enumerate(params)
        if design == "common"   # loc_oos.jl 과 같은 난수 (LO_SEED = 2026)
            rng = MersenneTwister(hash((2026, gi))); smp = r -> rand(r, Dirichlet(prm .* S .* nd.q_hat))
        else
            rng = MersenneTwister(hash((7071, seed_inst, gi))); smp = r -> tv_sphere(r, prm)
        end
        pairs, ndr = draw_pairs(rng, smp)
        for l in 1:3, (n, (p, pt)) in enumerate(pairs[l])
            emit(@sprintf("%.2f", prm), levels[l], n, p, pt, objs(p, pt))
        end
        @printf("  %s=%.2f (추출 %d 번) %.0fs\n", design == "common" ? "γ" : "r", prm, ndr, time() - t0); flush(stdout)
    end
elseif design == "tilt"
    q̂ = nd.q_hat; D = vec(sum(nd.meta[:xi]; dims=1)); z = -(D .- mean(D)) ./ std(D)
    tilt(θ) = (e = q̂ .* exp.(θ .* z); e ./ sum(e))
    function θ_for(r)
        r == 0 && return 0.0
        lo, hi = 0.0, 1.0
        while 0.5 * sum(abs.(tilt(sign(r) * hi) .- q̂)) < abs(r); hi *= 2; hi > 1e4 && error("기울임 θ 를 못 찾음 (r=$r)"); end
        for _ in 1:100; mid = (lo + hi) / 2; 0.5 * sum(abs.(tilt(sign(r) * mid) .- q̂)) < abs(r) ? (lo = mid) : (hi = mid); end
        return sign(r) * (lo + hi) / 2
    end
    rs = parse.(Float64, split(get(ENV, "NO_R", "-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5"), ","))
    for r in rs
        a = tilt(θ_for(r))
        for (mode, pt) in (("learn", a), ("fixed", q̂))
            emit(@sprintf("%.2f", r), mode, 1, a, pt, objs(a, pt))
        end
    end
    @printf("  tilt %d r, 총수요 = %s %.0fs
", length(rs), join(round.(D; digits=0), " "), time() - t0)
elseif design == "tiltm"
    q̂ = nd.q_hat; Ξ = nd.meta[:xi]; ncu = size(Ξ, 1)
    rs = parse.(Float64, split(get(ENV, "NO_R", "0.1,0.2,0.3,0.4,0.5"), ","))
    reg = nd.meta[:region]
    feats = vcat([Ξ[j, :] for j in 1:ncu], [vec(sum(Ξ; dims=1))], [vec(mean(Ξ[reg .== R, :]; dims=1)) for R in 1:3])
    for (j, F) in enumerate(feats), sg in (1, -1)
        std(F) < 1e-9 && continue
        z = sg .* (F .- mean(F)) ./ std(F)
        tl(θ) = (e = q̂ .* exp.(θ .* z); e ./ sum(e))
        for r in rs
            lo, hi = 0.0, 1.0
            while 0.5 * sum(abs.(tl(hi) .- q̂)) < r; hi *= 2; hi > 1e4 && error("기울임 θ 를 못 찾음 (j=$j r=$r)"); end
            for _ in 1:100; mid = (lo + hi) / 2; 0.5 * sum(abs.(tl(mid) .- q̂)) < r ? (lo = mid) : (hi = mid); end
            a = tl((lo + hi) / 2)
            for (mode, pt) in (("learn", a), ("fixed", q̂))
                emit(@sprintf("%.2f", r), mode, sg * j, a, pt, objs(a, pt))
            end
        end
    end
    @printf("  tiltm %d 방향 × %d r × 2 %.0fs
", 2 * length(feats), length(rs), time() - t0)
else
    q̂ = nd.q_hat
    rs = parse.(Float64, split(get(ENV, "NO_R", "0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6"), ","))
    for r in rs, i in 1:S
        a = (t = r / (1 - q̂[i]); (1 - t) .* q̂ .+ t .* [s == i ? 1.0 : 0.0 for s in 1:S])
        for (mode, pt) in (("learn", a), ("fixed", q̂))
            emit(@sprintf("%.2f", r), mode, i, a, pt, objs(a, pt))
        end
    end
    @printf("  stress %d r × %d i × 2 %.0fs\n", length(rs), S, time() - t0)
end
close(io)
println("NO_DONE s$seed_inst $design $(round(time() - t0; digits=1))s (φ 캐시 $(length(φcache)))"); flush(stdout)
