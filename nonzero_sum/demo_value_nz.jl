#=
demo_value_nz.jl — 학회 발표용 비제로섬 (경쟁 입지 + retailer 예약) 예시: 모델의 가치 (2026-10-20).
demo_value_zs.jl 과 같은 설계 (모델 4 종, belief 오차 τ 를 평가 축으로).

인스턴스: 기본은 Goyal 기본 사례 (SGB128 처음 8 도시, A 점포 1, B 후보 4, N_B ≤ 4) + 거점별 예약 (한도 DN_QUOTA),
          S 시나리오 (수요 U[30, 240]), β, ε. 원고 §6.1.2 의 기본 인스턴스와 같은 생성 (make_location_instance, coords = :sgb128).
1. 풀기 (Algorithm 2 = nz_belief_menu_benders, 본 실험과 같은 인자: α-B&B (쌍대 범위 조임) oracle, local cut, 허용오차 0.5%):
   nominal ε̂ = ε̃ = 0 / partial ε̂ = ε, ε̃ = 0 / product ε̂ = ε̃ = ε, δ = ∞ / coupled ε̂ = ε̃ = ε, δ ∈ DN_DELTAS
   주의: SGB128 거점별 한도의 λᵁ (기본 100) 는 아직 사후 인증 전. 평가 (2) 는 θ·λ 를 쓰지 않는 정확한 LP 라 영향 없음.
2. 평가 (같은 난수를 모든 모델에 씀): τ 마다 N 쌍
   p ~ Dir(DN_CP · S · q̂),  p̃ = (1 − τ) p + τ u,  u ~ Dir(1)
   h = retailer 최적 예약 (belief p̃ 아래 2 단계 LP, 꼭짓점 해),  φ_s = 그 예약에서 recourse 최적해 중 B 에 가장 불리한 값 (pessimistic)
   실현 비용 = q_Bᵀx + CVaR_β^p[φ]   (φ 는 리더 손실 = −v·B 판매량)
결과: logs/demo_value/sgb128/ (풀이 로그, x_<model>.txt, eval_draws.csv, eval_summary.csv). 그림은 true_dro/factor_5/demo_value_plot.jl.

환경변수: DN_S = 10   DN_BETA = 0.4   DN_EPS = 0.3   DN_DELTAS = 0.05,0.1,0.2   DN_QUOTA = 150   DN_SEED_INST = 1
          DN_LAMBDA = 100   DN_TAUS = 0,0.05,0.1,0.15,0.2,0.3,0.4   DN_N = 200   DN_CP = 5   DN_SEED = 2026
          DN_NWORKERS = 12   DN_TL = 3600   DN_STAGE = all | solve | eval | phaseab (원고 OOS 추출) | bias (retailer 수요 편향), 아래 참고   DN_TAG = sgb128 (결과 폴더 이름)
          DN_DEMAND = iid | factor (수요 생성, make_location_instance 의 demand)
실행: julia -t 14,1 nonzero_sum/demo_value_nz.jl
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

envf(k, d) = get(ENV, k, d); pf(k, d) = parse(Float64, envf(k, d)); pint(k, d) = parse(Int, envf(k, d))
envlist(k, d) = [strip(s) for s in split(envf(k, d), ",") if !isempty(strip(s))]

"CVaR_β^p[Z] (손실의 위쪽 꼬리) = max{νᵀZ : 0 ≤ ν ≤ p/(1−β), 1ᵀν = 1}"
function cvar_upper(Z::AbstractVector, p::AbstractVector, β::Float64)
    ord = sortperm(Z; rev=true); cap = p ./ (1 - β); rest = 1.0; val = 0.0
    for s in ord
        m = min(cap[s], rest); val += m * Z[s]; rest -= m
        rest <= 1e-12 && break
    end
    return val
end

"retailer 최적 예약: belief p̃ 아래 max c0ᵀh + Σ_s p̃_s cᵀy_s (exact_lb 의 follower LP 와 같은 식, 꼭짓점 해)"
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

S       = pint("DN_S", "10"); β = pf("DN_BETA", "0.4"); ε = pf("DN_EPS", "0.3")
deltas  = parse.(Float64, envlist("DN_DELTAS", "0.05,0.1,0.2"))
quota   = pf("DN_QUOTA", "150"); seed_inst = pint("DN_SEED_INST", "1"); lam = pf("DN_LAMBDA", "100")
demand  = Symbol(envf("DN_DEMAND", "iid"))     # :iid | :factor (지역 요인 수요, nz_data.jl)
taus    = parse.(Float64, envlist("DN_TAUS", "0,0.05,0.1,0.15,0.2,0.3,0.4"))
Ndraw   = pint("DN_N", "200"); cp = pf("DN_CP", "5"); seed = pint("DN_SEED", "2026")
nw      = pint("DN_NWORKERS", "12"); TL = pf("DN_TL", "3600"); stage = envf("DN_STAGE", "all")
outdir  = joinpath(root, "nonzero_sum", "logs", "demo_value", envf("DN_TAG", "sgb128"))   # 설정별 폴더 (DN_TAG)
mkpath(outdir)

function mk(eh, et, δ)
    nd = make_location_instance(; coords=:sgb128, S=S, seed=seed_inst, reservation=:pooled, pool_limit=:store, quota=quota,
                                eps_hat=eh, eps_tilde=et, beta=β, demand=demand)
    isfinite(δ) && (nd = nz_with_delta(nd, δ))
    return nz_with_bounds(nd; thetaU=nz_theta_circuit_exact(nd)[1], lambdaU=lam)
end

models = [("nominal", 0.0, 0.0, Inf), ("partial", ε, 0.0, Inf), ("product", ε, ε, Inf)]
for δ in deltas
    push!(models, ("coupled_d$(replace(string(δ), "." => "p"))", ε, ε, δ))
end
@printf("demo_value_nz: SGB128 거점별 한도 %.0f, S=%d (inst seed %d), β=%.2f, ε=%.2f, δ=%s, λᵁ=%.0f | τ=%s N=%d cp=%.1f seed=%d | stage=%s\n",
        quota, S, seed_inst, β, ε, string(deltas), lam, string(taus), Ndraw, cp, seed, stage)
flush(stdout)
xfile(name) = joinpath(outdir, "x_$(name).txt")
xstr(x) = "{" * join(findall(x .> 0.5), ",") * "}"

if stage in ("all", "solve")
    Threads.nthreads() >= nw + 2 || error("julia -t $(nw + 2),1 이상 필요")
    for (name, eh, et, δ) in models
        isfile(xfile(name)) && (@printf("  [skip] %s (x 있음)\n", name); flush(stdout); continue)
        nd = mk(eh, et, δ)
        open(joinpath(outdir, "solve_$(name).log"), "w") do io
            redirect_stdout(io) do
                @printf("%s: ε̂=%.2f ε̃=%.2f δ=%s β=%.2f S=%d θᵁ=%.2f λᵁ=%.0f (nz_belief_menu_benders, α-B&B dr, local_first)\n",
                        name, eh, et, string(δ), β, S, nd.thetaU, nd.lambdaU)
                t0 = time()
                r = nz_belief_menu_benders(nd; optimizer=GRB, tol=5e-3, oracle=:alpha_bnb, bnb=:dr, oracle_gap=1e-3, nworkers=nw,
                                           time_limit=TL, local_first=true, local_time=60.0, bnb_kw=(ipopt_time=120.0,), verbose=true)
                @printf("RESULT model=%s status=%s x=%s LB=%.6f UB=%.6f iters=%d time=%.1f\n", name, string(r[:status]),
                        xstr(r[:x]), r[:LB], r[:UB], r[:iters], time() - t0)
                flush(stdout)
                open(xfile(name), "w") do f; println(f, join(findall(r[:x] .> 0.5), ",")); end
            end
        end
        @printf("  [done] %s → %s\n", name, strip(read(xfile(name), String))); flush(stdout)
    end
end

if stage in ("all", "eval")
    nd = mk(0.0, 0.0, Inf)          # 평가는 θ·λ·ε 와 무관 (정확한 LP). 데이터 (수요, 비용, 용량) 만 씀
    xs_of = Dict{String,Vector{Float64}}()
    for (name, _, _, _) in models
        isfile(xfile(name)) || error("x 파일 없음: $(xfile(name)) (DN_STAGE=solve 먼저)")
        idx = [parse(Int, t) for t in split(strip(read(xfile(name), String)), ",") if !isempty(t)]
        x = zeros(nd.nx); x[idx] .= 1.0; xs_of[name] = x
        @printf("  x[%s] = %s\n", name, xstr(x))
    end
    q̂ = nd.q_hat
    rng = MersenneTwister(seed)
    draws = open(joinpath(outdir, "eval_draws.csv"), "w")
    println(draws, "tau,draw,tv,model,cost")
    summ = Dict{Tuple{Float64,String},Vector{Float64}}()
    for τ in taus, n in 1:Ndraw
        p = rand(rng, Dirichlet(cp .* S .* q̂))
        u = rand(rng, Dirichlet(ones(S)))
        pt = (1 - τ) .* p .+ τ .* u
        tv = 0.5 * sum(abs.(p .- pt))
        for (name, _, _, _) in models
            x = xs_of[name]
            h = follower_response(nd, x, pt)
            _, φ = nz_recourse_values(nd, x, h)
            c = dot(nd.qx, x) + cvar_upper(φ, p, β)
            @printf(draws, "%.3f,%d,%.5f,%s,%.6f\n", τ, n, tv, name, c)
            push!(get!(summ, (τ, name), Float64[]), c)
        end
        n % 50 == 0 && (@printf("  τ=%.2f: %d/%d\n", τ, n, Ndraw); flush(stdout); flush(draws))
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

# 원고 §6.2 OOS 와 같은 추출 (true_dro/factor_5/plot_phaseAB_3way.jl): γ 마다
#   Phase A (ζ = 0): p, p̃ 독립 ~ Dir(γ·S·q̂)
#   Phase B (ζ > 0): α = γ·S·q̂ ⊙ (1 + ζ·randn), max(α, 0.01) 로 p, p̃ 를 각각 독립 α 에서
# (q̂ 가 균등이면 원고의 Dir(γ·1) 과 같음). 실현 비용 = q_Bᵀx + CVaR_β^p[φ] (최소화, 낮을수록 좋음).
# DN_STAGE=phaseab 로만 돈다. 환경변수: DN_GAMMAS = 0.1,0.3,0.5,1.0   DN_ZETAS = 0,0.5,1,2,5   DN_M = 500
#   DN_OOS_SEED = 42   DN_EVAL_MODELS = (기본: 모든 모델)
if stage == "phaseab"
    nd = mk(0.0, 0.0, Inf)
    gammas = parse.(Float64, envlist("DN_GAMMAS", "0.1,0.3,0.5,1.0")); zetas = parse.(Float64, envlist("DN_ZETAS", "0,0.5,1,2,5"))
    M = pint("DN_M", "500"); oseed = pint("DN_OOS_SEED", "42")
    enames = envlist("DN_EVAL_MODELS", join(first.(models), ","))
    xs_of = Dict{String,Vector{Float64}}()
    for name in enames
        isfile(xfile(name)) || error("x 파일 없음: $(xfile(name)) (DN_STAGE=solve 먼저)")
        idx = [parse(Int, t) for t in split(strip(read(xfile(name), String)), ",") if !isempty(t)]
        x = zeros(nd.nx); x[idx] .= 1.0; xs_of[name] = x
        @printf("  x[%s] = %s\n", name, xstr(x))
    end
    # 같은 결정은 한 번만 평가 (결정이 같으면 비용도 같음)
    ukeys = unique(xstr(xs_of[n]) for n in enames); xu = Dict(xstr(xs_of[n]) => xs_of[n] for n in enames)
    φcache = Dict{Tuple{String,Vector{Float64}},Vector{Float64}}()     # (x, 예약 h) → φ (꼭짓점 h 는 자주 반복)
    base = nd.S .* nd.q_hat
    io = open(joinpath(outdir, "eval_phaseab.csv"), "w")
    println(io, "gamma,zeta,draw,model,x,cost")
    for γ in gammas, ζ in zetas
        rng = MersenneTwister(oseed); t0 = time()
        for n in 1:M
            if ζ == 0
                p = rand(rng, Dirichlet(γ .* base)); pt = rand(rng, Dirichlet(γ .* base))
            else
                α = max.(γ .* base .* (1 .+ ζ .* randn(rng, nd.S)), 0.01); p = rand(rng, Dirichlet(α))
                αt = max.(γ .* base .* (1 .+ ζ .* randn(rng, nd.S)), 0.01); pt = rand(rng, Dirichlet(αt))
            end
            cost = Dict{String,Float64}()
            for k in ukeys
                x = xu[k]; h = round.(follower_response(nd, x, pt); digits=6)
                φ = get!(() -> nz_recourse_values(nd, x, h)[2], φcache, (k, h))
                cost[k] = dot(nd.qx, x) + cvar_upper(φ, p, β)
            end
            for name in enames
                k = xstr(xs_of[name]); @printf(io, "%.2f,%.2f,%d,%s,%s,%.6f\n", γ, ζ, n, name, replace(k, "," => ";"), cost[k])
            end
        end
        flush(io)
        @printf("  γ=%.1f ζ=%.1f: %d 추출 %.0fs (φ 캐시 %d)\n", γ, ζ, M, time() - t0, length(φcache)); flush(stdout)
    end
    close(io)
    println("PHASEAB_DONE $(outdir)")
end

# retailer 믿음의 수요 편향 (2026-10-21 사용자 요청): 진짜 분포 p ~ Dir(γ·S·q̂), 믿음 p̃_s ∝ p_s · exp(κ z_s),
#   z = 시나리오 총수요의 표준화. κ > 0 = 낙관 (큰 수요 시나리오를 과대평가), κ < 0 = 비관, κ = 0 = 믿음이 맞음.
# 실현 비용은 phaseab 와 같음. DN_STAGE=bias 로만 돈다.
# 환경변수: DN_GAMMAS = 1,5   DN_KAPPAS = -1,-0.5,0,0.25,0.5,1,2   DN_M = 500   DN_OOS_SEED = 42   DN_EVAL_MODELS = (기본: 모든 모델)
if stage == "bias"
    nd = mk(0.0, 0.0, Inf)
    gammas = parse.(Float64, envlist("DN_GAMMAS", "1,5")); kappas = parse.(Float64, envlist("DN_KAPPAS", "-1,-0.5,0,0.25,0.5,1,2"))
    M = pint("DN_M", "500"); oseed = pint("DN_OOS_SEED", "42")
    enames = envlist("DN_EVAL_MODELS", join(first.(models), ","))
    xs_of = Dict{String,Vector{Float64}}()
    for name in enames
        isfile(xfile(name)) || error("x 파일 없음: $(xfile(name)) (DN_STAGE=solve 먼저)")
        idx = [parse(Int, t) for t in split(strip(read(xfile(name), String)), ",") if !isempty(t)]
        x = zeros(nd.nx); x[idx] .= 1.0; xs_of[name] = x
    end
    ukeys = unique(xstr(xs_of[n]) for n in enames); xu = Dict(xstr(xs_of[n]) => xs_of[n] for n in enames)
    φcache = Dict{Tuple{String,Vector{Float64}},Vector{Float64}}()
    rhs0(s) = nz_rhs(nd, zeros(nd.nx), zeros(nz_nh(nd)), s)
    D = [-sum(r for r in rhs0(s) if r < 0) for s in 1:nd.S]     # 시나리오 총수요 (음수 우변 = 수요 행 −ξ, 나머지는 예약·용량 ≥ 0)
    z = (D .- mean(D)) ./ std(D)
    @printf("  시나리오 총수요 = %s\n", string(round.(D; digits=0))); flush(stdout)
    io = open(joinpath(outdir, "eval_bias.csv"), "w")
    println(io, "gamma,kappa,draw,tv,model,x,cost")
    for γ in gammas, κ in kappas
        rng = MersenneTwister(oseed); t0 = time(); tvs = 0.0
        for n in 1:M
            p = rand(rng, Dirichlet(γ .* nd.S .* nd.q_hat)); p ./= sum(p)
            pt = p .* exp.(κ .* z); pt ./= sum(pt); tv = 0.5 * sum(abs.(p .- pt)); tvs += tv / M
            cost = Dict{String,Float64}()
            for k in ukeys
                x = xu[k]; h = round.(follower_response(nd, x, pt); digits=6)
                φ = get!(() -> nz_recourse_values(nd, x, h)[2], φcache, (k, h))
                cost[k] = dot(nd.qx, x) + cvar_upper(φ, p, β)
            end
            for name in enames
                k = xstr(xs_of[name]); @printf(io, "%.2f,%.2f,%d,%.5f,%s,%s,%.6f\n", γ, κ, n, tv, name, replace(k, "," => ";"), cost[k])
            end
        end
        flush(io)
        @printf("  γ=%.1f κ=%.2f: 평균 TV %.3f, %.0fs\n", γ, κ, tvs, time() - t0); flush(stdout)
    end
    close(io)
    println("BIAS_DONE $(outdir)")
end
