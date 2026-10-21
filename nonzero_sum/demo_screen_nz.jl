#=
demo_screen_nz.jl — 발표용 비제로섬 예시에 쓸 인스턴스 찾기 (2026-10-21, 사용자 요청 (b)).
SGB128 기본 사례 (+ 거점별 예약) 의 여러 설정에서 모델 4 종을 풀어, 결정 x 가 모델끼리 갈리는 인스턴스를 찾는다.
  nominal ε̂ = ε̃ = 0 / partial ε̂ = ε, ε̃ = 0 / product ε̂ = ε̃ = ε, δ = ∞ / coupled ε̂ = ε̃ = ε, δ = DS_DELTA
풀이는 demo_value_nz.jl 과 같은 Algorithm 2 (nz_belief_menu_benders, α-B&B dr, local_first, 허용오차 0.5%).
격자: DS_S × DS_EPS × DS_QUOTA × DS_SEEDS (β = DS_BETA 고정). 한 줄에 한 인스턴스를 CSV 로, 이미 있는 줄은 건너뜀 (이어서).
판정: n_distinct (서로 다른 x 개수), coupled≠product, partial≠coupled. 끝에 우선순위대로 정렬한 후보를 찍는다:
  1) coupled≠product 이고 partial≠coupled  2) coupled≠product  3) partial≠product  4) nominal≠나머지
환경변수: DS_S = 3,5,10   DS_EPS = 0.1,0.2,0.3,0.5   DS_QUOTA = 100,150,300   DS_SEEDS = 1,2,3,4,5   DS_BETA = 0.4
          DS_DELTA = 0.1   DS_LAMBDA = 100   DS_TL = 600   DS_NWORKERS = 12
결과: nonzero_sum/logs/demo_value/screen_sgb128.csv, 후보 요약은 stdout (마지막 BEST 줄)
실행: julia -t 14,1 nonzero_sum/demo_screen_nz.jl
=#
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_benders.jl"))
isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
isdefined(Main, :nz_alpha_bnb_dr) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
include(joinpath(root, "nonzero_sum", "nz_root_tighten.jl"))

envf(k, d) = get(ENV, k, d)
envlist(k, d) = [strip(s) for s in split(envf(k, d), ",") if !isempty(strip(s))]
Ss = parse.(Int, envlist("DS_S", "3,5,10")); epss = parse.(Float64, envlist("DS_EPS", "0.1,0.2,0.3,0.5"))
quotas = parse.(Float64, envlist("DS_QUOTA", "100,150,300")); seeds = parse.(Int, envlist("DS_SEEDS", "1,2,3,4,5"))
β = parse(Float64, envf("DS_BETA", "0.4")); δ = parse(Float64, envf("DS_DELTA", "0.1")); lam = parse(Float64, envf("DS_LAMBDA", "100"))
TL = parse(Float64, envf("DS_TL", "600")); nw = parse(Int, envf("DS_NWORKERS", "12"))
Threads.nthreads() >= nw + 2 || error("julia -t $(nw + 2),1 이상 필요")
outdir = joinpath(root, "nonzero_sum", "logs", "demo_value"); mkpath(outdir)
csv = joinpath(outdir, "screen_sgb128.csv")
xstr(x) = "{" * join(findall(x .> 0.5), ";") * "}"

function mk(S, seed, quota, eh, et, d)
    nd = make_location_instance(; coords=:sgb128, S=S, seed=seed, reservation=:pooled, pool_limit=:store, quota=quota,
                                eps_hat=eh, eps_tilde=et, beta=β)
    isfinite(d) && (nd = nz_with_delta(nd, d))
    return nz_with_bounds(nd; thetaU=nz_theta_circuit_exact(nd)[1], lambdaU=lam)
end
function solve(nd)
    r = nz_belief_menu_benders(nd; optimizer=GRB, tol=5e-3, oracle=:alpha_bnb, bnb=:dr, oracle_gap=1e-3, nworkers=nw,
                               time_limit=TL, local_first=true, local_time=60.0, bnb_kw=(ipopt_time=120.0,), verbose=false)
    return xstr(r[:x]), r[:UB], string(r[:status])
end

done = Set{String}()
if isfile(csv)
    for l in readlines(csv)[2:end]
        f = split(l, ","); push!(done, join(f[1:4], ","))
    end
else
    open(csv, "w") do io
        println(io, "S,eps,quota,seed,x_nominal,x_partial,x_product,x_coupled,v_nominal,v_partial,v_product,v_coupled,status,n_distinct,coupled_ne_product,partial_ne_coupled")
    end
end
@printf("demo_screen_nz: S=%s ε=%s quota=%s seeds=%s β=%.2f δ=%.2f λᵁ=%.0f TL=%.0f\n", string(Ss), string(epss), string(quotas),
        string(seeds), β, δ, lam, TL)
flush(stdout)
for S in Ss, ε in epss, quota in quotas, seed in seeds
    key = @sprintf("%d,%.2f,%.0f,%d", S, ε, quota, seed)
    key in done && continue
    t0 = time()
    res = [solve(mk(S, seed, quota, eh, et, d)) for (eh, et, d) in ((0.0, 0.0, Inf), (ε, 0.0, Inf), (ε, ε, Inf), (ε, ε, δ))]
    xs = [r[1] for r in res]; vs = [r[2] for r in res]; sts = join([r[3] for r in res], ";")
    nd_ = length(unique(xs)); cp_ = xs[4] != xs[3]; pc_ = xs[2] != xs[4]
    open(csv, "a") do io
        @printf(io, "%s,%s,%s,%s,%s,%.4f,%.4f,%.4f,%.4f,%s,%d,%d,%d\n", key, xs[1], xs[2], xs[3], xs[4], vs[1], vs[2], vs[3], vs[4],
                sts, nd_, cp_, pc_)
    end
    @printf("  %s: x = %s | %s | %s | %s  (distinct %d%s%s) %.0fs\n", key, xs[1], xs[2], xs[3], xs[4], nd_,
            cp_ ? ", coupled≠product" : "", pc_ ? ", partial≠coupled" : "", time() - t0)
    flush(stdout)
end

# 후보 정리
rows = [split(l, ",") for l in readlines(csv)[2:end]]
score(r) = (r[15] == "1" && r[16] == "1") ? 4 : r[15] == "1" ? 3 : r[6] != r[7] ? 2 : (r[5] != r[6] || r[5] != r[7]) ? 1 : 0
cands = sort([r for r in rows if score(r) > 0]; by=r -> (-score(r), -parse(Int, r[14]), -parse(Int, r[1])))
@printf("\n후보 %d 개 (전체 %d):\n", length(cands), length(rows))
for r in cands[1:min(15, end)]
    @printf("  score %d | S=%s ε=%s quota=%s seed=%s | x: nominal %s, partial %s, product %s, coupled %s\n",
            score(r), r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8])
end
if isempty(cands)
    println("BEST none")
else
    b = cands[1]
    @printf("BEST S=%s eps=%s quota=%s seed=%s score=%d\n", b[1], b[2], b[3], b[4], score(b))
end
