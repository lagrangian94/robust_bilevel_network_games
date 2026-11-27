# loc_orcl_check.jl (2026-11-28 새벽): 입지 (loc_tkp, B&P 와 같은 인스턴스) 고정 x̄ 에서 α-B&B oracle 설정별 하한·상한 비교.
#   zero-sum 에서 nz_alpha_bnb_dr 기본 (range_lp = true, 꼭짓점 LP 쌍대 범위 = submodularity 전제) 이 틀린 상한을 냄 (zs_orcl_check.jl) → 입지에서도 확인.
#   판정: 다른 설정의 하한 (incumbent 정확값) 이 기본 설정의 상한보다 크면 기본 설정이 틀림.
#   julia -t 12,1 loc_orcl_check.jl   환경 변수 GEO 1, S 10, XS "3:300;10:200;15:300", TLIM 300, MODES "dr,dr_mip,plain", HUCAP 0
root = "C:/Users/user/robust_bilevel_network_games"
using JuMP, Gurobi, Printf, LinearAlgebra, Random
import MathOptInterface as MOI
include(joinpath(root, "nonzero_sum", "loc_tkp.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
isdefined(Main, :nz_alpha_bnb) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
isdefined(Main, :nz_alpha_bnb_dr) || include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
GEO = parse(Int, get(ENV, "GEO", "1")); S = parse(Int, get(ENV, "S", "10")); TLIM = parse(Float64, get(ENV, "TLIM", "300"))
nd = make_loc_tkp(; g=GEO, S=S, seed=1, eps_hat=0.2, eps_tilde=0.2, delta=0.2 / 3, beta=0.4, lambdaU=1e4, NB=3)   # ccgbp.jl 과 같은 인자
K, tiers, nA = nd.meta[:K], nd.meta[:tiers], nd.meta[:nA]
xb = zeros(nd.nx)
for p in split(get(ENV, "XS", "3:300;10:200;15:300"), ";")
    i, tv = parse.(Int, split(p, ":")); xb[(i - nA - 1) * K + findfirst(==(Float64(tv)), tiers)] = 1.0
end
loc_tkp_xstr(nd, xb) == "{" * get(ENV, "XS", "3:300;10:200;15:300") * "}" || error("x 해석 오류 $(loc_tkp_xstr(nd, xb))")
@printf("geo %d S %d x %s, 스레드 %d, TLIM %.0f\n", GEO, S, loc_tkp_xstr(nd, xb), Threads.nthreads(), TLIM); flush(stdout)
for mode in split(get(ENV, "MODES", "dr,dr_mip,plain"), ",")
    t = @elapsed (r = mode == "dr"     ? nz_alpha_bnb_dr(nd, xb; nworkers=Threads.nthreads() - 1, time_limit=TLIM, rel_gap=1e-4, verbose=false, resp=true) :
                      mode == "dr_mip" ? nz_alpha_bnb_dr(nd, xb; nworkers=Threads.nthreads() - 1, time_limit=TLIM, rel_gap=1e-4, verbose=false, resp=true, range_lp=false) :
                                         nz_alpha_bnb(nd, xb; nworkers=Threads.nthreads() - 2, time_limit=TLIM, rel_gap=1e-4, verbose=false))
    @printf("%-8s V in [%.4f, %.4f], 노드 %d, %.1f s\n", mode, r[:LB], r[:UB], r[:nodes], t); flush(stdout)
end
