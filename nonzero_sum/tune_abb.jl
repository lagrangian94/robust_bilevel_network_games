"""
tune_abb.jl — 비제로섬 α-B&B 설정의 강건성 튜닝. 기본 인스턴스 (SGB128 + 예약 quota 150, θᵁ = 정확 circuit, WD) 의
고정 x̄ (기본 {1,2}, 가장 어려운 인증 후보) 에서 Ω 를 목표 gap 까지 풀고 시간·최종 gap 을 기록.

변형 (TA_VARIANTS, 쉼표):
  base      : 현재 기본 (branch_score=:viol, branch_point=:mid, dive_max=3, maxrounds=30). 1 단계 로그의 base 는 이전 기본 (:alphahat)
  alphahat  : branch_point=:alphahat (이전 기본)   mdive0 / mrounds10 / mweighted : mid 위에 하나씩
  weighted  : branch_score=:weighted (위반 × 목적 민감도)
  mid       : branch_point=:mid
  dive0     : dive_max=0          dive10 : dive_max=10
  rounds10  : maxrounds=10
  wmid      : weighted + mid
구성 격자: TA_S × TA_EPS × TA_DELTAS (ε̂ = ε̃ = ε).
환경변수: TA_S = "20,50"  TA_EPS = "0.2,0.3"  TA_DELTAS = "0.1"  TA_X = "1,2"  TA_TL = 300  TA_GAP = 1e-3  TA_WORKERS = 12
          TA_VARIANTS = "base,weighted,mid,dive0,dive10,rounds10"
출력: RESULT 줄 (구성, 변형, 상태, LB, UB, gap, 노드, 시간)
실행: julia -t 14,1 nonzero_sum/tune_abb.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
envf(k, d) = get(ENV, k, d); pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
pl(k, d) = pd.(split(envf(k, d), ","))
TL = parse(Float64, envf("TA_TL", "300")); GAP = parse(Float64, envf("TA_GAP", "1e-3"))
nw = parse(Int, envf("TA_WORKERS", "12"))
xi = parse.(Int, split(envf("TA_X", "1,2"), ","))
variants = Dict(
    "base"     => NamedTuple(),                          # 현재 기본 (2026-10-10 부터 branch_point = :mid)
    "alphahat" => (branch_point=:alphahat,),             # 이전 기본
    "mdive0"   => (dive_max=0,),                         # mid + diving 없음
    "mrounds10"=> (maxrounds=10,),                       # mid + 분리 10 라운드
    "mweighted"=> (branch_score=:weighted,),             # mid + 가중 분기
    "weighted" => (branch_score=:weighted,),
    "mid"      => (branch_point=:mid,),
    "dive0"    => (dive_max=0,),
    "dive10"   => (dive_max=10,),
    "rounds10" => (maxrounds=10,),
    "wmid"     => (branch_score=:weighted, branch_point=:mid),
)
vnames = String.(strip.(split(envf("TA_VARIANTS", "base,weighted,mid,dive0,dive10,rounds10"), ",")))
for v in vnames; haskey(variants, v) || error("알 수 없는 변형: $v"); end
@printf("tune_abb: x=%s TL=%.0fs 목표 gap=%.0e workers=%d 변형=%s\n", string(xi), TL, GAP, nw, join(vnames, ","))
flush(stdout)
for S in Int.(pl("TA_S", "20,50")), ε in pl("TA_EPS", "0.2,0.3"), δ in pl("TA_DELTAS", "0.1")
    nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=1,
                                               eps_hat=ε, eps_tilde=ε, beta=0.4), δ)
    nd = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
    x̄ = zeros(nd.nx); x̄[xi] .= 1
    for v in vnames
        t = @elapsed r = nz_alpha_bnb(nd, x̄; nworkers=nw, time_limit=TL, rel_gap=GAP, verbose=false, variants[v]...)
        gap = (r[:UB] - r[:LB]) / max(1.0, abs(r[:LB]))
        @printf("RESULT S=%d eps=%.2f delta=%s variant=%-9s %s LB=%.4f UB=%.4f gap=%.3e nodes=%d time=%.1f\n",
                S, ε, string(δ), v, r[:is_exact] ? "Optimal " : "TimeLim ", r[:LB], r[:UB], gap, r[:nodes], t)
        flush(stdout)
    end
end
