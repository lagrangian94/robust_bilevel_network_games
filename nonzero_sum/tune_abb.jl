"""
tune_abb.jl — 비제로섬 α-B&B 설정의 강건성 튜닝. 기본 인스턴스 (SGB128 + 예약 quota 150, θᵁ = 정확 circuit, WD) 의
고정 x̄ (기본 {1,2}, 가장 어려운 인증 후보) 에서 Ω 를 목표 gap 까지 풀고 시간·최종 gap 을 기록.

변형 (TA_VARIANTS, 쉼표):
  base      : 현재 기본 (branch_score=:viol, branch_point=:mid, dive_max=3, maxrounds=30). 1 단계 로그의 base 는 이전 기본 (:alphahat)
  alphahat  : branch_point=:alphahat (이전 기본)   mdive0 / mrounds10 / mweighted : mid 위에 하나씩
  cap500 / cap200 : 라운드당 추가 RLT 행 상한   active : 활성 시나리오만 분리   cap500act : 둘 다
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
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl")); include(joinpath(root, "nonzero_sum", "nz_lshaped.jl"))
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
    "cap500"   => (sep_maxadd=500,),                     # 라운드당 추가 RLT 행 500 개까지
    "cap200"   => (sep_maxadd=200,),
    "active"   => (sep_active_only=true,),               # 활성 시나리오 행만 분리
    "cap500act"=> (sep_maxadd=500, sep_active_only=true),
    "local8"   => (node_select=:local, local_k=8),       # 상한 상위 8 개 중 직전 상자에 가장 가까운 노드
    "local32"  => (node_select=:local, local_k=32),
    "local8d10"=> (node_select=:local, local_k=8, dive_max=10),
    "child3"   => (child_rounds=3,),                     # 루트가 아닌 노드의 분리 라운드 3 회까지
    "child1"   => (child_rounds=1,),
    "barrier"  => (lp_method=2,),                        # 노드 LP 를 barrier 로
    "lsleader" => (node_lp=:lshaped,),                   # L-shaped: 리더 블록만 분해
    "lsall"    => (node_lp=:lshaped, ls_mw=true, ls_follower=true),   # L-shaped: 리더·follower 분해 + MW
    "lsleader1" => (node_lp=:lshaped, ls_child_rounds=1, ls_root_rounds=30),   # base 와 같은 라운드 상한 (루트 30, 자식 1)
    "lsall1"   => (node_lp=:lshaped, ls_mw=true, ls_follower=true, ls_child_rounds=1, ls_root_rounds=30),
    "weighted" => (branch_score=:weighted,),
    "mid"      => (branch_point=:mid,),
    "dive0"    => (dive_max=0,),
    "dive10"   => (dive_max=10,),
    "rounds10" => (maxrounds=10,),
    "wmid"     => (branch_score=:weighted, branch_point=:mid),
    "dyn"      => (dyn_threads=true,),                    # 노드 LP 스레드 동적 배분 (k>1 이면 concurrent: barrier + dual simplex)
    "dynbar"   => (dyn_threads=true, dyn_method=2),       # k>1 이면 barrier
    "defer"    => (defer_rows=true,),                     # dual simplex + 마지막 라운드 행을 재풀이 없이 자식으로 (노드당 LP 1 회)
    "bar"      => (lp_method=2, lp_presolve=-1),          # 노드 LP barrier + presolve (child_rounds=1 기본)
    "dynbarc"  => (dyn_threads=true, lp_method=2, lp_presolve=-1),   # 1 스레드 barrier, 여러 스레드 concurrent
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
        @printf("RESULT S=%d eps=%.2f delta=%s variant=%-9s %s LB=%.4f UB=%.4f gap=%.3e nodes=%d time=%.1f | worker 합 node=%.0fs relax=%.0fs eval=%.0fs cut=%d\n",
                S, ε, string(δ), v, r[:is_exact] ? "Optimal " : "TimeLim ", r[:LB], r[:UB], gap, r[:nodes], t,
                get(r, :t_node, NaN), get(r, :t_relax, NaN), get(r, :t_eval, NaN), get(r, :ls_cuts, 0))
        if haskey(r, :thr_hist) && sum(r[:thr_hist]) > 0
            h = r[:thr_hist]; println("    스레드 배분 (노드 수): ", join(["$(k)→$(h[k])" for k in eachindex(h) if h[k] > 0], ", "))
        end
        flush(stdout)
    end
end
