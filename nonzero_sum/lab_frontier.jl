"""
lab_frontier.jl — α-B&B (nz_alpha_bnb_dr) 를 TL 초 돌리고 남은 전선 (열린 노드) 을 파일로 저장. lab_boxes.jl 의 입력.

완화를 바꿔 볼 때마다 B&B 전체를 다시 돌리지 않고, 같은 상자 묶음에서 노드 상한만 비교하기 위한 것.
환경변수: LF_S = 20  NZ_SEED = 3  LF_X = "1,2"  LF_TL = 300  LF_GAP = 1e-3  LF_WORKERS = 12  LF_OUT = (저장 경로, .jls)  LF_KW = (α-B&B 인자)
실행: julia -t 14,1 nonzero_sum/lab_frontier.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra, Serialization
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d)
S = parse(Int, envf("LF_S", "20")); seed = parse(Int, envf("NZ_SEED", "3"))
TL = parse(Float64, envf("LF_TL", "300")); GAP = parse(Float64, envf("LF_GAP", "1e-3")); nw = parse(Int, envf("LF_WORKERS", "12"))
xi = parse.(Int, split(envf("LF_X", "1,2"), ","))
out = envf("LF_OUT", joinpath(tempdir(), "nz_frontier_seed$(seed)_S$(S).jls"))
nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=seed,
                                           eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
nd = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
x̄ = zeros(nd.nx); x̄[xi] .= 1
kw = eval(Meta.parse(envf("LF_KW", "NamedTuple()")))            # α-B&B 인자 (예: "(fwd=true, sticky=true)")
t = @elapsed r = nz_alpha_bnb_dr(nd, x̄; nworkers=nw, time_limit=TL, rel_gap=GAP, verbose=true, log_every=60.0, ipopt_time=120.0, kw...)
@printf("LB=%.4f UB=%.4f gap=%.3e nodes=%d (%.0fs) 열린 노드 %d\n", r[:LB], r[:UB], (r[:UB] - r[:LB]) / max(1, abs(r[:LB])), r[:nodes], t,
        length(r[:state].open))
boxes = [(l=n[1], u=n[2], ub=n[3], lo=n[4], hi=n[5], phib=n[7], eq=collect(n[8].eq)) for n in r[:state].open]
serialize(out, (seed=seed, S=S, x=xi, LB=r[:LB], α=r[:α], root_UB=r[:root_UB], boxes=boxes))
println("저장: ", out)
