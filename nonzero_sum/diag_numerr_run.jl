"""
diag_numerr_run.jl — α-B&B 노드 LP NUMERICAL_ERROR 재현·덤프 (SGB128 pair, x=∅, δ=0.1, 기본 구성 worker 1 + 휴리스틱).
NZ_NUMERR_DUMP=<폴더> 로 실행하면 실패 LP 3 개를 .mps/.bas 로 저장.
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf
const GRB_ENV = Gurobi.Env(); const HEUR_ENV = Gurobi.Env()
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
nd = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, S=3, seed=1), parse(Float64, get(ENV, "DG_DELTA", "0.1")))
r = nz_alpha_bnb(nd, zeros(nd.nx); nworkers=1, envs=[GRB_ENV, HEUR_ENV], time_limit=parse(Float64, get(ENV, "DG_TL", "400")), verbose=false)
@printf("LB=%.6g UB=%.6g nodes=%d numerr=%d fail=%d\n", r[:LB], r[:UB], r[:nodes], r[:numerr], r[:numerr_fail])
