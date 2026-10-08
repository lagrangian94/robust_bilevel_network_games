"""
check_coupling_aux.jl — 결합 유효성 검증의 보조 확인 두 가지.
 (a) zero-sum grid3x3 (원고 생성, S=5, seed 1, β=0.7): 검증에 쓴 x 에서 V*_Inf 와 V*_0.1 (KKT) → 결합이 binding 했는가
 (b) SGB128 pair: x=∅ 에서 Gurobi 전역 Ω 의 incumbent 가 0 을 넘는가 (δ=Inf, 0.1; 600s)  — 기존 수치 특성인지 확인
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
include(joinpath(root, "network_generator.jl")); using .NetworkGenerator
include(joinpath(root, "true_dro", "true_dro_data.jl")); include(joinpath(root, "nonzero_sum", "nz_paper_instances.jl"))
ndz, _ = make_paper_maxflow("grid3x3"; S=5, seed=1, beta=0.7)
println("(a) zero-sum grid3x3 S=5 seed=1 β=0.7")
for xi in ([9, 10], [10, 11], [9, 11], Int[], [11], [10])
    x = zeros(ndz.nx); x[xi] .= 1
    v = [nz_kkt_value(nz_with_delta(ndz, δ), x; optimizer=GRB, time_limit=600.0)[:value] for δ in (Inf, 0.1, 0.0)]
    @printf("  x=%-8s V*_Inf=%.6f  V*_0.1=%.6f  V*_0=%.6f\n", string(xi), v...)
    flush(stdout)
end
nd = make_location_instance(; coords=:sgb128, reservation=:pair, S=3, seed=1)
println("(b) SGB128 pair x=∅, Gurobi 전역 Ω (기본 gap, 600s)")
for δ in (Inf, 0.1)
    n = nz_with_delta(nd, δ); O = build_nz_omega(n; optimizer=GRB)
    r = nz_solve!(O, n, zeros(n.nx); time_limit=600.0)
    @printf("  δ=%-4s incumbent=%.3e bound=%.3e %s  (참값 0)\n", string(δ), r[:Fval], r[:bound], r[:status])
    flush(stdout)
end
