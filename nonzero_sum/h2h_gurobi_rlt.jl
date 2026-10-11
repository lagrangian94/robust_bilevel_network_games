"""
h2h_gurobi_rlt.jl — Gurobi 전역 (Ω) 에 정적 RLT 를 넣은 변형 (nz_gurobi_rlt.jl) 의 비교. 고정 x̄, 목표 gap 까지의 시간·최종 gap.

변형 (GR_VARIANTS, 쉼표):
  g0      : 지금의 Gurobi 전역 (프리솔브 없음)
  p0      : 닫힌 점포 프리솔브 (nz_presolve_hU) 만
  p1 p2 p3: 프리솔브 + RLT level 1 / 2 / 3
환경변수: GR_S = "10,20"  GR_TL = 600  GR_GAP = 5e-3  GR_X = "1,2"  GR_EPS = 0.3  GR_DELTA = 0.1  GR_THREADS = 12
          GR_VARIANTS = "g0,p0,p1,p2,p3"  GR_LOG = 0 (1 이면 Gurobi 로그 출력)
실행: julia -t 4,1 nonzero_sum/h2h_gurobi_rlt.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_gurobi_rlt.jl"))
envf(k, d) = get(ENV, k, d); pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
TL = parse(Float64, envf("GR_TL", "600")); GAP = parse(Float64, envf("GR_GAP", "5e-3"))
ε = parse(Float64, envf("GR_EPS", "0.3")); δ = pd(envf("GR_DELTA", "0.1")); nthr = parse(Int, envf("GR_THREADS", "12"))
xi = parse.(Int, split(envf("GR_X", "1,2"), ","))
vnames = String.(strip.(split(envf("GR_VARIANTS", "g0,p0,p1,p2,p3"), ",")))
showlog = envf("GR_LOG", "0") == "1"
@printf("h2h_gurobi_rlt: x=%s ε=%.2f δ=%s TL=%.0fs 목표 gap=%.1e threads=%d\n", string(xi), ε, string(δ), TL, GAP, nthr); flush(stdout)
for S in parse.(Int, split(envf("GR_S", "10,20"), ","))
    nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=1,
                                               eps_hat=ε, eps_tilde=ε, beta=0.4), δ)
    ndf = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
    x̄ = zeros(ndf.nx); x̄[xi] .= 1
    hUp, nfix = nz_presolve_hU(ndf, x̄)
    ndp = nz_with_hU(ndf, hUp)
    for vn in vnames
        nd = vn == "g0" ? ndf : ndp
        level = vn in ("g0", "p0") ? 0 : parse(Int, vn[2:end])
        O = build_nz_omega_rlt(nd; optimizer=GRB, level=level, silent=!showlog)
        set_optimizer_attribute(O.model, "Threads", nthr)
        nz_set_objective!(O, nd, x̄)
        set_time_limit_sec(O.model, TL); set_optimizer_attribute(O.model, "MIPGap", GAP)
        t = @elapsed optimize!(O.model)
        st = termination_status(O.model)
        lb = has_values(O.model) ? objective_value(O.model) : NaN
        ub = objective_bound(O.model)
        @printf("RESULT S=%d %-3s %-10s LB=%.4f UB=%.4f gap=%.3e time=%.1f nodes=%d | 변수 %d 제약 %d\n",
                S, vn, string(st), lb, ub, (ub - lb) / max(1, abs(lb)), t, Int(round(MOI.get(O.model, MOI.NodeCount()))),
                num_variables(O.model), num_constraints(O.model; count_variable_in_set_constraints=false))
        flush(stdout)
    end
end
