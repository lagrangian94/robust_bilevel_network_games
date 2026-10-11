"""
scale_oracle.jl — 고정 x̄ 의 oracle 문제 (Ω) 를 목표 gap 까지 푸는 시간을 S 별로 측정 (Gurobi 전역 vs α-B&B).
"허용오차 0.5% 에서 S 를 어디까지 풀 수 있는가" 를 보기 위한 스크립트. 기본 인스턴스 (SGB128 + 예약 quota 150, θᵁ = circuit, WD).

환경변수: SO_S = "3,5,10,20,30"  SO_TL = 600  SO_GAP = 5e-3  SO_X = "1,2"  SO_EPS = 0.3  SO_DELTA = 0.1
          SO_SOLVERS = "gurobi,abb"  (추가: abbdr = 쌍대 범위 조임 시제품, abbdr0 = 같은 코드에서 조임 끔, abbdrnc = 상보성 행 없이)
          SO_WORKERS = 12  SO_ABB = "base" (base | bardx, 쉼표)
출력: RESULT 줄 (S, 해법, 상태, LB, UB, gap, 시간)
실행: julia -t 14,1 nonzero_sum/scale_oracle.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf
const GRB_ENV = Gurobi.Env(); GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d); pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
TL = parse(Float64, envf("SO_TL", "600")); GAP = parse(Float64, envf("SO_GAP", "5e-3"))
ε = parse(Float64, envf("SO_EPS", "0.3")); δ = pd(envf("SO_DELTA", "0.1"))
nw = parse(Int, envf("SO_WORKERS", "12"))
xi = parse.(Int, split(envf("SO_X", "1,2"), ","))
solvers = split(envf("SO_SOLVERS", "gurobi,abb"), ",")
abbkw = Dict("base" => NamedTuple(), "bardx" => (lp_method=2, lp_presolve=-1, defer_rows=true, lp_crossover=0))
abbvars = split(envf("SO_ABB", "base"), ",")
@printf("scale_oracle: x=%s ε=%.2f δ=%s TL=%.0fs 목표 gap=%.1e workers=%d\n", string(xi), ε, string(δ), TL, GAP, nw); flush(stdout)
for S in parse.(Int, split(envf("SO_S", "3,5,10,20,30"), ","))
    nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=1,
                                               eps_hat=ε, eps_tilde=ε, beta=0.4), δ)
    nd = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
    x̄ = zeros(nd.nx); x̄[xi] .= 1
    if "gurobi" in solvers
        O = build_nz_omega(nd; optimizer=GRB)
        set_optimizer_attribute(O.model, "Threads", nw)
        t = @elapsed g = nz_solve!(O, nd, x̄; time_limit=TL, gap=GAP)
        gp = (g[:bound] - g[:Fval]) / max(1, abs(g[:Fval]))
        @printf("RESULT S=%d gurobi     %s LB=%.4f UB=%.4f gap=%.3e time=%.1f\n",
                S, gp <= GAP * 1.0001 ? "Optimal" : "TimeLim", g[:Fval], g[:bound], gp, t)
        flush(stdout)
    end
    for (sv, kwd) in (("abbdr", (dual_range=true,)), ("abbdr0", (dual_range=false,)), ("abbdrnc", (dual_range=true, comp=false)),
                      ("abbdr1", (dual_range=true, vcut=false, cslift=false)),      # 1 차 시제품: 범위 (W, F, C) 만
                      ("abbdrv", (dual_range=true, vcut=true, cslift=false)), ("abbdrx", (dual_range=true, vcut=false, cslift=true)),
                      ("abbdrk", (dual_range=true, branch_point=:kink)), ("abbdrk20", (dual_range=true, branch_point=:kink, kink_width=20.0)),
                      ("abbdrk5", (dual_range=true, branch_point=:kink, kink_width=5.0)))
        sv in solvers || continue            # 쌍대 범위 조임 시제품 (nz_alpha_bnb_dr.jl). abbdr0 = 같은 코드에서 범위 조임만 끔
        t = @elapsed r = nz_alpha_bnb_dr(nd, x̄; nworkers=nw, time_limit=TL, rel_gap=GAP, verbose=true, log_every=60.0, kwd...)
        @printf("RESULT S=%d %-10s %s LB=%.4f UB=%.4f gap=%.3e time=%.1f nodes=%d rootUB=%.2f | relax %.0fs range %.0fs (MIP %d) 고정비율 %.2f 꺾임 분기 %d MIP 실패 %d\n",
                S, sv, r[:is_exact] ? "Optimal" : "TimeLim", r[:LB], r[:UB], (r[:UB] - r[:LB]) / max(1, abs(r[:LB])), t, r[:nodes], r[:root_UB],
                r[:t_relax], r[:t_range], r[:range_mips], r[:fixfrac], r[:kink_branches], r[:range_fail])
        flush(stdout)
    end
    if "abb" in solvers
        for v in abbvars
            t = @elapsed r = nz_alpha_bnb(nd, x̄; nworkers=nw, time_limit=TL, rel_gap=GAP, verbose=true, log_every=60.0, abbkw[v]...)
            @printf("RESULT S=%d abb-%-6s %s LB=%.4f UB=%.4f gap=%.3e time=%.1f nodes=%d rootUB=%.2f\n",
                    S, v, r[:is_exact] ? "Optimal" : "TimeLim", r[:LB], r[:UB], (r[:UB] - r[:LB]) / max(1, abs(r[:LB])), t, r[:nodes], r[:root_UB])
            flush(stdout)
        end
    end
end
