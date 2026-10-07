"""
compare_nz_dual_vi.jl — 비제로섬 dual-VI (nz_add_dual_vi!, McCormick 선형화) 를 넣었을 때와 안 넣었을 때 비교.

  1. x 별 VI 하한 L(x) (VI 만 있는 master 에서 x 고정) vs KKT V*(x): 유효성 L ≤ V*, 간격.
  2. root 하한: VI 만 있는 master 의 min q·x + L(x) vs 참 최적 min q·x + V*(x).
  3. (NZ_BENDERS=1) Benders (NZ_METHODS=menu,std) 를 vi = false / true 로 한 번씩.
환경변수: run_nz_benders.jl 과 같은 인스턴스 인자 (기본 NZ_COORDS=random NZ_RES=pooled NZ_SEED=4 NZ_S=3),
         NZ_BENDERS=1, NZ_METHODS=menu,std, NZ_LOCAL_WLS=1, NZ_KKT_TL=300
실행: julia -t 3,1 nonzero_sum/compare_nz_dual_vi.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl"))
include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl"))

envf(k, d) = get(ENV, k, d)
xs_str(x) = string(findall(x .> 0.5))

function main()
    kw = Dict{Symbol,Any}(:coords => Symbol(envf("NZ_COORDS", "random")), :reservation => Symbol(envf("NZ_RES", "pooled")),
        :quota => parse(Float64, envf("NZ_QUOTA", "100")), :wres => parse(Float64, envf("NZ_WRES", "300")),
        :S => parse(Int, envf("NZ_S", "3")), :seed => parse(Int, envf("NZ_SEED", "4")))
    nd = make_location_instance(; kw...)
    @printf("%s  θᵁ=%.3f λᵁ=%.1f  max π̂ᵁ=%.1f  예약 dual 상한 θᵁp=%.1f  hᵁ=%.1f\n", nd.name, nd.thetaU, nd.lambdaU,
            maximum(nd.piLU), nd.thetaU * nd.meta[:p], maximum(nd.hU)); flush(stdout)
    X = [Float64.(collect(b)) for b in Iterators.product(fill(0:1, nd.nx)...) if sum(b) <= nd.gamma]

    # ---- 1. x 별 L(x) vs V*(x) ----
    omp, x, t0 = _nz_omp(nd; optimizer=GRB)
    tb = @elapsed nz_add_dual_vi!(omp, x, t0, nd)
    @printf("VI: 변수 %d, 제약 %d (생성 %.2fs)\n", num_variables(omp), num_constraints(omp; count_variable_in_set_constraints=false), tb)
    tl = parse(Float64, envf("NZ_KKT_TL", "300"))
    # 진단 (NZ_VI_EXACT=1): h·π 를 bilinear 그대로 둔 VI 값 → "h 최소화" 손실과 McCormick 손실을 분리
    ex = envf("NZ_VI_EXACT", "0") == "1"
    if ex
        ompE, xE, t0E = _nz_omp(nd; optimizer=GRB)
        nz_add_dual_vi!(ompE, xE, t0E, nd; exact_h=true)
        set_attribute(ompE, "NonConvex", 2); set_attribute(ompE, "TimeLimit", tl)
    end
    nviol = 0; tot = Dict{Vector{Float64},Float64}(); totV = Dict{Vector{Float64},Float64}()
    for xv in X
        fix.(x, xv; force=true)
        optimize!(omp); L = value(t0)
        V = nz_kkt_value(nd, xv; optimizer=GRB, time_limit=tl)[:value]
        LE = NaN
        if ex
            fix.(xE, xv; force=true); optimize!(ompE)
            LE = termination_status(ompE) == MOI.OPTIMAL ? value(t0E) : objective_bound(ompE) - dot(nd.qx, xv)
        end
        ok = L <= V + 1e-6 * max(1, abs(V)); nviol += !ok
        tot[xv] = dot(nd.qx, xv) + L; totV[xv] = dot(nd.qx, xv) + V
        @printf("  x=%-12s L(x)=%11.4f  V*(x)=%11.4f  간격=%10.4f%s  %s\n", xs_str(xv), L, V, V - L,
                ex ? @sprintf("  | h·π exact: L=%11.4f (McCormick 손실 %.4f, h 최소화 손실 %.4f)", LE, LE - L, V - LE) : "",
                ok ? "" : "← 위반!")
        flush(stdout)
    end
    unfix.(x); set_binary.(x)
    optimize!(omp)
    xo = argmin(totV)
    @printf("유효성 위반 %d / %d.  root 하한 (VI 만) = %.4f at x=%s,  참 최적 = %.4f at x=%s,  VI 없으면 하한 −10⁷\n",
            nviol, length(X), objective_value(omp), xs_str(value.(x)), totV[xo], xs_str(xo))
    flush(stdout)

    envf("NZ_BENDERS", "1") == "1" || return
    # ---- 3. Benders vi = false / true ----
    local_wls = envf("NZ_LOCAL_WLS", "1") == "1"
    nw = local_wls ? 1 : parse(Int, envf("NZ_WORKERS", "12"))
    bkw = local_wls ? (envs=[GRB_ENV], heuristic=false) : NamedTuple()
    common = (optimizer=GRB, tol=1e-4, oracle_time=parse(Float64, envf("NZ_ORACLE_TIME", "300")),
              boost_time=parse(Float64, envf("NZ_BOOST", "1200")), nworkers=nw, bnb_kw=bkw, verbose=false)
    for meth in split(envf("NZ_METHODS", "menu,std"), ","), vi in (false, true)
        r = meth == "menu" ? nz_belief_menu_benders(nd; common..., vi=vi) : nz_standard_benders(nd; common..., vi=vi)
        first_lb = isempty(r[:hist]) ? NaN : r[:hist][1].LB
        @printf("%-4s vi=%-5s %-8s x*=%-8s LB=%11.4f UB=%11.4f  iter=%3d oracle=%3d%s  첫 LB=%.4g  wall=%.1fs\n",
                meth, vi, r[:status], xs_str(r[:x]), r[:LB], r[:UB], r[:iters], r[:oracle_calls],
                meth == "menu" ? @sprintf(" menu=%d", r[:menu_size]) : "", first_lb, r[:wall])
        flush(stdout)
    end
end
main()
