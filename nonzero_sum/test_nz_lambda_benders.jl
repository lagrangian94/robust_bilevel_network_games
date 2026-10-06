"""
test_nz_lambda_benders.jl — λᵁ 를 줄였을 때 (1) 모든 x 에서 여전히 exact 인지 (Ω 상한 ≤ KKT V* + tol),
(2) Benders cut 이 얼마나 강해지는지 (표준 / belief-menu 반복·oracle·LB 궤적).
seed 4, S=3. 환경변수 NZ_LAMBDAS="100,1000" (기본), NZ_SKIP_STD=1 이면 표준 Benders 생략.
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl"))
include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
include(joinpath(root, "nonzero_sum", "nz_benders.jl"))

all_x(nd) = [Float64.(collect(bits)) for bits in Iterators.product(fill(0:1, nd.nx)...) if sum(bits) <= nd.gamma]

function main()
    nd0 = make_location_instance(; S=3, seed=4, wres=100.0)
    lams = parse.(Float64, split(get(ENV, "NZ_LAMBDAS", "100,1000"), ","))
    tol = parse(Float64, get(ENV, "NZ_TOL", "1e-4"))
    otime = parse(Float64, get(ENV, "NZ_ORACLE_TIME", "600"))
    boost = parse(Float64, get(ENV, "NZ_BOOST", "2400"))
    X = all_x(nd0)
    kkt = get(ENV, "NZ_SKIP_ALLX", "0") == "1" ? Dict() : Dict(x => nz_kkt_value(nd0, x; optimizer=GRB)[:value] for x in X)
    for λ in lams
        nd = nz_with_bounds(nd0; lambdaU=λ)
        println("="^70, "\nλᵁ = $λ\n", "="^70)
        if get(ENV, "NZ_SKIP_ALLX", "0") != "1"
            O = build_nz_omega(nd; optimizer=GRB)
            worst = -Inf
            for x in X
                r = nz_solve!(O, nd, x; time_limit=300)
                over = r[:bound] - kkt[x]          # 상한이 V* 를 넘는 양 (> tol 이면 exact 증명 실패)
                worst = max(worst, r[:Fval] - kkt[x])
                @printf("  x=%-14s Ω=%11.4f bound=%11.4f KKT=%11.4f  inc−V*=%+.1e  bd−V*=%+.1e %s\n",
                        string(findall(x .> 0.5)), r[:Fval], r[:bound], kkt[x], r[:Fval] - kkt[x], over,
                        r[:status] == MOI.OPTIMAL ? "" : string(r[:status]))
                flush(stdout)
            end
            @printf("  max(Ω incumbent − V*) = %.2e  (> 0 이면 과대평가)\n", worst)
        end
        println("-- belief-menu Benders")
        rm = nz_belief_menu_benders(nd; optimizer=GRB, tol=tol, oracle_time=otime, boost_time=boost)
        @printf("  belief-menu: %s x*=%s LB=%.4f UB=%.4f iter=%d oracle=%d menu=%d wall=%.1fs\n", rm[:status],
                string(findall(rm[:x] .> 0.5)), rm[:LB], rm[:UB], rm[:iters], rm[:oracle_calls], rm[:menu_size], rm[:wall])
        if get(ENV, "NZ_SKIP_STD", "0") != "1"
            println("-- 표준 Benders")
            rs = nz_standard_benders(nd; optimizer=GRB, tol=tol, oracle_time=otime, boost_time=boost)
            @printf("  standard: %s x*=%s LB=%.4f UB=%.4f iter=%d oracle=%d wall=%.1fs\n", rs[:status],
                    string(findall(rs[:x] .> 0.5)), rs[:LB], rs[:UB], rs[:iters], rs[:oracle_calls], rs[:wall])
        end
        flush(stdout)
    end
end
main()
