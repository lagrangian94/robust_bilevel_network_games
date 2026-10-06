"""
test_nz_lambda.jl — follower 블록 exact penalty λᵁ 민감도.
λᵁ 가 작으면 V̄^F = −λ·gap 가 −∞ 대신 유한 벌점이 되어 반응집합 밖 α 를 허용 → Ω > V* (과대평가, cut 무효).
seed 4 (S=3) 의 몇 개 x 에서 λᵁ ∈ {0.1, 1, 10, 100, 1000} 별 Ω(x) 와 KKT V*(x) 를 비교한다.
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl"))
include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))

function main()
    nd0 = make_location_instance(; S=3, seed=4, wres=100.0)
    xs = [[0.0, 0, 1, 0], [0.0, 0, 0, 1], [0.0, 0, 1, 1], [1.0, 0, 1, 0]]
    kkt = Dict(x => nz_kkt_value(nd0, x; optimizer=GRB)[:value] for x in xs)
    @printf("%-8s %-10s %12s %12s %10s %8s\n", "λᵁ", "x", "Ω(x)", "KKT V*", "Ω−V*", "time")
    for λ in (0.1, 1.0, 10.0, 100.0, 1000.0)
        nd = nz_with_bounds(nd0; lambdaU=λ)
        O = build_nz_omega(nd; optimizer=GRB)
        for x in xs
            t = @elapsed (r = nz_solve!(O, nd, x; time_limit=300))
            @printf("%-8g %-10s %12.4f %12.4f %10.2e %7.1fs %s\n", λ, string(findall(x .> 0.5)), r[:Fval], kkt[x],
                    r[:Fval] - kkt[x], t, r[:status] == MOI.OPTIMAL ? "" : string(r[:status]))
            flush(stdout)
        end
    end
end
main()
