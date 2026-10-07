"""
enumerate_nz_kkt.jl — 모든 x ∈ X 에서 KKT 독립 평가 V*(x) 로 q·x + V*(x) 를 전수 열거 (참 최적, big-M 없음).
환경변수: run_nz_benders.jl 과 같은 인스턴스 인자 (NZ_COORDS, NZ_RES, NZ_QUOTA, NZ_WRES, NZ_S, NZ_SEED), NZ_TL=600.
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))

envf(k, d) = get(ENV, k, d)
function main()
    nd = make_location_instance(; coords=Symbol(envf("NZ_COORDS", "sgb128")), reservation=Symbol(envf("NZ_RES", "pair")),
        quota=parse(Float64, envf("NZ_QUOTA", "100")), wres=parse(Float64, envf("NZ_WRES", "300")),
        S=parse(Int, envf("NZ_S", "3")), seed=parse(Int, envf("NZ_SEED", "1")))
    tl = parse(Float64, envf("NZ_TL", "600"))
    println(nd.name); flush(stdout)
    X = [Float64.(collect(b)) for b in Iterators.product(fill(0:1, nd.nx)...) if sum(b) <= nd.gamma]
    res = []
    for x in X
        t = @elapsed (k = nz_kkt_value(nd, x; optimizer=GRB, time_limit=tl))
        tot = dot(nd.qx, x) + k[:value]
        push!(res, (x, tot, k[:value], k[:bound], k[:status]))
        @printf("  x=%-14s V*=%11.4f (bd %11.4f, %s, %.1fs)  q·x+V*=%11.4f\n", string(findall(x .> 0.5)), k[:value], k[:bound],
                k[:status], t, tot)
        flush(stdout)
    end
    sort!(res; by=r -> r[2])
    @printf("최적: x*=%s  q·x+V*=%.4f  (2등 %s %.4f)\n", string(findall(res[1][1] .> 0.5)), res[1][2],
            string(findall(res[2][1] .> 0.5)), res[2][2])
end
main()
