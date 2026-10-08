"""
sweep_coupling_kkt.jl — KKT 독립 평가기 (big-M 없음) 로 모든 x 와 δ 에서 V*_δ(x) 를 계산하고 δ 별 리더 최적해를 전수 열거.
location 처럼 x 가 적은 인스턴스용 (Benders·Ω 와 무관한 기준값).

환경변수: NZ_COORDS (sgb128) NZ_RES (pair) NZ_SEED (1) NZ_S (3) NZ_BETA (0.4)
          SW_EPS = "0.1:0.3;0.2:0.2"  (ε̂:ε̃ 쌍)   SW_DELTAS = "Inf,0.2,0.1,0.05,0.02,0.01,0"   SW_TL = 300
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl"))
include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
envf(k, d) = get(ENV, k, d)
pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
TL = parse(Float64, envf("SW_TL", "300"))
all_x(nd) = [Float64.(collect(bits)) for bits in Iterators.product(fill(0:1, nd.nx)...)
             if sum(bits) <= nd.gamma && all(nd.x_allowed[i] || bits[i] == 0 for i in 1:nd.nx)]
deltas = pd.(split(envf("SW_DELTAS", "Inf,0.2,0.1,0.05,0.02,0.01,0"), ","))
for pair in split(envf("SW_EPS", "0.1:0.3;0.2:0.2"), ";")
    ε̂, ε̃ = parse.(Float64, split(pair, ":"))
    nd0 = make_location_instance(; coords=Symbol(envf("NZ_COORDS", "sgb128")), reservation=Symbol(envf("NZ_RES", "pair")),
        S=parse(Int, envf("NZ_S", "3")), seed=parse(Int, envf("NZ_SEED", "1")), eps_hat=ε̂, eps_tilde=ε̃,
        beta=parse(Float64, envf("NZ_BETA", "0.4")))
    println("="^100, "\n", nd0.name, @sprintf("  ε̂=%.2f ε̃=%.2f β=%.2f", ε̂, ε̃, nd0.beta), "\n", "="^100)
    xs = all_x(nd0)
    V = Dict{Tuple{Int,Float64},Float64}(); B = Dict{Tuple{Int,Float64},Float64}()
    @printf("%-12s", "x \\ δ"); for δ in deltas; @printf(" %12s", string(δ)); end; println()
    for (ix, x) in enumerate(xs)
        @printf("%-12s", string(findall(x .> 0.5)))
        for δ in deltas
            r = try nz_kkt_value(nz_with_delta(nd0, δ), x; optimizer=GRB, time_limit=TL) catch; nothing end
            V[(ix, δ)] = r === nothing ? NaN : r[:value]; B[(ix, δ)] = r === nothing ? NaN : r[:bound]
            flag = (r !== nothing && r[:status] != MOI.OPTIMAL) ? "*" : " "
            @printf(" %11.4f%s", V[(ix, δ)], flag)
        end
        println(); flush(stdout)
    end
    println("(* = 시간 제한, incumbent 값)")
    # 단조성 확인
    nmono = 0
    ds = sort(deltas)
    for ix in eachindex(xs), k in 1:length(ds)-1
        V[(ix, ds[k])] > V[(ix, ds[k+1])] + 1e-5 * max(1, abs(V[(ix, ds[k+1])])) && (nmono += 1)
    end
    @printf("δ 단조성 위반: %d\n", nmono)
    for δ in deltas
        vals = sort([(dot(nd0.qx, xs[ix]) + V[(ix, δ)], findall(xs[ix] .> 0.5)) for ix in eachindex(xs)]; by=first)
        @printf("  δ=%-5s 리더 최적 x*=%-10s 값 %.4f   (2등 %s %.4f)\n", string(δ), string(vals[1][2]), vals[1][1],
                string(vals[2][2]), vals[2][1])
    end
    flush(stdout)
end
