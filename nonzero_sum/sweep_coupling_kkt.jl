"""
sweep_coupling_kkt.jl — KKT 독립 평가기 (big-M 없음) 로 모든 x 와 δ 에서 V*_δ(x) 를 계산하고 δ 별 리더 최적해를 전수 열거.
location 처럼 x 가 적은 인스턴스용 (Benders·Ω 와 무관한 기준값).

환경변수: NZ_COORDS (sgb128) NZ_RES (pair) NZ_QUOTA (100, 원고 w_i) NZ_WRES (300) NZ_SEED (1) NZ_S (3) NZ_BETA (0.4)
          SW_EPS = "0.1:0.3;0.2:0.2"  (ε̂:ε̃ 쌍)   SW_DELTAS = "Inf,0.2,0.1,0.05,0.02,0.01,0"   SW_TL = 300
          SW_EFF = 1  (유효 반경 rect 로 반경 효과 / 순수 결합 효과 분해)
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
    mk(eh, et) = make_location_instance(; coords=Symbol(envf("NZ_COORDS", "sgb128")), reservation=Symbol(envf("NZ_RES", "pair")),
        quota=parse(Float64, envf("NZ_QUOTA", "100")), wres=parse(Float64, envf("NZ_WRES", "300")),
        S=parse(Int, envf("NZ_S", "3")), seed=parse(Int, envf("NZ_SEED", "1")), eps_hat=eh, eps_tilde=et,
        beta=parse(Float64, envf("NZ_BETA", "0.4")))
    nd0 = mk(ε̂, ε̃)
    println("="^100, "\n", nd0.name, @sprintf("  ε̂=%.2f ε̃=%.2f β=%.2f quota=%s", ε̂, ε̃, nd0.beta, envf("NZ_QUOTA", "100")), "\n", "="^100)
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
    envf("SW_EFF", "1") == "1" || continue
    # 효과 분해 (Lemma geom): 결합은 (1) 투영 반경을 ε̂' = min(ε̂, ε̃+δ), ε̃' = min(ε̃, ε̂+δ) 로 줄이고 (2) 그 위에서 (p, p̃) 를 묶는다.
    # rect(ε̂', ε̃') = (1) 만 반영한 값. rect(ε̂,ε̃) − rect(ε̂',ε̃') = 반경 효과, rect(ε̂',ε̃') − V*_δ = 순수 결합 효과.
    println("효과 분해: rect(ε̂, ε̃) → rect(ε̂', ε̃') (반경) → V*_δ (순수 결합),  값 = qᵀx + V* (리더 최적 기준)")
    Veff = Dict{Tuple{Float64,Float64},Vector{Float64}}()
    for δ in deltas
        eh, et = min(ε̂, ε̃ + δ), min(ε̃, ε̂ + δ)
        if !haskey(Veff, (eh, et))
            nde = mk(eh, et)
            Veff[(eh, et)] = [(r = try nz_kkt_value(nde, x; optimizer=GRB, time_limit=TL) catch; nothing end;
                               r === nothing ? NaN : r[:value] + dot(nd0.qx, x)) for x in xs]
        end
        ve = Veff[(eh, et)]
        vd = [dot(nd0.qx, xs[ix]) + V[(ix, δ)] for ix in eachindex(xs)]
        vr = [dot(nd0.qx, xs[ix]) + V[(ix, Inf)] for ix in eachindex(xs)]
        ie, id, ir = argmin(ve), argmin(vd), argmin(vr)
        @printf("  δ=%-5s ε̂'=%.2f ε̃'=%.2f | rect x*=%-8s %.4f | rect(ε̂',ε̃') x*=%-8s %.4f | δ x*=%-8s %.4f | 반경 %.4f 순수 %.4f\n",
                string(δ), eh, et, string(findall(xs[ir] .> 0.5)), vr[ir], string(findall(xs[ie] .> 0.5)), ve[ie],
                string(findall(xs[id] .> 0.5)), vd[id], vr[ir] - ve[ie], ve[ie] - vd[id])
        flush(stdout)
    end
end
