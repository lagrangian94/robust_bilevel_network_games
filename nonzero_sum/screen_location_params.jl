"""
screen_location_params.jl — location 인스턴스 파라미터 탐색 (KKT 평가기, x 당 ~1s).
Goyal 기본 사례 값 (용량·출점비·수요·v) 은 고정하고, 원 예제에 없는 예약 파라미터 (p, f, wres) 와 seed 만 바꿔
q·x + V*(x) 가 x 마다 충분히 다른 (리더 결정이 자명하지 않은) 인스턴스를 찾는다.
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
all_x(nd) = [Float64.(collect(bits)) for bits in Iterators.product(fill(0:1, nd.nx)...) if sum(bits) <= nd.gamma]

S = parse(Int, get(ENV, "NZ_S", "3"))
cfg = get(ENV, "NZ_CFG", "base")
# base: Goyal 기본 사례 규모 (d=8, A 1곳 b=720, B 후보 4곳 b=360, N_B=4, 출점비 305, 수요 U[30,240])
# scal: Goyal scalability (d=15, 적격 5곳 중 A 2곳, N_B=2, b=150(d−5)/2=750, 출점비 750, 수요 U[50,150])
base_kw = cfg == "scal" ? (n_loc=15, nA=2, nB=3, NB=2, bA=750.0, bB=750.0, qB=750.0, dlo=50.0, dhi=150.0) : NamedTuple()
wlist = cfg == "scal" ? (200.0, 400.0) : (100.0, 200.0)
for seed in 1:6, (p, f, wres) in ((0.3, 0.1, wlist[1]), (0.2, 0.1, wlist[1]), (0.3, 0.1, wlist[2]))
    nd = make_location_instance(; base_kw..., S=S, seed=seed, p=p, f=f, wres=wres)
    vals = Dict{Vector{Float64},Float64}()
    t = @elapsed for x in all_x(nd)
        vals[x] = nz_kkt_value(nd, x; optimizer=GRB, time_limit=120)[:value]
    end
    tot = Dict(x => dot(nd.qx, x) + v for (x, v) in vals)
    srt = sort(collect(tot); by=last)
    nuniq = length(unique(round.(collect(values(vals)); digits=2)))
    @printf("seed=%d p=%.1f f=%.1f wres=%5.0f | best x=%-10s %.1f  2nd %-10s %.1f | V* 고유값 %2d/%d | %.0fs\n",
            seed, p, f, wres, string(findall(srt[1][1] .> 0.5)), srt[1][2],
            string(findall(srt[2][1] .> 0.5)), srt[2][2], nuniq, length(vals), t)
    flush(stdout)
end
