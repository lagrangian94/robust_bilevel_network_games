"""
test_kkt_reduced.jl — nz_kkt_value_reduced 가 nz_kkt_value 와 같은 V*(x) 를 주는지 (기본 인스턴스 모든 x) + 시간 비교,
그리고 무작위 d = 15 인스턴스에서 축소판이 풀리는지.

환경변수: TK_TL = 300   TK_RANDOM = 1 (무작위 d=15 시험 포함)
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
GRB() = Gurobi.Optimizer(GRB_ENV)
include(joinpath(root, "nonzero_sum", "nz_data.jl"))
include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl"))
TL = parse(Float64, get(ENV, "TK_TL", "300"))
all_x(nd) = [Float64.(collect(bits)) for bits in Iterators.product(fill(0:1, nd.nx)...)
             if sum(bits) <= nd.gamma && all(nd.x_allowed[i] || bits[i] == 0 for i in 1:nd.nx)]
xstr(x) = "{" * join(findall(x .> 0.5), ",") * "}"
run(f, nd, x) = (t = @elapsed(r = try f(nd, x; optimizer=GRB, time_limit=TL) catch e; e end);
                 r isa Exception ? (NaN, Symbol(sprint(showerror, r)), t) : (r[:value], Symbol(r[:status]), t))

for δ in (Inf, 0.1)
    nd = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=3, seed=1,
                                              eps_hat=0.3, eps_tilde=0.3, beta=0.4), δ)
    println("="^90, "\nSGB128 pair quota=150 S=3 ε=0.3 β=0.4 δ=", δ, "\n", "="^90)
    maxd = 0.0; tf = 0.0; tr = 0.0
    for x in all_x(nd)
        vf, sf, t1 = run(nz_kkt_value, nd, x); vr, sr, t2 = run(nz_kkt_value_reduced, nd, x)
        dv = abs(vf - vr) / max(1, abs(vf)); maxd = max(maxd, isnan(dv) ? Inf : dv); tf += t1; tr += t2
        @printf("  x=%-10s full %12.4f %-12s %6.1fs | reduced %12.4f %-12s %6.1fs | rel diff %.1e\n",
                xstr(x), vf, sf, t1, vr, sr, t2, dv)
        flush(stdout)
    end
    @printf("  최대 상대차 %.2e,  총 시간 full %.0fs / reduced %.0fs\n", maxd, tf, tr)
end

if get(ENV, "TK_RANDOM", "1") == "1"
    for (res, seed, S) in ((:pooled, 2, 3), (:pooled, 1, 3), (:pair, 1, 3))
        nd = nz_with_delta(make_location_instance(; coords=:random, n_loc=15, nA=2, nB=3, NB=3, S=S, seed=seed,
            bA=750.0, bB=750.0, qB=750.0, dlo=50.0, dhi=150.0, reservation=res, quota=300.0, wres=300.0,
            eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
        println("="^90, "\nrandom d=15 ", res, " seed=", seed, " S=", S, " δ=0.1 (reduced 만, TL ", TL, "s)\n", "="^90)
        for x in all_x(nd)
            vr, sr, t2 = run(nz_kkt_value_reduced, nd, x)
            @printf("  x=%-10s reduced %12.4f %-14s %6.1fs\n", xstr(x), vr, sr, t2)
            flush(stdout)
            sr == :OPTIMAL || break
        end
    end
end
