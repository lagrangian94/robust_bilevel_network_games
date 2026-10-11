"""
test_nz_alpha_bnb_dr.jl — 쌍대 범위 조임 α-B&B (nz_alpha_bnb_dr.jl) 의 유효성·효과 확인.
독립 KKT 평가기의 V*(x̄) 가 [LB, UB] 안에 있는지 (유효성), 같은 탐색에서 dual_range 끔/켬의 노드 수·시간·gap 비교.

환경변수: TD_S = "3,5"  TD_X = "1,2;1;2;"  (세미콜론으로 여러 x̄, 빈 칸은 ∅)  TD_TL = 300  TD_GAP = 1e-3  TD_WORKERS = 4
          TD_EPS = 0.3  TD_DELTA = 0.1  TD_KKT_TL = 600  TD_MODES = "off,on"  (off: 기존 완화, on: 범위 조임, nocomp: 상보성 행 없이)
실행: julia -t 6,1 nonzero_sum/test_nz_alpha_bnb_dr.jl
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf, LinearAlgebra
const GRB_ENV = Gurobi.Env()
const KKT_THR = parse(Int, get(ENV, "TD_KKT_THREADS", "4"))
GRB() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), KKT_THR); o)
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
include(joinpath(root, "nonzero_sum", "nz_alpha_bnb_dr.jl"))
envf(k, d) = get(ENV, k, d); pd(s) = strip(s) == "Inf" ? Inf : parse(Float64, s)
TL = parse(Float64, envf("TD_TL", "300")); GAP = parse(Float64, envf("TD_GAP", "1e-3")); nw = parse(Int, envf("TD_WORKERS", "4"))
ε = parse(Float64, envf("TD_EPS", "0.3")); δ = pd(envf("TD_DELTA", "0.1")); kktTL = parse(Float64, envf("TD_KKT_TL", "600"))
xs = [isempty(strip(t)) ? Int[] : parse.(Int, split(t, ",")) for t in split(envf("TD_X", "1,2;1;2;"), ";")]
modes = String.(strip.(split(envf("TD_MODES", "off,on"), ",")))
kw = Dict("off" => (dual_range=false,), "on" => NamedTuple(), "nocomp" => (comp=false,),
          "on1" => (vcut=false, cslift=false), "onv" => (cslift=false,), "onx" => (vcut=false,))
nbad = 0
for S in parse.(Int, split(envf("TD_S", "3,5"), ","))
    nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=1,
                                               eps_hat=ε, eps_tilde=ε, beta=0.4), δ)
    nd = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
    for xi in xs
        x̄ = zeros(nd.nx); x̄[xi] .= 1
        V = NaN; Vb = NaN
        if kktTL > 0
            k = nz_kkt_value(nd, x̄; optimizer=GRB, time_limit=kktTL)
            V = k[:value]; Vb = k[:bound]
            @printf("S=%d x=%s KKT %s V*=%.5f (bound %.5f)\n", S, string(xi), string(k[:status]), V, Vb); flush(stdout)
        end
        for md in modes
            t = @elapsed r = nz_alpha_bnb_dr(nd, x̄; nworkers=nw, time_limit=TL, rel_gap=GAP, verbose=false, kw[md]...)
            sc = max(1.0, abs(r[:LB]))
            # 유효성: 실현 가능해 값 LB ≤ V* (KKT 상한 Vb 기준), 상한 UB ≥ V* (KKT 실현 가능해 값 V 기준)
            ok = isnan(V) || (r[:LB] <= Vb + 1e-5 * sc && r[:UB] >= V - 1e-5 * sc)
            ok || (global nbad += 1)
            @printf("RESULT S=%d x=%-7s %-6s %s LB=%.4f UB=%.4f gap=%.3e nodes=%d time=%.1f rootUB=%.2f | relax %.0fs range %.0fs (MIP %d, 실패 %d) 고정비율 %.2f | %s\n",
                    S, string(xi), md, r[:is_exact] ? "Optimal" : "TimeLim", r[:LB], r[:UB], (r[:UB] - r[:LB]) / sc, r[:nodes], t, r[:root_UB],
                    r[:t_relax], r[:t_range], r[:range_mips], r[:range_fail], r[:fixfrac], ok ? "유효" : "**무효: V* 가 [LB, UB] 밖**")
            flush(stdout)
        end
    end
end
println(nbad == 0 ? "모든 실행에서 KKT V* ∈ [LB, UB]" : "무효 $nbad 건")
