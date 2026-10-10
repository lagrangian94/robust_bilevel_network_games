"""
diag_bnb_lpcount.jl — α-B&B 안의 노드 LP 호출 횟수·벽시계·Gurobi 시간 (barrier 가 따로 풀 때보다 노드당 훨씬 느린 원인).
환경변수: BC_S = 200  BC_TL = 300  BC_VAR = "bar" (bar | base | bard | bardx | bardxp | based)  BC_WORKERS = 12  BC_LOG = 1
"""
root = dirname(@__DIR__)
using JuMP, Gurobi, Printf
include(joinpath(root, "nonzero_sum", "nz_data.jl")); include(joinpath(root, "nonzero_sum", "nz_omega.jl"))
include(joinpath(root, "nonzero_sum", "nz_kkt_eval.jl")); include(joinpath(root, "nonzero_sum", "nz_alpha_bnb.jl"))
S = parse(Int, get(ENV, "BC_S", "200")); TL = parse(Float64, get(ENV, "BC_TL", "300")); var = get(ENV, "BC_VAR", "bar")
nd0 = nz_with_delta(make_location_instance(; coords=:sgb128, reservation=:pair, quota=150.0, wres=300.0, S=S, seed=1,
                                           eps_hat=0.3, eps_tilde=0.3, beta=0.4), 0.1)
nd = nz_with_bounds(nd0; thetaU=nz_theta_circuit_exact(nd0)[1])
x̄ = zeros(nd.nx); x̄[[1, 2]] .= 1
kw = Dict("bar" => (lp_method=2, lp_presolve=-1), "base" => NamedTuple(), "bard" => (lp_method=2, lp_presolve=-1, defer_rows=true),
          "bardx" => (lp_method=2, lp_presolve=-1, defer_rows=true, lp_crossover=0), "based" => (defer_rows=true,),
          "bardxp" => (lp_method=2, lp_presolve=-1, defer_rows=true, lp_crossover=0, purge_k=20))[var]
NW = parse(Int, get(ENV, "BC_WORKERS", "12"))
_NZ_LPLOG_ON[] = get(ENV, "BC_LOG", "1") == "1"
r = nz_alpha_bnb(nd, x̄; nworkers=NW, time_limit=TL, rel_gap=1e-3, verbose=true, log_every=60.0, kw...)
@printf("RESULT %s S=%d: nodes %d, relax(worker 합) %.0fs | LP 호출 %d, LP 벽시계 합 %.0fs, Gurobi solve_time 합 %.0fs, 수치오류 %d (실패 %d)\n",
        var, S, r[:nodes], r[:t_relax], _NZ_LPN[], _NZ_LPWALL[], _NZ_LPGRB[], r[:numerr], r[:numerr_fail])
println("LP 종료 상태: ", r[:lpstat], " | worker ", NW, ", 지운 행 ", r[:purged], ", 끝 RLT 행 수 ", r[:rows_end])
# ---- LP 기록 분석 (BC_LOG=1): 제약 수·동시 실행 LP 수와 Gurobi 시간의 관계 ----
if !isempty(_NZ_LPLOG)
    L = copy(_NZ_LPLOG)
    open(joinpath(root, "nonzero_sum", "logs", "lplog_$(var)_w$(NW)_S$(S).csv"), "w") do io
        println(io, "t_end,tid,ncons,grb,bariter,status")
        for x in L; println(io, join(x, ",")); end
    end
    t0 = minimum(x[1] - x[4] for x in L)
    iv = [(x[1] - x[4], x[1]) for x in L]
    # cc: 그 LP 구간 동안의 평균 동시 LP 수 (자기 포함)
    cc = map(eachindex(L)) do k
        a, b = iv[k]; b - a < 1e-3 && return 1.0
        sum(max(0.0, min(b, d) - max(a, c)) for (c, d) in iv) / (b - a)
    end
    nc = [x[3] for x in L]; g = [x[4] for x in L]; bi = [x[5] for x in L]
    q = sort(nc); cuts = [q[max(1, round(Int, f * length(q)))] for f in (0.25, 0.5, 0.75)]
    @printf("LP %d 개: Gurobi 시간 평균 %.1fs, 제약 수 %d~%d, barrier 반복 평균 %.0f\n", length(L), sum(g) / length(g), minimum(nc), maximum(nc), sum(max.(bi, 0)) / length(bi))
    for (lo, hi, nm) in ((0, cuts[1], "제약 하위 25%"), (cuts[1], cuts[2], "25~50%"), (cuts[2], cuts[3], "50~75%"), (cuts[3], typemax(Int), "상위 25%"))
        idx = findall(x -> lo < x <= hi || (lo == 0 && x <= hi), nc)
        isempty(idx) && continue
        @printf("  %-14s (%6d~%6d): LP %3d 개, 평균 %.1fs, 평균 동시 %.1f, barrier 반복 %.0f\n", nm, minimum(nc[idx]), maximum(nc[idx]),
                length(idx), sum(g[idx]) / length(idx), sum(cc[idx]) / length(idx), sum(max.(bi[idx], 0)) / length(idx))
    end
    for (lo, hi) in ((0, 3), (3, 8), (8, 100))
        idx = findall(c -> lo < c <= hi, cc)
        isempty(idx) && continue
        @printf("  동시 %d~%d: LP %3d 개, 평균 %.1fs, 평균 제약 %d, barrier 반복 %.0f, 반복당 %.3fs\n", lo, hi, length(idx), sum(g[idx]) / length(idx),
                round(Int, sum(nc[idx]) / length(idx)), sum(max.(bi[idx], 0)) / length(idx), sum(g[idx]) / max(1, sum(max.(bi[idx], 0))))
    end
end
