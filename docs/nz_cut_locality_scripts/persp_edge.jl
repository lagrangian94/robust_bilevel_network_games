include(joinpath(@__DIR__, "persp_defs.jl"))
# 변 위 검산: 다른 좌표는 정수, 한 좌표만 t. SA 는 이론상 현과 같아야 함
for (base, mv) in (([(1, 2)], (2, 1)), (Tuple{Int,Int}[], (3, 3)), ([(2, 1), (3, 2)], (1, 1)))
    x0 = xof(base); x1 = xof(vcat(base, [mv])); P0 = solveW(x0)[1]; P1 = solveW(x1)[1]
    for t in (0.25, 0.5)
        xt = x0 .+ t .* (x1 .- x0); w = solveW(xt)[1]; sa = solveSA(xt)[1]
        @printf("EDGE base=%-18s +%s t=%.2f: 현 %.1f | McCormick %.1f | perspective %.1f\n", string(base), string(mv), t, (1 - t) * P0 + t * P1, w, sa); flush(stdout)
    end
end
# 두 좌표가 함께 분수인 점: 두 거점을 각각 t 만큼
x0 = xof(Tuple{Int,Int}[]); for t in (0.25, 0.5)
    xt = copy(x0); xt[xe(1, 2)] = t; xt[xe(2, 2)] = t
    ends = [solveW(xof(s))[1] for s in (Tuple{Int,Int}[], [(1, 2)], [(2, 2)], [(1, 2), (2, 2)])]
    bil = (1 - t)^2 * ends[1] + t * (1 - t) * (ends[2] + ends[3]) + t^2 * ends[4]
    @printf("JOINT 두 거점 t=%.2f: 이중선형 보간 %.1f | McCormick %.1f | perspective %.1f\n", t, bil, solveW(xt)[1], solveSA(xt)[1]); flush(stdout)
end
