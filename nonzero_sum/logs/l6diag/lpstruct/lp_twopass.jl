#= lp_twopass.jl — 순차 2 단계가 정확한가: (1) 블록을 어떤 접점 ẑ 에서 한 번씩 풀어 절단 하나씩 → (2) 마스터 한 번.
   ẑ = 그 노드의 최적해 (oracle) / 다른 노드의 최적해. 이어서 Benders 로 정확해까지 몇 번 더 필요한지 =#
include(joinpath(@__DIR__, "lp_bcore.jl"))
function cuts_at(P::Split, xfull)
    L = P.L; pool = Any[]
    for g in 1:P.ng
        S, fix, ic, myouts = build_sub(P, g)
        # 출력 o 의 값 = 그 블록 부분의 행 값 (최적해에서)
        ov = [sum(a * xfull[j] for (j, a) in L.rows[P.outs[k][1]] if P.isblk[j] && P.grp[j] == g; init=0.0) for k in myouts]
        zv = vcat([xfull[j] for j in ic], ov)
        C.GRBsetdblattrlist(S, "LB", length(fix), Cint.(fix), zv); C.GRBsetdblattrlist(S, "UB", length(fix), Cint.(fix), zv)
        C.GRBoptimize(S); st(S) == "OPT" || error("블록 $g $(st(S))")
        Φ = getd(S, "ObjVal"); rc = [(r = Ref{Cdouble}(); C.GRBgetdblattrelement(S, "RC", f, r); r[]) for f in fix]
        keys = vcat([L.cn[j] for j in ic], [("out", anon(L.rn[P.outs[k][1]]) ? sig(L, P.outs[k][1]) : L.rn[P.outs[k][1]], P.outs[k][2]) for k in myouts])
        push!(pool, (g, Φ, collect(zip(keys, rc, zv)))); C.GRBfreemodel(S)
    end
    return pool
end
function fullsol(L); M, _ = build(L, collect(1:L.nr), collect(1:L.nc)); s, z, t = solve!(M); x = darr(M, "X", L.nc); C.GRBfreemodel(M); return (z, t, x); end
root = joinpath(raw"C:\Users\user\robust_bilevel_network_games", "nonzero_sum", "logs", "l6diag", "lpdump")
for (dir, a, b) in (("S20", "ok_300", "ok_750"), ("S50", "ok_60", "ok_75"))
    occursin(dir, get(ENV, "LB_CASES", "S20,S50")) || continue
    LA = readlp(joinpath(root, dir, a * ".mps")); zA, _, xA = fullsol(LA); PA = split_lp(LA)
    LB = readlp(joinpath(root, dir, b * ".mps")); zB, tB, xB = fullsol(LB); PB = split_lp(LB)
    @printf("\n#### %s %s (전체 LP %.2f s)\n", dir, b, tB)
    # 이름이 같은 열이면 다른 노드의 해를 그대로 접점으로 (열 수가 같고 이름 대응)
    xA_on_B = [get(Dict(LA.cn[j] => xA[j] for j in 1:LA.nc), LB.cn[j], 0.0) for j in 1:LB.nc]
    for (lbl, xz) in (("자기 최적해에서 절단 1 개씩", xB), ("다른 노드 ($a) 최적해에서 절단 1 개씩", xA_on_B))
        pool = cuts_at(PB, xz)
        run_benders(PB, zB; pool=pool, label="$b ← $lbl", maxit=400)
    end
end
