ENV["LS_NOMAIN"] = "1"; include(joinpath(@__DIR__, "lp_struct.jl"))
ENV["LA_NOMAIN"] = "1"; include(joinpath(@__DIR__, "lp_ablate.jl"))
root = joinpath(raw"C:\Users\user\robust_bilevel_network_games", "nonzero_sum", "logs", "l6diag", "lpdump")
for (dir, tn) in (("S50", "ok_60"), ("S50", "ok_75"))
    L = readlp(joinpath(root, dir, tn * ".mps")); cc = colclass(L); g = [rowgroup(L, cc, i) for i in 1:L.nr]
    drop(i) = g[i] in ("쌍대 범위 (ϖ_H, π̃_H)", "이름 있는 행 (Wcap)", "Z: α×belief (Zb)", "Z: α×belief (Zc)", "Z: α×belief (Za,Zc)")
    for rep in 1:2
        for (lbl, keep) in (("전체", collect(1:L.nr)), ("범위+Wcap+Zb+Zc 제거", [i for i in 1:L.nr if !drop(i)]))
            M, _ = build(L, keep, collect(1:L.nc)); s, z, t = solve!(M); it = Int(getd(M, "IterCount")); C.GRBfreemodel(M)
            @printf("%s/%s 반복%d %-22s %6d 행 z=%.4f %.2f s (%d iter)\n", dir, tn, rep, lbl, length(keep), z, t, it)
        end
    end
end
