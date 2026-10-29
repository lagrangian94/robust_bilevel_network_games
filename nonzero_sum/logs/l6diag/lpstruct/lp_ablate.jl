#= lp_ablate.jl — 노드 LP 의 노드 의존 부분 (B) 을 행 종류별로 하나씩 지우고 상한·시간 비교 (지우면 완화라 상한은 같거나 커짐) =#
ENV["LS_NOMAIN"] = "1"; include(joinpath(@__DIR__, "lp_struct.jl"))
function colclass(L)
    cc = Vector{String}(undef, L.nc)
    for j in 1:L.nc
        n = L.cn[j]
        if !anon(n); cc[j] = family(n); continue; end
        hasκ = any(any(!anon(L.cn[k]) && family(L.cn[k]) == "κ" for (k, _) in L.rows[i]) for (i, _) in L.cols[j])
        cc[j] = hasκ ? "Zκ" : "ζΠ"
    end
    return cc
end
function rowgroup(L, cc, i)
    n = L.rn[i]; startswith(n, "rltc") && return "sticky RLT"
    anon(n) || return "이름 있는 행 (" * family(n * "[]") * ")"
    f = Set(cc[j] for (j, _) in L.rows[i])
    ("ŷ" in f || "ȳ" in f) && return "WD/FWD/V (블록 강화)"
    z = [x for x in ("Za", "Zb", "Zc", "Ze") if x in f]
    !isempty(z) && return "Z: α×belief (" * join(z, ",") * ")"
    ("ζW" in f || "ζΠ" in f) && return "RLT: α×쌍대 (ζW, ζΠ)"
    "Zκ" in f && return "κ RLT (Zκ)"
    ("ζL" in f || "ζF" in f) && return "McCormick: α×(r,d) (ζL, ζF)"
    (f ⊆ Set(["r", "ϖ"]) || f ⊆ Set(["d", "π̃"])) && ("ϖ" in f || "π̃" in f) && return "쌍대 범위 (ϖ_H, π̃_H)"
    return "기타 (belief 등)"
end
function main_ablate()
root = joinpath(raw"C:\Users\user\robust_bilevel_network_games", "nonzero_sum", "logs", "l6diag", "lpdump")
for (dir, tn) in (("S20", "ok_300"), ("S20", "ok_750"), ("S50", "ok_60"), ("S50", "ok_75"))
    L = readlp(joinpath(root, dir, tn * ".mps")); cc = colclass(L)
    g = [rowgroup(L, cc, i) for i in 1:L.nr]
    groups = sort(unique(g)); cnt = Dict(x => count(==(x), g) for x in groups)
    M, _ = build(L, collect(1:L.nr), collect(1:L.nc)); s0, z0, t0 = solve!(M); C.GRBfreemodel(M)
    @printf("\n#### %s/%s: 전체 %s z=%.4f %.2f s (%d 행)\n", dir, tn, s0, z0, t0, L.nr)
    # Wcap 은 이름 있는 행이지만 쌍대 범위와 같은 종류라 함께 지움
    abl = [x for x in groups if !startswith(x, "이름 있는 행") && x != "기타 (belief 등)"]
    push!(abl, "쌍대 범위 + Wcap"); push!(abl, "Z 전부"); push!(abl, "(B) 전부: Z + RLT + 범위 + Wcap + sticky + κ")
    for a in abl
        keep = if a == "쌍대 범위 + Wcap"
            [i for i in 1:L.nr if !(g[i] == "쌍대 범위 (ϖ_H, π̃_H)" || g[i] == "이름 있는 행 (Wcap)")]
        elseif a == "Z 전부"
            [i for i in 1:L.nr if !startswith(g[i], "Z:")]
        elseif startswith(a, "(B) 전부")
            [i for i in 1:L.nr if !(startswith(g[i], "Z:") || g[i] in ("RLT: α×쌍대 (ζW, ζΠ)", "쌍대 범위 (ϖ_H, π̃_H)", "이름 있는 행 (Wcap)", "sticky RLT", "κ RLT (Zκ)"))]
        else
            [i for i in 1:L.nr if g[i] != a]
        end
        M, _ = build(L, keep, collect(1:L.nc)); s, z, t = solve!(M); C.GRBfreemodel(M)
        @printf("  − %-44s (%5d 행) %s z=%11.4f  상한 변화 %+8.3f%%  %.2f s\n", a, L.nr - length(keep), s, z, 100 * (z - z0) / abs(z0), t)
        flush(stdout)
    end
end
end; get(ENV, "LA_NOMAIN", "0") == "1" || main_ablate()
