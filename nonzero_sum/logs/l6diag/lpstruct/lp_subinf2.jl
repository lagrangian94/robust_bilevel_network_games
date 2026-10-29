#= lp_subinf2.jl — 목적 0 (순수 실행 가능성) 부분 LP: A (recourse 4 종 + WD 뺌), B (Ffeas·N1 뺌). 수치 오류 덤프 (numerr_*) 도 =#
ENV["LA_NOMAIN"] = "1"
include(raw"C:\Users\user\robust_bilevel_network_games\nonzero_sum\logs\l6diag\lpstruct\lp_ablate.jl")
const REC = ["이름 있는 행 (Ffeas)", "이름 있는 행 (N1)", "이름 있는 행 (Lflow)", "이름 있는 행 (Fflow)", "WD/FWD/V (블록 강화)"]
function run1(path; full=false)
    L = readlp(path); cc = colclass(L); g = [rowgroup(L, cc, i) for i in 1:L.nr]
    if full
        M, _ = build(L, collect(1:L.nr), collect(1:L.nc)); s0, z0, t0 = solve!(M); C.GRBfreemodel(M)
        @printf("  전체 (%5d 행)  %-4s z=%11.4f  %6.2f s\n", L.nr, s0, z0, t0)
    end
    for (lbl, drop) in (("A0: recourse 4 종 + WD 뺌, 목적 0", REC), ("B0: Ffeas·N1 뺌, 목적 0", REC[1:2]), ("B: Ffeas·N1 뺌", REC[1:2]))
        keep = [i for i in 1:L.nr if !(g[i] in drop)]
        M, _ = startswith(lbl, "B:") ? build(L, keep, collect(1:L.nc)) : build(L, keep, collect(1:L.nc); obj=zeros(L.nc))
        s, z, t = solve!(M); C.GRBfreemodel(M)
        @printf("  %-34s (%5d 행)  %-4s %6.2f s\n", lbl, length(keep), s, t); flush(stdout)
    end
end
r = raw"C:\Users\user\robust_bilevel_network_games\nonzero_sum\logs\l6diag"
for k in (1, 5, 8, 11); @printf("\n#### infeas_%d\n", k); run1(joinpath(r, "infdump", "S20b", "infeas_$k.mps")); end
for d in ("S20", "S50"), k in 1:3; @printf("\n#### (수치 오류) %s/numerr_%d\n", d, k); run1(joinpath(r, "lpdump", d, "numerr_$k.mps"); full=true); end
for (d, f) in (("S20", "ok_300"), ("S50", "ok_30"), ("S50", "ok_60")); @printf("\n#### (실행 가능) %s/%s\n", d, f); run1(joinpath(r, "lpdump", d, f * ".mps")); end
