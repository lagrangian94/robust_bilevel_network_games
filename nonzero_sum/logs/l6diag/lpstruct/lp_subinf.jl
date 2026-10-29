#= lp_subinf.jl — recourse 블록을 뺀 부분 LP (완화) 만으로 실행 불가능이 증명되는지, 그 비용. 실행 가능한 노드 (ok_*) 에서 헛도는 비용도 =#
ENV["LA_NOMAIN"] = "1"
include(raw"C:\Users\user\robust_bilevel_network_games\nonzero_sum\logs\l6diag\lpstruct\lp_ablate.jl")
const REC = ["이름 있는 행 (Ffeas)", "이름 있는 행 (N1)", "이름 있는 행 (Lflow)", "이름 있는 행 (Fflow)", "WD/FWD/V (블록 강화)"]
const SUBS = [("A: recourse 4 종 + WD 뺌", REC),
              ("B: Ffeas·N1 만 뺌", REC[1:2]),
              ("C: A 에서 FMC LMC Fpsi Hcon Wcap 도 뺌", vcat(REC, ["이름 있는 행 (FMC)", "이름 있는 행 (LMC)", "이름 있는 행 (Fpsi)", "이름 있는 행 (Hcon)", "이름 있는 행 (Wcap)"]))]
function run1(path)
    L = readlp(path); cc = colclass(L); g = [rowgroup(L, cc, i) for i in 1:L.nr]
    M, _ = build(L, collect(1:L.nr), collect(1:L.nc)); s0, z0, t0 = solve!(M); C.GRBfreemodel(M)
    @printf("  전체 (%5d 행)  %-4s z=%11.4f  %6.2f s\n", L.nr, s0, z0, t0)
    for (lbl, drop) in SUBS
        keep = [i for i in 1:L.nr if !(g[i] in drop)]
        M, _ = build(L, keep, collect(1:L.nc)); s, z, t = solve!(M); C.GRBfreemodel(M)
        @printf("  %-40s (%5d 행)  %-4s z=%11.4f  %6.2f s\n", lbl, length(keep), s, z, t); flush(stdout)
    end
end
idir = raw"C:\Users\user\robust_bilevel_network_games\nonzero_sum\logs\l6diag\infdump\S20b"
odir = raw"C:\Users\user\robust_bilevel_network_games\nonzero_sum\logs\l6diag\lpdump\S20"
for k in 1:12; @printf("\n#### infeas_%d\n", k); run1(joinpath(idir, "infeas_$k.mps")); end
for f in ("ok_300", "ok_450", "ok_750"); @printf("\n#### (실행 가능) S20/%s\n", f); run1(joinpath(odir, f * ".mps")); end
