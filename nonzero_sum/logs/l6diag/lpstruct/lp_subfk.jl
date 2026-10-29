#= lp_subfk.jl — 부분 LP (A0: recourse 4 종 + WD 뺌 / A1: 이름 있는 recourse 4 종만 뺌, 목적 0) 의 Farkas 쌍대가 엄밀 검증
   (_nz_dr_farkas_certified 와 같은 규칙, rtol 1e-9) 을 통과하는지. 부분 LP 의 증명서 = 뺀 행에 0 을 준 전체 LP 의 증명서 =#
ENV["LA_NOMAIN"] = "1"
include(raw"C:\Users\user\robust_bilevel_network_games\nonzero_sum\logs\l6diag\lpstruct\lp_ablate.jl")
const REC = ["이름 있는 행 (Ffeas)", "이름 있는 행 (N1)", "이름 있는 행 (Lflow)", "이름 있는 행 (Fflow)", "WD/FWD/V (블록 강화)"]
# Gurobi: λᵀAx ≤ λᵀb 가 유효 (≤ 행 λ ≥ 0, ≥ 행 λ ≤ 0). y = −λ 로 MOI 규칙 (Σ y a x ≥ Σ y b) 에 맞춤
function certify(L, keep, λ; rtol=1e-9)
    g = zeros(L.nc); ga = zeros(L.nc); β = 0.0; βa = 0.0
    for (k, i) in enumerate(keep)
        y = -λ[k]; s = Char(L.sense[i])
        y = s == '<' ? min(y, 0.0) : s == '>' ? max(y, 0.0) : y
        y == 0 && continue
        β += y * L.rhs[i]; βa += abs(y * L.rhs[i])
        for (j, a) in L.rows[i]; g[j] += y * a; ga[j] += abs(y * a); end
    end
    mx = 0.0; sl = 0.0
    for j in 1:L.nc
        g[j] == 0 && continue
        bj = g[j] > 0 ? L.ub[j] : L.lb[j]
        if abs(bj) < 1e29
            mx += g[j] * bj; sl += ga[j] * abs(bj)
        elseif abs(g[j]) > rtol * max(1.0, ga[j])
            return false, NaN
        end
    end
    return mx < β - rtol * (1.0 + βa + sl), (β - mx) / (1.0 + βa + sl)
end
function run1(path)
    L = readlp(path); cc = colclass(L); g = [rowgroup(L, cc, i) for i in 1:L.nr]
    for (lbl, drop) in (("A0", REC), ("A1 (이름 있는 4 종만)", REC[1:4]))
        keep = [i for i in 1:L.nr if !(g[i] in drop)]
        M, _ = build(L, keep, collect(1:L.nc); obj=zeros(L.nc)); C.GRBsetintparam(C.GRBgetenv(M), "InfUnbdInfo", 1)
        s, _, t = solve!(M)
        if s == "INF"
            λ = darr(M, "FarkasDual", length(keep)); ok, rel = certify(L, keep, λ)
            @printf("  %-22s (%5d 행) INF %5.2f s  Farkas 검증 %s (여유/규모 %.2e, 0 아닌 y %d)\n", lbl, length(keep), t, ok ? "통과" : "실패", rel, count(!=(0), λ))
        else
            @printf("  %-22s (%5d 행) %s %5.2f s\n", lbl, length(keep), s, t)
        end
        C.GRBfreemodel(M); flush(stdout)
    end
end
r = raw"C:\Users\user\robust_bilevel_network_games\nonzero_sum\logs\l6diag"
paths = vcat([joinpath(r, "infdump", "S20b", "infeas_$k.mps") for k in 1:12],
             [joinpath(r, "lpdump", d, "numerr_$k.mps") for d in ("S20", "S50") for k in 1:3],
             [joinpath(r, "infdump", "S50", "infeas_$k.mps") for k in 1:8])
for p in paths
    isfile(p) || continue
    @printf("\n#### %s\n", joinpath(splitpath(p)[end-1:end]...)); run1(p)
end
