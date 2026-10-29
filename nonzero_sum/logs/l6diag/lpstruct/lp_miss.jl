#= lp_miss.jl — 사전 판정 (infeas_pre) 이 못 잡은 실행 불가능 노드 (run23, infdump/S50miss) 의 원인:
   (a) 부분 LP (A0 = _nz_dr_pre_infeasible 과 같은 행 빼기) 가 30 s 안에 못 끝남, (b) 부분 LP 가 실행 가능 (= 뺀 recourse 행이 실행 불가능에 필요),
   (c) 부분 LP 는 INF 인데 Farkas 검증 실패. (b) 면 뺀 종류를 하나씩 되살려 어느 것이 필요한지 =#
ENV["LA_NOMAIN"] = "1"
# REC, certify: lp_subfk.jl 과 같음 (그 파일은 include 하면 예전 덤프를 다 다시 풀어서 복사)
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
function sub(L, g, drop; tl=300.0)
    keep = [i for i in 1:L.nr if !(g[i] in drop)]
    M, _ = build(L, keep, collect(1:L.nc); obj=zeros(L.nc)); e = C.GRBgetenv(M)
    C.GRBsetintparam(e, "InfUnbdInfo", 1); C.GRBsetdblparam(e, "TimeLimit", tl)
    s, _, t = solve!(M); it = Int(getd(M, "IterCount")); ok = ""
    if s == "INF"
        λ = darr(M, "FarkasDual", length(keep)); c, rel = certify(L, keep, λ); ok = c ? "검증 통과" : @sprintf("검증 실패 (%.1e)", rel)
    end
    C.GRBfreemodel(M)
    return s, t, it, length(keep), ok
end
dir = joinpath(dirname(@__DIR__), "infdump", "S50miss")
for k in 1:8
    p = joinpath(dir, "infeas_$k.mps"); isfile(p) || continue
    @printf("\n#### infeas_%d  (B&B 안: %s)\n", k, strip(read(joinpath(dir, "infeas_$k.txt"), String)))
    L = readlp(p); cc = colclass(L); g = [rowgroup(L, cc, i) for i in 1:L.nr]
    s, t, it, nk, ok = sub(L, g, REC)
    @printf("  A0 (사전 판정과 같음)   %5d 행  %-4s %7.2f s %7d iter  %s\n", nk, s, t, it, ok)
    if s != "INF"
        for r in REC
            s2, t2, it2, nk2, ok2 = sub(L, g, setdiff(REC, [r]))
            @printf("    + %-26s %5d 행  %-4s %7.2f s %7d iter  %s\n", r, nk2, s2, t2, it2, ok2)
        end
    end
    flush(stdout)
end
