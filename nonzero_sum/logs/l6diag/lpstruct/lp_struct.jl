#= lp_struct.jl — 덤프된 노드 LP (MPS/BAS) 로 (1) 활성 집합 예측 + 행/열 생성 (2) 시나리오 분해 (라그랑주) 측정 =#
using Gurobi, Printf
const C = Gurobi
const E0 = C.Env()
load(path) = (p = Ref{Ptr{Cvoid}}(); C.GRBreadmodel(E0, path, p) == 0 || error("read $path"); p[])
geti(m, a) = (r = Ref{Cint}(); C.GRBgetintattr(m, a, r); r[])
getd(m, a) = (r = Ref{Cdouble}(); C.GRBgetdblattr(m, a, r) == 0 ? r[] : NaN)
darr(m, a, n) = (v = zeros(n); n > 0 && C.GRBgetdblattrarray(m, a, 0, n, v); v)
iarr(m, a, n) = (v = zeros(Cint, n); n > 0 && C.GRBgetintattrarray(m, a, 0, n, v); v)
carr(m, a, n) = (v = zeros(Cchar, n); n > 0 && C.GRBgetcharattrarray(m, a, 0, n, v); v)
cname(m, i) = (r = Ref{Ptr{Cchar}}(); C.GRBgetstrattrelement(m, "ConstrName", i, r); unsafe_string(r[]))
vname(m, j) = (r = Ref{Ptr{Cchar}}(); C.GRBgetstrattrelement(m, "VarName", j, r); unsafe_string(r[]))
function prm!(m)
    e = C.GRBgetenv(m)
    C.GRBsetintparam(e, "OutputFlag", 0); C.GRBsetintparam(e, "Method", 1); C.GRBsetintparam(e, "Presolve", 0)
    C.GRBsetintparam(e, "Threads", 1); C.GRBsetdblparam(e, "FeasibilityTol", 1e-7); C.GRBsetdblparam(e, "OptimalityTol", 1e-7)
end
const STAT = Dict(2 => "OPT", 3 => "INF", 4 => "INF_OR_UNBD", 5 => "UNBD", 12 => "NUM", 13 => "SUBOPT")
st(m) = get(STAT, geti(m, "Status"), string(geti(m, "Status")))

struct LP
    nr::Int; nc::Int; rn::Vector{String}; cn::Vector{String}; sense::Vector{Cchar}; rhs::Vector{Float64}
    obj::Vector{Float64}; lb::Vector{Float64}; ub::Vector{Float64}; msense::Cint
    rows::Vector{Vector{Tuple{Int,Float64}}}; cols::Vector{Vector{Tuple{Int,Float64}}}
end
function readlp(path)
    m = load(path); nr, nc = geti(m, "NumConstrs"), geti(m, "NumVars")
    nz = Ref{Cint}(); C.GRBgetconstrs(m, nz, C_NULL, C_NULL, C_NULL, 0, nr)
    beg = zeros(Cint, nr); ind = zeros(Cint, nz[]); val = zeros(nz[]); C.GRBgetconstrs(m, nz, beg, ind, val, 0, nr)
    rows = [Tuple{Int,Float64}[] for _ in 1:nr]; cols = [Tuple{Int,Float64}[] for _ in 1:nc]
    for i in 1:nr, k in (beg[i] + 1):(i < nr ? beg[i+1] : nz[])
        j = ind[k] + 1; push!(rows[i], (j, val[k])); push!(cols[j], (i, val[k]))
    end
    L = LP(nr, nc, [cname(m, i) for i in 0:nr-1], [vname(m, j) for j in 0:nc-1], carr(m, "Sense", nr), darr(m, "RHS", nr),
           darr(m, "Obj", nc), darr(m, "LB", nc), darr(m, "UB", nc), geti(m, "ModelSense"), rows, cols)
    C.GRBfreemodel(m)
    return L
end
# 덤프 basis: 묶인 행 (CBasis ≠ 0), 기저 열 (VBasis = 0) — 이름으로
function basis_sets(path)
    m = load(path); C.GRBread(m, replace(path, ".mps" => ".bas")); nr, nc = geti(m, "NumConstrs"), geti(m, "NumVars")
    prm!(m); C.GRBoptimize(m)       # 저장 basis 가 최적 basis 라 0 iter 로 끝남
    cb = iarr(m, "CBasis", nr); vb = iarr(m, "VBasis", nc)
    br = Set(cname(m, i - 1) for i in 1:nr if cb[i] != 0); bc = Set(vname(m, j - 1) for j in 1:nc if vb[j] == 0)
    @printf("  [basis %s] iter %d, 묶인 행 %d, 기저 열 %d\n", basename(path), Int(getd(m, "IterCount")), length(br), length(bc))
    C.GRBfreemodel(m); return br, bc
end
# 부분 LP 만들기 (rows, cols: 원래 인덱스). obj 덮어쓰기 가능
function build(L, rows, cols; obj=L.obj[cols])
    p = Ref{Ptr{Cvoid}}(); C.GRBnewmodel(E0, p, "sub", 0, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL); M = p[]; prm!(M)
    C.GRBsetintattr(M, "ModelSense", L.msense)
    C.GRBaddvars(M, length(cols), 0, C_NULL, C_NULL, C_NULL, obj, L.lb[cols], L.ub[cols], C_NULL, C_NULL)
    pos = Dict{Int,Int}(c => k - 1 for (k, c) in enumerate(cols))
    beg = Cint[]; ind = Cint[]; val = Float64[]
    for r in rows
        push!(beg, length(ind))
        for (j, a) in L.rows[r]; q = get(pos, j, -1); q >= 0 && (push!(ind, q); push!(val, a)); end
    end
    C.GRBaddconstrs(M, length(rows), length(ind), beg, ind, val, L.sense[rows], L.rhs[rows], C_NULL); C.GRBupdatemodel(M)
    return M, pos
end
solve!(M) = (t = @elapsed C.GRBoptimize(M); (st(M), getd(M, "ObjVal"), t))

isR(n) = startswith(n, "N1[") || startswith(n, "Ffeas[")
isY(n) = startswith(n, "ŷ[") || startswith(n, "ȳ[")
idx(n) = parse.(Int, split(n[nextind(n, findfirst('[', n)):prevind(n, lastindex(n))], ","))     # [j,s]
family(n) = n[1:prevind(n, findfirst('[', n))]

# ---------------- (1) 활성 집합 + 행/열 생성 ----------------
# 시작 집합 (prow, pcol) 으로 줄인 LP 를 풀고, 빠진 N1/Ffeas 행의 위반과 빠진 ŷ/ȳ 열의 감소비용을 검사해 다시 넣기를 반복 (warm basis)
function colgen(L, prow::Set{String}, pcol::Set{String}; tol=1e-7, maxround=50)
    inR = [!isR(L.rn[i]) || L.rn[i] in prow for i in 1:L.nr]; inC = [!isY(L.cn[j]) || L.cn[j] in pcol for j in 1:L.nc]
    r0, c0 = count(inR), count(inC); ttot = 0.0; it = 0; addR = 0; addC = 0; vb = Dict{Int,Cint}(); cb = Dict{Int,Cint}()
    z = NaN; s = ""
    for round in 1:maxround
        keepR = findall(inR); keepC = findall(inC)
        M, pos = build(L, keepR, keepC)
        if !isempty(vb)
            C.GRBsetintattrarray(M, "VBasis", 0, length(keepC), Cint[get(vb, j, -1) for j in keepC])
            C.GRBsetintattrarray(M, "CBasis", 0, length(keepR), Cint[get(cb, i, 0) for i in keepR])
        end
        s, z, t = solve!(M); ttot += t; it += Int(getd(M, "IterCount"))
        s == "OPT" || (C.GRBfreemodel(M); return (r0, c0, round, addR, addC, ttot, it, z, s))
        x = zeros(L.nc); xv = darr(M, "X", length(keepC)); for (k, j) in enumerate(keepC); x[j] = xv[k]; end
        pi = zeros(L.nr); pv = darr(M, "Pi", length(keepR)); for (k, i) in enumerate(keepR); pi[i] = pv[k]; end
        vbv = iarr(M, "VBasis", length(keepC)); cbv = iarr(M, "CBasis", length(keepR))
        for (k, j) in enumerate(keepC); vb[j] = vbv[k]; end; for (k, i) in enumerate(keepR); cb[i] = cbv[k]; end
        C.GRBfreemodel(M)
        nv = Int[]; na = Int[]
        for i in 1:L.nr
            inR[i] && continue
            act = sum(a * x[j] for (j, a) in L.rows[i]; init=0.0); rhs = L.rhs[i]; sn = Char(L.sense[i])
            ((sn == '>' && act < rhs - tol * max(1, abs(rhs))) || (sn == '<' && act > rhs + tol * max(1, abs(rhs)))) && push!(nv, i)
        end
        sg = L.msense == -1 ? 1.0 : -1.0          # 최대화: 감소비용 > 0 이면 들어올 열
        for j in 1:L.nc
            inC[j] && continue
            rc = L.obj[j] - sum(a * pi[i] for (i, a) in L.cols[j] if inR[i]; init=0.0)
            sg * rc > tol * max(1, abs(L.obj[j])) && push!(na, j)
        end
        (isempty(nv) && isempty(na)) && return (r0, c0, round, addR, addC, ttot, it, z, s)
        for i in nv; inR[i] = true; cb[i] = 0; end; for j in na; inC[j] = true; vb[j] = -1; end
        addR += length(nv); addC += length(na)
    end
    return (r0, c0, maxround, addR, addC, ttot, it, z, "MAXROUND")
end

# ---------------- (2) 시나리오 분해 ----------------
const GLOBALFAM = Set(["α", "κ", "ρ01", "ρ02", "ρ03"])
function scen_labels(L)
    lab = fill(-1, L.nc)    # -1 미정, 0 전역, s 시나리오
    for j in 1:L.nc
        n = L.cn[j]; occursin('[', n) || continue
        f = family(n); ix = idx(n)
        lab[j] = f in GLOBALFAM ? 0 : ix[end]
    end
    lab0 = copy(lab)   # 이름으로 정한 것만 근거로 (이미 정한 익명 열을 쓰면 Σ_s 결합 행이 한 시나리오로 보임)
    for _ in 1:1
        ch = false
        for j in 1:L.nc
            lab[j] == -1 || continue
            # 이 열이 들어간 행 중 시나리오 하나짜리 행의 시나리오 (Σ_s 결합 행은 무시)
            ss = Set{Int}()
            for (i, _) in L.cols[j]
                rsi = Set{Int}(lab0[k] for (k, _) in L.rows[i] if lab0[k] > 0); length(rsi) == 1 && push!(ss, first(rsi))
            end
            length(ss) == 1 && (lab[j] = first(ss); ch = true)
        end
        ch || break
    end
    nun = count(==(-1), lab)
    for j in 1:L.nc; lab[j] == -1 && (lab[j] = 0); end    # 시나리오 이웃이 없는 익명 열 = 전역 (Zκ 등)
    rs = [Set{Int}(lab[j] for (j, _) in L.rows[i] if lab[j] > 0) for i in 1:L.nr]
    return lab, rs, nun
end
anon(n) = !occursin('[', n)
sig(L, i) = join(sort([L.cn[j] for (j, _) in L.rows[i]]), "|") * "|" * string(Char(L.sense[i]))

# 결합 행 (시나리오 둘 이상) 을 승수 π 로, 시나리오 행 안의 전역 열은 시나리오별 복사본 + 승수 w (=Σ_{r∈s} a π) 로 풀어
# 시나리오 블록 S 개 + 전역 블록 1 개의 독립 LP 로. 어떤 π (부호 맞음), w 에서도 원래 LP 최적값의 상한.
function lagrange(L, lab, rs, pi::Vector{Float64}; S)
    coup = [i for i in 1:L.nr if length(rs[i]) >= 2]; glr = [i for i in 1:L.nr if isempty(rs[i])]
    incoup = falses(L.nr); incoup[coup] .= true
    glc = [j for j in 1:L.nc if lab[j] == 0]
    cl = [L.obj[j] - sum(a * pi[i] for (i, a) in L.cols[j] if incoup[i]; init=0.0) for j in 1:L.nc]
    val = sum(pi[i] * L.rhs[i] for i in coup; init=0.0)
    w = Dict{Tuple{Int,Int},Float64}()
    rows_of = [Int[] for _ in 1:S]; for i in 1:L.nr; length(rs[i]) == 1 && push!(rows_of[first(rs[i])], i); end
    cols_of = [Int[] for _ in 1:S]; for j in 1:L.nc; lab[j] > 0 && push!(cols_of[lab[j]], j); end
    tsum = 0.0; tmax = 0.0; sts = Dict{String,Int}(); szr = 0; szc = 0
    for s in 1:S
        rows = rows_of[s]; cols = cols_of[s]; insr = falses(L.nr); insr[rows] .= true
        gl = sort(unique(j for i in rows for (j, _) in L.rows[i] if lab[j] == 0))
        for g in gl; w[(s, g)] = sum(a * pi[i] for (i, a) in L.cols[g] if insr[i]; init=0.0); end
        allc = vcat(cols, gl)
        M, _ = build(L, rows, allc; obj=vcat(cl[cols], [w[(s, g)] for g in gl]))
        sname, z, t = solve!(M); C.GRBfreemodel(M)
        sts[sname] = get(sts, sname, 0) + 1; tsum += t; tmax = max(tmax, t); szr = max(szr, length(rows)); szc = max(szc, length(allc))
        val += sname == "OPT" ? z : Inf
    end
    og = [cl[j] - sum(get(w, (s, j), 0.0) for s in 1:S) for j in glc]
    M, _ = build(L, glr, glc; obj=og); sname, z, t = solve!(M); C.GRBfreemodel(M)
    sts["G:" * sname] = 1; val += sname == "OPT" ? z : Inf
    return val, tsum, tmax, sts, szr, szc
end

function main()
    root = joinpath(raw"C:\Users\user\robust_bilevel_network_games", "nonzero_sum", "logs", "l6diag", "lpdump")
    which = get(ENV, "LS_CASES", "S20,S50")
    cases = [("S20", "ok_150", ["ok_300", "ok_450", "ok_750"], 20), ("S50", "ok_30", ["ok_60", "ok_75"], 50)]
    for (dir, pn, tns, S) in cases
        occursin(dir, which) || continue
        ppath = joinpath(root, dir, pn * ".mps"); prow, pcol = basis_sets(ppath)
        jr = Set((family(n), idx(n)[1]) for n in prow if isR(n)); jc = Set((family(n), idx(n)[1]) for n in pcol if isY(n))
        PL = readlp(ppath)
        Mp, _ = build(PL, collect(1:PL.nr), collect(1:PL.nc)); solve!(Mp); piP = darr(Mp, "Pi", PL.nr); C.GRBfreemodel(Mp)
        pmap = Dict{String,Float64}(); for i in 1:PL.nr; pmap[anon(PL.rn[i]) ? sig(PL, i) : PL.rn[i]] = piP[i]; end
        for tn in tns
            tpath = joinpath(root, dir, tn * ".mps"); L = readlp(tpath)
            M, _ = build(L, collect(1:L.nr), collect(1:L.nc)); sF, zF, tF = solve!(M)
            piF = darr(M, "Pi", L.nr); itF = Int(getd(M, "IterCount")); C.GRBfreemodel(M)
            @printf("\n######## %s/%s (예측 노드 %s): %d 행 × %d 열, 전체 cold %s z=%.8f %.2f s (%d iter)\n", dir, tn, pn, L.nr, L.nc, sF, zF, tF, itF)
            @printf("  Pi 부호 확인: ≤ 행 Pi<-1e-9 %d 개, ≥ 행 Pi>1e-9 %d 개 (0 이면 가정 맞음)\n",
                    count(i -> Char(L.sense[i]) == '<' && piF[i] < -1e-9, 1:L.nr), count(i -> Char(L.sense[i]) == '>' && piF[i] > 1e-9, 1:L.nr))
            orow, ocol = basis_sets(tpath)
            urow = Set(n for n in L.rn if isR(n) && (family(n), idx(n)[1]) in jr); ucol = Set(n for n in L.cn if isY(n) && (family(n), idx(n)[1]) in jc)
            nR = count(isR, L.rn); nY = count(isY, L.cn)
            actR = [n for n in orow if isR(n)]; actC = [n for n in ocol if isY(n)]
            for (lbl, pr, pc) in (("자기 basis (oracle)", orow, ocol), ("다른 노드 basis", prow, pcol), ("다른 노드 호 합집합", urow, ucol))
                rec_r = count(n -> n in pr, actR) / max(1, length(actR)); rec_c = count(n -> n in pc, actC) / max(1, length(actC))
                spr = Set(n for n in pr if isR(n)); spc = Set(n for n in pc if isY(n))
                r0, c0, rd, aR, aC, tt, it, z, s = colgen(L, spr, spc)
                @printf("  %-14s 시작 %6d×%6d (N1/Ffeas %5d/%d, ŷ/ȳ %5d/%d) 재현율 행 %.3f 열 %.3f | 라운드 %d, +행 %d +열 %d | %.2f s (%d iter) | %s z=%.8f (상대차 %.1e)\n",
                        lbl, r0, c0, count(n -> n in spr, L.rn), nR, count(n -> n in spc, L.cn), nY, rec_r, rec_c,
                        rd, aR, aC, tt, it, s, z, (z - zF) / max(1, abs(zF)))
                flush(stdout)
            end
            lab, rs, nun = scen_labels(L)
            coup = [i for i in 1:L.nr if length(rs[i]) >= 2]; glr = count(i -> isempty(rs[i]), 1:L.nr); glc = count(==(0), lab)
            lnk = count(i -> length(rs[i]) == 1 && any(lab[j] == 0 for (j, _) in L.rows[i]), 1:L.nr)
            cnames = Dict{String,Int}(); for i in coup; k = anon(L.rn[i]) ? "anon" : first(split(L.rn[i], '[')); k = startswith(k, "rltc") ? "rltc" : k; cnames[k] = get(cnames, k, 0) + 1; end
            @printf("  구조: 결합 행 %d %s, 전역 행 %d, 전역 열 %d (이름으로 못 정한 익명 열 %d), 전역 열을 포함한 시나리오 행 %d\n", length(coup),
                    string(cnames), glr, glc, nun, lnk)
            v, ts, tm, sts, szr, szc = lagrange(L, lab, rs, piF; S)
            @printf("  라그랑주 (자기 쌍대): 값 %.8f (상대차 %.1e), 블록 %d 개 최대 %d×%d, 합 %.2f s, 최대 %.3f s, 상태 %s\n", v, (v - zF) / max(1, abs(zF)), S, szr, szc, ts, tm, string(sts))
            pi2 = zeros(L.nr); miss = 0
            for i in 1:L.nr
                key = anon(L.rn[i]) ? sig(L, i) : L.rn[i]
                haskey(pmap, key) ? (pi2[i] = pmap[key]) : (miss += 1)
                sn = Char(L.sense[i])
                # 최대화 문제: ≤ 행 π ≥ 0, ≥ 행 π ≤ 0 (Gurobi Pi 부호) 로 잘라 유효한 상한 유지
                sn == '<' && (pi2[i] = max(pi2[i], 0.0)); sn == '>' && (pi2[i] = min(pi2[i], 0.0))
            end
            v2, ts2, _, sts2, _, _ = lagrange(L, lab, rs, pi2; S)
            @printf("  라그랑주 (다른 노드 쌍대, 못 맞춘 행 %d/%d): 값 %.8f (LP 대비 상대 %.2e), 합 %.2f s, 상태 %s\n", miss, L.nr, v2, (v2 - zF) / max(1, abs(zF)), ts2, string(sts2))
            flush(stdout)
        end
    end
end
get(ENV, "LS_NOMAIN", "0") == "1" || main()
