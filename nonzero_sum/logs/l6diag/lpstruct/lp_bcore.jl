#= lp_benders.jl — 노드 LP 를 (마스터: 노드 의존 행 + 접점 변수) + (시나리오 × {리더, follower} 블록: 노드와 무관한 recourse 원·쌍대 LP) 로
   나눠 multi-cut Benders 로 풀기. 블록 행은 노드마다 같으므로 한 노드에서 얻은 절단을 다른 노드에 그대로 쓴다.
   블록 변수: ŷ ȳ ρ̂ ρ̃, ϖ/π̃ 중 H 행 (예약, Wcap 에 나오는 k) 이 아닌 성분. 나머지는 마스터 변수 (블록 행에 나오면 접점).
   블록 변수를 포함하지만 여러 시나리오에 걸친 행 (Flam, Fpsi) 은 시나리오별 출력 변수 o 로 쪼갬 (블록 쪽 등식은 벌점 M 탄성).
=#
ENV["LS_NOMAIN"] = "1"; include(joinpath(@__DIR__, "lp_struct.jl"))
const BLKFAM = Set(["ŷ", "ȳ", "ρ̂1", "ρ̂2", "ρ̂3", "ρ̃1", "ρ̃2", "ρ̃3"])
const LEADFAM = Set(["ŷ", "ρ̂1", "ρ̂2", "ρ̂3", "ϖ"])
const MPEN = parse(Float64, get(ENV, "LB_M", "1e5"))

struct Split
    L::LP; lab::Vector{Int}; isblk::BitVector; grp::Vector{Int}          # grp: 블록 변수의 그룹 (2(s−1)+1 리더, +2 follower)
    mrows::Vector{Int}; srows::Vector{Vector{Int}}                        # 마스터 행, 그룹별 블록 행
    crows::Vector{Int}                                                    # 출력 변수로 쪼갤 결합 행
    outs::Vector{Tuple{Int,Int}}                                          # (결합 행, 그룹) → 출력 변수
    iface::Vector{Vector{Int}}                                            # 그룹별 접점 (원래 열 인덱스)
    ng::Int
end
function split_lp(L)
    lab, rs, _ = scen_labels(L); S = maximum(lab)
    H = Set{Int}(); for i in 1:L.nr; startswith(L.rn[i], "Wcap") || continue; for (j, _) in L.rows[i]; family(L.cn[j]) == "ϖ" && push!(H, idx(L.cn[j])[1]); end; end
    isblk = falses(L.nc); grp = zeros(Int, L.nc)
    for j in 1:L.nc
        n = L.cn[j]; anon(n) && continue; f = family(n)
        b = f in BLKFAM || (f in ("ϖ", "π̃") && !(idx(n)[1] in H))
        b || continue
        isblk[j] = true; grp[j] = 2 * (lab[j] - 1) + (f in LEADFAM ? 1 : 2)
    end
    ng = 2S; mrows = Int[]; srows = [Int[] for _ in 1:ng]; crows = Int[]
    for i in 1:L.nr
        gs = Set(grp[j] for (j, _) in L.rows[i] if isblk[j])
        if isempty(gs); push!(mrows, i)
        elseif length(gs) == 1 && all(isblk[j] || lab[j] == 0 || (lab[j] - 1) == (first(gs) - 1) ÷ 2 for (j, _) in L.rows[i])
            push!(srows[first(gs)], i)
        else
            push!(crows, i); push!(mrows, i)
        end
    end
    outs = [(i, g) for i in crows for g in sort(unique(grp[j] for (j, _) in L.rows[i] if isblk[j]))]
    iface = [sort(unique(j for i in srows[g] for (j, _) in L.rows[i] if !isblk[j])) for g in 1:ng]
    @printf("  분할: H=%s, 블록 변수 %d, 마스터 행 %d (출력으로 쪼갠 결합 행 %d: %s), 블록 행 %d, 출력 변수 %d, 그룹 %d, 접점 크기 최대 %d\n",
            string(sort(collect(H))), count(isblk), length(mrows), length(crows), join(unique(first(split(L.rn[i], '[')) for i in crows), ","),
            sum(length, srows), length(outs), ng, maximum(length, iface))
    return Split(L, lab, isblk, grp, mrows, srows, crows, outs, iface, ng)
end

# 마스터: 열 = 블록 아닌 원래 열 + 출력 + θ_g. 결합 행은 블록 부분을 Σ 출력으로 바꿈
function build_master(P::Split; θU=1e6)
    L = P.L; mcols = findall(.!P.isblk); nm = length(mcols); no = length(P.outs)
    p = Ref{Ptr{Cvoid}}(); C.GRBnewmodel(E0, p, "master", 0, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL); M = p[]; prm!(M)
    C.GRBsetintattr(M, "ModelSense", L.msense)
    obj = vcat(L.obj[mcols], zeros(no), ones(P.ng)); lb = vcat(L.lb[mcols], fill(-1e100, no), fill(-1e100, P.ng)); ub = vcat(L.ub[mcols], fill(1e100, no), fill(θU, P.ng))
    C.GRBaddvars(M, length(obj), 0, C_NULL, C_NULL, C_NULL, obj, lb, ub, C_NULL, C_NULL)
    pos = Dict{Int,Int}(c => k - 1 for (k, c) in enumerate(mcols))
    opos = Dict{Tuple{Int,Int},Int}(o => nm + k - 1 for (k, o) in enumerate(P.outs))
    beg = Cint[]; ind = Cint[]; val = Float64[]
    for i in P.mrows
        push!(beg, length(ind))
        for (j, a) in L.rows[i]; P.isblk[j] || (push!(ind, pos[j]); push!(val, a)); end
        for g in unique(P.grp[j] for (j, _) in L.rows[i] if P.isblk[j]); push!(ind, opos[(i, g)]); push!(val, 1.0); end
    end
    C.GRBaddconstrs(M, length(P.mrows), length(ind), beg, ind, val, L.sense[P.mrows], L.rhs[P.mrows], C_NULL); C.GRBupdatemodel(M)
    θpos = [nm + no + g - 1 for g in 1:P.ng]
    return M, pos, opos, θpos
end
# 블록 g: 열 = 블록 변수 + 접점 (고정) + 출력 (고정) + 탄성 (p, n). 목적 = 블록 목적 − M (p + n)
function build_sub(P::Split, g)
    L = P.L; bcols = [j for j in 1:L.nc if P.isblk[j] && P.grp[j] == g]; ic = P.iface[g]
    myouts = [k for (k, (i, gg)) in enumerate(P.outs) if gg == g]; no = length(myouts)
    p = Ref{Ptr{Cvoid}}(); C.GRBnewmodel(E0, p, "sub", 0, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL); S = p[]; prm!(S)
    C.GRBsetintattr(S, "ModelSense", L.msense)
    nb, ni = length(bcols), length(ic)
    obj = vcat(L.obj[bcols], zeros(ni), zeros(no), fill(-MPEN, 2no))
    lb = vcat(L.lb[bcols], zeros(ni), zeros(no), zeros(2no)); ub = vcat(L.ub[bcols], zeros(ni), zeros(no), fill(1e100, 2no))
    C.GRBaddvars(S, length(obj), 0, C_NULL, C_NULL, C_NULL, obj, lb, ub, C_NULL, C_NULL)
    pos = Dict{Int,Int}(); for (k, j) in enumerate(bcols); pos[j] = k - 1; end; for (k, j) in enumerate(ic); pos[j] = nb + k - 1; end
    beg = Cint[]; ind = Cint[]; val = Float64[]
    for i in P.srows[g]
        push!(beg, length(ind)); for (j, a) in L.rows[i]; push!(ind, pos[j]); push!(val, a); end
    end
    rows_sense = L.sense[P.srows[g]]; rows_rhs = L.rhs[P.srows[g]]
    for (q, k) in enumerate(myouts)      # o − Σ_{블록 g} a x + p − n = 0
        i = P.outs[k][1]; push!(beg, length(ind))
        push!(ind, nb + ni + q - 1); push!(val, 1.0)
        for (j, a) in L.rows[i]; (P.isblk[j] && P.grp[j] == g) && (push!(ind, pos[j]); push!(val, -a)); end
        push!(ind, nb + ni + no + 2q - 2); push!(val, 1.0); push!(ind, nb + ni + no + 2q - 1); push!(val, -1.0)
        push!(rows_sense, Cchar('=')); push!(rows_rhs, 0.0)
    end
    C.GRBaddconstrs(S, length(beg), length(ind), beg, ind, val, rows_sense, rows_rhs, C_NULL); C.GRBupdatemodel(S)
    fix = vcat([nb + k - 1 for k in 1:ni], [nb + ni + q - 1 for q in 1:no])   # 고정할 열 (sub 인덱스)
    return S, fix, ic, myouts
end

# 절단 = (그룹, Φ, [(키, 기울기, ẑ)]). 키 = 원래 열 이름 또는 ("out", 행 이름/서명, 그룹)
function run_benders(P::Split, zref; pool=Any[], maxit=400, tol=1e-6, tlim=600.0, label="")
    L = P.L
    M, mpos, opos, θpos = build_master(P)
    keyof_out(k) = ("out", anon(L.rn[P.outs[k][1]]) ? sig(L, P.outs[k][1]) : L.rn[P.outs[k][1]], P.outs[k][2])
    colkey = Dict{Any,Int}(); for (j, q) in mpos; colkey[L.cn[j]] = q; end
    for (k, o) in enumerate(P.outs); colkey[keyof_out(k)] = opos[o]; end
    function addcut!(g, Φ, terms)
        ind = Cint[θpos[g]]; val = [1.0]; rhs = Φ
        for (key, sl, z0) in terms
            q = get(colkey, key, -1); q < 0 && return false
            push!(ind, q); push!(val, -sl); rhs -= sl * z0
        end
        C.GRBaddconstr(M, length(ind), ind, val, Cchar('<'), rhs, C_NULL); return true
    end
    nadd = count(c -> addcut!(c...), pool); C.GRBupdatemodel(M)
    subs = [build_sub(P, g) for g in 1:P.ng]
    newpool = Any[]; tm = 0.0; ts = 0.0; tsmax_sum = 0.0; it = 0; UB = Inf; LBv = -Inf; t0 = time()
    nmc = length(findall(.!P.isblk))
    while it < maxit && time() - t0 < tlim
        it += 1
        tm += @elapsed C.GRBoptimize(M)
        st(M) == "OPT" || (println("  마스터 ", st(M)); break)
        UB = getd(M, "ObjVal"); xm = darr(M, "X", nmc + length(P.outs) + P.ng)
        θv = xm[θpos .+ 1]; Φs = zeros(P.ng); itmax = 0.0; ncut = 0; pen = 0.0
        for g in 1:P.ng
            S, fix, ic, myouts = subs[g]
            zv = vcat([xm[mpos[j] + 1] for j in ic], [xm[opos[P.outs[k]] + 1] for k in myouts])
            C.GRBsetdblattrlist(S, "LB", length(fix), Cint.(fix), zv); C.GRBsetdblattrlist(S, "UB", length(fix), Cint.(fix), zv)
            t = @elapsed C.GRBoptimize(S); ts += t; itmax = max(itmax, t)
            st(S) == "OPT" || error("블록 $g: $(st(S))")
            Φs[g] = getd(S, "ObjVal"); rc = [(r = Ref{Cdouble}(); C.GRBgetdblattrelement(S, "RC", f, r); r[]) for f in fix]
            nv = geti(S, "NumVars"); xs = darr(S, "X", nv); pen += MPEN * sum(xs[end-2length(myouts)+1:end]; init=0.0)
            if θv[g] > Φs[g] + 1e-9 * max(1, abs(Φs[g]))
                keys = vcat([L.cn[j] for j in ic], [keyof_out(k) for k in myouts])
                cut = (g, Φs[g], collect(zip(keys, rc, zv))); push!(newpool, cut); addcut!(cut...); ncut += 1
            end
        end
        tsmax_sum += itmax
        LBv = UB - sum(θv) + sum(Φs)
        gap = (UB - LBv) / max(1, abs(UB))
        (it <= 3 || it % 10 == 0 || gap <= tol) && @printf("    [%s] it %3d  UB %.8f  LB %.8f  gap %.2e  새 절단 %4d  벌점 %.1e  (마스터 %.2f s, 블록 합 %.2f s)\n", label, it, UB, LBv, gap, ncut, pen, tm, ts)
        (gap <= tol || ncut == 0) && break
        C.GRBupdatemodel(M)
    end
    @printf("  [%s] 결과: %d 반복, UB %.8f (전체 LP 대비 %.1e), 시작 절단 %d, 새 절단 %d | 마스터 %.2f s + 블록 합 %.2f s (블록 12 병렬 가정 ≈ %.2f s) | 마스터 크기 %d×%d\n",
            label, it, UB, (UB - zref) / max(1, abs(zref)), nadd, length(newpool), tm, ts, tm + ts / 12, geti(M, "NumConstrs"), geti(M, "NumVars"))
    C.GRBfreemodel(M); for s in subs; C.GRBfreemodel(s[1]); end
    return newpool
end

