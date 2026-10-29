#= lp_decouple.jl — 노드 LP R(K) 대 순차형 상한 (시나리오별 h 분리 + follower 블록 제거) 비교.
   drop_f : follower 블록 (ȳ π̃ ρ̃ ζF ζΠ d e cpl κ ρ⁰ Ze Zc Zκ) 과 그것이 들어간 행을 지움. 지운 목적 항은 ≤ 0 (λ 벌점) 이라 유효한 상한.
   split  : α 를 시나리오별 복사본 α_s (같은 상자) 로. 한 시나리오 행은 α→α_s, 시나리오 없는 α 행 (Hcon) 은 시나리오마다 복제,
            여러 시나리오에 걸치고 α 를 포함한 행 (Σ_s Z_is = α_i 등) 은 지움. → 원래 R(K) 의 완화.
=#
ENV["LS_NOMAIN"] = "1"; include(joinpath(@__DIR__, "lp_struct.jl"))
const FFAM = Set(["ȳ", "π̃", "ρ̃1", "ρ̃2", "ρ̃3", "ζF", "d", "e", "cpl", "κ", "ρ01", "ρ02", "ρ03", "Ze", "Zc"])
function variant(L, lab, rs; drop_f::Bool, split::Bool, S)
    isf = falses(L.nc)
    for j in 1:L.nc; anon(L.cn[j]) || (isf[j] = family(L.cn[j]) in FFAM); end
    for j in 1:L.nc     # 익명 열 (ζΠ, Zκ): follower 열과 같은 행에 나오면 follower
        anon(L.cn[j]) || continue
        isf[j] = any(any(!anon(L.cn[k]) && family(L.cn[k]) in FFAM for (k, _) in L.rows[i]) for (i, _) in L.cols[j])
    end
    isα = [!anon(n) && family(n) == "α" for n in L.cn]
    keepc = [j for j in 1:L.nc if !(drop_f && isf[j]) && !(split && isα[j])]
    newid = Dict(j => k for (k, j) in enumerate(keepc)); nc = length(keepc)
    obj = L.obj[keepc]; lb = L.lb[keepc]; ub = L.ub[keepc]
    αs = Dict{Tuple{Int,Int},Int}()
    if split
        any(L.obj[j] != 0 for j in 1:L.nc if isα[j]) && println("  주의: α 에 목적 계수 있음 (S 로 나눠 복사본에 분배)")
        for s in 1:S, j in 1:L.nc
            isα[j] || continue
            nc += 1; αs[(j, s)] = nc; push!(obj, L.obj[j] / S); push!(lb, L.lb[j]); push!(ub, L.ub[j])
        end
    end
    rows = Tuple{Vector{Tuple{Int,Float64}},Cchar,Float64}[]; ndrop = 0; nrep = 0
    for i in 1:L.nr
        ent = L.rows[i]
        if drop_f && any(isf[j] for (j, _) in ent); ndrop += 1; continue; end
        hasα = any(isα[j] for (j, _) in ent)
        if !split || !hasα
            push!(rows, ([(newid[j], a) for (j, a) in ent], L.sense[i], L.rhs[i])); continue
        end
        sc = rs[i]
        if length(sc) == 1
            s = first(sc); push!(rows, ([(isα[j] ? αs[(j, s)] : newid[j], a) for (j, a) in ent], L.sense[i], L.rhs[i]))
        elseif isempty(sc) && all(isα[j] for (j, _) in ent)
            for s in 1:S; push!(rows, ([(αs[(j, s)], a) for (j, a) in ent], L.sense[i], L.rhs[i])); end; nrep += 1
        else
            ndrop += 1      # 여러 시나리오 + α, 또는 α 와 전역 열이 섞인 행 → 지움 (완화)
        end
    end
    p = Ref{Ptr{Cvoid}}(); C.GRBnewmodel(E0, p, "v", 0, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL); M = p[]; prm!(M)
    C.GRBsetintattr(M, "ModelSense", L.msense)
    C.GRBaddvars(M, nc, 0, C_NULL, C_NULL, C_NULL, obj, lb, ub, C_NULL, C_NULL)
    beg = Cint[]; ind = Cint[]; val = Float64[]
    for (e, _, _) in rows; push!(beg, length(ind)); for (k, a) in e; push!(ind, k - 1); push!(val, a); end; end
    C.GRBaddconstrs(M, length(rows), length(ind), beg, ind, val, Cchar[r[2] for r in rows], [r[3] for r in rows], C_NULL); C.GRBupdatemodel(M)
    s, z, t = solve!(M); nr_ = geti(M, "NumConstrs"); C.GRBfreemodel(M)
    return s, z, t, nr_, nc, ndrop
end

root = joinpath(raw"C:\Users\user\robust_bilevel_network_games", "nonzero_sum", "logs", "l6diag", "lpdump")
for (dir, S, names) in (("S20", 20, ["ok_150", "ok_300", "ok_450", "ok_750"]), ("S50", 50, ["ok_30", "ok_60", "ok_75"]))
    for tn in names
        L = readlp(joinpath(root, dir, tn * ".mps")); lab, rs, _ = scen_labels(L)
        αb = [(L.lb[j], L.ub[j]) for j in 1:L.nc if !anon(L.cn[j]) && family(L.cn[j]) == "α"]
        @printf("\n#### %s/%s  α 상자 폭 %s\n", dir, tn, join([@sprintf("%.0f", u - l) for (l, u) in αb], ","))
        for (lbl, df, sp) in (("R(K) (노드 LP)", false, false), ("follower 제거", true, false), ("h 시나리오별 분리", false, true), ("둘 다 (제안한 순차형)", true, true))
            s, z, t, nr, nc, nd = variant(L, lab, rs; drop_f=df, split=sp, S)
            @printf("  %-22s %s z=%12.4f  %6d×%6d (지운 행 %5d)  %.2f s\n", lbl, s, z, nr, nc, nd, t)
            flush(stdout)
        end
    end
end
