#= lp_warm.jl — warm start 의 가치: 노드 B 를 (cold / 다른 노드 A 의 최적 basis / 자기 basis) 로 풀 때 반복·시간 =#
ENV["LS_NOMAIN"] = "1"; include(joinpath(@__DIR__, "lp_struct.jl"))
rkey(L, i) = anon(L.rn[i]) ? sig(L, i) : L.rn[i]
function opt_basis(path)
    m = load(path); C.GRBread(m, replace(path, ".mps" => ".bas")); prm!(m); C.GRBoptimize(m)
    nr, nc = geti(m, "NumConstrs"), geti(m, "NumVars"); cb = iarr(m, "CBasis", nr); vb = iarr(m, "VBasis", nc); C.GRBfreemodel(m)
    return cb, vb
end
function mapped_basis(LA, cbA, vbA, LB)
    q = Dict{String,Vector{Cint}}(); for i in 1:LA.nr; push!(get!(q, rkey(LA, i), Cint[]), cbA[i]); end
    cb = Cint[(v = get(q, rkey(LB, i), Cint[]); isempty(v) ? Cint(0) : popfirst!(v)) for i in 1:LB.nr]
    vd = Dict(zip(LA.cn, vbA)); vb = Cint[get(vd, LB.cn[j], -1) for j in 1:LB.nc]
    nb = count(==(0), cb) + count(==(0), vb); nmatch = count(i -> haskey(q, rkey(LB, i)), 1:LB.nr)
    # 기저 수를 행 수에 맞춤: 넘치면 기저 slack 일부를 비기저로, 모자라면 비기저 행 일부를 기저로
    if nb > LB.nr
        for i in 1:LB.nr; nb == LB.nr && break; cb[i] == 0 && (cb[i] = -1; nb -= 1); end
    elseif nb < LB.nr
        for i in 1:LB.nr; nb == LB.nr && break; cb[i] != 0 && (cb[i] = 0; nb += 1); end
    end
    return cb, vb, nmatch
end
function run(pathB, pathA)
    LA = readlp(pathA); LB = readlp(pathB); cbA, vbA = opt_basis(pathA)
    for lbl in ("cold", "다른 노드 basis", "자기 basis (.bas)")
        m = load(pathB); prm!(m)
        if lbl == "다른 노드 basis"
            cb, vb, nm = mapped_basis(LA, cbA, vbA, LB)
            C.GRBsetintattrarray(m, "VBasis", 0, LB.nc, vb); C.GRBsetintattrarray(m, "CBasis", 0, LB.nr, cb)
            @printf("    (행 맞춤 %d / %d)\n", nm, LB.nr)
        elseif startswith(lbl, "자기")
            C.GRBread(m, replace(pathB, ".mps" => ".bas"))
        end
        t = @elapsed C.GRBoptimize(m)
        @printf("  %-18s %s z=%.6f  %7d iter  %.2f s\n", lbl, st(m), getd(m, "ObjVal"), Int(getd(m, "IterCount")), t)
        C.GRBfreemodel(m)
    end
end
root = joinpath(raw"C:\Users\user\robust_bilevel_network_games", "nonzero_sum", "logs", "l6diag", "lpdump")
for (dir, a, b) in (("S20", "ok_300", "ok_450"), ("S20", "ok_450", "ok_750"), ("S50", "ok_30", "ok_60"), ("S50", "ok_60", "ok_75"))
    @printf("\n#### %s: %s 를 %s 의 basis 로\n", dir, b, a); run(joinpath(root, dir, b * ".mps"), joinpath(root, dir, a * ".mps"))
end
