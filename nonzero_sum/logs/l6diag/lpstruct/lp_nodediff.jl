#= lp_nodediff.jl — 같은 트리의 두 노드 LP 에서 어떤 행이 노드마다 바뀌는지 (계수·우변), 바뀌는 행에 들어가는 열 (= 노드 의존 접점) =#
ENV["LS_NOMAIN"] = "1"; include(joinpath(@__DIR__, "lp_struct.jl"))
root = joinpath(raw"C:\Users\user\robust_bilevel_network_games", "nonzero_sum", "logs", "l6diag", "lpdump")
for (dir, a, b) in (("S20", "ok_300", "ok_750"), ("S50", "ok_60", "ok_75"))
    LA = readlp(joinpath(root, dir, a * ".mps")); LB = readlp(joinpath(root, dir, b * ".mps"))
    key(L, i) = anon(L.rn[i]) ? sig(L, i) : L.rn[i]
    content(L, i) = (sort([(L.cn[j], round(v; sigdigits=12)) for (j, v) in L.rows[i]]), round(L.rhs[i]; sigdigits=12))
    # 서명이 겹치는 익명 행이 있으므로 key → 내용의 다중집합
    MA = Dict{String,Vector{Any}}(); for i in 1:LA.nr; push!(get!(MA, key(LA, i), Any[]), content(LA, i)); end
    fam(L, i) = anon(L.rn[i]) ? "anon{" * join(sort(unique(anon(L.cn[j]) ? "C" : family(L.cn[j]) for (j, _) in L.rows[i])), ",") * "}" :
                (startswith(L.rn[i], "rltc") ? "rltc" : family(L.rn[i] * "[]"))
    same = Dict{String,Int}(); diff = Dict{String,Int}(); touched = Set{String}()
    for i in 1:LB.nr
        k = key(LB, i); c = content(LB, i); f = fam(LB, i)
        lst = get(MA, k, Any[]); p = findfirst(==(c), lst)
        if p === nothing
            diff[f] = get(diff, f, 0) + 1
            for (j, _) in LB.rows[i]; push!(touched, anon(LB.cn[j]) ? "C(anon)" : family(LB.cn[j])); end
        else
            same[f] = get(same, f, 0) + 1; deleteat!(lst, p)
        end
    end
    bdiff = count(j -> (LA.lb[j], LA.ub[j]) != (LB.lb[j], LB.ub[j]), 1:min(LA.nc, LB.nc))
    bfam = unique(family(LB.cn[j]) for j in 1:min(LA.nc, LB.nc) if !anon(LB.cn[j]) && (LA.lb[j], LA.ub[j]) != (LB.lb[j], LB.ub[j]))
    @printf("\n#### %s %s vs %s: 같은 행 %d, 다른 행 %d, 상한/하한 바뀐 열 %d (%s)\n", dir, a, b, sum(values(same)), sum(values(diff)), bdiff, join(bfam, ","))
    println("  바뀐 행 종류: ", sort(collect(diff); by=x -> -x[2]))
    println("  안 바뀐 행 종류 (상위): ", sort(collect(same); by=x -> -x[2])[1:min(end, 12)])
    println("  바뀐 행에 들어가는 열 종류: ", sort(collect(touched)))
end
