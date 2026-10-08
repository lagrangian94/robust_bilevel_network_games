#= diag_numerr_box.jl — 저장된 노드 LP 의 α 상자 폭 (u − l) 비교: 실패 노드가 좁은 상자 (깊은 노드) 인가 =#
using Gurobi, Printf
const C = Gurobi
env = C.Env()
geti(m, a) = (r = Ref{Cint}(); C.GRBgetintattr(m, a, r); r[])
vname(m, j) = (r = Ref{Ptr{Cchar}}(); C.GRBgetstrattrelement(m, "VarName", j, r); unsafe_string(r[]))
for sub in ("numerr_dump_inf", "numerr_dump_d01")
    dir = joinpath(@__DIR__, "logs", sub); println("== ", sub)
    for f in sort(filter(endswith(".mps"), readdir(dir)); by=x -> (startswith(x, "ok") ? 0 : 1, parse(Int, match(r"\d+", x).match)))
        p = Ref{Ptr{Cvoid}}(); C.GRBreadmodel(env, joinpath(dir, f), p); m = p[]
        nc = geti(m, "NumVars"); lb = zeros(nc); ub = zeros(nc)
        C.GRBgetdblattrarray(m, "LB", 0, nc, lb); C.GRBgetdblattrarray(m, "UB", 0, nc, ub)
        ia = [j for j in 1:nc if startswith(vname(m, j - 1), "α[")]
        w = ub[ia] .- lb[ia]
        @printf("  %-14s α 개수 %d  폭: 최소 %.2e  중앙 %.2e  최대 %.2e   폭<1e-3 인 α %d 개\n", f, length(ia),
                minimum(w), sort(w)[cld(length(w), 2)], maximum(w), count(<(1e-3), w))
        C.GRBfreemodel(m)
    end
end
