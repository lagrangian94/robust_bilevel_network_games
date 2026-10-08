#= diag_numerr_inactive.jl — 실패 LP 에서 비활성 RLT 행 (rhs = ±1e30) 처리 방식이 조건수에 미치는 영향.
   (1) 그대로 (2) 비활성 행 삭제 (3) 비활성 행 rhs 를 ±GRB_INFINITY (1e100, Gurobi 가 무한으로 취급) 로.
   각각 cold dual simplex (Presolve 0) → status, Kappa, KappaExact, 반복 수 =#
using Gurobi, Printf
const C = Gurobi
env = C.Env()
geti(m, a) = (r = Ref{Cint}(); C.GRBgetintattr(m, a, r); r[])
getd(m, a) = (r = Ref{Cdouble}(); C.GRBgetdblattr(m, a, r) == 0 ? r[] : NaN)
dir = joinpath(@__DIR__, "logs", "numerr_dump_d01")
for k in 1:3
    f = joinpath(dir, "numerr_$k.mps")
    for mode in (:asis, :delete, :inf)
        p = Ref{Ptr{Cvoid}}(); C.GRBreadmodel(env, f, p); m = p[]
        e = C.GRBgetenv(m); C.GRBsetintparam(e, "OutputFlag", 0); C.GRBsetintparam(e, "Method", 1)
        C.GRBsetintparam(e, "Presolve", 0); C.GRBsetintparam(e, "Threads", 1)
        nr = geti(m, "NumConstrs"); rhs = zeros(nr); C.GRBgetdblattrarray(m, "RHS", 0, nr, rhs)
        idx = [i - 1 for i in 1:nr if abs(rhs[i]) >= 1e29]
        if mode == :delete
            C.GRBdelconstrs(m, length(idx), Cint.(idx))
        elseif mode == :inf
            for i in idx; C.GRBsetdblattrelement(m, "RHS", i, sign(rhs[i + 1]) * 1e100); end
        end
        C.GRBupdatemodel(m)
        C.GRBoptimize(m)
        @printf("  numerr_%d %-7s rows %6d  status %d  Kappa %.2e  KappaExact %.2e  iter %d\n", k, string(mode),
                geti(m, "NumConstrs"), geti(m, "Status"), getd(m, "Kappa"), getd(m, "KappaExact"), round(Int, getd(m, "IterCount")))
        C.GRBfreemodel(m)
    end
end
