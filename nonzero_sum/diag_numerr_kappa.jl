#= diag_numerr_kappa.jl — 저장된 노드 LP (ok_*, numerr_*) 의 최적 basis 조건수 비교 (δ=Inf vs δ=0.1).
   각 파일: 모델 + 저장 basis 로 dual simplex (Presolve 0) → Kappa, KappaExact, 행 수, 비활성 행 (rhs ±1e30) 수, 큰 값 =#
using Gurobi, Printf
const C = Gurobi
env = C.Env()
geti(m, a) = (r = Ref{Cint}(); C.GRBgetintattr(m, a, r); r[])
getd(m, a) = (r = Ref{Cdouble}(); C.GRBgetdblattr(m, a, r) == 0 ? r[] : NaN)
for sub in ("numerr_dump_inf", "numerr_dump_d01")
    dir = joinpath(@__DIR__, "logs", sub)
    println("== ", sub)
    for f in sort(filter(endswith(".mps"), readdir(dir)); by=x -> (startswith(x, "ok") ? 0 : 1, parse(Int, match(r"\d+", x).match)))
        p = Ref{Ptr{Cvoid}}(); C.GRBreadmodel(env, joinpath(dir, f), p); m = p[]
        e = C.GRBgetenv(m); C.GRBsetintparam(e, "OutputFlag", 0); C.GRBsetintparam(e, "Method", 1)
        C.GRBsetintparam(e, "Presolve", 0); C.GRBsetintparam(e, "Threads", 1)
        C.GRBread(m, joinpath(dir, replace(f, ".mps" => ".bas")))
        C.GRBoptimize(m)
        nr = geti(m, "NumConstrs"); rhs = zeros(nr); C.GRBgetdblattrarray(m, "RHS", 0, nr, rhs)
        nc = geti(m, "NumVars"); x = zeros(nc); C.GRBgetdblattrarray(m, "X", 0, nc, x)
        @printf("  %-14s rows %6d  inactive(|rhs|≥1e29) %5d  status %d  Kappa %.2e  KappaExact %.2e  max|x| %.2e\n",
                f, nr, count(r -> abs(r) >= 1e29, rhs), geti(m, "Status"), getd(m, "Kappa"), getd(m, "KappaExact"),
                maximum(abs.(x)))
        C.GRBfreemodel(m)
    end
end
