#=
diag_numerr_analyze.jl — 저장된 실패 노드 LP (logs/numerr_dump/numerr_k.mps/.bas) 분석. Gurobi C API 를 직접 쓴다.

  (A) 재현: 노드 LP 설정 그대로 (dual simplex, Presolve 0) — 저장 basis warm start / cold start
  (B) 계수 범위 (GRBprintstats)
  (C) 조건수: NumericFocus 3 으로 푼 뒤 Kappa, KappaExact
  (D) ablation: 결합 RLT 행 (이름 rltc_*) 만 지우고 cold start / 같은 수의 다른 RLT 행을 지운 대조군
  (E) 해의 크기: 최적해에서 |값| 이 큰 변수, 결합 변수 (cpl, Zc) 값
=#

using Gurobi, Printf, Random
const C = Gurobi
dir = joinpath(@__DIR__, "logs", "numerr_dump")
env = C.Env()

function load(path)
    p = Ref{Ptr{Cvoid}}()
    C.GRBreadmodel(env, path, p) == 0 || error("read $path")
    return p[]
end
setp(m, k, v::Integer) = C.GRBsetintparam(C.GRBgetenv(m), k, v)
setp(m, k, v::Real) = C.GRBsetdblparam(C.GRBgetenv(m), k, Float64(v))
geti(m, a) = (r = Ref{Cint}(); C.GRBgetintattr(m, a, r); r[])
getd(m, a) = (r = Ref{Cdouble}(); C.GRBgetdblattr(m, a, r) == 0 ? r[] : NaN)
const STATUS = Dict(2 => "OPTIMAL", 3 => "INFEASIBLE", 5 => "UNBOUNDED", 4 => "INF_OR_UNBD", 12 => "NUMERIC", 13 => "SUBOPTIMAL", 9 => "TIME_LIMIT")
st(m) = get(STATUS, geti(m, "Status"), string(geti(m, "Status")))
cname(m, i) = (r = Ref{Ptr{Cchar}}(); C.GRBgetstrattrelement(m, "ConstrName", i, r); unsafe_string(r[]))
vname(m, j) = (r = Ref{Ptr{Cchar}}(); C.GRBgetstrattrelement(m, "VarName", j, r); unsafe_string(r[]))

function solve(path; basis=false, nf=0, method=1, del=Int[], quiet=true, iterlog=false)
    m = load(path)
    setp(m, "OutputFlag", quiet ? 0 : 1); setp(m, "Method", method); setp(m, "Presolve", 0); setp(m, "NumericFocus", nf)
    setp(m, "Threads", 1)
    if !isempty(del)
        C.GRBdelconstrs(m, length(del), Cint.(del)); C.GRBupdatemodel(m)
    end
    basis && C.GRBread(m, replace(path, ".mps" => ".bas"))
    C.GRBoptimize(m)
    return m
end

for k in 1:3
    path = joinpath(dir, "numerr_$k.mps"); isfile(path) || continue
    m0 = load(path)
    nr, nc = geti(m0, "NumConstrs"), geti(m0, "NumVars")
    names = [cname(m0, i) for i in 0:nr-1]
    rltc = [i - 1 for i in 1:nr if startswith(names[i], "rltc_")]
    cplb = count(n -> startswith(n, "cpl_"), names)
    println("="^90)
    @printf("numerr_%d: 행 %d, 열 %d, 결합 RLT 행 (rltc_*) %d, 결합 기본 행 %d\n", k, nr, nc, length(rltc), cplb)
    # (B) 계수 범위
    nstat = count(n -> startswith(n, "rltc_static"), names)
    @printf("  결합 RLT 행 중 정적 (w−Wα)·cpl ≥ 0: %d, 분리된 bound-factor 행: %d
", nstat, length(rltc) - nstat)
    # 계수 범위: Gurobi 가 optimize 시작 때 출력하는 Coefficient statistics (시간 0 으로 통계만)
    setp(m0, "OutputFlag", 1); setp(m0, "TimeLimit", 0.0); setp(m0, "Presolve", 0); C.GRBoptimize(m0); flush(stdout)
    # (A) 재현
    for (lbl, kw) in (("warm (저장 basis), dual simplex", (basis=true,)), ("cold, dual simplex", (basis=false,)),
                      ("cold, primal simplex", (method=0,)), ("cold, barrier", (method=2,)),
                      ("warm, NumericFocus 3", (basis=true, nf=3)))
        m = solve(path; kw...)
        @printf("  (A) %-34s → %-9s obj=%.6g  iter=%d\n", lbl, st(m), getd(m, "ObjVal"), round(Int, getd(m, "IterCount")))
    end
    # (C) 조건수
    m = solve(path; basis=true, nf=3)
    if st(m) == "OPTIMAL"
        @printf("  (C) Kappa=%.3e  KappaExact=%.3e\n", getd(m, "Kappa"), getd(m, "KappaExact"))
        # (E) 해의 크기
        x = zeros(nc); C.GRBgetdblattrarray(m, "X", 0, nc, x)
        ord = sortperm(abs.(x); rev=true)[1:8]
        println("  (E) |값| 상위 변수: ", join([@sprintf("%s=%.3g", vname(m, j - 1), x[j]) for j in ord], ", "))
        zc = [(vname(m, j - 1), x[j]) for j in 1:nc if startswith(vname(m, j - 1), "Zc") || startswith(vname(m, j - 1), "cpl")]
        @printf("  (E) 결합 변수 %d 개, max |값| = %.3g\n", length(zc), maximum(abs(v) for (_, v) in zc; init=0.0))
    end
    # (D) ablation (cold start 기준 — 행을 지우면 저장 basis 를 쓸 수 없음)
    m = solve(path; del=rltc)
    @printf("  (D) 결합 RLT 행 %d 개 제거, cold dual simplex → %s obj=%.6g\n", length(rltc), st(m), getd(m, "ObjVal"))
    rlt_other = [i - 1 for i in 1:nr if isempty(names[i]) || startswith(names[i], "R")]
    Random.seed!(k)
    ctrl = sort(shuffle(rlt_other)[1:min(length(rltc), length(rlt_other))])
    m = solve(path; del=ctrl)
    @printf("  (D) 대조군: 이름 없는 다른 행 %d 개 무작위 제거, cold dual simplex → %s\n", length(ctrl), st(m))
    flush(stdout)
end
