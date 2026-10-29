#= lp_miss3.jl — (lp_miss2 + 재시도 설정 비교) 사전 판정 함수 (_nz_dr_pre_infeasible 본문 그대로, JuMP 대신 Gurobi 포인터) 를 덤프 모델에 그대로 적용:
   시작 basis (infeas_k_start.bas = 직전 노드의 최적 basis) 를 읽은 모델 / 안 읽은 모델에서 복사해 풀 때 상태·검증 여유가 다른지 =#
ENV["LA_NOMAIN"] = "1"
include(raw"C:\Users\user\robust_bilevel_network_games\nonzero_sum\logs\l6diag\lpstruct\lp_ablate.jl")
function pre_raw(o; rtol=1e-9, tlim=30.0, extra=())
    Gurobi.GRBupdatemodel(o)
    m = Gurobi.GRBcopymodel(o)
    m == C_NULL && return false, -1
    try
        ri = Ref{Cint}()
        Gurobi.GRBgetintattr(m, "NumConstrs", ri); nr = Int(ri[]); Gurobi.GRBgetintattr(m, "NumVars", ri); nc = Int(ri[])
        sp = Vector{Ptr{Cchar}}(undef, max(nr, nc))
        Gurobi.GRBgetstrattrarray(m, "VarName", 0, nc, sp)
        rec = BitVector(undef, nc)
        for j in 1:nc; s = unsafe_string(sp[j]); rec[j] = startswith(s, "ŷ[") || startswith(s, "ȳ["); end
        Gurobi.GRBgetstrattrarray(m, "ConstrName", 0, nr, sp)
        nz = Ref{Cint}(); Gurobi.GRBgetconstrs(m, nz, C_NULL, C_NULL, C_NULL, 0, nr)
        beg = zeros(Cint, nr); ind = zeros(Cint, nz[]); val = zeros(nz[]); Gurobi.GRBgetconstrs(m, nz, beg, ind, val, 0, nr)
        drop = Cint[]
        for i in 1:nr
            s = unsafe_string(sp[i]); k = findfirst('[', s)
            if k === nothing                                  # 이름 없는 행: ŷ·ȳ 가 들면 지움 (WD/FWD/V)
                e = i < nr ? beg[i+1] : nz[]
                any(rec[ind[q] + 1] for q in (beg[i] + 1):e) && push!(drop, i - 1)
            elseif s[1:prevind(s, k)] in ("Ffeas", "N1", "Lflow", "Fflow")
                push!(drop, i - 1)
            end
        end
        Gurobi.GRBdelconstrs(m, length(drop), drop)
        Gurobi.GRBsetdblattrarray(m, "Obj", 0, nc, zeros(nc))
        e = Gurobi.GRBgetenv(m)
        Gurobi.GRBsetintparam(e, "Method", 1); Gurobi.GRBsetintparam(e, "Presolve", 0); Gurobi.GRBsetintparam(e, "InfUnbdInfo", 1)
        Gurobi.GRBsetintparam(e, "NumericFocus", 0); Gurobi.GRBsetdblparam(e, "TimeLimit", max(tlim, 0.1))
        for (k, v) in extra; v isa Int ? Gurobi.GRBsetintparam(e, k, v) : Gurobi.GRBsetdblparam(e, k, v); end
        Gurobi.GRBupdatemodel(m)
        Gurobi.GRBoptimize(m)
        Gurobi.GRBgetintattr(m, "Status", ri); st = Int(ri[])
        st == 3 || return false, st, NaN, -1                         # 3 = GRB_INFEASIBLE
        Gurobi.GRBgetintattr(m, "NumConstrs", ri); n2 = Int(ri[])
        λ = zeros(n2); Gurobi.GRBgetdblattrarray(m, "FarkasDual", 0, n2, λ) == 0 || return false, st, NaN, -1
        sense = zeros(Cchar, n2); Gurobi.GRBgetcharattrarray(m, "Sense", 0, n2, sense)
        rhs = zeros(n2); Gurobi.GRBgetdblattrarray(m, "RHS", 0, n2, rhs)
        lb = zeros(nc); ub = zeros(nc); Gurobi.GRBgetdblattrarray(m, "LB", 0, nc, lb); Gurobi.GRBgetdblattrarray(m, "UB", 0, nc, ub)
        g = zeros(nc); ga = zeros(nc); β = 0.0; βa = 0.0
        b1 = zeros(Cint, 1)
        for i in 1:n2
            y = -λ[i]; sg = Char(sense[i])
            y = sg == '<' ? min(y, 0.0) : sg == '>' ? max(y, 0.0) : y
            y == 0 && continue
            β += y * rhs[i]; βa += abs(y * rhs[i])
            Gurobi.GRBgetconstrs(m, nz, C_NULL, C_NULL, C_NULL, i - 1, 1); ki = zeros(Cint, nz[]); kv = zeros(nz[])
            Gurobi.GRBgetconstrs(m, nz, b1, ki, kv, i - 1, 1)
            for q in 1:nz[]; j = ki[q] + 1; g[j] += y * kv[q]; ga[j] += abs(y * kv[q]); end
        end
        mx = 0.0; slack = 0.0
        for j in 1:nc
            g[j] == 0 && continue
            bj = g[j] > 0 ? ub[j] : lb[j]
            if abs(bj) < 1e29
                mx += g[j] * bj; slack += ga[j] * abs(bj)
            elseif abs(g[j]) > rtol * max(1.0, ga[j])
                return false, st, NaN, -1
            end
        end
        it = Ref{Cdouble}(); Gurobi.GRBgetdblattr(m, "IterCount", it)
        return (mx < β - rtol * (1.0 + βa + slack), st, (β - mx) / (1.0 + βa + slack), Int(it[]))
    finally
        Gurobi.GRBfreemodel(m)
    end
end

dir = joinpath(dirname(@__DIR__), "infdump", "S50miss")
const OPTS = [("기본", ()), ("Seed 1", (("Seed", 1),)), ("Seed 2", (("Seed", 2),)), ("NumericFocus 2", (("NumericFocus", 2),)),
              ("NumericFocus 3", (("NumericFocus", 3),)), ("primal", (("Method", 0),)), ("barrier", (("Method", 2),)), ("Presolve 켬", (("Presolve", -1),))]
for k in 1:8
    p = joinpath(dir, "infeas_$k.mps"); isfile(p) || continue
    @printf("\n#### infeas_%d\n", k)
    for (lbl, ex) in OPTS
        o = load(p); prm!(o); C.GRBread(o, replace(p, ".mps" => "_start.bas"))
        t = @elapsed (r = pre_raw(o; extra=ex))
        @printf("  %-16s 증명 %-5s 상태 %2d  여유 %+.2e  %6d iter  %.2f s\n", lbl, r[1], r[2], r[3], r[4], t)
        C.GRBfreemodel(o); flush(stdout)
    end
end
