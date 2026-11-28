# ddmip_sdcut.jl (2026-11-28, 사용자 "내역부터 재" 의 다음): 저장한 DD KKT-MIP 에 강쌍대 부등식을 McCormick 으로 선형화해 넣고 다시 푼다.
#   KKT-MIP (nz_alpha_bnb_dr.jl 의 _nz_dr_build_mip) 은 상보성을 indicator 로만 갖고 있어 LP 완화가 상보성을 보지 못한다 (나무 안 MIP 하나 노드 중앙 1,233 개).
#   KKT 점은 강쌍대 b(α)ᵀσ = cᵀy 를 만족한다 (약쌍대 b(α)ᵀσ ≥ cᵀy 는 원래 성립). b(α)_k = b0_k + Σ_i a_ki α_i 라 α_i σ_k 곱이 생기므로,
#   a_ki > 0 이면 ζ_ki ≤ α_i σ_k 인 McCormick 아래 envelope (ζ ≥ Lα σ + Lσ α − Lα Lσ, ζ ≥ Uα σ + Uσ α − Uα Uσ), a_ki < 0 이면 위 envelope 를 써서
#       Σ_k b0_k σ_k + Σ a_ki ζ_ki ≤ cᵀy
#   를 넣는다. KKT 점 (ζ = ασ) 은 이 식을 만족하므로 해를 자르지 않는다 (유효). α·σ 의 범위는 그 MIP 의 변수 bound (상자 [l, u], sigbd 의 σ 범위) 를 쓴다.
#   b0, a, c 는 indicator 쌍에서 읽는다: 같은 이진 z 에 (σ_k = 0 @1, sP_k = 0 @0) 과 (y_j = 0 @1, sD_j = 0 @0).
#   sP_k = b0_k + aα − Ay 가 "aα − Ay = −b0" 으로, sD_j = Aᵀσ − c_j 가 "Aᵀσ = c_j" 로 저장돼 있다.
#   검사: 원래 판과 이 판이 둘 다 OPTIMAL 이면 목적값이 같아야 한다 (다르면 오류).
#   julia -t 10,1 ddmip_sdcut.jl <DDP_DIR>   환경 변수 CONFS "orig,sd,sd_cuts0,orig_cuts0"
using Gurobi, Printf, Statistics
const DIR = ARGS[1]
const CONFS = split(get(ENV, "CONFS", "orig,sd,sd_cuts0,orig_cuts0"), ",")
rows = [split(l, ",") for l in readlines(joinpath(DIR, "ddmip.csv"))[2:end]]
const TIGHT = get(ENV, "TIGHT", "0") == "1"   # 허용오차 검사용 (FeasibilityTol·IntFeasTol·OptimalityTol 1e-9)
const ONLY = split(get(ENV, "ONLY", ""), ","; keepempty=false)   # 저장 파일 이름 (예 "m0030.mps") 만 풀기
dumped = [r for r in rows if !isempty(r[13]) && (isempty(ONLY) || r[13] in ONLY)]
const ENVS = [Gurobi.Env() for _ in 1:Threads.maxthreadid()]
const INF = 1e100

function vname(m, j)
    p = Ref{Ptr{Cchar}}(); Gurobi.GRBgetstrattrelement(m, "VarName", j, p) == 0 || error("VarName"); unsafe_string(p[])
end
function bnd(m, a, j)
    d = Ref{Cdouble}(); Gurobi.GRBgetdblattrelement(m, a, j, d) == 0 || error(a); d[]
end
"indicator id 의 (이진 변수, 활성 값, 변수 목록, 계수, 우변)"
function getind(m, id)
    bv = Ref{Cint}(); bval = Ref{Cint}(); nv = Ref{Cint}(); sn = Ref{Cchar}(); rhs = Ref{Cdouble}()
    Gurobi.GRBgetgenconstrIndicator(m, id, bv, bval, nv, C_NULL, C_NULL, sn, rhs) == 0 || error("GRBgetgenconstrIndicator")
    ind = zeros(Cint, nv[]); val = zeros(Cdouble, nv[])
    Gurobi.GRBgetgenconstrIndicator(m, id, bv, bval, nv, ind, val, sn, rhs) == 0 || error("GRBgetgenconstrIndicator")
    Char(sn[]) == '=' || error("indicator 의 등호가 아님")
    return Int(bv[]), Int(bval[]), Int.(ind), val, rhs[]
end

"강쌍대 McCormick 행을 넣는다. 반환 = (H 곱 수, 무한 bound 로 빠진 envelope 수)"
function add_sd!(m)
    i = Ref{Cint}(); Gurobi.GRBgetintattr(m, "NumVars", i); nv = Int(i[]); Gurobi.GRBgetintattr(m, "NumGenConstrs", i); ng = Int(i[])
    nm = [vname(m, j) for j in 0:nv-1]
    kind = [startswith(s, "σ[") ? :σ : startswith(s, "y[") ? :y : startswith(s, "α[") ? :α : :other for s in nm]
    byz = Dict{Int,Vector{Any}}()
    for id in 0:ng-1; t = getind(m, id); push!(get!(byz, t[1], Any[]), t); end
    prim = Tuple{Int,Float64,Vector{Tuple{Int,Float64}}}[]   # (σ 열, b0, [(α 열, a)])
    cy = Tuple{Int,Float64}[]                                  # (y 열, c)
    for (z, L) in byz
        length(L) == 2 || continue
        one = findfirst(t -> t[2] == 1, L); zero_ = findfirst(t -> t[2] == 0, L)
        (one === nothing || zero_ === nothing) && continue
        t1, t0 = L[one], L[zero_]
        (length(t1[3]) == 1 && t1[4][1] == 1.0 && t1[5] == 0.0) || continue
        v = t1[3][1] + 1
        all(kind[c + 1] == :σ for c in t0[3]) && kind[v] == :σ && continue   # 등식 쌍의 정규형 (σ_k = 0 | σ_k⁺ = 0) 은 상보성 쌍이 아님
        if kind[v] == :σ
            all(kind[c + 1] in (:α, :y) for c in t0[3]) || error("sP 행에 α·y 밖의 변수")
            push!(prim, (v - 1, -t0[5], [(c, a) for (c, a) in zip(t0[3], t0[4]) if kind[c + 1] == :α]))
        elseif kind[v] == :y
            all(kind[c + 1] == :σ for c in t0[3]) || error("sD 행에 σ 밖의 변수")
            push!(cy, (v - 1, t0[5]))
        end
    end
    nσ = count(==(:σ), kind); ny = count(==(:y), kind)
    (length(prim) == nσ && length(cy) == ny) || error("indicator 쌍 수가 안 맞음: σ $(length(prim))/$(nσ), y $(length(cy))/$(ny)")
    ind = Cint[]; val = Cdouble[]; nprod = 0; nskip = 0
    for (sc, b0, terms) in prim
        b0 != 0 && (push!(ind, sc); push!(val, b0))
        for (ac, a) in terms
            Lα, Uα = bnd(m, "LB", ac), bnd(m, "UB", ac); Lσ, Uσ = bnd(m, "LB", sc), bnd(m, "UB", sc)
            if Lα == Uα                                   # α 고정: 곱이 선형 (a Lα σ)
                push!(ind, sc); push!(val, a * Lα); continue
            end
            nprod += 1
            Gurobi.GRBaddvar(m, 0, C_NULL, C_NULL, 0.0, -INF, INF, Gurobi.GRB_CONTINUOUS, C_NULL) == 0 || error("GRBaddvar")
            Gurobi.GRBupdatemodel(m); Gurobi.GRBgetintattr(m, "NumVars", i); zc = Int(i[]) - 1
            push!(ind, zc); push!(val, a)
            # a > 0: ζ ≥ envelope (아래), a < 0: ζ ≤ envelope (위). 무한 bound 를 쓰는 envelope 는 뺀다
            envs = a > 0 ? ((Lα, Lσ), (Uα, Uσ)) : ((Lα, Uσ), (Uα, Lσ))
            nadd = 0
            for (pα, pσ) in envs
                (abs(pα) >= INF / 10 || abs(pσ) >= INF / 10) && (nskip += 1; continue)
                # ζ − pα σ − pσ α (≥ | ≤) − pα pσ
                Gurobi.GRBaddconstr(m, 3, Cint[zc, sc, ac], Cdouble[1.0, -pα, -pσ], a > 0 ? Gurobi.GRB_GREATER_EQUAL : Gurobi.GRB_LESS_EQUAL, -pα * pσ, C_NULL) == 0 || error("GRBaddconstr")
                nadd += 1
            end
            nadd == 0 && error("곱 (σ 열 $(sc), α 열 $(ac)) 의 envelope 가 모두 무한")
        end
    end
    for (yc, c) in cy; c != 0 && (push!(ind, yc); push!(val, -c)); end
    Gurobi.GRBaddconstr(m, length(ind), ind, val, Gurobi.GRB_LESS_EQUAL, 0.0, C_NULL) == 0 || error("GRBaddconstr (강쌍대)")
    Gurobi.GRBupdatemodel(m)
    return nprod, nskip
end

function solve_one(path, conf, tl, os, tid)
    mref = Ref{Ptr{Cvoid}}()
    Gurobi.GRBreadmodel(ENVS[tid], path, mref) == 0 || error("GRBreadmodel 실패 $path")
    m = mref[]; e = Gurobi.GRBgetenv(m)
    Gurobi.GRBsetintparam(e, "OutputFlag", 0); Gurobi.GRBsetintparam(e, "Threads", 1)
    Gurobi.GRBsetdblparam(e, "MIPGap", 0.0); Gurobi.GRBsetdblparam(e, "MIPGapAbs", 0.0)
    isfinite(tl) && Gurobi.GRBsetdblparam(e, "TimeLimit", tl)
    abs(os) < 1e90 && Gurobi.GRBsetdblparam(e, "BestObjStop", os)
    endswith(conf, "cuts0") && Gurobi.GRBsetintparam(e, "Cuts", 0)
    if TIGHT
        Gurobi.GRBsetdblparam(e, "FeasibilityTol", 1e-9); Gurobi.GRBsetdblparam(e, "IntFeasTol", 1e-9); Gurobi.GRBsetdblparam(e, "OptimalityTol", 1e-9)
    end
    np, ns = startswith(conf, "sd") ? add_sd!(m) : (0, 0)
    Gurobi.GRBoptimize(m) == 0 || error("GRBoptimize 실패 $path")
    d = Ref{Cdouble}(); i = Ref{Cint}()
    Gurobi.GRBgetdblattr(m, "Runtime", d); rt = d[]
    Gurobi.GRBgetintattr(m, "Status", i); st = Int(i[])
    Gurobi.GRBgetdblattr(m, "NodeCount", d); nn = d[]
    bd = Gurobi.GRBgetdblattr(m, "ObjBound", d) == 0 ? d[] : NaN
    Gurobi.GRBgetintattr(m, "SolCount", i); ov = i[] > 0 && Gurobi.GRBgetdblattr(m, "ObjVal", d) == 0 ? d[] : NaN
    Gurobi.GRBfreemodel(m)   # 모형의 env 도 함께 해제됨
    return (rt=rt, st=st, nn=nn, bd=bd, ov=ov, np=np, ns=ns)
end

res = Dict{String,Vector{Any}}()
for conf in CONFS
    R = Vector{Any}(undef, length(dumped))
    t = @elapsed Threads.@threads :static for q in eachindex(dumped)
        r = dumped[q]
        R[q] = solve_one(joinpath(DIR, r[13]), conf, parse(Float64, r[12]), parse(Float64, r[11]), Threads.threadid())
    end
    res[conf] = R
    rt = [x.rt for x in R]; nn = [x.nn for x in R]
    @printf("[%-10s] %.0f s: Runtime 합 %.1f s (평균 %.3f, 중앙 %.3f, 최대 %.2f) | 노드 중앙 %.0f, 최대 %.0f, 0~1 개 %d / %d | 상태 %s | 곱 %d, 빠진 envelope %d\n",
            conf, t, sum(rt), mean(rt), median(rt), maximum(rt), median(nn), maximum(nn), count(<=(1), nn), length(nn),
            join(["$(s)×$(count(x -> x.st == s, R))" for s in sort(unique([x.st for x in R]))], " "), R[1].np, R[1].ns); flush(stdout)
end
# 검사: 원래 판과 sd 판이 둘 다 OPTIMAL (2) 이면 목적값이 같아야 함
if "orig" in CONFS
    bad = Tuple{String,String,Float64,Float64}[]
    for c in CONFS
        c == "orig" && continue
        nb = 0; mx = 0.0
        for q in eachindex(dumped)
            a, b = res["orig"][q], res[c][q]
            (a.st == 2 && b.st == 2) || continue
            nb += 1; dv = abs(a.ov - b.ov) / max(1.0, abs(a.ov)); mx = max(mx, dv)
            dv > 1e-6 && push!(bad, (c, dumped[q][13], a.ov, b.ov))
        end
        @printf("검사 %s: 둘 다 OPTIMAL 인 %d 개에서 목적값 최대 상대 차 %.1e\n", c, nb, mx)
    end
    for (c, f, x, y) in bad; @printf("  어긋남 %s %s: orig %.9f, %s %.9f\n", c, f, x, c, y); end
    isempty(bad) || println(stderr, "검사 실패: 목적값이 어긋난 MIP $(length(bad)) 개 (위 목록)"); global CHECK_BAD = length(bad)
end
# MIP 별 결과 (OUT, 기본 <DIR>/sdcut[_tight].csv): 설정, 파일, 방향, 상태, 목적값, bound
let out = get(ENV, "OUT", joinpath(DIR, TIGHT ? "sdcut_tight.csv" : "sdcut.csv"))
    sense = Dict(r[13] => r[10] for r in dumped)
    open(out, "w") do io
        println(io, "conf,dump,sense,status,obj,bound,runtime,nodes")
        for c in CONFS, (q, r) in enumerate(dumped)
            x = res[c][q]; println(io, join([c, r[13], sense[r[13]], x.st, x.ov, x.bd, x.rt, x.nn], ","))
        end
    end
    println("→ ", out)
end
isdefined(Main, :CHECK_BAD) && CHECK_BAD > 0 && error("검사 실패: 목적값이 어긋난 MIP $(CHECK_BAD) 개 (위 목록, CSV 는 저장함)")
