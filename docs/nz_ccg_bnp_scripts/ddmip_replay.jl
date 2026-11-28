# ddmip_replay.jl (2026-11-28, 사용자 "내역부터 재"): ddmip_probe.jl 이 저장한 DD KKT-MIP (.mps) 를 따로 다시 풀어 시간 내역을 잰다.
#   나무 안과 같은 설정 (Threads 1, MIPGap 0, MIPGapAbs 0, 그 MIP 의 TimeLimit·BestObjStop) 으로 풀고 Gurobi 로그에서
#   presolve 시간, 루트 LP 시간, 전체 시간, 노드 수를 읽는다. CONFS 로 Gurobi 설정을 바꿔 같은 MIP 들을 다시 푼다.
#   julia -t 10,1 ddmip_replay.jl <DDP_DIR>   환경 변수 CONFS "base" (예 "base,pre0,pre2,focus3,heur0,cuts0"), OUT (기본 <DDP_DIR>/replay.csv)
using Gurobi, Printf, Statistics
const DIR = ARGS[1]
const CONFS = split(get(ENV, "CONFS", "base"), ",")
const CONFP = Dict("base" => Pair{String,Any}[], "pre0" => ["Presolve" => 0], "pre2" => ["Presolve" => 2],
                   "focus3" => ["MIPFocus" => 3], "heur0" => ["Heuristics" => 0.0], "cuts0" => ["Cuts" => 0],
                   "pre0heur0" => ["Presolve" => 0, "Heuristics" => 0.0])
all(c -> haskey(CONFP, c), CONFS) || error("CONFS 에 모르는 설정: $(CONFS)")
# 기록 읽기 (ddmip.csv: k,what,status,wall,runtime,nodes,iters,obj,bound,sense,objstop,timelimit,dump,thread)
rows = [split(l, ",") for l in readlines(joinpath(DIR, "ddmip.csv"))[2:end]]
dumped = [r for r in rows if !isempty(r[13])]
@printf("기록 %d 개, 저장 %d 개, 설정 %s, 스레드 %d\n", length(rows), length(dumped), join(CONFS, " "), Threads.nthreads()); flush(stdout)
const ENVS = [Gurobi.Env() for _ in 1:Threads.maxthreadid()]   # 상호작용 스레드가 있으면 threadid 가 nthreads 보다 클 수 있음

function parse_log(f)
    tp = 0.0; tr = 0.0; tt = NaN; nn = NaN; solved_pre = false
    for l in eachline(f)
        (m = match(r"^Presolve time: ([0-9.eE+-]+)s", l)) !== nothing && (tp = parse(Float64, m[1]))
        (m = match(r"^Root relaxation: .*, ([0-9.eE+-]+) seconds", l)) !== nothing && (tr = parse(Float64, m[1]))
        (m = match(r"^Explored ([0-9]+) nodes .* in ([0-9.eE+-]+) seconds", l)) !== nothing && (nn = parse(Float64, m[1]); tt = parse(Float64, m[2]))
        occursin("Presolve: All rows and columns removed", l) && (solved_pre = true)
        occursin(r"^Presolve removed .* rows and .* columns", l) && occursin("Infeasible", l) && (solved_pre = true)
    end
    return tp, tr, tt, nn, solved_pre
end

function solve_one(path, conf, tl, os, tid)
    mref = Ref{Ptr{Cvoid}}()
    Gurobi.GRBreadmodel(ENVS[tid], path, mref) == 0 || error("GRBreadmodel 실패 $path")
    m = mref[]; e = Gurobi.GRBgetenv(m)
    logf = path * "." * conf * ".log"; isfile(logf) && rm(logf)
    Gurobi.GRBsetintparam(e, "Threads", 1); Gurobi.GRBsetdblparam(e, "MIPGap", 0.0); Gurobi.GRBsetdblparam(e, "MIPGapAbs", 0.0)
    Gurobi.GRBsetintparam(e, "LogToConsole", 0); Gurobi.GRBsetstrparam(e, "LogFile", logf)
    isfinite(tl) && Gurobi.GRBsetdblparam(e, "TimeLimit", tl)
    abs(os) < 1e90 && Gurobi.GRBsetdblparam(e, "BestObjStop", os)
    for (k, v) in CONFP[conf]
        v isa Int ? Gurobi.GRBsetintparam(e, k, v) : Gurobi.GRBsetdblparam(e, k, Float64(v))
    end
    wall = @elapsed (Gurobi.GRBoptimize(m) == 0 || error("GRBoptimize 실패 $path"))
    d = Ref{Cdouble}(); i = Ref{Cint}()
    Gurobi.GRBgetdblattr(m, "Runtime", d); rt = d[]
    Gurobi.GRBgetintattr(m, "Status", i); st = Int(i[])
    bd = Gurobi.GRBgetdblattr(m, "ObjBound", d) == 0 ? d[] : NaN
    sz = Int[]; for a in ("NumVars", "NumBinVars", "NumConstrs", "NumGenConstrs"); Gurobi.GRBgetintattr(m, a, i); push!(sz, Int(i[])); end
    Gurobi.GRBfreemodel(m)   # 모형의 env (e) 도 함께 해제됨 — 이 뒤로 e 를 쓰지 말 것
    tp, tr, tt, nn, sp = parse_log(logf)
    return (wall=wall, rt=rt, st=st, bd=bd, tp=tp, tr=tr, nn=nn, sp=sp, sz=sz)
end

out = get(ENV, "OUT", joinpath(DIR, "replay.csv"))
res = Dict{String,Vector{Any}}()
for conf in CONFS
    R = Vector{Any}(undef, length(dumped))
    t = @elapsed Threads.@threads :static for q in eachindex(dumped)   # :static = 반복마다 스레드 고정 (env 를 스레드별로 씀)
        r = dumped[q]
        R[q] = solve_one(joinpath(DIR, r[13]), conf, parse(Float64, r[12]), parse(Float64, r[11]), Threads.threadid())
    end
    res[conf] = R
    rt = [x.rt for x in R]; tp = [x.tp for x in R]; tr = [x.tr for x in R]; nn = [x.nn for x in R]
    @printf("[%s] %.0f s 벽시계: Runtime 합 %.1f s (평균 %.3f, 중앙 %.3f, 최대 %.2f), presolve %.0f%%, 루트 LP %.0f%%, 나머지 %.0f%% | 노드 0~1 개 %d / %d, 노드 중앙 %.0f, 최대 %.0f | 상태 %s\n",
            conf, t, sum(rt), mean(rt), median(rt), maximum(rt), 100sum(tp) / sum(rt), 100sum(tr) / sum(rt), 100(sum(rt) - sum(tp) - sum(tr)) / sum(rt),
            count(<=(1), nn), length(nn), median(nn), maximum(nn),
            join(["$(s)×$(count(x -> x.st == s, R))" for s in sort(unique([x.st for x in R]))], " ")); flush(stdout)
end
sz = res[CONFS[1]][1].sz
@printf("모형 크기 (첫 MIP): 변수 %d (이진 %d), 제약 %d, indicator %d\n", sz...)
open(out, "w") do io
    println(io, "dump,tree_wall,tree_runtime,tree_status," * join(["$(c)_rt,$(c)_st,$(c)_presolve,$(c)_rootlp,$(c)_nodes,$(c)_bound" for c in CONFS], ","))
    for (q, r) in enumerate(dumped)
        println(io, join([r[13], r[4], r[5], r[3]], ",") * "," *
                    join([join([res[c][q].rt, res[c][q].st, res[c][q].tp, res[c][q].tr, res[c][q].nn, res[c][q].bd], ",") for c in CONFS], ","))
    end
end
println("→ ", out)
