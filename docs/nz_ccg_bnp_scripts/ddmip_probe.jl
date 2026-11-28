# ddmip_probe.jl (2026-11-28, 사용자 "내역부터 재"): DD 의 KKT-MIP 한 개 시간이 어디에 쓰이는지 재는 덮어쓰기 (ABB_PATCH 로 줌).
#   abb_dd_patch.jl 을 include 한 뒤 저장소 _nz_dr_opt! (nz_alpha_bnb_dr.jl) 를 같은 동작 + 기록으로 재정의한다. 저장소 파일은 그대로.
#   기록 (what 이 "dd" 로 시작하는 MIP 만): optimize! 벽시계, Gurobi Runtime, NodeCount, 단체법 반복, 상태, 목적값, 한계, 방향, BestObjStop, 시간 제한.
#   DDP_EVERY (기본 5) 번째마다 그 MIP 를 Gurobi 가 가진 그대로 (GRBwrite .mps, indicator 포함) DDP_DIR 에 저장, 최대 DDP_MAX (기본 400) 개.
#   저장은 optimize! 앞에서 하고 벽시계에는 넣지 않는다. 끝나면 (atexit) DDP_DIR/ddmip.csv.
include(joinpath(@__DIR__, "abb_dd_patch.jl"))
using Printf

const _DDP_DIR = get(ENV, "DDP_DIR", joinpath(@__DIR__, "logs", "ddmip")); mkpath(_DDP_DIR)
const _DDP_EVERY = parse(Int, get(ENV, "DDP_EVERY", "5")); const _DDP_MAX = parse(Int, get(ENV, "DDP_MAX", "400"))
const _DDP_LOCK = ReentrantLock(); const _DDP_REC = Any[]
const _DDP_N = Threads.Atomic{Int}(0); const _DDP_ND = Threads.Atomic{Int}(0)

function _nz_dr_opt!(DR::_NZDR, M::_NZRangeMIP, l, u, what)
    isdd = startswith(what, "dd")
    for attempt in 1:2
        try
            k = 0; dump = ""
            if isdd
                k = Threads.atomic_add!(_DDP_N, 1) + 1
                if k % _DDP_EVERY == 0 && _DDP_ND[] < _DDP_MAX
                    j = Threads.atomic_add!(_DDP_ND, 1) + 1
                    if j <= _DDP_MAX
                        MOI.Utilities.state(backend(M.model)) == MOI.Utilities.ATTACHED_OPTIMIZER || MOI.Utilities.attach_optimizer(M.model)
                        o = unsafe_backend(M.model); Gurobi.GRBupdatemodel(o)
                        dump = joinpath(_DDP_DIR, @sprintf("m%04d.mps", j))
                        Gurobi.GRBwrite(o, dump) == 0 || error("ddmip_probe: GRBwrite 실패 ($dump)")
                    end
                end
            end
            t0 = time()
            optimize!(M.model); DR.nmip += 1
            if isdd
                wall = time() - t0
                st = termination_status(M.model)
                rt = MOI.get(M.model, MOI.SolveTimeSec()); nn = MOI.get(M.model, MOI.NodeCount()); it = MOI.get(M.model, MOI.SimplexIterations())
                ov = primal_status(M.model) == MOI.FEASIBLE_POINT ? objective_value(M.model) : NaN
                bd = st in (MOI.OPTIMAL, MOI.TIME_LIMIT, MOI.OBJECTIVE_LIMIT) ? objective_bound(M.model) : NaN
                os = get_optimizer_attribute(M.model, "BestObjStop"); tl = something(time_limit_sec(M.model), Inf)
                sense = objective_sense(M.model) == MAX_SENSE ? "max" : "min"
                lock(_DDP_LOCK) do
                    push!(_DDP_REC, (k, replace(what, "," => ";"), string(st), wall, rt, nn, it, ov, bd, sense, os, tl, basename(dump), Threads.threadid()))
                end
            end
            return true
        catch e
            e isa ErrorException || rethrow()
            if attempt == 2
                DR.nfail += 1
                if Threads.atomic_add!(_NZ_DR_FAIL_SHOWN, 1) < 3
                    println(stderr, "nz_alpha_bnb_dr: 범위 MIP 실패 ($what): ", e.msg, "\n  l = ", l, "\n  u = ", u)
                end
            end
        end
    end
    return false
end

atexit() do
    f = joinpath(_DDP_DIR, "ddmip.csv")
    open(f, "w") do io
        println(io, "k,what,status,wall,runtime,nodes,iters,obj,bound,sense,objstop,timelimit,dump,thread")
        for r in sort(_DDP_REC; by=first); println(io, join(r, ",")); end
    end
    @printf("ddmip_probe: DD MIP %d 개 기록, %d 개 저장 → %s\n", length(_DDP_REC), min(_DDP_ND[], _DDP_MAX), f)
end
