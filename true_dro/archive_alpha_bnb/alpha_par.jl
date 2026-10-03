"""
alpha_par.jl — RLT 분리 α-B&B 의 노드 병렬 처리.

worker 마다 독립 Gurobi env + 노드 LP 모델 (_SepLP) + α 고정 평가 LP.
공유: 열린 노드 목록, LB, 처리 중 노드의 부모 상한 (전역 UB 계산용) — lock 으로 보호.
worker 별 diving: 분기 후 한 자식은 직접 이어서 풀고, 다른 자식은 공유 목록으로.
"""

using JuMP, Printf
import Gurobi

function alpha_bnb_par(td, x̄; nworkers=Threads.nthreads(), time_limit=300.0, rel_gap=1e-4,
                       dive_max=3, maxrounds=30, min_width=1e-7, verbose=true, log_every=30.0, LB0=-Inf,
                       global_lb=false, global_threads=4,
                       local_heur=false, ipopt_time=120.0, gurobi_local_time=60.0, local_solver=:both)
    t0 = time()
    K, S, w = td.num_arcs, td.S, td.w
    mk(env) = () -> (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true);
                     MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
    envs = [Gurobi.Env() for _ in 1:nworkers]
    Ps = Vector{Any}(undef, nworkers); Es = Vector{Any}(undef, nworkers)
    for i in 1:nworkers      # 모델 생성은 순차 (JuMP 빌드 안전)
        Ps[i] = _build_sep_lp(td, x̄; optimizer=mk(envs[i]))
        Es[i] = _build_alpha_lp(td, x̄; optimizer=mk(envs[i]), rlt=:basic)
    end
    t_build = time() - t0
    tol(v) = rel_gap * (isfinite(v) ? max(1.0, abs(v)) : 1.0)

    lk = ReentrantLock()
    open = Any[(zeros(K), fill(w, K), Inf)]
    inflight = fill(-Inf, nworkers)
    LB = Ref(LB0); best_α = Ref(zeros(K))
    nodes = Ref(0)
    done = Ref(false)
    root_UB = Ref(Inf)
    root_rows = Ref{Any}(nothing)        # root 에서 분리된 RLT 행 (k, j, islo, b) — 모든 상자에서 유효
    node_cands = Tuple{Float64,Vector{Float64}}[]   # 처리된 노드의 (상한, α̂) — 로컬 heuristic 출발점 후보
    t_end = t0 + time_limit

    globalUB() = max(LB[], isempty(open) ? -Inf : maximum(n[3] for n in open), maximum(inflight))

    function worker(wid)
        P = Ps[wid]; E = Es[wid]
        localnode = nothing
        dive_left = 0
        synced = false
        while true
            node = nothing
            lock(lk)
            try
                if done[]
                    localnode !== nothing && push!(open, localnode)
                    return
                end
                if localnode !== nothing && dive_left > 0 && localnode[3] > LB[] + tol(LB[])
                    node = localnode; dive_left -= 1
                else
                    localnode !== nothing && localnode[3] > LB[] + tol(LB[]) && push!(open, localnode)
                    if !isempty(open)
                        i = argmax([n[3] for n in open])
                        node = popat!(open, i); dive_left = dive_max
                    end
                end
                localnode = nothing
                node !== nothing && (inflight[wid] = node[3])
            finally
                unlock(lk)
            end
            if node === nothing
                sleep(0.05); continue
            end
            l, u, ubp = node
            # root 행 복사 (root 를 처리한 worker 제외, 한 번만)
            if !synced && root_rows[] !== nothing && isempty(P.allrows)
                for (k, j, islo, b) in root_rows[]
                    _sep_add!(P, k, j, islo, b)
                end
                synced = true
            end
            z = -Inf; α̂ = nothing; children = nothing; zE = -Inf
            unfinished = false
            if ubp > LB[] + tol(LB[]) && sum(l) <= w + 1e-9
                _sep_set_box!(P, l, u)
                set_time_limit_sec(P.model, max(t_end - time(), 0.1))      # 시간 엄수
                z, _, _ = _sep_solve!(P; maxrounds=maxrounds, time_budget=max(t_end - time(), 0.0))
                unfinished = isnan(z)
                z = unfinished ? ubp : min(z, ubp)
                if !unfinished && z > LB[] + tol(LB[])
                    α̂ = clamp.(value.(P.vars[:α]), l, u)
                    rv = value.(P.lead); dv = value.(P.fol)
                    ζLv = value.(P.vars[:ζL]); ζFv = value.(P.vars[:ζF])
                    set_time_limit_sec(E.model, max(t_end - time(), 0.1))
                    _set_box!(E, α̂, α̂); zE = _solve_alpha_lp!(E)
                    isnan(zE) && (zE = -Inf)
                    viol = [(u[k] - l[k] > min_width) ?
                            sum(abs(ζLv[k, s] - α̂[k] * rv[s]) + abs(ζFv[k, s] - α̂[k] * dv[s]) for s in 1:S) : 0.0
                            for k in 1:K]
                    k = argmax(viol)
                    if viol[k] > 1e-9
                        width = u[k] - l[k]; t = α̂[k]
                        (t - l[k] < 0.1 * width || u[k] - t < 0.1 * width) && (t = 0.5 * (l[k] + u[k]))
                        u1 = copy(u); u1[k] = t; l2 = copy(l); l2[k] = t
                        children = ((copy(l), u1, z), (l2, copy(u), z))
                    else
                        zE = max(zE, z)          # 완화가 exact
                    end
                end
            end
            lock(lk)
            try
                if unfinished              # 시간 초과로 못 푼 노드는 부모 상한 그대로 되돌림
                    push!(open, node); inflight[wid] = -Inf; dive_left = 0
                    continue
                end
                nodes[] += 1
                if ubp == Inf
                    root_UB[] = z
                    # root 행 사양 저장: root 상자 [0,w] 에서 만든 행은 모든 노드에서 유효
                    root_rows[] = [(r[1], r[6], r[2], r[3]) for r in P.allrows]
                end
                if zE > LB[]
                    LB[] = zE; α̂ !== nothing && (best_α[] = copy(α̂))
                end
                if local_heur && α̂ !== nothing
                    push!(node_cands, (z, copy(α̂)))
                    length(node_cands) > 200 && popfirst!(node_cands)
                end
                if children !== nothing
                    push!(open, children[2])
                    localnode = children[1]
                else
                    dive_left = 0
                end
                inflight[wid] = -Inf
            finally
                unlock(lk)
            end
        end
    end

    tasks = [Threads.@spawn worker(i) for i in 1:nworkers]
    # (옵션) 로컬 NLP primal heuristic: Gurobi 로컬 1회 → 상한 높은 노드의 α̂ 에서 Ipopt 반복.
    #        로컬 해의 α 를 고정해 정확한 LP 로 재평가한 값만 LB 로 사용.
    hl = Dict{Symbol,Any}(:gurobi_local => -Inf, :ipopt_calls => 0, :ipopt_best => -Inf)
    if local_heur
        push!(tasks, Threads.@spawn begin
            henv = Gurobi.Env()
            Eh = _build_alpha_lp(td, x̄; optimizer=mk(henv), rlt=:basic)
            function offer(a)
                a = _capw(a, w)
                _set_box!(Eh, a, a)
                z = _solve_alpha_lp!(Eh)
                lock(lk)
                try
                    z > LB[] && (LB[] = z; best_α[] = copy(a))
                finally
                    unlock(lk)
                end
                return z
            end
            tried = Set{Vector{Float64}}()
            if local_solver == :both
                gm, gv = build_true_dro_subproblem(td, x̄; optimizer=() -> Gurobi.Optimizer(henv),
                                                   silent=true, rho_upper_bound=10.0)
                set_optimizer_attribute(gm, "NonConvex", 2)
                set_optimizer_attribute(gm, "OptimalityTarget", 1)
                set_time_limit_sec(gm, max(min(gurobi_local_time, t_end - time()), 1.0))
                optimize!(gm)
                has_values(gm) && (hl[:gurobi_local] = offer(value.(gv[:α])))
            elseif local_solver == :ipopt
                # 출발점이 아직 없으므로 균등 α 에서 첫 Ipopt
                a0 = fill(w / K, K)
                push!(tried, round.(a0; digits=3))
                aloc = ipopt_local_alpha(td, x̄, Eh, a0; max_time=min(ipopt_time, t_end - time()), deadline=t_end)
                hl[:ipopt_calls] += 1
                aloc === nothing || (hl[:ipopt_best] = max(hl[:ipopt_best], offer(aloc)))
            else
                error("local_solver must be :both or :ipopt (got $local_solver)")
            end
            while !done[] && t_end - time() > 30.0      # 모델 생성 오버헤드 고려, 남은 시간이 짧으면 새 호출 안 함
                a0 = nothing
                lock(lk)
                try
                    best = -Inf
                    for (z, a) in node_cands
                        key = round.(a; digits=3)
                        (z > best && !(key in tried)) && (best = z; a0 = a)
                    end
                finally
                    unlock(lk)
                end
                if a0 === nothing
                    sleep(0.5); continue
                end
                push!(tried, round.(a0; digits=3))
                aloc = ipopt_local_alpha(td, x̄, Eh, _capw(a0, w); max_time=min(ipopt_time, t_end - time()), deadline=t_end)
                hl[:ipopt_calls] += 1
                aloc === nothing || (hl[:ipopt_best] = max(hl[:ipopt_best], offer(aloc)))
            end
        end)
    end
    # (옵션) Gurobi global (Ω, NonConvex) 를 병행해 incumbent 를 공유 LB 로 사용
    gl = Dict{Symbol,Any}()
    if global_lb
        push!(tasks, Threads.@spawn begin
            genv = Gurobi.Env()
            gm, gv = build_true_dro_subproblem(td, x̄; optimizer=() -> Gurobi.Optimizer(genv), silent=true)
            set_optimizer_attribute(gm, "NonConvex", 2)
            set_optimizer_attribute(gm, "Threads", global_threads)
            set_optimizer_attribute(gm, "MIPGap", rel_gap)
            set_time_limit_sec(gm, max(t_end - time(), 1.0))
            MOI.set(gm, Gurobi.CallbackFunction(), (cb_data, cb_where) -> begin
                if cb_where == Gurobi.GRB_CB_MIPSOL
                    objP = Ref{Cdouble}()
                    Gurobi.GRBcbget(cb_data, cb_where, Gurobi.GRB_CB_MIPSOL_OBJ, objP)
                    lock(lk)
                    try
                        objP[] > LB[] && (LB[] = objP[])
                    finally
                        unlock(lk)
                    end
                end
            end)
            optimize!(gm)
            gl[:status] = termination_status(gm)
            gl[:bound] = try objective_bound(gm) catch; Inf end
            gl[:value] = has_values(gm) ? objective_value(gm) : -Inf
        end)
    end
    t_log = time()
    while true
        sleep(0.2)
        fin = false; UBn = Inf
        lock(lk)
        try
            UBn = globalUB()
            idle = isempty(open) && all(inflight .== -Inf)
            fin = (time() - t0 > time_limit) || (UBn - LB[] <= tol(LB[])) || (idle && nodes[] > 0)
            fin && (done[] = true)
        finally
            unlock(lk)
        end
        if verbose && time() - t_log > log_every
            @printf("    [α-par] t=%6.1fs nodes=%5d open=%5d LB=%.6f UB=%.6f gap=%.3e\n",
                    time() - t0, nodes[], length(open), LB[], UBn, (UBn - LB[]) / max(1.0, abs(LB[])))
            flush(stdout); t_log = time()
        end
        fin && break
    end
    foreach(wait, tasks)
    UB = max(LB[], isempty(open) ? -Inf : maximum(n[3] for n in open))
    if global_lb && haskey(gl, :bound)
        UB = min(UB, gl[:bound])              # 두 상한 중 작은 것
        LB[] = max(LB[], gl[:value])
    end
    return Dict(:LB => LB[], :UB => UB, :α => best_α[], :nodes => nodes[], :time => time() - t0,
                :t_build => t_build, :root_UB => root_UB[], :nworkers => nworkers,
                :is_exact => (UB - LB[] <= tol(LB[])), :global => gl, :local => hl)
end
