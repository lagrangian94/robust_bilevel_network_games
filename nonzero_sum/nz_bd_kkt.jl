"""
nz_bd_kkt.jl — 벤치마크 BD-KKT (2026-10-21 사용자 결정): Algorithm 2 (nz_belief_menu_benders, local_first = true) 에서
belief cut 을 빼고, 전역 subproblem 을 우리 spatial B&B 대신 monolithic KKT 모형 (nz_kkt_value, Gurobi NonConvex) 으로 푼다.

  같은 것: master (_nz_omp), local cut (Ω local 해 → α 고정 LP, MW), 허용오차·시간 한도, 전역 호출의 목표값 조기 종료
           (target = t₀(x̄) + 0.99·tol·max(1,|LB|)/(1+tol), Gurobi BestObjStop = BestBdStop = target), 남은 시간 전체를 전역 호출에.
  다른 것: belief cut 없음. KKT 에 시작해 (local 해의 belief 와 q̂ 에서 LP 로 만든 KKT 점, nz_kkt_eval._nz_kkt_start) 를 준다. 전역 = KKT 모형 (θ·λ·Ω 를 쓰지 않음). exact cut = KKT 해의 α* 를 고정한 LP (restricted ISP) 의 cut.
           UB = q_xᵀx̄ + max(KKT 상한, α* 고정 LP 값).
  KKT 호출이 incumbent 없이 상한 ≤ target 으로 멈추면 cut 없이 UB 만 갱신한다 (x̄ 에서 인증된 것).

필요: nz_data.jl, nz_omega.jl, nz_kkt_eval.jl, nz_benders.jl (_nz_omp, _nz_local_solve!, _nz_alpha_cut, _nz_lp_cut!, …), nz_alpha_bnb.jl (nz_build_fixed_alpha, nz_eval_alpha!)
"""

function nz_bd_kkt_benders(nd::NZData; optimizer, kkt_optimizer=optimizer, max_iter=500, tol=5e-3, oracle_gap=1e-3,
                           time_limit=3600.0, local_time=60.0, mw::Bool=true, verbose=true, oracle::Symbol=:kkt)
    oracle in (:kkt, :omega) || error("oracle = :kkt (BD-KKT) | :omega (BD-G)")
    isfinite(time_limit) || error("BD-KKT/BD-G: 유한한 time_limit 이 필요 (전역 호출에 남은 시간 전체를 준다)")
    verbose && (println(oracle == :kkt ?
        "  [설정] BD-KKT: 전역 = monolithic KKT (Gurobi NonConvex), 남은 시간 전체, 목표값 조기 종료, local cut 켬, belief cut 없음" :
        "  [설정] BD-G: 전역 = Ω (Gurobi NonConvex), 남은 시간 전체, 목표값 조기 종료, local cut 켬, belief cut 없음"); flush(stdout))
    Og = oracle == :omega ? build_nz_omega(nd; optimizer=kkt_optimizer) : nothing
    x_core = nz_core_point(nd); n_mw = 0
    omp, x, t0 = _nz_omp(nd; optimizer=optimizer)
    Oloc = build_nz_omega(nd; optimizer=optimizer)
    set_optimizer_attribute(Oloc.model, "OptimalityTarget", 1)
    Fx = Main.nz_build_fixed_alpha(nd, zeros(nd.nx); optimizer=optimizer)
    LB, UB, best_x = -Inf, Inf, zeros(nd.nx)
    hist = NamedTuple[]; cuts = NamedTuple[]
    n_local = 0; ncalls = 0
    wall = time(); iter = 0; status = :MaxIter
    while iter < max_iter
        iter += 1
        optimize!(omp)
        x̄ = Float64.(value.(x) .> 0.5)
        LB = objective_value(omp); w_LB = time() - wall
        _rel(LB, UB) <= tol && (status = :Optimal; break)
        time() - wall > time_limit && (status = :TimeLimit; break)
        t0v = _t0_at(omp, t0, x̄)

        # ---- local 단계 (Algorithm 2 와 같음) ----
        t_l = @elapsed (rl = _nz_local_solve!(Oloc, nd, x̄; time_limit=_remaining(wall, time_limit, local_time)))
        d_loc = nothing
        if rl !== nothing
            d_loc = nz_read_belief(Oloc, nd)[:d]          # KKT 시작해 후보 (cut 계산 전에 읽음)
            cut, ok = _nz_alpha_cut(Oloc, Fx, nd, x̄; mw=mw, x_core=x_core)
            cval = cut[:intercept] + dot(cut[:slope], x̄)
            if cval > t0v + tol * max(1.0, abs(t0v))
                _add_cut!(omp, x, t0, cut); _log_cut!(cuts, :local, iter, x̄, cut, ok); n_local += 1; n_mw += ok
                push!(hist, (iter=iter, LB=LB, UB=UB, x=findall(x̄ .> 0.5), kind=:local, w=w_LB))
                verbose && @printf("  [loc]  it %3d LB=%11.4f UB=%11.4f x=%-10s Ω_loc=%11.4f (%.1fs)\n",
                                   iter, LB, UB, string(findall(x̄ .> 0.5)), rl[:Fval], t_l)
                verbose && flush(stdout)
                continue
            end
        end

        # ---- 전역 단계: monolithic KKT ----
        lim = _remaining(wall, time_limit, Inf)
        ncalls += 1
        target = t0v + 0.99 * tol * max(1.0, abs(LB)) / (1 + tol)
        gap_o = _oracle_gap_eff(oracle_gap, tol, LB, t0v)
        t_o = @elapsed (k = oracle == :kkt ?
            nz_kkt_value(nd, x̄; optimizer=kkt_optimizer, time_limit=lim, gap=gap_o,
                         best_obj_stop=target, best_bd_stop=target, allow_no_incumbent=true,
                         start_beliefs=(d_loc === nothing ? [copy(nd.q_hat)] : [d_loc, copy(nd.q_hat)]),
                         lp_optimizer=optimizer) :
            _nz_omega_global!(Og, nd, x̄; time_limit=lim, gap=gap_o, target=target))
        zbd = k[:bound]; zinc = k[:value]
        if k[:α] !== nothing
            nz_set_objective!(Fx, nd, x̄)
            zinc = Main.nz_eval_alpha!(Fx, nd, max.(k[:α], 0.0))
            cut, ok = _nz_lp_cut!(Fx, nd, x̄, zinc; mw=mw, x_core=x_core); n_mw += ok
            _log_cut!(cuts, :restricted, iter, x̄, cut, ok)
            _add_cut!(omp, x, t0, cut)
            zbd = max(zbd, zinc)
        end
        ub_here = dot(nd.qx, x̄) + zbd
        ub_here < UB && (UB = ub_here; best_x = copy(x̄))
        push!(hist, (iter=iter, LB=LB, UB=UB, x=findall(x̄ .> 0.5), kind=oracle, w=w_LB))
        verbose && @printf("  [%s]  it %3d LB=%11.4f UB=%11.4f x=%-10s KKT=%11.4f bd=%11.4f start=%.4f %s (%.1fs)\n",
                           oracle == :kkt ? "kkt" : "omg", iter, LB, UB, string(findall(x̄ .> 0.5)), k[:value], k[:bound], k[:start_val], string(k[:status]), t_o)
        verbose && flush(stdout)
    end
    return Dict(:status => status, :LB => LB, :UB => UB, :x => best_x, :iters => iter,
                :oracle_calls => ncalls, :mw_cuts => n_mw, :local_cuts => n_local, :wall => time() - wall,
                :hist => hist, :cuts => cuts, :oracle => oracle)
end

"BD-G 의 전역 단계: Ω 를 Gurobi NonConvex 로 (목표값 조기 종료). nz_kkt_value 와 같은 키 (:value, :bound, :status, :α, :start_val) 로 돌려준다."
function _nz_omega_global!(O::NZOmega, nd::NZData, x̄; time_limit, gap, target)
    nz_set_objective!(O, nd, x̄)
    set_time_limit_sec(O.model, time_limit)
    set_optimizer_attribute(O.model, "MIPGap", gap)
    set_optimizer_attribute(O.model, "BestObjStop", target)
    set_optimizer_attribute(O.model, "BestBdStop", target)
    optimize!(O.model)
    st = termination_status(O.model)
    st in (MOI.OPTIMAL, MOI.TIME_LIMIT, MOI.OBJECTIVE_LIMIT) || error("BD-G Ω: $st")
    hv = has_values(O.model)
    return Dict(:value => hv ? objective_value(O.model) : -Inf, :bound => objective_bound(O.model), :status => st,
                :α => hv ? [max(value(O.v[:α][i]), 0.0) for i in 1:nz_nh(nd)] : nothing, :start_val => NaN)
end
