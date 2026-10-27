"""
bd_kkt_benders.jl — zero-sum 네트워크 차단의 벤치마크 BD-KKT (2026-10-21 사용자 결정).

belief_menu_benders_optimize! (우리 Algorithm 2 의 zero-sum 판) 에서 belief menu 를 빼고, 전역 subproblem 을 α-B&B 대신
monolithic KKT 모형 (true_dro_structured_eval.jl 의 monolithic 경로, leader = :bilinear) 으로 푼다. 결합 집합 D_δ 를 반영하려고
그 모형에 d_TV(a, p̃) ≤ δ 를 더한 evaluate_monolithic_coupled 를 여기 둔다 (기존 evaluate_monolithic 은 결합 이전 코드라 δ 가 없다).
결합이 있으면 leader KKT (순수 MILP) 는 쌍대 목적에 (쌍대변수 × p̃) 곱이 생겨 정확하지 않으므로 leader 는 bilinear (r·t, S 개) 로 둔다.

같은 것: OMP + min-cut VI, Ipopt local 해 (출발점 균등 α) → α 고정 Ω (LP) 로 cut, oracle 시간 제한 (oracle_time_limit → boost_time_limit),
         목표값 조기 종료 (target = t₀ + 0.99·tol·max(1,|t₀|), BestObjStop = BestBdStop = target), 허용오차, 시간 한도.
다른 것: belief menu 없음, 전역 = KKT 모형 (Gurobi NonConvex). UB = KKT 상한.
필요: true_dro_data.jl, true_dro_build_omp.jl, true_dro_build_subproblem.jl, true_dro_benders.jl, true_dro_mincut_vi.jl,
      belief_menu_benders.jl (_oracle_solve!), true_dro_structured_eval.jl, global_bilinear_solver.jl (_ipopt_local_alpha, _build_fixed_alpha_lp, _capw)
"""

"""
    evaluate_monolithic_coupled(sd; optimizer, delta_couple, time_limit, mip_gap, best_obj_stop, best_bd_stop)

evaluate_monolithic (leader = :bilinear) 에 결합 d_TV(a, p̃) ≤ δ 를 더한 V*(x̄) 의 monolithic KKT 모형.
반환 Dict: :V (incumbent, 없으면 -Inf), :V_bound, :α (없으면 nothing), :status, :p_tilde
"""
function evaluate_monolithic_coupled(sd::StructEvalData; optimizer, delta_couple::Real=Inf, encoding::Symbol=:indicator,
                                     M_dual_F::Float64=2.0, time_limit=nothing, mip_gap=1e-6, silent::Bool=true,
                                     best_obj_stop=nothing, best_bd_stop=nothing,
                                     mip_start::Bool=true, n_start::Int=50, lp_optimizer=nothing)
    _check_encoding(encoding)
    td = sd.td
    S = td.S
    model = _make_mip(optimizer; silent, time_limit, mip_gap)
    best_obj_stop === nothing || set_optimizer_attribute(model, "BestObjStop", best_obj_stop)
    best_bd_stop === nothing || set_optimizer_attribute(model, "BestBdStop", best_bd_stop)

    p̃ = @variable(model, [1:S], lower_bound = 0.0, base_name = "p̃")
    η = @variable(model, [1:S], lower_bound = 0.0, base_name = "η")
    @constraint(model, sum(p̃) == 1)
    @constraint(model, [s=1:S], η[s] >= p̃[s] - sd.q_tilde[s])
    @constraint(model, [s=1:S], η[s] >= sd.q_tilde[s] - p̃[s])
    @constraint(model, 0.5 * sum(η) <= td.eps_tilde)

    P = _add_follower_primal!(model, sd)
    D = _add_follower_dual!(model, sd)
    @constraint(model, [s=1:S], D.ntsμ[s] == p̃[s])
    Z = _add_pattern_certificate!(model, sd, P, D; encoding, M_dual_F)
    E = _add_eval_block!(model, sd, P.α)
    L = _add_leader_bilinear!(model, sd, E.t)
    if isfinite(delta_couple)                     # 결합: d_TV(a, p̃) ≤ δ  (a = truth, p̃ = belief)
        cpl = @variable(model, [1:S], lower_bound = 0.0, base_name = "cpl")
        @constraint(model, [s=1:S], cpl[s] >= L.a[s] - p̃[s])
        @constraint(model, [s=1:S], cpl[s] >= p̃[s] - L.a[s])
        @constraint(model, sum(cpl) <= 2 * delta_couple)
    end
    @objective(model, Max, L.obj)

    if mip_start                                   # evaluate_monolithic 과 같은 follower KKT 시작점 (leader 는 Gurobi 가 채움)
        lpo = lp_optimizer === nothing ? optimizer : lp_optimizer
        cands = [copy(sd.q_tilde)]
        if td.eps_tilde > 0
            V = _tv_ball_vertices(sd.q_tilde, td.eps_tilde)
            step = max(1, cld(length(V), n_start))
            append!(cands, V[1:step:end])
        end
        best = nothing
        for pc in cands
            st = _kkt_start_values(sd, pc; lp_optimizer=lpo)
            st === nothing && continue
            (best === nothing || st.val > best.val) && (best = st)
        end
        best !== nothing && _apply_kkt_start!(Z, L, best)
    end
    optimize!(model)
    st = termination_status(model)
    # incumbent 없이 멈춤: 상한 ≤ best_bd_stop, 또는 시간 제한 (벤치마크가 미해결로 기록)
    if !has_values(model) && st in (MOI.OBJECTIVE_LIMIT, MOI.TIME_LIMIT)
        return Dict{Symbol,Any}(:V => -Inf, :V_bound => objective_bound(model), :α => nothing, :status => st)
    end
    (has_values(model) && st in (MOI.OPTIMAL, MOI.TIME_LIMIT, MOI.OBJECTIVE_LIMIT)) ||
        error("monolithic KKT (coupled): $st (가능해 없음)")
    return Dict{Symbol,Any}(:V => objective_value(model), :V_bound => objective_bound(model),
                            :α => max.(value.(P.α), 0.0), :status => st, :p_tilde => value.(p̃))
end


"""
    bd_kkt_benders_optimize!(td; mip_optimizer, nlp_optimizer, lp_optimizer, kkt_optimizer, ...)

인자는 belief_menu_benders_optimize! 와 같은 뜻 (oracle 관련은 KKT 로 고정). 반환 Dict 도 같은 키.
"""
function bd_kkt_benders_optimize!(td::TrueDROData;
        mip_optimizer, nlp_optimizer, lp_optimizer=nlp_optimizer, kkt_optimizer=mip_optimizer,
        oracle_time_limit=15.0, boost_time_limit=3600.0, oracle_gap=5e-3, repeat_boost::Bool=true, local_time=60.0,
        max_iter=1000, tol=5e-3, verbose=true, valid_inequality::Symbol=:mincut, source_sink_cut=nothing,
        wall_time_limit=7200.0, delta_couple::Real=Inf, oracle::Symbol=:kkt)
    oracle in (:kkt, :omega) || error("oracle = :kkt (BD-KKT) | :omega (BD-G)")
    wall_start = time()
    K = td.num_arcs
    td, δc = reduce_coupling(td, delta_couple)

    omp_model, omp_vars = build_true_dro_omp(td; optimizer=mip_optimizer, silent=true)
    if source_sink_cut !== nothing
        x = omp_vars[:x]
        haskey(source_sink_cut, :source_arcs) &&
            @constraint(omp_model, sum(x[a] for a in source_sink_cut[:source_arcs]) <= length(source_sink_cut[:source_arcs]) - 1)
        haskey(source_sink_cut, :sink_arcs) &&
            @constraint(omp_model, sum(x[a] for a in source_sink_cut[:sink_arcs]) <= length(source_sink_cut[:sink_arcs]) - 1)
    end
    valid_inequality == :mincut && add_phase1_mincut_vi!(omp_model, omp_vars, td)

    # α 고정 Ω (cut 용, belief_menu 의 eval_fixed 와 같음)
    sub_model, sub_vars = build_true_dro_subproblem(td, zeros(K); optimizer=nlp_optimizer, silent=true, delta_couple=δc)
    set_optimizer_attribute(sub_model, "NonConvex", 2)
    Floc = Main._build_fixed_alpha_lp(td, zeros(K); optimizer=lp_optimizer, delta_couple=δc)
    function eval_fixed(αv, x̄)
        α_v = sub_vars[:α]
        for k in 1:K; fix(α_v[k], αv[k]; force=true); end
        set_time_limit_sec(sub_model, nothing)
        set_optimizer_attribute(sub_model, "MIPGap", 1e-6)
        set_optimizer_attribute(sub_model, "BestObjStop", 1e100)
        set_optimizer_attribute(sub_model, "BestBdStop", -1e100)
        inf_ = _oracle_solve!(sub_model, sub_vars, td, x̄)
        for k in 1:K; unfix(α_v[k]); set_lower_bound(α_v[k], 0.0); set_upper_bound(α_v[k], td.w); end
        return inf_
    end

    LB, UB = -Inf, Inf
    best_x = zeros(K)
    cut_count = 0; oracle_calls = 0; local_hits = 0; local_t = 0.0; kkt_calls = 0; kkt_t = 0.0
    oracle_tl = Dict{Vector{Float64},Float64}(); oracle_seen = Set{Vector{Float64}}()
    hist = Dict(:LB => Float64[], :UB => Float64[], :t => Float64[])
    rel(a, b) = abs(b - a) / max(abs(b), 1e-10)
    status = :MaxIter; iter = 0
    add_cut!(info, xs) = (cut_count += 1;
                          add_true_dro_optimality_cut!(omp_model, omp_vars, compute_true_dro_outer_cut(td, info, xs), cut_count))
    verbose && (println(oracle == :kkt ? "  [설정] BD-KKT (zero-sum): 전역 = monolithic KKT (leader bilinear, δ=$(δc)), local = Ipopt, belief menu 없음" :
                                         "  [설정] BD-G (zero-sum): 전역 = Ω (Gurobi NonConvex, δ=$(δc)), local = Ipopt, belief menu 없음"); flush(stdout))
    # BD-G 전역용 Ω (cut 용 sub_model 과 따로: 고정·해제가 섞이지 않게)
    Og, Ogv = oracle == :omega ? build_true_dro_subproblem(td, zeros(K); optimizer=kkt_optimizer, silent=true, delta_couple=δc) : (nothing, nothing)
    oracle == :omega && set_optimizer_attribute(Og, "NonConvex", 2)

    while iter < max_iter
        iter += 1
        if wall_time_limit !== nothing && time() - wall_start > wall_time_limit
            status = :WallTimeLimit; break
        end
        optimize!(omp_model)
        termination_status(omp_model) == MOI.OPTIMAL || error("OMP: $(termination_status(omp_model))")
        x̄ = Float64.(value.(omp_vars[:x]) .> 0.5)
        t0 = value(omp_vars[:t_0])
        LB = max(LB, t0)
        rel(LB, UB) <= tol && (status = :Optimal; break)

        tl = get(oracle_tl, x̄, (repeat_boost && x̄ in oracle_seen) ? boost_time_limit : oracle_time_limit)
        wall_time_limit !== nothing && (tl = min(tl, max(wall_time_limit - (time() - wall_start), 1.0)))
        push!(oracle_seen, x̄)
        oracle_calls += 1
        target = t0 + 0.99 * tol * max(1.0, abs(t0))
        hit = false; info = nothing; Zbd = Inf; src = ""
        t_or = @elapsed begin
            # local (belief_menu 의 local_first 와 같음: Ipopt, 출발점 균등 α)
            t_l = @elapsed begin
                update_true_dro_subproblem_objective!(Floc.model, Floc.vars, td, x̄)
                aloc = Main._ipopt_local_alpha(td, x̄, Floc, Main._capw(fill(td.w / K, K), td.w);
                                               max_time=local_time, deadline=time() + local_time, delta_couple=δc)
                if aloc !== nothing
                    inf_ = eval_fixed(aloc, x̄)
                    inf_[:Z0_val] >= target && (info = inf_; hit = true; src = "local")
                end
            end
            local_t += t_l; hit && (local_hits += 1)
            if !hit
                if oracle == :kkt
                    sd = make_struct_eval_data(td, x̄; lp_optimizer=lp_optimizer)
                    t_k = @elapsed (k = evaluate_monolithic_coupled(sd; optimizer=kkt_optimizer, delta_couple=δc,
                                                                   time_limit=tl, mip_gap=oracle_gap,
                                                                   best_obj_stop=target, best_bd_stop=target, lp_optimizer=lp_optimizer))
                    αg, Zbd = k[:α], k[:V_bound]
                else                                   # BD-G: Ω 를 Gurobi 로 (목표값 조기 종료)
                    t_k = @elapsed begin
                        update_true_dro_subproblem_objective!(Og, Ogv, td, x̄)
                        set_time_limit_sec(Og, tl); set_optimizer_attribute(Og, "MIPGap", oracle_gap)
                        set_optimizer_attribute(Og, "BestObjStop", target); set_optimizer_attribute(Og, "BestBdStop", target)
                        optimize!(Og)
                        stg = termination_status(Og)
                        stg in (MOI.OPTIMAL, MOI.TIME_LIMIT, MOI.OBJECTIVE_LIMIT) || error("BD-G Ω: $stg")
                        αg = has_values(Og) ? max.([value(Ogv[:α][k]) for k in 1:K], 0.0) : nothing
                        Zbd = stg == MOI.OPTIMAL ? objective_value(Og) : _bound_or_inf(Og)
                    end
                end
                kkt_calls += 1; kkt_t += t_k
                src = oracle == :kkt ? "kkt" : "omega"
                αg !== nothing && (info = eval_fixed(αg, x̄); Zbd = max(Zbd, info[:Z0_val]))
            end
        end
        Zinc = info === nothing ? -Inf : info[:Z0_val]
        Zbd < UB && (UB = Zbd; best_x = copy(x̄))
        info !== nothing && add_cut!(info, x̄)
        if Zinc <= t0 + 1e-6 * max(1.0, abs(t0)) && rel(LB, UB) > tol
            if tl >= boost_time_limit
                status = :Stalled
                verbose && @printf("  Iter %d: KKT 가 x̄ 에서 boost 후에도 진전 없음 → 종료 (gap %.2e)\n", iter, rel(LB, UB))
                break
            end
            oracle_tl[x̄] = boost_time_limit
        end
        verbose && @printf("  Iter %d: LB=%.6f UB=%.6f  oracle[%s %.0fs] inc=%.6f bd=%.6f (%.1fs) x=%s\n",
                           iter, LB, UB, src, tl, Zinc, Zbd, t_or, string(findall(x̄ .> 0.5)))
        flush(stdout)
        push!(hist[:LB], LB); push!(hist[:UB], UB); push!(hist[:t], time() - wall_start)
    end
    wall = time() - wall_start
    verbose && @printf("%s %s: LB=%.6f UB=%.6f gap=%.2e iters=%d oracle=%d (local %d / %.1fs, KKT %d / %.1fs) wall=%.1fs\n",
                       oracle == :kkt ? "BD-KKT" : "BD-G", status, LB, UB, rel(LB, UB), iter, oracle_calls, local_hits, local_t, kkt_calls, kkt_t, wall)
    return Dict(:status => status, :Z0 => UB, :x => best_x, :lower_bound => LB, :upper_bound => UB,
                :iters => iter, :oracle_calls => oracle_calls, :history => hist, :wall_time => wall,
                :local_hits => local_hits, :local_time => local_t, :kkt_calls => kkt_calls, :kkt_time => kkt_t)
end
