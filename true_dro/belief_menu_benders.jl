"""
belief_menu_benders.jl — 설계 A: belief-menu Benders (docs/benders_belief_menu_design.md §1).

Ω (build_true_dro_subproblem) 는 제약이 x 와 무관하고 x 는 목적 계수에만 있다.
belief b = (a, b, r; d, e) 를 고정하면 bilinear 항 α·r, α·d 가 선형 → v_b(x) 는 LP, v_b(x) ≤ V*(x).
최적 belief 는 유한 집합 (belief cover) 에서 고를 수 있으므로 V*(x) = max_{b ∈ 𝔅} v_b(x).

알고리즘
  OMP → x̄, LB
  menu 단계 (저렴): 모든 b ∈ 𝔅 에 대해 LP v_b(x̄). v_b(x̄) > t₀ 이면 그 LP 해로 cut 추가 → OMP 재풀이.
  oracle 단계 (비쌈, menu 가 x̄ 에서 수렴했을 때만): 전역 평가 V*(x̄)
      UB ← min(UB, 상한), oracle 해로 cut 추가, oracle 해의 belief 가 menu 보다 좋으면 𝔅 에 추가.
      같은 x̄ 에서 진전이 없으면 oracle 시간 제한을 늘림 (boost).
유효성: menu cut 과 oracle cut 모두 Ω 의 가능해 → 전역 유효. UB 는 oracle 의 상한 (dual bound).
"""

using JuMP, Printf, LinearAlgebra


# ---------------------------------------------------------------------
# belief 고정 LP: ζL_ks = α_k·lead_s, ζF_ks = α_k·d_s 를 (belief 고정 계수로) 선형 등식으로
# ---------------------------------------------------------------------
mutable struct _BeliefLP
    model::Model
    vars::Dict
    cL::Matrix{ConstraintRef}
    cF::Matrix{ConstraintRef}
    fam::Vector{Symbol}
end

function _build_belief_lp(td; optimizer)
    K, S = td.num_arcs, td.S
    model, vars = build_true_dro_subproblem(td, zeros(K); optimizer=optimizer, silent=true)
    for name in (:ζL_def, :ζF_def)
        delete.(model, model[name]); unregister(model, name)
    end
    α = vars[:α]
    cL = Matrix{ConstraintRef}(undef, K, S); cF = Matrix{ConstraintRef}(undef, K, S)
    for k in 1:K, s in 1:S
        cL[k, s] = @constraint(model, vars[:ζL][k, s] - 0.0 * α[k] == 0)
        cF[k, s] = @constraint(model, vars[:ζF][k, s] - 0.0 * α[k] == 0)
    end
    fam = vars[:_use_cvar] ? [:a, :b, :r, :d, :e] : [:a, :b, :d, :e]
    return _BeliefLP(model, vars, cL, cF, fam)
end

_read_belief(vars, fam) = Dict(f => [value(v) for v in vec(vars[f])] for f in fam)
_belief_key(b) = Tuple(round(x; digits=6) for f in sort(collect(keys(b))) for x in b[f])

function _set_belief!(B::_BeliefLP, b)
    K, S = size(B.cL)
    α = B.vars[:α]
    lead = haskey(b, :r) ? b[:r] : b[:a]
    for k in 1:K, s in 1:S
        set_normalized_coefficient(B.cL[k, s], α[k], -lead[s])
        set_normalized_coefficient(B.cF[k, s], α[k], -b[:d][s])
    end
    for f in B.fam
        X = vec(B.vars[f])
        for s in 1:S
            fix(X[s], b[f][s]; force=true)
        end
    end
end


# ---------------------------------------------------------------------
# 주 함수
# ---------------------------------------------------------------------
"""
    belief_menu_benders_optimize!(td; mip_optimizer, nlp_optimizer, lp_optimizer, oracle, ...)

oracle = :gurobi (Ω NonConvex, 시간 제한 oracle_time_limit → 정체 시 boost_time_limit, MIPGap 0.5%)
       | :alpha_bnb (global_bilinear_solve, 시간 제한 oracle_time_limit, 목표 gap 0.5%;
                     global_bilinear_solver.jl include 및 julia -t (nworkers+2) 필요)
반환 Dict: :status, :Z0, :x, :lower_bound, :upper_bound, :iters, :oracle_calls, :menu_size, :history, :wall_time
"""
function belief_menu_benders_optimize!(td::TrueDROData;
        mip_optimizer, nlp_optimizer, lp_optimizer=nlp_optimizer,
        oracle::Symbol=:gurobi, oracle_time_limit=15.0, boost_time_limit=3600.0, oracle_gap=5e-3,
        nworkers=12, max_iter=1000, tol=5e-3, verbose=true,
        valid_inequality::Symbol=:mincut, source_sink_cut=nothing, wall_time_limit=7200.0)
    oracle in (:gurobi, :alpha_bnb) || error("oracle must be :gurobi or :alpha_bnb (got $oracle)")
    if oracle == :alpha_bnb
        isdefined(Main, :global_bilinear_solve) || error("oracle=:alpha_bnb: global_bilinear_solver.jl 을 먼저 include 하세요")
        Threads.nthreads() >= nworkers + 2 || error("oracle=:alpha_bnb: julia -t $(nworkers + 2) 이상 필요")
    end
    wall_start = time()
    K = td.num_arcs

    # ---- OMP (기존과 동일 구성) ----
    omp_model, omp_vars = build_true_dro_omp(td; optimizer=mip_optimizer, silent=true)
    if source_sink_cut !== nothing
        x = omp_vars[:x]
        haskey(source_sink_cut, :source_arcs) &&
            @constraint(omp_model, sum(x[a] for a in source_sink_cut[:source_arcs]) <= length(source_sink_cut[:source_arcs]) - 1)
        haskey(source_sink_cut, :sink_arcs) &&
            @constraint(omp_model, sum(x[a] for a in source_sink_cut[:sink_arcs]) <= length(source_sink_cut[:sink_arcs]) - 1)
    end
    valid_inequality == :mincut && add_phase1_mincut_vi!(omp_model, omp_vars, td)

    # ---- oracle 용 Ω (Gurobi NonConvex) 와 belief LP ----
    sub_model, sub_vars = build_true_dro_subproblem(td, zeros(K); optimizer=nlp_optimizer, silent=true)
    set_optimizer_attribute(sub_model, "NonConvex", 2)
    B = _build_belief_lp(td; optimizer=lp_optimizer)

    menu = Vector{Dict{Symbol,Vector{Float64}}}()
    menu_keys = Set{Any}()
    LB, UB = -Inf, Inf
    best_x = zeros(K)
    cut_count = 0
    oracle_calls = 0
    oracle_tl = Dict{Vector{Float64},Float64}()        # x̄ 별 현재 oracle 시간 제한 (정체 시 boost)
    hist = Dict(:LB => Float64[], :UB => Float64[], :menu => Int[], :oracle => Bool[], :t => Float64[])
    rel(a, b) = abs(b - a) / max(abs(b), 1e-10)
    status = :MaxIter
    iter = 0

    add_cut!(info, xs) = (cut_count += 1;
                          add_true_dro_optimality_cut!(omp_model, omp_vars, compute_true_dro_outer_cut(td, info, xs), cut_count))

    while iter < max_iter
        iter += 1
        if wall_time_limit !== nothing && time() - wall_start > wall_time_limit
            status = :WallTimeLimit; break
        end
        optimize!(omp_model)
        termination_status(omp_model) == MOI.OPTIMAL || error("OMP: $(termination_status(omp_model))")
        x̄ = round.(value.(omp_vars[:x]))
        t0 = value(omp_vars[:t_0])
        LB = max(LB, t0)
        if rel(LB, UB) <= tol
            status = :Optimal; break
        end

        # ---- menu 단계 ----
        V_menu = -Inf; nviol = 0
        for b in menu
            _set_belief!(B, b)
            info = solve_true_dro_subproblem!(B.model, B.vars, td, x̄; is_global=false)
            V_menu = max(V_menu, info[:Z0_val])
            if info[:Z0_val] > t0 + 1e-6 * max(1.0, abs(t0))
                add_cut!(info, x̄); nviol += 1
            end
        end
        push!(hist[:oracle], nviol == 0)
        if nviol > 0
            verbose && @printf("  Iter %d: LB=%.6f UB=%.6f  menu cut %d/%d (V_menu=%.6f)\n", iter, LB, UB, nviol, length(menu), V_menu)
            push!(hist[:LB], LB); push!(hist[:UB], UB); push!(hist[:menu], length(menu)); push!(hist[:t], time() - wall_start)
            continue
        end

        # ---- oracle 단계 (menu 가 x̄ 에서 수렴) ----
        tl = get(oracle_tl, x̄, oracle_time_limit)
        oracle_calls += 1
        t_or = @elapsed begin
            if oracle == :gurobi
                set_time_limit_sec(sub_model, tl)
                set_optimizer_attribute(sub_model, "MIPGap", oracle_gap)
                info = solve_true_dro_subproblem!(sub_model, sub_vars, td, x̄; is_global=true)
                Zbd = info[:Z0_bound]
                b_star = _read_belief(sub_vars, B.fam)
            else
                ab = Main.global_bilinear_solve(td, x̄; nworkers=nworkers, time_limit=tl,
                                                rel_gap=oracle_gap, verbose=false)
                α_v = sub_vars[:α]
                for k in 1:K; fix(α_v[k], ab[:α][k]; force=true); end
                set_time_limit_sec(sub_model, nothing)
                set_optimizer_attribute(sub_model, "MIPGap", 1e-6)
                info = solve_true_dro_subproblem!(sub_model, sub_vars, td, x̄; is_global=true)
                b_star = _read_belief(sub_vars, B.fam)          # 고정 해제 전에 읽기 (수정하면 해가 무효화)
                for k in 1:K; unfix(α_v[k]); set_lower_bound(α_v[k], 0.0); set_upper_bound(α_v[k], td.w); end
                Zbd = ab[:UB]
            end
        end
        Zinc = info[:Z0_val]
        if Zbd < UB
            UB = Zbd; best_x = copy(x̄)
        end
        add_cut!(info, x̄)
        new_belief = Zinc > V_menu + 1e-6 * max(1.0, abs(Zinc))
        if new_belief && !(_belief_key(b_star) in menu_keys)
            push!(menu, b_star); push!(menu_keys, _belief_key(b_star))
        end
        # 같은 x̄ 에서 oracle 이 진전 없이 상한만 남기면 (incumbent ≤ t₀ < 상한) 시간 제한을 늘림
        if Zinc <= t0 + 1e-6 * max(1.0, abs(t0)) && rel(LB, UB) > tol
            newtl = oracle == :gurobi ? boost_time_limit : min(4 * tl, boost_time_limit)
            if tl >= boost_time_limit
                status = :Stalled
                verbose && @printf("  Iter %d: oracle 이 x̄ 에서 boost 후에도 진전 없음 → 종료 (gap %.2e)\n", iter, rel(LB, UB))
                break
            end
            oracle_tl[x̄] = newtl
        end
        verbose && @printf("  Iter %d: LB=%.6f UB=%.6f  oracle[%s %.0fs] inc=%.6f bd=%.6f (%.1fs) menu=%d%s x=%s\n",
                           iter, LB, UB, oracle, tl, Zinc, Zbd, t_or, length(menu), new_belief ? " (+belief)" : "",
                           string(findall(x̄ .> 0.5)))
        flush(stdout)
        push!(hist[:LB], LB); push!(hist[:UB], UB); push!(hist[:menu], length(menu)); push!(hist[:t], time() - wall_start)
    end
    wall = time() - wall_start
    verbose && @printf("Belief-menu Benders %s: LB=%.6f UB=%.6f gap=%.2e iters=%d oracle=%d menu=%d wall=%.1fs\n",
                       status, LB, UB, rel(LB, UB), iter, oracle_calls, length(menu), wall)
    return Dict(:status => status, :Z0 => UB, :x => best_x, :lower_bound => LB, :upper_bound => UB,
                :iters => iter, :oracle_calls => oracle_calls, :menu_size => length(menu),
                :history => hist, :wall_time => wall)
end
