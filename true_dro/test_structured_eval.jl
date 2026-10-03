"""
test_structured_eval.jl — ccg.tex 세 평가 경로 검증.

Stage 1: 고정 x̄에서 V*(x̄) 비교
  global bilinear (Gurobi NonConvex=2)  vs  monolithic / decomposed / disjunctive × indicator / bigM
  + α* 고정 LP 값(Z0_lp)이 MILP 값과 일치하는지 (cut tightness)
Stage 2: Benders 전체 비교
  기존 true_dro_benders_optimize! (exact, inexact=false)  vs  structured_benders_optimize!

실행: julia --project test_structured_eval.jl   (또는 REPL에서 include)
"""

using JuMP, Gurobi, HiGHS
using Printf, Random, LinearAlgebra

if !@isdefined(NetworkGenerator)
    include("../network_generator.jl")
end
using .NetworkGenerator

include("true_dro_data.jl")
include("true_dro_build_omp.jl")
include("true_dro_build_subproblem.jl")
include("true_dro_build_isp_leader.jl")
include("true_dro_build_isp_follower.jl")
include("true_dro_benders.jl")
include("true_dro_mincut_vi.jl")
include("true_dro_structured_eval.jl")

const GRB_ENV = Gurobi.Env()
grb() = Gurobi.Optimizer(GRB_ENV)
grb_lp() = (o = Gurobi.Optimizer(GRB_ENV); MOI.set(o, MOI.Silent(), true); o)


function make_instance(; m=3, n=3, S=3, seed=42, eps_hat=0.1, eps_tilde=0.2,
                       gamma=2, w=2.0, beta=nothing)
    network = generate_grid_network(m, n; seed=seed)
    scen, _ = generate_capacity_scenarios_uniform_model(length(network.arcs), S;
                  interdictable_arcs=network.interdictable_arcs, seed=seed)
    q_hat = fill(1.0 / S, S)
    td = make_true_dro_data(network, scen, q_hat, eps_hat, eps_tilde;
                            w=w, lambda_U=10.0, gamma=gamma, beta=beta)
    return td
end


"""기존 global bilinear 평가 (exact M1)."""
function global_eval(td, x̄; time_limit=nothing)
    model, vars = build_true_dro_subproblem(td, x̄; optimizer=grb, silent=true)
    set_optimizer_attribute(model, "NonConvex", 2)
    set_optimizer_attribute(model, "MIPGap", 1e-7)
    time_limit === nothing || set_time_limit_sec(model, time_limit)
    t = @elapsed info = solve_true_dro_subproblem!(model, vars, td, x̄)
    # 시간 제한 시 [incumbent, bound] 구간 반환
    return info[:Z0_val], t, info[:Z0_bound], info[:is_optimal]
end


function random_xs(td, nx; seed=1)
    rng = MersenneTwister(seed)
    idx = findall(td.interdictable_arcs[1:td.num_arcs])
    xs = [zeros(td.num_arcs)]
    for _ in 1:nx-1
        x = zeros(td.num_arcs)
        for k in randperm(rng, length(idx))[1:min(td.gamma, length(idx))]
            x[idx[k]] = 1.0
        end
        push!(xs, x)
    end
    return xs
end


function stage1(td; nx=4, encodings=(:indicator, :bigM),
                methods=(:monolithic, :decomposed, :disjunctive), tol=1e-4,
                cover_kwargs=(;), cover_method=:vertex, global_time_limit=nothing)
    println("\n", "="^90)
    @printf("Stage 1: K=%d S=%d ε̂=%.2f ε̃=%.2f β=%s γ=%d w=%.1f\n",
            td.num_arcs, td.S, td.eps_hat, td.eps_tilde, string(td.beta), td.gamma, td.w)
    println("="^90)
    n_fail = 0
    # 판정: 값 일치(V, LP(α*) 모두 global과) — cover 불완전 여부는 별도 표기
    function _report(meth, enc, ev, x̄, Vg, Vgb; extra="")
        _, Z0lp, _, zf = structured_outer_cut(td, x̄, ev[:α]; lp_optimizer=grb_lp)
        # global 미해결이면 incumbent ≤ V ≤ bound 구간으로 판정
        lo, hi = Vg - tol * max(1, abs(Vg)), Vgb + tol * max(1, abs(Vgb))
        ok = lo <= ev[:V] <= hi && abs(Z0lp - ev[:V]) <= tol * max(1, abs(ev[:V]))
        n_fail += !ok
        @printf("   %-12s %-9s V=%.6f  LP(α*)=%.6f  V̄F=%.1e  bin=%d  %.2fs%s  %s%s\n",
                meth, enc, ev[:V], Z0lp, zf, ev[:n_bin], ev[:time], extra,
                ok ? "OK" : "**MISMATCH**", ev[:is_exact] ? "" : " (inexact)"); flush(stdout)
    end
    for (ix, x̄) in enumerate(random_xs(td, nx))
        Vg, tg, Vgb, gopt = global_eval(td, x̄; time_limit=global_time_limit)
        @printf("x̄[%d]=%s  global V=%.6f bound=%.6f opt=%s (%.2fs)\n", ix,
                string(findall(x̄ .> 0.5)), Vg, Vgb, gopt, tg); flush(stdout)
        sd = make_struct_eval_data(td, x̄; lp_optimizer=grb_lp)
        cover = nothing
        for enc in encodings
            if :monolithic in methods
                _report(:monolithic, enc, evaluate_monolithic(sd; optimizer=grb, encoding=enc), x̄, Vg, Vgb)
            end
            if :decomposed in methods || :disjunctive in methods
                if cover_method == :pattern
                    cover = generate_belief_cover(sd; optimizer=grb, encoding=enc, cover_kwargs...)
                    @printf("   cover(pattern,%s): patterns=%d beliefs=%d complete=%s %.2fs\n",
                            enc, length(cover[:patterns]), length(cover[:beliefs]),
                            cover[:complete], cover[:time]); flush(stdout)
                elseif cover === nothing
                    cover = generate_belief_vertex_cover(sd; lp_optimizer=grb_lp, cover_kwargs...)
                    @printf("   cover(vertex): beliefs=%d oracle=%d env=%d tv=%d rays=%d complete=%s %.2fs\n",
                            length(cover[:beliefs]), cover[:n_oracle], cover[:n_cuts_env],
                            cover[:n_cuts_tv], cover[:n_rays], cover[:complete], cover[:time]); flush(stdout)
                    td.S <= 3 && println("      beliefs=", [round.(b; digits=4) for b in cover[:beliefs]])
                end
                for meth in (:decomposed, :disjunctive)
                    meth in methods || continue
                    f = meth == :decomposed ? evaluate_decomposed : evaluate_disjunctive
                    ev = f(sd, cover[:beliefs]; optimizer=grb, encoding=enc)
                    ev[:is_exact] &= cover[:complete]
                    _report(meth, enc, ev, x̄, Vg, Vgb)
                end
            end
        end
    end
    @printf("Stage 1 failures: %d\n", n_fail); flush(stdout)
    return n_fail
end


function stage2(td; methods=(:monolithic, :decomposed, :disjunctive), encoding=:indicator,
                tol=1e-4)
    println("\n", "="^90)
    @printf("Stage 2 (Benders): K=%d S=%d ε̂=%.2f ε̃=%.2f β=%s\n",
            td.num_arcs, td.S, td.eps_hat, td.eps_tilde, string(td.beta))
    println("="^90)
    ref = true_dro_benders_optimize!(td; mip_optimizer=grb, nlp_optimizer=grb,
                                     max_iter=300, tol=tol, verbose=false, inexact=false)
    @printf("  %-22s Z0=%.6f iters=%3d time=%7.2fs x*=%s\n", "bilinear (existing)",
            ref[:Z0], ref[:iters], ref[:wall_time], string(findall(ref[:x] .> 0.5))); flush(stdout)
    n_fail = 0
    for meth in methods
        r = structured_benders_optimize!(td; method=meth, encoding=encoding,
                                         mip_optimizer=grb, lp_optimizer=grb_lp,
                                         max_iter=300, tol=tol, verbose=false)
        ok = r[:status] == :Optimal && abs(r[:Z0] - ref[:Z0]) <= 10tol * max(1, abs(ref[:Z0]))
        n_fail += !ok
        @printf("  %-22s Z0=%.6f iters=%3d time=%7.2fs x*=%s  %s\n",
                "$(meth)/$(encoding)", r[:Z0], r[:iters], r[:wall_time],
                string(findall(r[:x] .> 0.5)), ok ? "OK" : "**MISMATCH**"); flush(stdout)
    end
    return n_fail
end


# ============================================================================
# Stage 3: factor_5 batch 인스턴스에서 고정 x̄ 평가 시간 비교 (global bilinear vs monolithic)
# ============================================================================

"""factor_5/run_baseline_batch.jl과 동일한 인스턴스 생성."""
function make_batch_instance(name::String; beta=0.4, eps=0.2, S=10, double=true)
    if name == "grid5x5"
        net = generate_grid_network(5, 5; seed=42)
        intd = net.interdictable_arcs
    else
        gen = Dict("polska" => generate_polska_network, "abilene" => generate_abilene_network,
                   "nobel_us" => generate_nobel_us_network,
                   "sioux_falls" => generate_sioux_falls_network)
        net0 = gen[name]()
        intd = fill(true, length(net0.arcs))
        net = RealWorldNetworkData(net0.name, net0.original_node_names, net0.nodes, net0.arcs,
                                   net0.N, intd, net0.arc_adjacency, net0.node_arc_incidence)
    end
    K = length(net.arcs) - 1
    γ = 2
    caps, _ = generate_capacity_scenarios_factor_additive(length(net.arcs), S;
                  interdictable_arcs=intd, seed=42, num_factors=5)
    intd_idx = findall(intd[1:K])
    w = round(0.5 * γ * median(caps[intd_idx, :]); digits=4)
    Random.seed!(42)
    v = zeros(K, S)
    for k in 1:K, s in 1:S
        v[k, s] = intd[k] ? (rand() < 0.75 ? 1.0 : 0.0) : 0.0
    end
    return make_true_dro_data(net, caps, fill(1.0 / S, S), eps, double ? eps : 0.0;
                              w=w, lambda_U=10.0, gamma=γ, beta=beta, v_scenarios=v)
end


function stage3(td; nx=3, time_limit=600.0, encodings=(:indicator, :bigM),
                methods=(:monolithic, :decomposed, :disjunctive), run_global=true,
                cover_time_limit=time_limit)
    println("\n", "="^90)
    @printf("Stage 3 (timing): K=%d S=%d ε̂=%.2f ε̃=%.2f β=%s w=%.3f\n",
            td.num_arcs, td.S, td.eps_hat, td.eps_tilde, string(td.beta), td.w); flush(stdout)
    println("="^90)
    for (ix, x̄) in enumerate(random_xs(td, nx))
        if run_global
            model, vars = build_true_dro_subproblem(td, x̄; optimizer=grb, silent=true)
            set_optimizer_attribute(model, "NonConvex", 2)
            set_optimizer_attribute(model, "MIPGap", 1e-6)
            set_time_limit_sec(model, time_limit)
            tg = @elapsed info = solve_true_dro_subproblem!(model, vars, td, x̄)
            @printf("x̄[%d]=%s  global: V=%.6f bound=%.6f opt=%s  %.2fs\n", ix,
                    string(findall(x̄ .> 0.5)), info[:Z0_val], info[:Z0_bound], info[:is_optimal], tg); flush(stdout)
        else
            @printf("x̄[%d]=%s\n", ix, string(findall(x̄ .> 0.5))); flush(stdout)
        end
        sd = make_struct_eval_data(td, x̄; lp_optimizer=grb_lp)
        cover = generate_belief_vertex_cover(sd; lp_optimizer=grb_lp, time_limit=cover_time_limit)
        @printf("   cover(vertex): beliefs=%d oracle=%d env=%d tv=%d rays=%d complete=%s %.2fs\n",
                length(cover[:beliefs]), cover[:n_oracle], cover[:n_cuts_env],
                cover[:n_cuts_tv], cover[:n_rays], cover[:complete], cover[:time]); flush(stdout)
        for enc in encodings
            for (meth, f) in ((:monolithic, sd -> evaluate_monolithic(sd; optimizer=grb, encoding=enc, time_limit=time_limit)),
                              (:decomposed, sd -> evaluate_decomposed(sd, cover[:beliefs]; optimizer=grb, encoding=enc, time_limit=time_limit)),
                              (:disjunctive, sd -> evaluate_disjunctive(sd, cover[:beliefs]; optimizer=grb, encoding=enc, time_limit=time_limit)))
                meth in methods || continue
                if meth != :monolithic && !cover[:complete]
                    @printf("   %-11s %-9s SKIPPED (cover incomplete: %d beliefs so far)\n",
                            meth, enc, length(cover[:beliefs])); flush(stdout)
                    continue
                end
                ev = try
                    f(sd)
                catch err
                    @printf("   %-11s %-9s FAILED: %s\n", meth, enc, sprint(showerror, err)); flush(stdout)
                    continue
                end
                _, Z0lp, _, _ = structured_outer_cut(td, x̄, ev[:α]; lp_optimizer=grb_lp)
                @printf("   %-11s %-9s V=%.6f bound=%.6f opt=%s LP(α*)=%.6f bin=%d %.2fs%s\n",
                        meth, enc, ev[:V], ev[:V_bound], ev[:is_exact], Z0lp, ev[:n_bin], ev[:time],
                        haskey(ev, :n_pruned) ? @sprintf(" pruned=%d/%d", ev[:n_pruned], ev[:n_beliefs]) : ""); flush(stdout)
            end
        end
    end
end


# ============================================================================
# Stage 4: 부분 cover (q̃ + TV 볼 꼭짓점) — 하한/valid cut 품질, MIP start 효과
# ============================================================================
function stage4(td; nx=2, time_limit=600.0, run_global=true, global_first_only=false,
                variants=((:decomposed, :bilinear), (:disjunctive, :bilinear),
                          (:monolithic, :bilinear), (:monolithic, :kkt)))
    println("\n", "="^90)
    @printf("Stage 4 (S large): K=%d S=%d ε̂=%.2f ε̃=%.2f β=%s\n",
            td.num_arcs, td.S, td.eps_hat, td.eps_tilde, string(td.beta)); flush(stdout)
    println("="^90)
    for (ix, x̄) in enumerate(random_xs(td, nx))
        if run_global && (!global_first_only || ix == 1)
            Vg, tg, Vgb, gopt = global_eval(td, x̄; time_limit=time_limit)
            @printf("x̄[%d]=%s  global-bilinear V=%.6f bound=%.6f opt=%s (%.2fs)\n", ix,
                    string(findall(x̄ .> 0.5)), Vg, Vgb, gopt, tg); flush(stdout)
        else
            @printf("x̄[%d]=%s\n", ix, string(findall(x̄ .> 0.5))); flush(stdout)
        end
        sd = make_struct_eval_data(td, x̄; lp_optimizer=grb_lp)
        B = vcat([copy(sd.q_tilde)], td.eps_tilde > 0 ? _tv_ball_vertices(sd.q_tilde, td.eps_tilde) : Vector{Float64}[])
        for (meth, ldr) in variants
            ev = try
                if meth == :decomposed
                    evaluate_decomposed(sd, B; optimizer=grb, time_limit=time_limit, leader=ldr)
                elseif meth == :disjunctive
                    evaluate_disjunctive(sd, B; optimizer=grb, time_limit=time_limit, leader=ldr)
                else
                    evaluate_monolithic(sd; optimizer=grb, encoding=:indicator, time_limit=time_limit,
                                        leader=ldr, mip_start=true, lp_optimizer=grb_lp)
                end
            catch err
                @printf("   %-11s leader=%-8s FAILED: %s\n", meth, ldr, sprint(showerror, err)); flush(stdout)
                continue
            end
            _, Z0lp, _, zf = structured_outer_cut(td, x̄, ev[:α]; lp_optimizer=grb_lp)
            tag = meth == :monolithic ? "exact-form" : @sprintf("partial J=%d (하한)", length(B))
            @printf("   %-11s leader=%-8s V=%.6f bound=%.6f opt=%s LP(α*)=%.6f V̄F=%.1e %.2fs  [%s]%s\n",
                    meth, ldr, ev[:V], ev[:V_bound], ev[:is_exact], Z0lp, zf, ev[:time], tag,
                    haskey(ev, :n_pruned) ? @sprintf(" pruned=%d", ev[:n_pruned]) : ""); flush(stdout)
        end
    end
end


# ============================================================================
# Stage 5: minimal cover (국소 결정 Lemma + 극대 패턴) — 크기/정확성/속도
# ============================================================================
function stage5(td; nx=2, time_limit=300.0, run_global=true, compare_vertex=true,
                leaders=(:bilinear,), methods=(:decomposed, :disjunctive))
    println("\n", "="^90)
    @printf("Stage 5 (minimal cover): K=%d S=%d ε̂=%.2f ε̃=%.2f β=%s\n",
            td.num_arcs, td.S, td.eps_hat, td.eps_tilde, string(td.beta)); flush(stdout)
    println("="^90)
    n_fail = 0
    for (ix, x̄) in enumerate(random_xs(td, nx))
        Vg, Vgb = NaN, NaN
        if run_global
            Vg, tg, Vgb, gopt = global_eval(td, x̄; time_limit=time_limit)
            @printf("x̄[%d]=%s  global V=%.6f bound=%.6f opt=%s (%.2fs)\n", ix,
                    string(findall(x̄ .> 0.5)), Vg, Vgb, gopt, tg); flush(stdout)
        else
            @printf("x̄[%d]=%s\n", ix, string(findall(x̄ .> 0.5))); flush(stdout)
        end
        sd = make_struct_eval_data(td, x̄; lp_optimizer=grb_lp)
        cov = generate_belief_minimal_cover(sd; optimizer=grb, lp_optimizer=grb_lp,
                                            time_limit=time_limit, verbose=true)
        @printf("   cover(minimal): J=%d pieces=%d sep=%d complete=%s %.2fs (sep %.2fs, pat %.2fs)\n",
                length(cov[:beliefs]), cov[:n_pieces], cov[:n_sep], cov[:complete], cov[:time],
                cov[:sep_time], cov[:pat_time]); flush(stdout)
        if compare_vertex
            cv = generate_belief_vertex_cover(sd; lp_optimizer=grb_lp, time_limit=60.0)
            @printf("   cover(vertex):  J=%d complete=%s %.2fs\n", length(cv[:beliefs]),
                    cv[:complete], cv[:time]); flush(stdout)
        end
        for meth in methods, ldr in leaders
            f = meth == :decomposed ? evaluate_decomposed : evaluate_disjunctive
            ev = try
                f(sd, cov[:beliefs]; optimizer=grb, time_limit=time_limit, leader=ldr)
            catch err
                @printf("   %-11s leader=%-8s FAILED: %s\n", meth, ldr, sprint(showerror, err)); flush(stdout)
                continue
            end
            _, Z0lp, _, zf = structured_outer_cut(td, x̄, ev[:α]; lp_optimizer=grb_lp)
            certified = ev[:is_exact] && cov[:complete]
            ok = !run_global || (Vg - 1e-4 * max(1, abs(Vg)) <= ev[:V] <= Vgb + 1e-4 * max(1, abs(Vgb)))
            n_fail += !ok
            @printf("   %-11s leader=%-8s V=%.6f bound=%.6f certified=%s LP(α*)=%.6f V̄F=%.1e %.2fs%s  %s\n",
                    meth, ldr, ev[:V], ev[:V_bound], certified, Z0lp, zf, ev[:time],
                    haskey(ev, :n_pruned) ? @sprintf(" pruned=%d/%d", ev[:n_pruned], length(cov[:beliefs])) : "",
                    ok ? "OK" : "**MISMATCH**"); flush(stdout)
        end
    end
    @printf("Stage 5 failures: %d\n", n_fail); flush(stdout)
    return n_fail
end


# ============================================================================
# Stage 6: belief-bilinear (bilinear 2S개) vs global bilinear (Ω, 2KS개)
# ============================================================================
function stage6(td; nx=2, time_limit=300.0, leaders=(:bilinear, :kkt), tol=1e-4)
    println("\n", "="^90)
    @printf("Stage 6 (belief-bilinear): K=%d S=%d ε̂=%.2f ε̃=%.2f β=%s  [Ω bilinear 2KS=%d, belief 2S=%d]\n",
            td.num_arcs, td.S, td.eps_hat, td.eps_tilde, string(td.beta), 2td.num_arcs * td.S, 2td.S); flush(stdout)
    println("="^90)
    n_fail = 0
    for (ix, x̄) in enumerate(random_xs(td, nx))
        Vg, tg, Vgb, gopt = global_eval(td, x̄; time_limit=time_limit)
        @printf("x̄[%d]=%s  global(Ω)        V=%.6f bound=%.6f opt=%-5s %7.2fs\n", ix,
                string(findall(x̄ .> 0.5)), Vg, Vgb, gopt, tg); flush(stdout)
        sd = make_struct_eval_data(td, x̄; lp_optimizer=grb_lp)
        for ldr in leaders
            ev = try evaluate_belief_bilinear(sd; optimizer=grb, leader=ldr, time_limit=time_limit) catch err
                @printf("   belief-bilinear %-8s FAILED: %s\n", ldr, sprint(showerror, err)); flush(stdout); continue
            end
            _, Z0lp, _, zf = structured_outer_cut(td, x̄, ev[:α]; lp_optimizer=grb_lp)
            # 상호 일관성: 두 구간 [V, bound]가 겹쳐야 함 (둘 다 V* 포함)
            lo = max(Vg, ev[:V]); hi = min(Vgb, ev[:V_bound])
            ok = lo <= hi + tol * max(1, abs(hi)) && abs(Z0lp - ev[:V]) <= tol * max(1, abs(ev[:V]))
            n_fail += !ok
            @printf("   belief-bilinear %-8s V=%.6f bound=%.6f opt=%-5s %7.2fs  LP(α*)=%.6f %s\n",
                    ldr, ev[:V], ev[:V_bound], ev[:is_exact], ev[:time], Z0lp, ok ? "OK" : "**INCONSISTENT**"); flush(stdout)
        end
    end
    @printf("Stage 6 failures: %d\n", n_fail); flush(stdout)
    return n_fail
end


# ============================================================================
# Stage 7: belief-space B&B vs global(Ω)
# ============================================================================
function stage7(td; nx=2, time_limit=300.0, run_global=true, w_leader=:bilinear, tol=1e-4)
    println("\n", "="^90)
    @printf("Stage 7 (belief B&B): K=%d S=%d ε̂=%.2f ε̃=%.2f β=%s  w_leader=%s\n",
            td.num_arcs, td.S, td.eps_hat, td.eps_tilde, string(td.beta), w_leader); flush(stdout)
    println("="^90)
    n_fail = 0
    for (ix, x̄) in enumerate(random_xs(td, nx))
        Vg, Vgb = NaN, NaN
        if run_global
            Vg, tg, Vgb, gopt = global_eval(td, x̄; time_limit=time_limit)
            @printf("x̄[%d]=%s  global(Ω)   V=%.6f bound=%.6f opt=%-5s %7.2fs\n", ix,
                    string(findall(x̄ .> 0.5)), Vg, Vgb, gopt, tg); flush(stdout)
        else
            @printf("x̄[%d]=%s\n", ix, string(findall(x̄ .> 0.5))); flush(stdout)
        end
        sd = make_struct_eval_data(td, x̄; lp_optimizer=grb_lp)
        ev = evaluate_belief_bnb(sd; lp_optimizer=grb_lp, mip_optimizer=grb, w_leader=w_leader,
                                 time_limit=time_limit, verbose=(td.S >= 10))
        _, Z0lp, _, zf = structured_outer_cut(td, x̄, ev[:α]; lp_optimizer=grb_lp)
        ok = true
        if run_global
            lo = max(Vg, ev[:V]); hi = min(Vgb, ev[:V_bound])
            ok = lo <= hi + tol * max(1, abs(hi))
        end
        ok &= Z0lp >= ev[:V] - tol * max(1, abs(ev[:V]))      # α*가 보고값 이상 달성
        n_fail += !ok
        @printf("   belief-B&B  V=%.6f bound=%.6f opt=%-5s %7.2fs  nodes=%d nW=%d rootUB=%.4f LP(α*)=%.6f %s\n",
                ev[:V], ev[:V_bound], ev[:is_exact], ev[:time], ev[:nodes], ev[:n_W], ev[:root_ub],
                Z0lp, ok ? "OK" : "**INCONSISTENT**"); flush(stdout)
    end
    @printf("Stage 7 failures: %d\n", n_fail); flush(stdout)
    return n_fail
end


# ============================================================================
# Stage 8: follower-response CCG (bilinear master / MILP master) vs global(Ω)
# ============================================================================
function stage8(td; nx=2, time_limit=300.0, run_global=true, masters=(:bilinear, :milp), tol=1e-4)
    println("\n", "="^90)
    @printf("Stage 8 (response CCG): K=%d S=%d ε̂=%.2f ε̃=%.2f β=%s\n",
            td.num_arcs, td.S, td.eps_hat, td.eps_tilde, string(td.beta)); flush(stdout)
    println("="^90)
    n_fail = 0
    for (ix, x̄) in enumerate(random_xs(td, nx))
        Vg, Vgb = NaN, NaN
        if run_global
            Vg, tg, Vgb, gopt = global_eval(td, x̄; time_limit=time_limit)
            @printf("x̄[%d]=%s  global(Ω)    V=%.6f bound=%.6f opt=%-5s %7.2fs\n", ix,
                    string(findall(x̄ .> 0.5)), Vg, Vgb, gopt, tg); flush(stdout)
        else
            @printf("x̄[%d]=%s\n", ix, string(findall(x̄ .> 0.5))); flush(stdout)
        end
        sd = make_struct_eval_data(td, x̄; lp_optimizer=grb_lp)
        for ms in masters
            ev = try
                evaluate_response_ccg(sd; master=ms, mip_optimizer=grb, lp_optimizer=grb_lp,
                                      time_limit=time_limit, verbose=(td.S >= 10))
            catch err
                @printf("   ccg-%-8s FAILED: %s\n", ms, sprint(showerror, err)); flush(stdout); continue
            end
            _, Z0lp, _, zf = structured_outer_cut(td, x̄, ev[:α]; lp_optimizer=grb_lp)
            ok = Z0lp >= ev[:V] - tol * max(1, abs(ev[:V])) && zf > -1e-4
            if run_global
                lo = max(Vg, ev[:V]); hi = min(Vgb, ev[:V_bound])
                ok &= lo <= hi + tol * max(1, abs(hi))
            end
            n_fail += !ok
            @printf("   ccg-%-8s V=%.6f bound=%.6f opt=%-5s %7.2fs  iters=%d pieces=%d master=%.1fs%s LP(α*)=%.6f %s\n",
                    ms, ev[:V], ev[:V_bound], ev[:is_exact], ev[:time], ev[:iters], ev[:n_pieces],
                    ev[:master_time], ms == :milp ? @sprintf(" |V|=%d", ev[:n_vertices]) : "",
                    Z0lp, ok ? "OK" : "**INCONSISTENT**"); flush(stdout)
        end
    end
    @printf("Stage 8 failures: %d\n", n_fail); flush(stdout)
    return n_fail
end


# ============================================================================
# Stage 9: S 교차점 실험 (같은 네트워크, S만 변화)
# ============================================================================
function stage9(name::String; S_list=(3, 5, 7, 10), nx=2, time_limit=300.0, beta=0.4, eps=0.2)
    rows = NamedTuple[]
    for S in S_list
        td = make_batch_instance(name; beta=beta, eps=eps, S=S)
        println("\n", "="^90)
        @printf("Stage 9 [%s] S=%d K=%d β=%.2f ε=%.2f\n", name, S, td.num_arcs, beta, eps); flush(stdout)
        println("="^90)
        for (ix, x̄) in enumerate(random_xs(td, nx))
            xs = string(findall(x̄ .> 0.5))
            Vg, tg, Vgb, gopt = global_eval(td, x̄; time_limit=time_limit)
            @printf("  S=%2d x̄=%-9s global       V=%.6f bound=%.6f opt=%-5s %7.2fs\n",
                    S, xs, Vg, Vgb, gopt, tg); flush(stdout)
            push!(rows, (S=S, x=xs, method="global", V=Vg, bound=Vgb, opt=gopt, time=tg, size=0))
            sd = make_struct_eval_data(td, x̄; lp_optimizer=grb_lp)
            # follower 반응 CCG (MILP master)
            ev = try evaluate_response_ccg(sd; master=:milp, mip_optimizer=grb, lp_optimizer=grb_lp,
                                           time_limit=time_limit) catch err
                @printf("  S=%2d x̄=%-9s ccg-milp     FAILED %s\n", S, xs, sprint(showerror, err)); nothing
            end
            if ev !== nothing
                @printf("  S=%2d x̄=%-9s ccg-milp     V=%.6f bound=%.6f opt=%-5s %7.2fs  iters=%d pieces=%d |V|=%d\n",
                        S, xs, ev[:V], ev[:V_bound], ev[:is_exact], ev[:time], ev[:iters], ev[:n_pieces], ev[:n_vertices]); flush(stdout)
                push!(rows, (S=S, x=xs, method="ccg-milp", V=ev[:V], bound=ev[:V_bound], opt=ev[:is_exact], time=ev[:time], size=ev[:n_vertices]))
            end
            # minimal cover + disjunctive (KKT leader)
            t0 = time()
            cov = generate_belief_minimal_cover(sd; optimizer=grb, lp_optimizer=grb_lp, sep_time_limit=time_limit)
            rem = max(time_limit - (time() - t0), 1.0)
            ev2 = try
                isempty(cov[:beliefs]) ? nothing :
                    evaluate_disjunctive(sd, cov[:beliefs]; optimizer=grb, leader=:kkt, time_limit=rem)
            catch err
                nothing
            end
            tt = time() - t0
            if ev2 === nothing
                @printf("  S=%2d x̄=%-9s mincover     (평가 실패/미완료) cover J=%d complete=%s %7.2fs\n",
                        S, xs, length(cov[:beliefs]), cov[:complete], tt); flush(stdout)
                push!(rows, (S=S, x=xs, method="mincover", V=NaN, bound=NaN, opt=false, time=tt, size=length(cov[:beliefs])))
            else
                cert = ev2[:is_exact] && cov[:complete]
                @printf("  S=%2d x̄=%-9s mincover     V=%.6f bound=%.6f opt=%-5s %7.2fs  J=%d pieces=%d cover_complete=%s\n",
                        S, xs, ev2[:V], ev2[:V_bound], cert, tt, length(cov[:beliefs]), cov[:n_pieces], cov[:complete]); flush(stdout)
                push!(rows, (S=S, x=xs, method="mincover", V=ev2[:V], bound=ev2[:V_bound], opt=cert, time=tt, size=length(cov[:beliefs])))
            end
        end
    end
    println("\nSUMMARY (S, x̄, method, V, bound, opt, time, size)")
    for r in rows
        @printf("SUM,%d,%s,%s,%.6f,%.6f,%s,%.2f,%d\n", r.S, r.x, r.method, r.V, r.bound, r.opt, r.time, r.size)
    end
    flush(stdout)
    return rows
end
