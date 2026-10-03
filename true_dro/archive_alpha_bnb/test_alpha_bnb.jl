"""
test_alpha_bnb.jl — α-공간 B&B vs Gurobi global (Ω, NonConvex=2) 비교.

check(): 작은 S에서 두 방법의 최적값 일치 확인
sweep(): S = 10, 20, 50, 100, 200 에서 같은 시간 제한으로 LB/UB(dual bound) 비교
실행: julia test_alpha_bnb.jl [check|sweep]
"""

include("test_structured_eval.jl")
include("alpha_bnb.jl")

function grb_lp_alpha()
    o = Gurobi.Optimizer(GRB_ENV)
    MOI.set(o, MOI.Silent(), true)
    return o
end

"""Gurobi global on Ω. (V, bound, opt, time)"""
function global_omega(td, x̄; time_limit, mip_gap=1e-4)
    model, vars = build_true_dro_subproblem(td, x̄; optimizer=grb, silent=true)
    set_optimizer_attribute(model, "NonConvex", 2)
    set_optimizer_attribute(model, "MIPGap", mip_gap)
    set_time_limit_sec(model, time_limit)
    t = @elapsed info = solve_true_dro_subproblem!(model, vars, td, x̄)
    return info[:Z0_val], info[:Z0_bound], info[:is_optimal], t
end

gap(lb, ub) = (ub - lb) / max(1.0, abs(lb))

function run_pair(name, S, x̄; time_limit, beta=0.4, eps=0.2)
    td = make_batch_instance(name; beta=beta, eps=eps, S=S)
    xs = string(findall(x̄ .> 0.5))
    Vg, Bg, og, tg = global_omega(td, x̄; time_limit=time_limit)
    @printf("  S=%3d x̄=%-8s global : LB=%.6f UB=%.6f gap=%.2e opt=%-5s %7.1fs\n",
            S, xs, Vg, Bg, gap(Vg, Bg), og, tg); flush(stdout)
    r = alpha_bnb(td, x̄; optimizer=grb_lp_alpha, time_limit=time_limit, verbose=true)
    @printf("  S=%3d x̄=%-8s α-B&B  : LB=%.6f UB=%.6f gap=%.2e opt=%-5s %7.1fs nodes=%d rootUB=%.6f build=%.1fs\n",
            S, xs, r[:LB], r[:UB], gap(r[:LB], r[:UB]), r[:is_exact], r[:time], r[:nodes], r[:root_UB], r[:t_build])
    @printf("SUM,%d,%s,%d,%.6f,%.6f,%s,%.1f,%.6f,%.6f,%s,%.1f,%d,%.6f\n", S, xs, td.num_arcs,
            Vg, Bg, og, tg, r[:LB], r[:UB], r[:is_exact], r[:time], r[:nodes], r[:root_UB])
    flush(stdout)
end

function check()
    println("=== check: 작은 S 정확성 (Abilene β=0.4 ε=0.2) ===")
    for S in (3, 5)
        td = make_batch_instance("abilene"; S=S)
        for x̄ in random_xs(td, 2)
            run_pair("abilene", S, x̄; time_limit=300.0)
        end
    end
end

function sweep(; S_list=(10, 20, 50, 100, 200), time_limit=300.0)
    println("=== sweep: Abilene β=0.4 ε=0.2, time_limit=$(time_limit)s ===")
    for S in S_list
        td = make_batch_instance("abilene"; S=S)
        for x̄ in random_xs(td, 2)
            run_pair("abilene", S, x̄; time_limit=time_limit)
        end
    end
end


"""α-B&B만 (global 결과는 이전 sweep 재사용). rlt=:full 기본."""
function sweep_alpha_only(; S_list=(10, 20, 50, 100, 200), time_limit=300.0, rlt=:full)
    println("=== sweep (α-B&B only, rlt=$rlt): Abilene β=0.4 ε=0.2, time_limit=$(time_limit)s ===")
    for S in S_list
        td = make_batch_instance("abilene"; S=S)
        for x̄ in random_xs(td, 2)
            xs = string(findall(x̄ .> 0.5))
            r = alpha_bnb(td, x̄; optimizer=grb_lp_alpha, time_limit=time_limit, verbose=true, rlt=rlt)
            @printf("  S=%3d x̄=%-8s α-B&B[%s]: LB=%.6f UB=%.6f gap=%.2e opt=%-5s %7.1fs nodes=%d rootUB=%.6f build=%.1fs\n",
                    S, xs, rlt, r[:LB], r[:UB], gap(r[:LB], r[:UB]), r[:is_exact], r[:time], r[:nodes], r[:root_UB], r[:t_build])
            @printf("SUMA,%d,%s,%s,%.6f,%.6f,%s,%.1f,%d,%.6f\n", S, xs, rlt, r[:LB], r[:UB], r[:is_exact], r[:time], r[:nodes], r[:root_UB])
            flush(stdout)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    mode = isempty(ARGS) ? "check" : ARGS[1]
    mode == "check" ? check() : mode == "full" ? sweep_alpha_only() : sweep()
end
