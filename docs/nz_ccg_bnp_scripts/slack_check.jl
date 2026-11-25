# 진단 (저장소 안 고침): _bccg_value! 의 follower 최적 행 여유 κ (지금 1e-7) 를 줄이면 평가값 v 가 어떻게 되는지
include(joinpath(@__DIR__, "zero_belief_hdr.jl"))
function value_kappa!(ev, α, κ)
    S, m = nd.S, nz_m(nd); Q = zeros(S); φ = zeros(S)
    for s in 1:S
        for k in 1:m
            rr = nd.u[k, s] + ev.gx[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * α[nd.hrow[k]] : 0.0)
            set_normalized_rhs(ev.MQcon[k], rr); set_normalized_rhs(ev.Mφcon[k], rr)
        end
        optimize!(ev.MQ); Q[s] = objective_value(ev.MQ)
        set_normalized_rhs(ev.MφcQ, Q[s] - 1e-9 * max(1.0, abs(Q[s]))); optimize!(ev.Mφ); φ[s] = objective_value(ev.Mφ)
    end
    lhs = dot(nd.c0, α)
    for s in 1:S; set_normalized_coefficient(ev.EFO, ev.Ed[s], Q[s]); set_objective_coefficient(ev.E, ev.Er[s], φ[s]); end
    set_normalized_rhs(ev.EFO, -lhs - κ * (1 + abs(lhs) + maximum(abs, Q)))
    optimize!(ev.E); termination_status(ev.E) == MOI.OPTIMAL || return -Inf, nothing, nothing
    return objective_value(ev.E), [value(ev.Ed[s]) for s in 1:S], [value(ev.Er[s]) for s in 1:S]
end
for xs in split(envf("XS", "{10:400}"), "|")
    x̄ = parsex(String(xs)); _bccg_setx!(ev, nd, x̄)
    worst = sort([(d0, _bccg_g!(ev, nd, d0)) for d0 in cands]; by=t -> -(t[2][1] === nothing ? -Inf : t[2][1]))
    shown = 0
    for (d0, (v, dd, r, α)) in worst
        dd === nothing && continue
        Vx = pessval(x̄, dd, r); v - Vx > 1e-4 * abs(v) || continue
        shown += 1; shown > 3 && break
        @printf("x=%s  지금 v %.3f, V_dd %.3f\n", xs, v, Vx)
        for κ in (1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 0.0)
            vk, dk, rk = value_kappa!(ev, α, κ)
            if dk === nothing; @printf("   κ %.0e: E 실행 불가 (%s)\n", κ, string(termination_status(ev.E))); continue; end
            @printf("   κ %.0e: v %.3f, 그 belief 의 V_d %.3f, α 비최적 %.2e, 블록 %.3f\n", κ, vk, pessval(x̄, dk, rk), fsub(x̄, dk, α)[1], blockval(x̄, dk, rk, MU))
        end
    end
end
