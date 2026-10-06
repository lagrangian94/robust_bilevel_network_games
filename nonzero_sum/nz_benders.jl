"""
nz_benders.jl — 비제로섬 Ω 위의 Benders 두 가지.

  nz_standard_benders  : 매 반복 x̄ 에서 전역 Ω (Gurobi NonConvex) → cut, UB.
  nz_belief_menu_benders: 설계 A (true_dro/belief_menu_benders.jl) 를 확장 belief 로.
      menu 원소 = (belief (a,b,r,d,e), follower 인증서 σ).  고정하면 Ω 가 LP (html "확장 belief").
      menu 단계: 모든 원소의 LP 값 v_b(x̄) 가 t₀ 보다 크면 그 LP 해로 cut.
      oracle 단계 (menu 가 x̄ 에서 수렴): 전역 Ω(x̄) → UB, cut, 새 belief + 인증서를 menu 에 추가.
      인증서: cert = :recompute (html (b): (x̄, α̂) 에서 follower LP simplex dual, 기본)
                   | :solution  (html (a): ϖ̂ˢ / r̂_s)
      검증 기록: 새 원소를 넣을 때 그 LP 를 x̄ 에서 풀어 Ω(x̄) 와 같은지 (exactness) :exact_gap 에 남김.
OMP: min q_xᵀx + t₀, 1ᵀx ≤ γ, t₀ ≥ intercept + slopeᵀx.
"""

using JuMP, Printf, LinearAlgebra
import MathOptInterface as MOI

function _nz_omp(nd::NZData; optimizer, t_lb=-1e7)
    omp = Model(optimizer); set_silent(omp)
    @variable(omp, x[1:nd.nx], Bin)
    @variable(omp, t0 >= t_lb)
    @constraint(omp, sum(x) <= nd.gamma)
    for i in 1:nd.nx
        nd.x_allowed[i] || fix(x[i], 0.0; force=true)
    end
    @objective(omp, Min, dot(nd.qx, x) + t0)
    return omp, x, t0
end

_add_cut!(omp, x, t0, cut) = @constraint(omp, t0 >= cut[:intercept] + dot(cut[:slope], x))
_rel(a, b) = abs(b - a) / max(abs(b), 1.0)


function nz_standard_benders(nd::NZData; optimizer, max_iter=200, tol=1e-4, oracle_time=600.0, verbose=true)
    omp, x, t0 = _nz_omp(nd; optimizer=optimizer)
    O = build_nz_omega(nd; optimizer=optimizer)
    LB, UB, best_x = -Inf, Inf, zeros(nd.nx)
    hist = NamedTuple[]
    wall = time(); iter = 0; status = :MaxIter
    while iter < max_iter
        iter += 1
        optimize!(omp)
        x̄ = Float64.(value.(x) .> 0.5)
        LB = objective_value(omp)
        _rel(LB, UB) <= tol && (status = :Optimal; break)
        t_o = @elapsed (res = nz_solve!(O, nd, x̄; time_limit=oracle_time))
        cut = nz_cut_from_solution(O, nd, x̄)
        _add_cut!(omp, x, t0, cut)
        ub_here = dot(nd.qx, x̄) + res[:bound]
        ub_here < UB && (UB = ub_here; best_x = copy(x̄))
        push!(hist, (iter=iter, LB=LB, UB=UB, x=findall(x̄ .> 0.5), F=res[:Fval], t=t_o))
        verbose && @printf("  [std] it %3d LB=%11.4f UB=%11.4f x=%-10s Ω=%11.4f (%.1fs)\n",
                           iter, LB, UB, string(findall(x̄ .> 0.5)), res[:Fval], t_o)
        verbose && flush(stdout)
    end
    return Dict(:status => status, :LB => LB, :UB => UB, :x => best_x, :iters => iter,
                :oracle_calls => iter - (status == :Optimal ? 1 : 0), :wall => time() - wall, :hist => hist)
end


function nz_belief_menu_benders(nd::NZData; optimizer, lp_optimizer=optimizer, cert::Symbol=:recompute,
                                cert_rows::Symbol=:hrows, max_iter=500, tol=1e-4, oracle_time=600.0, verbose=true)
    cert in (:recompute, :solution) || error("cert = :recompute | :solution")
    omp, x, t0 = _nz_omp(nd; optimizer=optimizer)
    O = build_nz_omega(nd; optimizer=optimizer)
    B = build_nz_omega(nd; optimizer=lp_optimizer, belief_lp=true, cert_rows=cert_rows)
    menu = Tuple{Dict{Symbol,Any},Matrix{Float64}}[]
    LB, UB, best_x = -Inf, Inf, zeros(nd.nx)
    oracle_calls = 0; menu_cuts = 0
    exact_log = NamedTuple[]
    hist = NamedTuple[]
    wall = time(); iter = 0; status = :MaxIter
    while iter < max_iter
        iter += 1
        optimize!(omp)
        x̄ = Float64.(value.(x) .> 0.5)
        LB = objective_value(omp)
        t0v = value(t0)
        _rel(LB, UB) <= tol && (status = :Optimal; break)

        # ---- menu 단계 ----
        V_menu = -Inf; nviol = 0
        for (bel, σ) in menu
            nz_set_belief!(B, nd, bel, σ)
            res = nz_solve!(B, nd, x̄)
            V_menu = max(V_menu, res[:Fval])
            if res[:Fval] > t0v + 1e-6 * max(1.0, abs(t0v))
                _add_cut!(omp, x, t0, nz_cut_from_solution(B, nd, x̄)); nviol += 1; menu_cuts += 1
            end
        end
        if nviol > 0
            push!(hist, (iter=iter, LB=LB, UB=UB, x=findall(x̄ .> 0.5), kind=:menu, n=nviol))
            verbose && @printf("  [menu] it %3d LB=%11.4f UB=%11.4f x=%-10s menu cut %d/%d (V_menu=%.4f)\n",
                               iter, LB, UB, string(findall(x̄ .> 0.5)), nviol, length(menu), V_menu)
            verbose && flush(stdout)
            continue
        end

        # ---- oracle 단계 ----
        oracle_calls += 1
        t_o = @elapsed (res = nz_solve!(O, nd, x̄; time_limit=oracle_time))
        cut = nz_cut_from_solution(O, nd, x̄)
        bel_raw = nz_read_belief(O, nd)
        _add_cut!(omp, x, t0, cut)
        ub_here = dot(nd.qx, x̄) + res[:bound]
        ub_here < UB && (UB = ub_here; best_x = copy(x̄))
        new_b = res[:Fval] > V_menu + 1e-6 * max(1.0, abs(res[:Fval]))
        if new_b
            bel = nz_clean_belief(bel_raw, nd)
            σ = if cert == :recompute
                nz_follower_duals(nd, x̄, bel_raw[:α]; lp_optimizer=lp_optimizer)[1]
            else
                nz_repair_cert(nd, bel_raw[:ϖ] ./ reshape(max.(bel_raw[:r], 1e-12), 1, nd.S); lp_optimizer=lp_optimizer)
            end
            push!(menu, (bel, σ))
            # exactness: 새 (belief, 인증서) 고정 LP 의 x̄ 값 = Ω(x̄) ?
            nz_set_belief!(B, nd, bel, σ)
            vb = nz_solve!(B, nd, x̄)[:Fval]
            push!(exact_log, (iter=iter, x=findall(x̄ .> 0.5), omega=res[:Fval], lp=vb, gap=res[:Fval] - vb))
        end
        push!(hist, (iter=iter, LB=LB, UB=UB, x=findall(x̄ .> 0.5), kind=:oracle, n=length(menu)))
        verbose && @printf("  [orcl] it %3d LB=%11.4f UB=%11.4f x=%-10s Ω=%11.4f (%.1fs) menu=%d%s\n",
                           iter, LB, UB, string(findall(x̄ .> 0.5)), res[:Fval], t_o, length(menu),
                           new_b ? @sprintf(" (+belief, LP@x̄=%.4f, gap %.1e)", exact_log[end].lp, exact_log[end].gap) : "")
        verbose && flush(stdout)
    end
    return Dict(:status => status, :LB => LB, :UB => UB, :x => best_x, :iters => iter,
                :oracle_calls => oracle_calls, :menu_cuts => menu_cuts, :menu => menu, :menu_size => length(menu),
                :exact_log => exact_log, :wall => time() - wall, :hist => hist)
end
