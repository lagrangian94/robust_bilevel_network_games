"""
nz_benders.jl — 비제로섬 Ω 위의 Benders 두 가지.

기본 구성 (zero-sum 과 동일 방침): oracle = α-B&B (nz_alpha_bnb), cut = Magnanti–Wong 강화 (mw = true).
  α-B&B oracle 은 nz_alpha_bnb.jl include 와 julia -t (nworkers+2),1 이 필요. Gurobi NonConvex oracle 은 oracle = :gurobi.
  MW 는 LP (belief LP, α 고정 LP) 에서만 건다. 비볼록 Ω 에 걸면 문제가 어려워지므로 :gurobi oracle 의 cut 은 기본 cut.

  nz_standard_benders  : 매 반복 x̄ 에서 전역 Ω → cut, UB.
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

# 기본 oracle 이 α-B&B 이므로 미리 불러 둔다 (nz_omega.jl 이 먼저 include 되어 있어야 함). 실행은 julia -t (nworkers+2),1.
isdefined(Main, :nz_alpha_bnb) || include(joinpath(@__DIR__, "nz_alpha_bnb.jl"))

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

"""
    nz_add_dual_vi!(omp, x, t0, nd) — 비제로섬 dual-VI (원고 Proposition dual-VI 의 비제로섬판, McCormick 선형화).

  V*(x) ≥ E^q̂ φ(x, h̄, ξ) ≥ min_{h∈H} Σ_s q̂_s φ_s(x, h)        (q̂ ∈ D̂ ∩ D̃, CVaR ≥ E, h̄ ∈ Ψ(x, q̂) ⊂ H)
  φ_s(x, h) ≤ r_s(x,h)ᵀπ_s − θᵁ cᵀy'_s   (Aᵀπ_s ≥ ℓ + θᵁc, π_s ≥ 0;  A y'_s ≤ r_s(x,h), y'_s ≥ 0)
     약쌍대 + Q_s ≥ cᵀy'_s 로 항상 성립, θᵁ ≥ θ* 이면 등호 가능 (exact penalty).
  h̄ = 0 은 쓸 수 없음 (φ 가 h 에 단조가 아님) → h 를 변수로 두고 최소화 (follower 선주문을 리더에게 가장 유리하게).
  x_i·π_k : 이진×연속, π_k ≤ piLU[k] 로 exact McCormick.
  h·π_k   : 연속×연속, π_k ≤ θᵁ·varpiU[k] 로 McCormick 완화 (master 가 t₀ 를 누르는 방향이므로 완화해도 유효).
     exact_h = true 는 진단용: h·π 를 bilinear 그대로 둠 (McCormick 손실과 "h 최소화" 손실을 분리).
     예약 행 dual 상한 θᵁ·p: 같은 점포의 예약/현장 열은 ℓ 이 같고 비용 차가 p 라, 예약 행 dual 을 최소로 고르면 ≤ θᵁ p.
"""
function nz_add_dual_vi!(omp, x, t0, nd::NZData; exact_h::Bool=false)
    m, ny, S = nz_m(nd), nz_ny(nd), nd.S
    xr = [k for k in 1:m if nd.xrow[k] > 0]
    hr = [k for k in 1:m if nd.hrow[k] > 0]
    all(isfinite, nd.varpiU[hr]) || error("nz_add_dual_vi!: h 행의 varpiU 가 유한해야 함")
    h = @variable(omp, [j=1:nz_nh(nd)], lower_bound=0, upper_bound=nd.hU[j], base_name="vi_h")
    @constraint(omp, nd.W * h .<= nd.wvec)
    rhs = exact_h ? QuadExpr() : AffExpr(0.0)
    for s in 1:S
        π = @variable(omp, [1:m], lower_bound=0, base_name="vi_pi_$s")
        y = @variable(omp, [1:ny], lower_bound=0, base_name="vi_y_$s")
        @constraint(omp, nd.A' * π .>= nd.ell .+ nd.thetaU .* nd.c)
        r = [nd.u[k, s] + (nd.hrow[k] > 0 ? nd.hcoef[k] * h[nd.hrow[k]] : 0.0) +
             (nd.xrow[k] > 0 ? nd.g[k, s] * x[nd.xrow[k]] : 0.0) for k in 1:m]
        @constraint(omp, nd.A * y .<= r)
        obj = exact_h ? QuadExpr() : AffExpr(0.0)
        add_to_expression!(obj, -nd.thetaU, dot(nd.c, y))
        for k in 1:m
            add_to_expression!(obj, nd.u[k, s], π[k])
        end
        for k in xr                                   # w = x_i π_k (exact)
            U = nd.piLU[k]; xi = x[nd.xrow[k]]
            w = @variable(omp, lower_bound=0, upper_bound=U)
            @constraint(omp, π[k] <= U)
            @constraint(omp, w <= U * xi); @constraint(omp, w <= π[k]); @constraint(omp, w >= π[k] - U * (1 - xi))
            add_to_expression!(obj, nd.g[k, s], w)
        end
        for k in hr                                   # w ≈ h_j π_k (McCormick 완화)
            U = nd.thetaU * nd.varpiU[k]; j = nd.hrow[k]; H = nd.hU[j]
            if exact_h                                # 진단용: h·π 를 그대로 (비볼록, Gurobi NonConvex)
                @constraint(omp, π[k] <= U)
                add_to_expression!(obj, nd.hcoef[k], h[j] * π[k])
                continue
            end
            w = @variable(omp)
            @constraint(omp, π[k] <= U)
            @constraint(omp, w >= 0); @constraint(omp, w >= H * π[k] + U * h[j] - H * U)
            @constraint(omp, w <= H * π[k]); @constraint(omp, w <= U * h[j])
            add_to_expression!(obj, nd.hcoef[k], w)
        end
        add_to_expression!(rhs, nd.q_hat[s], obj)
    end
    return @constraint(omp, t0 >= rhs)
end

# 같은 x̄ 에서 oracle 을 다시 부르면 (상한이 시간 제한 안에 안 닫혀 LB < UB 가 남은 경우) 시간 제한을 4배 (최대 boost_time).
# 이미 boost_time 으로 풀었던 x̄ 가 또 나오면 더 할 수 있는 게 없으므로 :Stalled.
function _oracle_limit!(tl::Dict, x̄, oracle_time, boost_time)
    haskey(tl, x̄) || return (tl[x̄] = oracle_time; (oracle_time, false))
    tl[x̄] >= boost_time && return (tl[x̄], true)
    tl[x̄] = min(4 * tl[x̄], boost_time)
    return (tl[x̄], false)
end

"""
전역 Ω 풀이 (oracle). 반환 (Fval = incumbent 값, bound = 상한, src = 해가 들어 있는 모델).
  :gurobi    — Ω NonConvex (O)
  :alpha_bnb — nz_alpha_bnb (nz_alpha_bnb.jl include, julia -t (nworkers+2),1 필요), incumbent α 를 고정한 LP (Fx) 를
               다시 풀어 그 해로 cut·belief 를 만든다.  target: 목표값 조기 종료 (LB ≥ target 또는 UB ≤ target).
"""
function _nz_oracle!(O, Fx, nd, x̄; oracle, time_limit, gap, nworkers, target=nothing, bnb_kw=NamedTuple())
    if oracle == :gurobi
        res = nz_solve!(O, nd, x̄; time_limit=time_limit, gap=gap)
        return res[:Fval], res[:bound], O
    end
    r = Main.nz_alpha_bnb(nd, x̄; nworkers=nworkers, time_limit=time_limit, rel_gap=gap, verbose=false, target=target,
                          bnb_kw...)
    nz_set_objective!(Fx, nd, x̄)
    z = Main.nz_eval_alpha!(Fx, nd, r[:α])
    return z, max(r[:UB], z), Fx
end
_rel(a, b) = abs(b - a) / max(abs(b), 1.0)

"""MW core point: conv(X) 의 상대적 내부. 출점 가능 위치에 min(γ/n, 0.5) (zero-sum 의 γ/n 과 같고, γ ≥ n 이면 0.5)."""
function nz_core_point(nd::NZData)
    n = count(nd.x_allowed)
    return [nd.x_allowed[i] ? min(nd.gamma / n, 0.5) : 0.0 for i in 1:nd.nx]
end

"""
Magnanti–Wong 강화 (zero-sum `_mw_menu_cut!` 과 같은 Phase 2).
L 은 x̄ 에서 막 풀린 LP (belief LP 또는 α 고정 LP), zstar 는 그 최적값.
x̄ 에서의 값을 유지하는 (F(x̄) ≥ zstar − tol) 해 중 core point 에서 F 가 최대인 해로 cut → x̄ 에서 값은 같고 다른 x 에서 더 강함.
실패 (OPTIMAL 아님) 시 nothing → 호출 측이 기본 cut 사용.
"""
function _nz_mw_cut!(L::NZOmega, nd::NZData, x̄, zstar, x_core)
    con = @constraint(L.model, objective_function(L.model) >= zstar - 1e-6 * max(1.0, abs(zstar)))
    nz_set_objective!(L, nd, x_core)
    optimize!(L.model)
    cut = termination_status(L.model) == MOI.OPTIMAL ? nz_cut_from_solution(L, nd, x_core) : nothing
    delete(L.model, con)
    nz_set_objective!(L, nd, x̄)
    return cut
end

"LP 해 (값 z) 로 cut. mw 면 MW 를 시도하고 실패 시 기본 cut. 반환 (cut, mw 성공 여부)"
function _nz_lp_cut!(L::NZOmega, nd::NZData, x̄, z; mw::Bool, x_core)
    if mw
        c = _nz_mw_cut!(L, nd, x̄, z, x_core)
        c === nothing || return c, true
        nz_set_objective!(L, nd, x̄); optimize!(L.model)        # 기본 cut 을 위해 x̄ 해 복원
    end
    return nz_cut_from_solution(L, nd, x̄), false
end


function nz_standard_benders(nd::NZData; optimizer, max_iter=200, tol=1e-4, oracle_time=600.0, boost_time=3600.0,
                             oracle::Symbol=:alpha_bnb, oracle_gap=1e-5, nworkers=12, bnb_kw=NamedTuple(),
                             mw::Bool=true, vi::Bool=false, verbose=true)
    oracle in (:gurobi, :alpha_bnb) || error("oracle = :gurobi | :alpha_bnb")
    x_core = nz_core_point(nd); n_mw = 0
    omp, x, t0 = _nz_omp(nd; optimizer=optimizer)
    vi && nz_add_dual_vi!(omp, x, t0, nd)
    O = oracle == :gurobi ? build_nz_omega(nd; optimizer=optimizer) : nothing
    Fx = oracle == :alpha_bnb ? Main.nz_build_fixed_alpha(nd, zeros(nd.nx); optimizer=optimizer) : nothing
    LB, UB, best_x = -Inf, Inf, zeros(nd.nx)
    hist = NamedTuple[]
    tl = Dict{Vector{Float64},Float64}(); ncalls = 0
    wall = time(); iter = 0; status = :MaxIter
    while iter < max_iter
        iter += 1
        optimize!(omp)
        x̄ = Float64.(value.(x) .> 0.5)
        LB = objective_value(omp)
        _rel(LB, UB) <= tol && (status = :Optimal; break)
        lim, stalled = _oracle_limit!(tl, x̄, oracle_time, boost_time)
        stalled && (status = :Stalled; break)
        ncalls += 1
        t_o = @elapsed ((zinc, zbd, src) = _nz_oracle!(O, Fx, nd, x̄; oracle=oracle, time_limit=lim, gap=oracle_gap,
                                                       nworkers=nworkers, bnb_kw=bnb_kw))
        if oracle == :alpha_bnb                     # src = α 고정 LP → MW 가능
            cut, ok = _nz_lp_cut!(src, nd, x̄, zinc; mw=mw, x_core=x_core); n_mw += ok
        else
            cut = nz_cut_from_solution(src, nd, x̄)
        end
        _add_cut!(omp, x, t0, cut)
        ub_here = dot(nd.qx, x̄) + zbd
        ub_here < UB && (UB = ub_here; best_x = copy(x̄))
        push!(hist, (iter=iter, LB=LB, UB=UB, x=findall(x̄ .> 0.5), F=zinc, t=t_o))
        verbose && @printf("  [std] it %3d LB=%11.4f UB=%11.4f x=%-10s Ω=%11.4f bd=%11.4f (%.1fs)\n",
                           iter, LB, UB, string(findall(x̄ .> 0.5)), zinc, zbd, t_o)
        verbose && flush(stdout)
    end
    return Dict(:status => status, :LB => LB, :UB => UB, :x => best_x, :iters => iter,
                :oracle_calls => ncalls, :mw_cuts => n_mw, :wall => time() - wall, :hist => hist)
end


function nz_belief_menu_benders(nd::NZData; optimizer, lp_optimizer=optimizer, cert::Symbol=:recompute,
                                cert_rows::Symbol=:hrows, max_iter=500, tol=1e-4, oracle_time=600.0, boost_time=3600.0,
                                oracle::Symbol=:alpha_bnb, oracle_gap=1e-5, nworkers=12, bnb_kw=NamedTuple(),
                                target_stop::Bool=(oracle == :alpha_bnb), mw::Bool=true, vi::Bool=false,
                                verbose=true)
    cert in (:recompute, :solution) || error("cert = :recompute | :solution")
    oracle in (:gurobi, :alpha_bnb) || error("oracle = :gurobi | :alpha_bnb")
    x_core = nz_core_point(nd); n_mw = 0
    target_stop && oracle != :alpha_bnb && error("target_stop 은 oracle=:alpha_bnb 에서만")
    omp, x, t0 = _nz_omp(nd; optimizer=optimizer)
    vi && nz_add_dual_vi!(omp, x, t0, nd)
    O = oracle == :gurobi ? build_nz_omega(nd; optimizer=optimizer) : nothing
    Fx = oracle == :alpha_bnb ? Main.nz_build_fixed_alpha(nd, zeros(nd.nx); optimizer=optimizer) : nothing
    B = build_nz_omega(nd; optimizer=lp_optimizer, belief_lp=true, cert_rows=cert_rows)
    menu = Tuple{Dict{Symbol,Any},Matrix{Float64}}[]
    LB, UB, best_x = -Inf, Inf, zeros(nd.nx)
    oracle_calls = 0; menu_cuts = 0
    exact_log = NamedTuple[]
    hist = NamedTuple[]
    tl = Dict{Vector{Float64},Float64}()
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
                cut, ok = _nz_lp_cut!(B, nd, x̄, res[:Fval]; mw=mw, x_core=x_core); n_mw += ok
                _add_cut!(omp, x, t0, cut); nviol += 1; menu_cuts += 1
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
        lim, stalled = _oracle_limit!(tl, x̄, oracle_time, boost_time)
        stalled && (status = :Stalled; break)
        oracle_calls += 1
        # 목표값: menu 가 x̄ 에서 수렴했으므로 oracle 은 "t₀ 보다 의미 있게 큰 값이 있는가" 만 판정하면 됨
        target = target_stop ? t0v + 0.99 * tol * max(1.0, abs(t0v)) : nothing
        t_o = @elapsed ((zinc, zbd, src) = _nz_oracle!(O, Fx, nd, x̄; oracle=oracle, time_limit=lim, gap=oracle_gap,
                                                       nworkers=nworkers, target=target, bnb_kw=bnb_kw))
        res = Dict(:Fval => zinc, :bound => zbd)
        bel_raw = nz_read_belief(src, nd)                 # MW 재풀이 전에 읽음 (재풀이가 해를 바꿈)
        if oracle == :alpha_bnb
            cut, ok = _nz_lp_cut!(src, nd, x̄, zinc; mw=mw, x_core=x_core); n_mw += ok
        else
            cut = nz_cut_from_solution(src, nd, x̄)
        end
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
                :oracle_calls => oracle_calls, :menu_cuts => menu_cuts, :mw_cuts => n_mw, :menu => menu, :menu_size => length(menu),
                :exact_log => exact_log, :wall => time() - wall, :hist => hist)
end
