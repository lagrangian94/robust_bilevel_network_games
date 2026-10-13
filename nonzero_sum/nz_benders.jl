"""
nz_benders.jl — 비제로섬 Ω 위의 Benders 두 가지.

기본 구성 (zero-sum 과 동일 방침): oracle = α-B&B, cut = Magnanti–Wong 강화 (mw = true).
  전역 oracle 은 두 인자로 정한다 (실행을 시작할 때 "[설정]" 줄로 출력, 결과 Dict 의 :oracle, :bnb 에도 기록):
    oracle = :alpha_bnb (기본) | :gurobi (Ω NonConvex)
    bnb    = :dr (기본: nz_alpha_bnb_dr, 상자별 쌍대 범위 조임) | :legacy (nz_alpha_bnb, 2026-10-10 까지의 구현)   — oracle = :alpha_bnb 일 때만 의미
    oracle_resume = true (기본): bnb = :dr 일 때 같은 x̄ 의 재호출은 이전 호출이 멈춘 곳에서 이어 풂
    bnb_kw = α-B&B 에 그대로 넘길 인자 (예: (ipopt_time = 120.0,)). 인자 목록은 각 함수의 docstring. 두 구현의 인자는 서로 다르다.
  α-B&B oracle 은 julia -t (nworkers+2),1 이 필요.
  MW 는 LP (belief LP, α 고정 LP) 에서만 건다. 비볼록 Ω 에 걸면 문제가 어려워지므로 :gurobi oracle 의 cut 은 기본 cut.

  nz_standard_benders  : 매 반복 x̄ 에서 전역 Ω → cut, UB.
  nz_belief_menu_benders: 설계 A (true_dro/belief_menu_benders.jl) 를 확장 belief 로.
      menu 원소 = (belief (a,b,r,d,e), follower 의 recourse 쌍대해 σ) — 원고의 belief–subgradient pair.  고정하면 Ω 가 LP (html "확장 belief").
      menu 단계: 모든 원소의 LP 값 v_b(x̄) 가 t₀ 보다 크면 그 LP 해로 cut.
      local 단계 (local_first = true 일 때만): menu 가 수렴한 x̄ 에서 Ω 의 local 해를 보고, 위반 cut 이 있으면 cut + belief 를 menu 에.
      oracle 단계 (menu 가 x̄ 에서 수렴): 전역 Ω(x̄) → UB, cut, 새 belief + recourse 쌍대해를 menu 에 추가.
      recourse 쌍대해: cert = :recompute (html (b): (x̄, α̂) 에서 follower LP simplex dual, 기본)
                   | :solution  (html (a): ϖ̂ˢ / r̂_s)
      검증 기록: 새 원소를 넣을 때 그 LP 를 x̄ 에서 풀어 Ω(x̄) 와 같은지 (exactness) :exact_gap 에 남김.
OMP: min q_xᵀx + t₀, 1ᵀx ≤ γ, t₀ ≥ intercept + slopeᵀx.
"""

using JuMP, Printf, LinearAlgebra
import MathOptInterface as MOI

# 기본 oracle 이 α-B&B 이므로 미리 불러 둔다 (nz_omega.jl 이 먼저 include 되어 있어야 함). 실행은 julia -t (nworkers+2),1.
isdefined(Main, :nz_alpha_bnb) || include(joinpath(@__DIR__, "nz_alpha_bnb.jl"))
isdefined(Main, :nz_alpha_bnb_dr) || include(joinpath(@__DIR__, "nz_alpha_bnb_dr.jl"))

"α-B&B 구현 선택: :dr = nz_alpha_bnb_dr (기본), :legacy = nz_alpha_bnb"
_nz_bnb_solver(bnb::Symbol) = bnb == :dr ? Main.nz_alpha_bnb_dr : bnb == :legacy ? Main.nz_alpha_bnb : error("bnb = :dr | :legacy")
"이어 풀기 상태를 보관할 x̄ 의 수 (메모리 상한용)"
const _NZ_CACHE_MAX = 6
"실행 시작 시 한 줄: 어떤 전역 oracle 로 도는지"
_nz_oracle_banner(oracle, bnb, nworkers, gap, bnb_kw, resume=false) = oracle == :gurobi ?
    @sprintf("  [설정] 전역 oracle = Gurobi Ω (NonConvex), gap %.0e", gap) :
    @sprintf("  [설정] 전역 oracle = α-B&B (%s), worker %d, gap %.0e, 이어 풀기 %s, bnb_kw = %s", string(nameof(_nz_bnb_solver(bnb))),
             nworkers, gap, resume ? "켬" : "끔", string(bnb_kw))

function _nz_omp(nd::NZData; optimizer, t_lb=-1e7)
    omp = Model(optimizer); set_silent(omp)
    @variable(omp, x[1:nd.nx], Bin)
    @variable(omp, t0 >= t_lb)
    @constraint(omp, sum(x) <= nd.gamma)
    # 추가 카디널리티 (원고 실험의 source/sink cut: 출발·도착 arc 를 모두 막지 못하게): meta[:omp_card] = [(idx, rhs), ...]
    for (idx, rhs) in get(nd.meta, :omp_card, Tuple{Vector{Int},Int}[])
        @constraint(omp, sum(x[i] for i in idx) <= rhs)
    end
    for i in 1:nd.nx
        nd.x_allowed[i] || fix(x[i], 0.0; force=true)
    end
    @objective(omp, Min, dot(nd.qx, x) + t0)
    return omp, x, t0
end

_add_cut!(omp, x, t0, cut) = @constraint(omp, t0 >= cut[:intercept] + dot(cut[:slope], x))

"cut 기록 (검증용): kind = :restricted (α 고정 LP) | :omega (Gurobi Ω 해) | :belief (menu belief LP), mw = MW 강화 여부"
_log_cut!(cuts, kind, iter, x̄, cut, mw) =
    push!(cuts, (kind=kind, iter=iter, x=copy(x̄), intercept=cut[:intercept], slope=copy(cut[:slope]), mw=mw))

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
"oracle 호출 시간을 전체 시간 제한의 남은 시간으로 자름 (최소 1 s). 전체 제한을 넘겨 도는 일을 막는다."
_remaining(wall, time_limit, lim) = isfinite(time_limit) ? min(lim, max(time_limit - (time() - wall), 1.0)) : lim

function _oracle_limit!(tl::Dict, x̄, oracle_time, boost_time)
    haskey(tl, x̄) || return (tl[x̄] = oracle_time; (oracle_time, false))
    tl[x̄] >= boost_time && return (tl[x̄], true)
    tl[x̄] = min(4 * tl[x̄], boost_time)
    return (tl[x̄], false)
end

"""
전역 Ω 풀이 (oracle). 반환 (Fval = incumbent 값, bound = 상한, src = 해가 들어 있는 모델).
  :gurobi    — Ω NonConvex (O)
  :alpha_bnb — α-B&B (bnb = :dr → nz_alpha_bnb_dr, :legacy → nz_alpha_bnb. julia -t (nworkers+2),1 필요), incumbent α 를 고정한 LP (Fx) 를
               다시 풀어 그 해로 cut·belief 를 만든다.  target: 목표값 조기 종료 (LB ≥ target 또는 UB ≤ target).
"""
function _nz_oracle!(O, Fx, nd, x̄; oracle, time_limit, gap, nworkers, target=nothing, bnb::Symbol=:dr, bnb_kw=NamedTuple(),
                     cache=nothing)
    if oracle == :gurobi
        res = nz_solve!(O, nd, x̄; time_limit=time_limit, gap=gap)
        return res[:Fval], res[:bound], O
    end
    # 구현은 bnb 로 고른다 (:dr 기본 | :legacy). 반환 Dict 의 :α, :UB 만 쓰므로 인터페이스는 같다.
    # cache (x̄ → (순번, α-B&B 상태)): 같은 x̄ 를 다시 부르면 이전 호출이 멈춘 곳에서 이어 푼다 (nz_alpha_bnb_dr 의 resume, :dr 전용).
    #   Ω(x̄) 는 x̄ 만으로 정해지므로 이전 호출의 열린 노드·상한·incumbent 가 그대로 유효하다. 최근 _NZ_CACHE_MAX 개의 x̄ 만 보관.
    if bnb == :dr && cache !== nothing
        prev = get(cache, x̄, nothing)
        r = Main.nz_alpha_bnb_dr(nd, x̄; nworkers=nworkers, time_limit=time_limit, rel_gap=gap, verbose=false, target=target,
                                 resume=(prev === nothing ? nothing : prev[2]), bnb_kw...)
        stamp = isempty(cache) ? 1 : maximum(v[1] for v in values(cache)) + 1
        cache[copy(x̄)] = (stamp, r[:state])
        while length(cache) > _NZ_CACHE_MAX
            delete!(cache, argmin(k -> cache[k][1], collect(keys(cache))))
        end
    else
        r = _nz_bnb_solver(bnb)(nd, x̄; nworkers=nworkers, time_limit=time_limit, rel_gap=gap, verbose=false, target=target, bnb_kw...)
    end
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
    # x̄ 에서의 값 손실 허용치는 menu 의 위반 판정 여유 (1e-6 상대) 보다 훨씬 작아야 한다. 같은 크기면 t₀ 가 그 사이에 끼어
    # 같은 cut 이 무한히 반복된다 (Abilene δ=0.1 에서 발생). 채택 여부는 _nz_lp_cut! 가 x̄ 값으로 다시 확인.
    con = @constraint(L.model, objective_function(L.model) >= zstar - 1e-9 * max(1.0, abs(zstar)))
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
        # x̄ 에서 값을 (거의) 유지하는 경우에만 MW cut 채택
        c !== nothing && c[:intercept] + dot(c[:slope], x̄) >= z - 1e-7 * max(1.0, abs(z)) && return c, true
        nz_set_objective!(L, nd, x̄); optimize!(L.model)        # 기본 cut 을 위해 x̄ 해 복원
    end
    return nz_cut_from_solution(L, nd, x̄), false
end


"local Ω 풀이: 해가 있으면 (:status, :Fval), 시간 제한에 해가 없으면 nothing, 그 밖의 상태는 오류"
function _nz_local_solve!(Oloc::NZOmega, nd::NZData, x̄; time_limit)
    nz_set_objective!(Oloc, nd, x̄)
    set_time_limit_sec(Oloc.model, time_limit)
    optimize!(Oloc.model)
    st = termination_status(Oloc.model)
    # 한도 도달 (시간·반복) 은 정상 종료: 해가 있으면 쓰고 없으면 전역으로. 그 밖의 상태는 오류
    lim = st == MOI.TIME_LIMIT || st == MOI.ITERATION_LIMIT
    lim && !has_values(Oloc.model) && return nothing
    ok = st == MOI.LOCALLY_SOLVED || st == MOI.OPTIMAL || (lim && has_values(Oloc.model))
    ok || error("nz Ω (local): $st")
    return Dict(:status => st, :Fval => objective_value(Oloc.model))
end

"""
Ω 해 (Gurobi 전역 또는 local) 로 cut. mw 면 그 해의 α (h^r) 를 고정한 LP 를 x̄ 에서 다시 풀고 (값 ≥ Ω 해의 값) 그 LP 에서
MW 강화 (α-B&B 경로와 같은 _nz_lp_cut!). α 고정 LP 가 실패하면 Ω 해 그대로의 cut. 반환 (cut, MW 적용 여부).
"""
function _nz_alpha_cut(O::NZOmega, Fx, nd::NZData, x̄; mw::Bool, x_core)
    (mw && Fx !== nothing) || return nz_cut_from_solution(O, nd, x̄), false
    α̂ = [max(value(O.v[:α][i]), 0.0) for i in 1:nz_nh(nd)]
    nz_set_objective!(Fx, nd, x̄)
    z = Main.nz_eval_alpha!(Fx, nd, α̂)
    isfinite(z) || return nz_cut_from_solution(O, nd, x̄), false
    return _nz_lp_cut!(Fx, nd, x̄, z; mw=true, x_core=x_core)
end


function nz_standard_benders(nd::NZData; optimizer, max_iter=200, tol=1e-4, oracle_time=600.0, boost_time=3600.0,
                             oracle::Symbol=:alpha_bnb, bnb::Symbol=:dr, oracle_gap=1e-5, nworkers=12, bnb_kw=NamedTuple(),
                             mw::Bool=true, vi::Bool=false, verbose=true, time_limit=Inf,
                             local_first::Bool=false, local_time=60.0, oracle_remaining::Bool=false, oracle_resume::Bool=true)
    # oracle_resume: 같은 x̄ 에서 α-B&B (bnb = :dr) 를 다시 부르면 이전 호출이 멈춘 곳에서 이어 풂 (_nz_oracle! 의 cache)
    bnb_cache = (oracle == :alpha_bnb && bnb == :dr && oracle_resume) ? Dict{Vector{Float64},Any}() : nothing
    oracle in (:gurobi, :alpha_bnb) || error("oracle = :gurobi | :alpha_bnb")
    oracle == :alpha_bnb && _nz_bnb_solver(bnb)
    verbose && (println(_nz_oracle_banner(oracle, bnb, nworkers, oracle_gap, bnb_kw, bnb_cache !== nothing)); flush(stdout))
    # oracle_remaining: 전역 호출에 처음부터 남은 시간 전체를 줌 (SB-G / SB-A: 전역은 local cut 이 막힌 뒤에만 불리므로
    # 600 → 2,400 단계 증가 대신). time_limit 이 유한해야 함.
    oracle_remaining && !isfinite(time_limit) && error("oracle_remaining 은 유한한 time_limit 이 필요")
    x_core = nz_core_point(nd); n_mw = 0
    omp, x, t0 = _nz_omp(nd; optimizer=optimizer)
    vi && nz_add_dual_vi!(omp, x, t0, nd)
    O = oracle == :gurobi ? build_nz_omega(nd; optimizer=optimizer) : nothing
    # 원고 SB-G / SB-A: Ω 를 local 해법 (Gurobi OptimalityTarget = 1, barrier 기반 local) 으로 먼저 풀고,
    # 그 해의 cut 이 x̄ 에서 위반되면 그것만 추가. 위반되지 않을 때만 전역 oracle. Ω 의 모든 실행가능해는 유효 cut 을
    # 주므로 (Theorem structural-minsup) local 해의 cut 도 유효하지만, local 값은 V*(x̄) 의 하한일 뿐이라 UB 에는 쓰지 않음.
    Oloc = nothing
    if local_first
        Oloc = build_nz_omega(nd; optimizer=optimizer)
        set_optimizer_attribute(Oloc.model, "OptimalityTarget", 1)
    end
    n_local = 0
    # α (h^r) 고정 LP: α-B&B 의 cut 생성, 그리고 MW 를 켜면 Gurobi 전역 해·local 해의 α 를 고정해 같은 MW 를 적용
    Fx = (oracle == :alpha_bnb || mw) ? Main.nz_build_fixed_alpha(nd, zeros(nd.nx); optimizer=optimizer) : nothing
    LB, UB, best_x = -Inf, Inf, zeros(nd.nx)
    hist = NamedTuple[]
    tl = Dict{Vector{Float64},Float64}(); ncalls = 0
    cuts = NamedTuple[]
    wall = time(); iter = 0; status = :MaxIter
    while iter < max_iter
        iter += 1
        optimize!(omp)
        x̄ = Float64.(value.(x) .> 0.5)
        LB = objective_value(omp); w_LB = time() - wall     # 이 LB 가 정해진 시각 (hist 의 w)
        _rel(LB, UB) <= tol && (status = :Optimal; break)
        time() - wall > time_limit && (status = :TimeLimit; break)
        if local_first
            # local 해가 시간 안에 하나도 없으면 (TIME_LIMIT, 해 없음) 전역으로. 그 밖의 비정상 상태는 오류 (Ω 는 항상 실행가능)
            t_l = @elapsed (rl = _nz_local_solve!(Oloc, nd, x̄; time_limit=_remaining(wall, time_limit, local_time)))
            if rl !== nothing
                cut, ok = _nz_alpha_cut(Oloc, Fx, nd, x̄; mw=mw, x_core=x_core)
                cval = cut[:intercept] + dot(cut[:slope], x̄)
                if cval > value(t0) + tol * max(1.0, abs(value(t0)))
                    _add_cut!(omp, x, t0, cut); _log_cut!(cuts, :local, iter, x̄, cut, ok); n_local += 1; n_mw += ok
                    push!(hist, (iter=iter, LB=LB, UB=UB, x=findall(x̄ .> 0.5), F=rl[:Fval], t=t_l, w=w_LB))
                    verbose && @printf("  [loc] it %3d LB=%11.4f UB=%11.4f x=%-10s Ω_loc=%11.4f (%.1fs)\n",
                                       iter, LB, UB, string(findall(x̄ .> 0.5)), rl[:Fval], t_l)
                    verbose && flush(stdout)
                    continue
                end
            end
        end
        if oracle_remaining
            lim = _remaining(wall, time_limit, Inf)
        else
            lim, stalled = _oracle_limit!(tl, x̄, oracle_time, boost_time)
            stalled && (status = :Stalled; break)
            lim = _remaining(wall, time_limit, lim)
        end
        ncalls += 1
        t_o = @elapsed ((zinc, zbd, src) = _nz_oracle!(O, Fx, nd, x̄; oracle=oracle, time_limit=lim, gap=oracle_gap,
                                                       nworkers=nworkers, bnb=bnb, bnb_kw=bnb_kw, cache=bnb_cache))
        t_o < 0.9 * lim && delete!(tl, x̄)           # 시간 제한에 걸린 호출만 stall 판정에 셈
        if oracle == :alpha_bnb                     # src = α 고정 LP → MW 가능
            cut, ok = _nz_lp_cut!(src, nd, x̄, zinc; mw=mw, x_core=x_core); n_mw += ok
            _log_cut!(cuts, :restricted, iter, x̄, cut, ok)
        else
            cut, ok = _nz_alpha_cut(src, Fx, nd, x̄; mw=mw, x_core=x_core); n_mw += ok
            _log_cut!(cuts, :omega, iter, x̄, cut, ok)
        end
        _add_cut!(omp, x, t0, cut)
        ub_here = dot(nd.qx, x̄) + zbd
        ub_here < UB && (UB = ub_here; best_x = copy(x̄))
        push!(hist, (iter=iter, LB=LB, UB=UB, x=findall(x̄ .> 0.5), F=zinc, t=t_o, w=w_LB))
        verbose && @printf("  [std] it %3d LB=%11.4f UB=%11.4f x=%-10s Ω=%11.4f bd=%11.4f (%.1fs)\n",
                           iter, LB, UB, string(findall(x̄ .> 0.5)), zinc, zbd, t_o)
        verbose && flush(stdout)
    end
    return Dict(:status => status, :LB => LB, :UB => UB, :x => best_x, :iters => iter,
                :oracle_calls => ncalls, :mw_cuts => n_mw, :local_cuts => n_local, :wall => time() - wall,
                :hist => hist, :cuts => cuts, :oracle => oracle, :bnb => (oracle == :alpha_bnb ? bnb : nothing))
end


function nz_belief_menu_benders(nd::NZData; optimizer, lp_optimizer=optimizer, cert::Symbol=:recompute,
                                cert_rows::Symbol=:hrows, max_iter=500, tol=1e-4, oracle_time=600.0, boost_time=3600.0,
                                oracle::Symbol=:alpha_bnb, bnb::Symbol=:dr, oracle_gap=1e-5, nworkers=12, bnb_kw=NamedTuple(),
                                target_stop::Bool=(oracle == :alpha_bnb), mw::Bool=true, vi::Bool=false,
                                verbose=true, time_limit=Inf, local_first::Bool=false, local_time=60.0,
                                oracle_resume::Bool=true,
                                oracle_remaining::Bool=(target_stop && isfinite(time_limit)))
    # oracle_remaining (기본: 목표값 조기 종료를 쓰고 전체 시간 제한이 유한하면 켬): 전역 oracle 에 남은 시간 전체를 준다.
    #   목표값 조기 종료가 있으면 oracle 이 오래 도는 것은 "위반 값이 없어 상한을 증명해야 하는" 경우뿐이고, 그때 호출을 끊어도
    #   새 cut 이 없어 OMP 가 같은 x̄ 를 다시 내놓는다 → 600 s / 2,400 s 로 쪼개는 것은 재시작 비용만 든다
    #   (seed 2, S=30: 4 번으로 쪼개진 Algorithm 2 가 gap 1.9%, 한 번에 돌린 SB-A 가 0.75%. seed 3, S=20: 530 s 를 남기고 :Stalled).
    #   켜면 oracle_time / boost_time 단계와 :Stalled 종료를 쓰지 않고 전체 time_limit 으로만 끝난다.
    #   Gurobi oracle (target_stop 없음) 은 x̄ 마다 끝까지 증명하려 하므로 기본 끔 (단계별 제한 유지).
    oracle_remaining && !isfinite(time_limit) && error("oracle_remaining 은 유한한 time_limit 이 필요")
    # oracle_resume: 같은 x̄ 에서 α-B&B (bnb = :dr) 를 다시 부르면 이전 호출이 멈춘 곳에서 이어 풂 (_nz_oracle! 의 cache).
    #   목표값 조기 종료 (target_stop) 로 끝난 호출 뒤 같은 x̄ 가 다시 오는 것이 흔하므로 (cut 을 넣어도 x̄ 가 그대로 최적 후보)
    #   이어 풀지 않으면 같은 B&B 를 처음부터 반복한다.
    bnb_cache = (oracle == :alpha_bnb && bnb == :dr && oracle_resume) ? Dict{Vector{Float64},Any}() : nothing
    # local_first: menu 가 x̄ 에서 수렴한 뒤 전역 oracle 을 부르기 전에 Ω 의 local 해 (Gurobi OptimalityTarget = 1, SB-G / SB-A 와 같음) 를
    #   먼저 본다. 그 해의 cut (α 고정 LP, MW) 이 x̄ 에서 tol 이상 위반되면 cut 을 넣고, 그 해의 belief + recourse 쌍대해도 menu 에 넣는다
    #   (local 해도 Ω 의 실행가능해라 cut 과 menu 원소 모두 유효. 값은 하한일 뿐이라 UB 에는 쓰지 않음). 위반이 없을 때만 전역 oracle.
    cert in (:recompute, :solution) || error("cert = :recompute | :solution")
    oracle in (:gurobi, :alpha_bnb) || error("oracle = :gurobi | :alpha_bnb")
    oracle == :alpha_bnb && _nz_bnb_solver(bnb)
    verbose && (println(_nz_oracle_banner(oracle, bnb, nworkers, oracle_gap, bnb_kw, bnb_cache !== nothing));
                println("  [설정] oracle 시간: ", oracle_remaining ? "남은 시간 전체 (끊지 않음)" : "$(oracle_time) s, 재방문 시 최대 $(boost_time) s",
                        ", 목표값 조기 종료 ", target_stop ? "켬" : "끔", ", local cut ", local_first ? "켬" : "끔"); flush(stdout))
    x_core = nz_core_point(nd); n_mw = 0
    target_stop && oracle != :alpha_bnb && error("target_stop 은 oracle=:alpha_bnb 에서만")
    omp, x, t0 = _nz_omp(nd; optimizer=optimizer)
    vi && nz_add_dual_vi!(omp, x, t0, nd)
    O = oracle == :gurobi ? build_nz_omega(nd; optimizer=optimizer) : nothing
    Fx = (oracle == :alpha_bnb || (local_first && mw)) ? Main.nz_build_fixed_alpha(nd, zeros(nd.nx); optimizer=optimizer) : nothing
    Oloc = nothing; n_local = 0
    if local_first
        Oloc = build_nz_omega(nd; optimizer=optimizer)
        set_optimizer_attribute(Oloc.model, "OptimalityTarget", 1)
    end
    B = build_nz_omega(nd; optimizer=lp_optimizer, belief_lp=true, cert_rows=cert_rows)
    menu = Tuple{Dict{Symbol,Any},Matrix{Float64}}[]
    LB, UB, best_x = -Inf, Inf, zeros(nd.nx)
    oracle_calls = 0; menu_cuts = 0
    exact_log = NamedTuple[]
    hist = NamedTuple[]
    tl = Dict{Vector{Float64},Float64}()
    cuts = NamedTuple[]
    wall = time(); iter = 0; status = :MaxIter
    while iter < max_iter
        iter += 1
        optimize!(omp)
        x̄ = Float64.(value.(x) .> 0.5)
        LB = objective_value(omp); w_LB = time() - wall     # 이 LB 가 정해진 시각 (hist 의 w)
        t0v = value(t0)
        _rel(LB, UB) <= tol && (status = :Optimal; break)
        time() - wall > time_limit && (status = :TimeLimit; break)

        # ---- menu 단계 ----
        V_menu = -Inf; nviol = 0
        for (bel, σ) in menu
            nz_set_belief!(B, nd, bel, σ)
            res = nz_solve!(B, nd, x̄)
            V_menu = max(V_menu, res[:Fval])
            if res[:Fval] > t0v + 1e-6 * max(1.0, abs(t0v))
                cut, ok = _nz_lp_cut!(B, nd, x̄, res[:Fval]; mw=mw, x_core=x_core); n_mw += ok
                _log_cut!(cuts, :belief, iter, x̄, cut, ok)
                _add_cut!(omp, x, t0, cut); nviol += 1; menu_cuts += 1
            end
        end
        if nviol > 0
            push!(hist, (iter=iter, LB=LB, UB=UB, x=findall(x̄ .> 0.5), kind=:menu, n=nviol, w=w_LB))
            verbose && @printf("  [menu] it %3d LB=%11.4f UB=%11.4f x=%-10s menu cut %d/%d (V_menu=%.4f)\n",
                               iter, LB, UB, string(findall(x̄ .> 0.5)), nviol, length(menu), V_menu)
            verbose && flush(stdout)
            continue
        end

        # ---- local 단계 (local_first) ----
        if local_first
            t_l = @elapsed (rl = _nz_local_solve!(Oloc, nd, x̄; time_limit=_remaining(wall, time_limit, local_time)))
            if rl !== nothing
                bel_loc = nz_read_belief(Oloc, nd)          # cut 계산 (Fx 를 푼다) 과 무관하게 Oloc 의 해에서 읽음
                cut, ok = _nz_alpha_cut(Oloc, Fx, nd, x̄; mw=mw, x_core=x_core)
                cval = cut[:intercept] + dot(cut[:slope], x̄)
                if cval > t0v + tol * max(1.0, abs(t0v))
                    _add_cut!(omp, x, t0, cut); _log_cut!(cuts, :local, iter, x̄, cut, ok); n_local += 1; n_mw += ok
                    bel = nz_clean_belief(bel_loc, nd)
                    σ = cert == :recompute ? nz_follower_duals(nd, x̄, bel_loc[:α]; lp_optimizer=lp_optimizer)[1] :
                        nz_repair_cert(nd, bel_loc[:ϖ] ./ reshape(max.(bel_loc[:r], 1e-12), 1, nd.S); lp_optimizer=lp_optimizer)
                    push!(menu, (bel, σ))
                    push!(hist, (iter=iter, LB=LB, UB=UB, x=findall(x̄ .> 0.5), kind=:local, n=length(menu), w=w_LB))
                    verbose && @printf("  [loc]  it %3d LB=%11.4f UB=%11.4f x=%-10s Ω_loc=%11.4f (%.1fs) menu=%d (+belief)
",
                                       iter, LB, UB, string(findall(x̄ .> 0.5)), rl[:Fval], t_l, length(menu))
                    verbose && flush(stdout)
                    continue
                end
            end
        end

        # ---- oracle 단계 ----
        if oracle_remaining
            lim = _remaining(wall, time_limit, Inf)
        else
            lim, stalled = _oracle_limit!(tl, x̄, oracle_time, boost_time)
            stalled && (status = :Stalled; break)
            lim = _remaining(wall, time_limit, lim)
        end
        oracle_calls += 1
        # 목표값: menu 가 x̄ 에서 수렴했으므로 oracle 은 "t₀ 보다 의미 있게 큰 값이 있는가" 만 판정하면 됨
        # 여유는 수렴 판정 (_rel(LB, UB) = |UB − LB| / max(|UB|, 1) ≤ tol, 전체 목적값 기준) 과 같은 기준이어야 한다.
        #   이전에는 |t₀| 기준 (t₀ + 0.99·tol·|t₀|) 이라 q_xᵀx̄ ≠ 0 이면 판정 기준과 어긋났다: SGB128 S=20 에서 |t₀| ≈ 991, |UB| ≈ 682
        #   (출점비 305) → oracle 은 상한 ≤ target 으로 멈추는데 gap 은 0.53% > tol 이고 새 cut 도 없어 같은 x̄ 를 시간 제한까지 반복.
        #   oracle 상한 zbd ≤ t₀ + δ 이면 UB = q_xᵀx̄ + zbd ≤ LB + δ, |UB| ≥ |LB| − δ 이므로 δ = 0.99·tol·|LB| / (1 + tol) 이면 δ ≤ tol·|UB|.
        target = target_stop ? t0v + 0.99 * tol * max(1.0, abs(LB)) / (1 + tol) : nothing
        t_o = @elapsed ((zinc, zbd, src) = _nz_oracle!(O, Fx, nd, x̄; oracle=oracle, time_limit=lim, gap=oracle_gap,
                                                       nworkers=nworkers, target=target, bnb=bnb, bnb_kw=bnb_kw, cache=bnb_cache))
        # 시간 제한에 걸린 호출만 stall 판정에 셈. 목표값 조기 종료 (target_stop) 로 끝난 호출은 같은 x̄ 를 다시 불러도
        # 정상이다 (t₀ 가 올라간 뒤 다시 확인하는 것). 이전에는 같은 x̄ 세 번째 호출에서 무조건 :Stalled.
        t_o < 0.9 * lim && delete!(tl, x̄)
        res = Dict(:Fval => zinc, :bound => zbd)
        bel_raw = nz_read_belief(src, nd)                 # MW 재풀이 전에 읽음 (재풀이가 해를 바꿈)
        if oracle == :alpha_bnb
            cut, ok = _nz_lp_cut!(src, nd, x̄, zinc; mw=mw, x_core=x_core); n_mw += ok
            _log_cut!(cuts, :restricted, iter, x̄, cut, ok)
        else
            cut = nz_cut_from_solution(src, nd, x̄)
            _log_cut!(cuts, :omega, iter, x̄, cut, false)
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
            # exactness: 새 (belief, recourse 쌍대해) 고정 LP 의 x̄ 값 = Ω(x̄) ?
            nz_set_belief!(B, nd, bel, σ)
            vb = nz_solve!(B, nd, x̄)[:Fval]
            push!(exact_log, (iter=iter, x=findall(x̄ .> 0.5), omega=res[:Fval], lp=vb, gap=res[:Fval] - vb))
        end
        push!(hist, (iter=iter, LB=LB, UB=UB, x=findall(x̄ .> 0.5), kind=:oracle, n=length(menu), w=w_LB))
        verbose && @printf("  [orcl] it %3d LB=%11.4f UB=%11.4f x=%-10s Ω=%11.4f (%.1fs) menu=%d%s\n",
                           iter, LB, UB, string(findall(x̄ .> 0.5)), res[:Fval], t_o, length(menu),
                           new_b ? @sprintf(" (+belief, LP@x̄=%.4f, gap %.1e)", exact_log[end].lp, exact_log[end].gap) : "")
        verbose && flush(stdout)
    end
    return Dict(:status => status, :LB => LB, :UB => UB, :x => best_x, :iters => iter,
                :oracle_calls => oracle_calls, :menu_cuts => menu_cuts, :mw_cuts => n_mw, :menu => menu, :menu_size => length(menu),
                :local_cuts => n_local,
                :exact_log => exact_log, :wall => time() - wall, :hist => hist, :cuts => cuts,
                :oracle => oracle, :bnb => (oracle == :alpha_bnb ? bnb : nothing))
end
