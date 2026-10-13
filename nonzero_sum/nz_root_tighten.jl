"""
nz_root_tighten.jl — 반응 정의역 프리솔브: 루트 노드 LP 에서 좌표별 max α_i 를 구해, follower 의 최적 반응이 될 수 없는 예약 좌표를 0 으로 고정한다.
nz_alpha_bnb.jl, nz_alpha_bnb_dr.jl 을 먼저 include. **시험 단계: nz_alpha_bnb_dr 에는 아직 연결하지 않았다** (lab_rtight_bench.jl 이 시험용으로 연결).

  hᵁ′_i = max { α_i : 루트 노드 LP (기본 구성의 조임 전부, RLT 분리까지) 의 실행가능 영역 },   hᵁ′_i ≤ 1e-6 이면 0

유효성. 루트 노드 LP 는 Ω 의 완화에 "최적값을 바꾸지 않는 제한" (루트 상자의 쌍대 범위·차이 범위, Wcap 등) 을 더한 것이다. 그 제한들은 α 를 건드리지 않고
쌍대 (ϖ, π̃) 의 선택만 줄이며, 모든 belief 의 모든 최적 반응 α 에서 그 제한을 만족하는 완성이 있다 (nz_alpha_bnb_dr.jl 의 "유효성"). 따라서 α ∈ Ψ(x̄, 𝔅) 는 모두
LP 실행가능 영역의 α 사영 안에 있고, 사영 밖의 α 를 버려도 V*(x̄) 는 그대로다. incumbent 가 필요 없다 (목적 조건을 쓰지 않는 실행가능성 기반 조임).
0 으로 고정되는 좌표는 "쌍대 범위의 상한으로도 예약의 한계 가치 Σ_s d_s σ_ks 가 예약비에 못 미치는" 예약이다 (정상성 여유 > 0 → 상보성 RLT 가 α_i = 0 을 강제).

효과 (SGB128, x̄ = {1,2}, 활성 좌표 9): 좌표 4 개가 0 으로 고정 → 분기 차원 9 → 5, 쌍대 차이 범위의 쌍 36 → 10.
  oracle 단독 (gap 0.1%): seed 3 S = 20 이 83.5 s → 28.9 s, seed 2 S = 30 이 447~522 s → 208.5 s (`logs/lab_root_tighten_*.log`, 프리솔브 7~18 s 별도).
"""

"""
    nz_response_presolve(nd, x̄; env, fixtol=1e-6) -> (hU′, 0 으로 고정한 좌표 수, 걸린 시간)

nd 는 nz_presolve_hU 를 거친 것 (닫힌 점포의 좌표는 이미 0). 루트 LP 가 실행 불가능·수치 오류면 hᵁ 를 그대로 돌려준다.
"""
function nz_response_presolve(nd::NZData, x̄; env, fixtol=1e-6)
    t0 = time()
    mk() = (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true); MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
    nh, S = nz_nh(nd), nd.S
    act = findall(nd.hU .> 0)
    P = _nz_build_sep(nd, x̄; optimizer=mk); mdl = P.O.model; αv = P.O.v[:α]
    for a in ("FeasibilityTol", "OptimalityTol"); set_optimizer_attribute(mdl, a, 1e-7); end
    set_optimizer_attribute(mdl, "NumericFocus", 0); mdl.ext[:lp_tol] = 1e-7; mdl.ext[:retry_fresh] = true
    DR = _nz_dr_attach!(P, nd, x̄; optimizer=mk)
    lo0, hi0 = _nz_dr_global(nd, DR); dl0, dh0 = _nz_dr_dglobal(nd, DR)
    l = zeros(nh); u = copy(nd.hU)
    _nz_sep_set_box!(P, l, u); _nz_sep_set_wbox!(P, P.Lw, P.Uw)
    rg = _nz_dr_ranges(DR, nd, l, u, lo0, hi0, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), fill(Inf, S))
    rg === nothing && return copy(nd.hU), 0, time() - t0
    lo, hi, wt, phib = rg
    dl, dh = _nz_dr_dranges!(DR, nd, l, u, lo, hi, dl0, dh0, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), wt)
    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; dlo=dl, dhi=dh)
    z = _nz_sep_solve!(P; maxrounds=30)
    isfinite(z) || return copy(nd.hU), 0, time() - t0
    hU2 = copy(nd.hU)
    for i in act
        @objective(mdl, Max, αv[i]); optimize!(mdl)
        termination_status(mdl) == MOI.OPTIMAL || continue            # 못 풀면 그 좌표는 그대로
        v = objective_value(mdl)
        v <= fixtol && (hU2[i] = 0.0)                                 # 상한 쪽의 작은 조임 (hᵁ 의 0.9~1.0 배) 은 쓰지 않는다: 0 고정만
    end
    return hU2, count(i -> hU2[i] == 0, act), time() - t0
end
