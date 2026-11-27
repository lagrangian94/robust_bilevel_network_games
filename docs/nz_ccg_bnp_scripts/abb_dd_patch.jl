# abb_dd_patch.jl — gen_abb_dd_patch.py 가 만든 파일 (손으로 고치지 말 것). 저장소 nz_alpha_bnb_dr 의 재정의:
#   꼭짓점 LP 범위 (range_lp) 와 쌍대 차이 범위 (dd = true; dd_mode :screen 기본, :full) 를 함께 쓸 수 있게 함. 나머지는 저장소 판 그대로.
println(stderr, "[abb_dd_patch] nz_alpha_bnb_dr 재정의: range_lp + dd 허용 (실험용)")

"패치 (시간 내역, 2026-11-27): 선별 DD 의 꼭짓점·면 LP 시간·수, MIP 시간, DD 로 다시 푼 뒤 가지친 노드. 호출마다 0 으로"
const _ABB_DD_T = (tlp=Threads.Atomic{Float64}(0.0), nlp=Threads.Atomic{Int}(0), tmip=Threads.Atomic{Float64}(0.0), nprune=Threads.Atomic{Int}(0),
                   nrechk=Threads.Atomic{Int}(0), nrechk_inf=Threads.Atomic{Int}(0), nobjstop=Threads.Atomic{Int}(0))   # 재확인 MIP 수 / 재확인에서도 실행 불가능
"패치 (screen): 두 꼭짓점의 최적 쌍대 면에서 차이가 모인 (쌍, s) 를 골라 KKT-MIP 로 차이 범위를 인증. 반환 (:none | :tight | :empty, (dlo, dhi), MIP 수)"
function _abb_dd_screen!(CL, DR, nd, x̄, l, u, lo, hi, dlo0, dhi0; mip_tl=2.0, deadline=Inf, fixtol=1e-4, sigbd::Bool=true)
    S, m, nh = nd.S, nz_m(nd), nz_nh(nd); dlo = copy(dlo0); dhi = copy(dhi0); nmip = 0; tight = false
    for s in 1:S
        time() > deadline && break
        ps = Int[]
        for (p, (a, b)) in enumerate(DR.pairs)
            (hi[a, s] - lo[a, s] > fixtol && hi[b, s] - lo[b, s] > fixtol) || continue
            wl = max(dlo[p, s], lo[a, s] - hi[b, s]); wh = min(dhi[p, s], hi[a, s] - lo[b, s])
            wh - wl > 1.0 && push!(ps, p)
        end
        isempty(ps) && continue
        C = CL[s]; ext = Dict{Int,Tuple{Float64,Float64}}()
        tlp0 = time()
        for αc in (l, u)
            rr = nz_rhs(nd, x̄, αc, s)
            for k in 1:m; nd.hrow[k] > 0 && set_normalized_coefficient(C.opt, C.σ[k], rr[k]); end
            set_normalized_rhs(C.opt, 1e20)
            @objective(C.model, Min, sum(rr[k] * C.σ[k] for k in 1:m if rr[k] != 0)); optimize!(C.model)
            termination_status(C.model) == MOI.OPTIMAL || error("_abb_dd_screen!: 꼭짓점 LP (s=$s) $(termination_status(C.model))")
            zc = objective_value(C.model); set_normalized_rhs(C.opt, zc + 1e-7 * max(1.0, abs(zc)))
            for p in ps
                a, b = DR.pairs[p]; e = C.σ[DR.Hk[a]] - C.σ[DR.Hk[b]]
                @objective(C.model, Min, e); optimize!(C.model)
                termination_status(C.model) == MOI.OPTIMAL || error("_abb_dd_screen!: 꼭짓점 차이 LP (s=$s) $(termination_status(C.model))")
                mn = objective_value(C.model)
                @objective(C.model, Max, e); optimize!(C.model)
                termination_status(C.model) == MOI.OPTIMAL || error("_abb_dd_screen!: 꼭짓점 차이 LP (s=$s) $(termination_status(C.model))")
                mx = objective_value(C.model)
                o = get(ext, p, (Inf, -Inf)); ext[p] = (min(o[1], mn), max(o[2], mx))
            end
        end
        Threads.atomic_add!(_ABB_DD_T.tlp, time() - tlp0); Threads.atomic_add!(_ABB_DD_T.nlp, 2 + 4 * length(ps))
        M = DR.mips[s]; boxset = false
        for p in ps
            a, b = DR.pairs[p]
            wl = max(dlo[p, s], lo[a, s] - hi[b, s]); wh = min(dhi[p, s], hi[a, s] - lo[b, s])
            ext[p][2] - ext[p][1] < 0.5 * (wh - wl) || continue          # 두 꼭짓점의 차이가 함의 폭의 절반 안에 모인 쌍만
            if !boxset
                for i in 1:nh; set_lower_bound(M.α[i], l[i]); set_upper_bound(M.α[i], u[i]); end
                sigbd && _abb_sigbd!(M, DR, nd, lo, hi, s)
                boxset = true
            end
            for ismax in (false, true)
                if time() > deadline
                    set_time_limit_sec(M.model, nothing); sigbd && _abb_sigbd!(M, DR, nd, nothing, nothing, s)
                    return (tight ? :tight : :none), (dlo, dhi), nmip
                end
                @objective(M.model, ismax ? MAX_SENSE : MIN_SENSE, M.σ[DR.Hk[a]] - M.σ[DR.Hk[b]])
                set_time_limit_sec(M.model, max(0.01, min(mip_tl, deadline - time())))
                nmip += 1
                tm0 = time(); okm = _nz_dr_opt!(DR, M, l, u, "dd screen s=$s 쌍 ($(DR.Hk[a]), $(DR.Hk[b]))"); Threads.atomic_add!(_ABB_DD_T.tmip, time() - tm0)
                okm || continue
                st = termination_status(M.model)
                getbd(st) = st == MOI.OPTIMAL ? objective_bound(M.model) : (st == MOI.TIME_LIMIT ? _nz_dr_bound_or_nan(M.model) : NaN)
                bd = getbd(st)
                if sigbd && st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)   # 경계를 빼고 다시 풀어 확인 (허용오차 거짓 판정 방지)
                    _abb_sigbd!(M, DR, nd, nothing, nothing, s); nmip += 1; Threads.atomic_add!(_ABB_DD_T.nrechk, 1)
                    ok = _nz_dr_opt!(DR, M, l, u, "dd screen 재확인 s=$s")
                    # 2026-11-27 고침: 상태·bound 를 경계를 되돌리기 전에 읽음 (되돌리면 JuMP 가 결과를 지워 상태가 OPTIMIZE_NOT_CALLED 가 되어
                    #   재확인 결과가 버려지고 있었음 — 유효성은 그대로, 그 쌍을 덜 조였을 뿐)
                    ok && (st = termination_status(M.model); bd = getbd(st))
                    sigbd && _abb_sigbd!(M, DR, nd, lo, hi, s)
                    ok || continue
                    st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED) && Threads.atomic_add!(_ABB_DD_T.nrechk_inf, 1)
                end
                if st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)       # 상자 안에 recourse 최적 쌍대를 갖는 α 없음
                    set_time_limit_sec(M.model, nothing); sigbd && _abb_sigbd!(M, DR, nd, nothing, nothing, s)
                    return :empty, (dlo, dhi), nmip
                end
                isfinite(bd) || continue
                if ismax
                    bd + 1e-6 < dhi[p, s] && (dhi[p, s] = bd + 1e-6; tight = true)
                else
                    bd - 1e-6 > dlo[p, s] && (dlo[p, s] = bd - 1e-6; tight = true)
                end
            end
            dhi[p, s] < dlo[p, s] && (dhi[p, s] = dlo[p, s] = 0.5 * (dlo[p, s] + dhi[p, s]))
        end
        boxset && set_time_limit_sec(M.model, nothing)
        boxset && sigbd && _abb_sigbd!(M, DR, nd, nothing, nothing, s)
    end
    return (tight ? :tight : :none), (dlo, dhi), nmip
end

"패치 (dd_inc): 선별만 — _abb_dd_screen! 의 앞부분. 반환 [(모임 비율, 쌍 p, 시나리오 s, 꼭짓점 구간 c1, c2)]"
function _abb_dd_cands(CL, DR, nd, x̄, l, u, lo, hi, dlo, dhi; deadline=Inf, fixtol=1e-4, only_s::Int=0)
    S, m = nd.S, nz_m(nd); out = Tuple{Float64,Int,Int,Float64,Float64}[]
    for s in (only_s > 0 ? (only_s:only_s) : (1:S))
        time() > deadline && break
        ps = Int[]
        for (p, (a, b)) in enumerate(DR.pairs)
            (hi[a, s] - lo[a, s] > fixtol && hi[b, s] - lo[b, s] > fixtol) || continue
            wl = max(dlo[p, s], lo[a, s] - hi[b, s]); wh = min(dhi[p, s], hi[a, s] - lo[b, s])
            wh - wl > 1.0 && push!(ps, p)
        end
        isempty(ps) && continue
        C = CL[s]; ext = Dict{Int,Tuple{Float64,Float64}}(); tlp0 = time()
        for αc in (l, u)
            rr = nz_rhs(nd, x̄, αc, s)
            for k in 1:m; nd.hrow[k] > 0 && set_normalized_coefficient(C.opt, C.σ[k], rr[k]); end
            set_normalized_rhs(C.opt, 1e20)
            @objective(C.model, Min, sum(rr[k] * C.σ[k] for k in 1:m if rr[k] != 0)); optimize!(C.model)
            termination_status(C.model) == MOI.OPTIMAL || error("_abb_dd_cands: 꼭짓점 LP (s=$s) $(termination_status(C.model))")
            zc = objective_value(C.model); set_normalized_rhs(C.opt, zc + 1e-7 * max(1.0, abs(zc)))
            for p in ps
                a, b = DR.pairs[p]; e = C.σ[DR.Hk[a]] - C.σ[DR.Hk[b]]
                @objective(C.model, Min, e); optimize!(C.model)
                termination_status(C.model) == MOI.OPTIMAL || error("_abb_dd_cands: 꼭짓점 차이 LP (s=$s) $(termination_status(C.model))")
                mn = objective_value(C.model)
                @objective(C.model, Max, e); optimize!(C.model)
                termination_status(C.model) == MOI.OPTIMAL || error("_abb_dd_cands: 꼭짓점 차이 LP (s=$s) $(termination_status(C.model))")
                mx = objective_value(C.model)
                o = get(ext, p, (Inf, -Inf)); ext[p] = (min(o[1], mn), max(o[2], mx))
            end
        end
        Threads.atomic_add!(_ABB_DD_T.tlp, time() - tlp0); Threads.atomic_add!(_ABB_DD_T.nlp, 2 + 4 * length(ps))
        for p in ps
            a, b = DR.pairs[p]
            wl = max(dlo[p, s], lo[a, s] - hi[b, s]); wh = min(dhi[p, s], hi[a, s] - lo[b, s])
            c1, c2 = ext[p]
            c2 - c1 < 0.5 * (wh - wl) && push!(out, ((c2 - c1) / (wh - wl), p, s, c1, c2))
        end
    end
    return out
end

"패치 (dd_inc): (쌍 p, 시나리오 s) 차이의 한쪽 (ismax) 을 KKT-MIP 로 인증해 dlo·dhi 를 제자리 조임. 반환 (:empty | :tight | :none, MIP 수)"
function _abb_dd_mip!(DR, nd, l, u, lo, hi, dlo, dhi, p, s, ismax; mip_tl=2.0, deadline=Inf, sigbd::Bool=true, objstop::Float64=NaN, gap::Float64=NaN)
    M = DR.mips[s]; a, b = DR.pairs[p]; nh = nz_nh(nd); n = 0
    time() > deadline && return :none, 0
    for i in 1:nh; set_lower_bound(M.α[i], l[i]); set_upper_bound(M.α[i], u[i]); end
    sigbd && _abb_sigbd!(M, DR, nd, lo, hi, s)
    @objective(M.model, ismax ? MAX_SENSE : MIN_SENSE, M.σ[DR.Hk[a]] - M.σ[DR.Hk[b]])
    set_time_limit_sec(M.model, max(0.01, min(mip_tl, deadline - time())))
    getbd(st) = st == MOI.OPTIMAL ? objective_bound(M.model) : (st in (MOI.TIME_LIMIT, MOI.OBJECTIVE_LIMIT) ? _nz_dr_bound_or_nan(M.model) : NaN)
    # objstop (dd_objstop): 이 쪽 MIP 가 노드 LP 해를 자르려면 최댓값 < δ̂ (최솟값이면 > δ̂) 여야 함 → 해가 δ̂ 이상 (이하) 이면 자를 수 없으니 멈춤 (BestObjStop).
    #   멈춰도 objective_bound 는 유효한 범위. gap (dd_gap): MIPGap. 둘 다 풀이 뒤 원래 값으로
    ob0 = isfinite(objstop) ? get_optimizer_attribute(M.model, "BestObjStop") : NaN; gp0 = isfinite(gap) ? get_optimizer_attribute(M.model, "MIPGap") : NaN
    isfinite(objstop) && set_optimizer_attribute(M.model, "BestObjStop", objstop); isfinite(gap) && set_optimizer_attribute(M.model, "MIPGap", gap)
    tm0 = time(); n += 1
    ok = _nz_dr_opt!(DR, M, l, u, "dd inc s=$s 쌍 ($(DR.Hk[a]), $(DR.Hk[b]))")
    st = ok ? termination_status(M.model) : MOI.OTHER_ERROR; bd = ok ? getbd(st) : NaN
    if ok && sigbd && st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)     # 경계를 빼고 다시 풀어 확인 (결과는 되돌리기 전에 읽음)
        _abb_sigbd!(M, DR, nd, nothing, nothing, s); n += 1; Threads.atomic_add!(_ABB_DD_T.nrechk, 1)
        ok = _nz_dr_opt!(DR, M, l, u, "dd inc 재확인 s=$s")
        st = ok ? termination_status(M.model) : MOI.OTHER_ERROR; bd = ok ? getbd(st) : NaN
        ok && st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED) && Threads.atomic_add!(_ABB_DD_T.nrechk_inf, 1)
    end
    Threads.atomic_add!(_ABB_DD_T.tmip, time() - tm0)
    ok && st == MOI.OBJECTIVE_LIMIT && Threads.atomic_add!(_ABB_DD_T.nobjstop, 1)
    isfinite(objstop) && set_optimizer_attribute(M.model, "BestObjStop", ob0); isfinite(gap) && set_optimizer_attribute(M.model, "MIPGap", gp0)
    set_time_limit_sec(M.model, nothing); sigbd && _abb_sigbd!(M, DR, nd, nothing, nothing, s)
    ok || return :none, n
    st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED) && return :empty, n      # 상자 안에 recourse 최적 쌍대를 갖는 α 없음
    isfinite(bd) || return :none, n
    tight = false
    if ismax
        bd + 1e-6 < dhi[p, s] && (dhi[p, s] = bd + 1e-6; tight = true)
    else
        bd - 1e-6 > dlo[p, s] && (dlo[p, s] = bd - 1e-6; tight = true)
    end
    dhi[p, s] < dlo[p, s] && (dhi[p, s] = dlo[p, s] = 0.5 * (dlo[p, s] + dhi[p, s]))
    return (tight ? :tight : :none), n
end

"패치 (dd_inc = :viol1): 노드 LP 해의 (쌍 p, 시나리오 s) follower 쌍대 차이가 꼭짓점 구간 [c1, c2] 에서 벗어난 양 (× d̂_s) 과 쪽 (+1 위, −1 아래, 0 안)"
function _abb_dd_lpviol(P, DR, p, s, c1, c2)
    v = P.O.v; a, b = DR.pairs[p]; ka, kb = DR.Hk[a], DR.Hk[b]; ds = value(v[:d][s])
    ds > 1e-7 || return 0.0, 0, NaN
    δ = (value(v[:π̃][ka, s]) - value(v[:π̃][kb, s])) / ds
    return (δ > c2 ? (δ - c2) * ds : (δ < c1 ? (c1 - δ) * ds : 0.0)), (δ > c2 ? 1 : (δ < c1 ? -1 : 0)), δ
end

"패치 (dd_sigbd): 시나리오 s 의 범위 MIP 에 H 행 쌍대 경계 [max(0, lo), hi] 를 넣거나 (lo = nothing 이면) 원래 경계 [0, ϖᵁ] 로 되돌림"
function _abb_sigbd!(M, DR, nd, lo, hi, s)
    for (a, k) in enumerate(DR.Hk)
        if lo === nothing
            set_lower_bound(M.σ[k], 0.0); isfinite(nd.varpiU[k]) ? set_upper_bound(M.σ[k], nd.varpiU[k]) : (has_upper_bound(M.σ[k]) && delete_upper_bound(M.σ[k]))
        else
            set_lower_bound(M.σ[k], max(0.0, lo[a, s])); set_upper_bound(M.σ[k], max(hi[a, s], max(0.0, lo[a, s])))
        end
    end
end

"패치 (dd_adapt): 이 노드에서 선별 DD 를 돌릴지. MIP 솔버가 분리기 호출 빈도를 성공률로 조절하는 것과 같은 꼴 —
 DD 를 받은 노드가 warm 개 이상인데 상한이 내려간 비율이 minrate 미만이면 자격 있는 노드 probe 개 중 하나만 (비율은 그 노드들로 계속 갱신).
 돌리지 않는 것은 덜 조일 뿐이라 유효성과 무관"
function _abb_dd_go(adapt, nelig, nnode, ncut, nskip, warm, minrate, probe)
    adapt || return true
    k = Threads.atomic_add!(nelig, 1)
    nn = nnode[]; nc = ncut[]
    (nn < warm || nc >= minrate * nn) && return true
    k % probe == 0 && return true
    Threads.atomic_add!(nskip, 1)
    return false
end

function nz_alpha_bnb_dr(nd::NZData, x̄;
                         # 호출마다 정하는 것
                         nworkers=Threads.nthreads() - 2, time_limit=300.0, rel_gap=1e-4, target=nothing,
                         verbose=true, log_every=30.0,
                         # 자원
                         ipopt_time=60.0, heuristic::Bool=true, envs=nothing,
                         # 탐색 (기본값 = 측정에 쓴 값)
                         branch_point::Symbol=:mid, child_rounds::Int=1, maxrounds=30, dive_max=3, sep_maxadd::Int=5000,
                         min_width=1e-7, presolve::Bool=true,
                         # 강화 스위치 (ablation 전용, 기본 모두 켬)
                         dual_range::Bool=true, comp::Bool=true, vcut::Bool=true, cslift::Bool=true,
                         fwd::Bool=true, kmc::Bool=true, dd::Bool=false,
                         # 탐색·LP 구성 (ablation 전용, 기본 켬)
                         cbranch::Bool=true, sticky::Bool=true, lp_tol::Float64=1e-7,
                         # primal 휴리스틱: 노드 완화의 belief 에서 follower 의 최적 반응을 평가 (nz_resp_heuristic.jl)
                         resp::Bool=false,
                         # range_mip = false: 쌍대 범위 KKT-MIP 를 풀지 않고 범위를 전역 값 [0, ϖᵁ] 으로 둔다 (dual_range 의 나머지 강화 — FWD, KMC,
                         #   상보성·상보성 분기, RLT — 는 그대로). recourse 가 커서 범위 MIP 가 노드 시간을 다 쓰는 인스턴스용 (L6, 문서 §10.26)
                         range_mip::Bool=true,
                         # range_nodes: 범위 MIP 를 처음 range_nodes 개 노드에서만 푼다. 그 뒤 노드는 부모 범위를 물려받는다 (부모 상자의 범위는
                         #   자식 상자에서도 유효). range_mip = false 와 기본 (모든 노드) 의 중간 (L6, 문서 §10.26)
                         range_nodes::Int=typemax(Int),
                         # range_time: 호출 시작 뒤 range_time 초까지만 범위 MIP 를 푼다 (그 뒤 노드는 부모 범위). range_nodes 와 함께 쓰면 둘 다 만족할 때만
                         range_time::Float64=Inf,
                         # range_lp = true: 범위 MIP 대신 상자 두 꼭짓점의 LP (_nz_dr_corner_ranges, 모든 노드). 전제 = recourse 값함수의 submodularity (시제품, 문서 §10.26).
                         #   값 MIP (φ̄) 와 차이 범위 (dd) 는 풀지 않음 → dd = false 와 함께, range_time·range_nodes 는 범위 MIP 용이라 같이 쓰지 않음
                         range_lp::Bool=true,
                         # resp_fix = true (range_lp 전용): 꼭짓점 범위로 follower 한계 이득의 부호가 상자에서 정해지는 예약 좌표를 l 또는 u 로 고정 (_nz_dr_resp_fix)
                         resp_fix::Bool=false,
                         # farkas = true: 노드 LP 가 실행 불가능이면 Gurobi 의 Farkas 쌍대를 먼저 검증 (_nz_dr_farkas_certified). 증명되면 재풀이 없이 가지치기,
                         #   아니면 기존 재확인. 환경변수 NZ_FARKAS_CHECK=1 이면 증명된 노드도 재확인을 돌려 결과가 다르면 센다 (검증용)
                         farkas::Bool=true,
                         # 노드 LP 풀이 방법 (기본 = 기존 dual simplex, Presolve 끔): lp_method = 1 dual simplex | 2 barrier | -1 자동, lp_presolve = 0 끔 | -1 자동,
                         #   lp_crossover = -1 자동 | 0 끔 (내부점 해: _nz_solve_lp! 가 max(primal, dual 목적값) 을 상한으로). L6 S=50 은 B&B 안의 노드 LP 가
                         #   12 개 동시 cold 와 같은 시간이라 (warm start 가 거의 안 들음) barrier 를 시험하려고 둠 (문서 §10.29)
                         lp_method::Int=1, lp_presolve::Int=0, lp_crossover::Int=-1,
                         # infeas_pre = true: 루트가 아닌 노드에서 노드 LP 전에 recourse 블록을 뺀 목적 0 부분 LP 로 실행 불가능을 먼저 판정
                         #   (_nz_dr_pre_infeasible, Farkas 증명서 검증). 증명되면 노드 LP 없이 가지치기 (문서 §10.29)
                         infeas_pre::Bool=true,
                         # infeas_pre_retry = true (infeas_pre 전용): 부분 LP 가 NUMERIC 이거나 검증 실패 INFEASIBLE 이면 NumericFocus 3, 그다음 Seed 2 로 다시 풂 (부분 LP 의 수치 취약성, 문서 §10.29)
                         infeas_pre_retry::Bool=true,
                         # parent_basis = true: 노드 LP 를 푼 직후의 basis 를 자식에 실어 두고, 꺼낸 (pop) 노드의 시작 basis 로 넣음 (_nz_dr_get_basis,
                         #   _nz_dr_set_basis!). dive 는 이미 부모 basis 에서 시작. LP 값에는 영향 없음 (문서 §10.29)
                         parent_basis::Bool=true,
                         # share_rows = true: 분리 행 (sticky) 집합을 worker 끼리 맞춤 (한 worker 가 만든 행을 다른 worker 도 추가). LP 가 커짐 (문서 §10.29)
                         share_rows::Bool=false,
                         # node_lp_cap: 루트가 아닌 노드의 LP 작업 (분리 라운드·수치 오류 재시도·실행 불가능 재확인) 시간 상한 (초). 넘으면 수치 오류 노드처럼
                         #   부모 상한을 물려 가장 넓은 좌표를 반으로 나눔 (노드 하나가 오래 전역 UB 를 붙잡지 않게, 문서 §10.29). Inf = 전체 마감까지
                         node_lp_cap::Float64=120.0,
                         # hucap = true: 루트 전에 h 상한을 recourse 실행 가능성으로 조임 (nz_presolve_hU_feas, location 에선 약정 ≤ 거점 용량).
                         #   최적 반응은 모두 남아 상한·incumbent 유효성 그대로. 재현에서 인증 약 2 배 (문서 docs/nz_ccg_bnp_2026-11-23.md 11.6, 2026-11-25)
                         hucap::Bool=true,
                         # corner_reuse = true (range_lp 전용): 꼭짓점 범위는 hi 를 α = l, lo 를 α = u 의 LP 로 구함. 왼쪽 자식 [l, u′] 은 부모와 l 이, 오른쪽 자식 [l′, u] 는
                         #   u 가 같아 그 꼭짓점 LP 는 부모와 같은 계산 → 건너뛰고 부모 범위 (상보성 분기의 둘째 자식은 상자가 같아 둘 다). 노드 비용의 23% 가 범위 (문서 11.21)
                         # ev_adapt = true: 노드마다 하는 α̂ 정확 평가 (resp 포함) 가 ev_warm 번 연속 incumbent 를 못 바꾸면 ev_probe 노드에 한 번만. incumbent 가 바뀌면 처음으로.
                         #   평가는 incumbent (하한) 에만 쓰여 상한·가지치기 유효성과 무관 (문서 11.21)
                         corner_reuse::Bool=true, ev_adapt::Bool=true, ev_warm::Int=300, ev_probe::Int=10,
                         # lp_scale: 노드 LP 의 Gurobi ScaleFlag (2 = 기하평균). 저장한 dive 노드 LP 59 개에서 시간 0.46 배·가지치기 판정 변화 0 (정밀 기준 대비),
                         #   나무 안에서 지리 5 막힌 점 같은 상한까지 약 20% 빠름, 지리 6 중립 (문서 11.22). −1 = Gurobi 기본 (자동)
                         lp_scale::Int=2,
                         # 이어 풀기: 같은 x̄ 의 이전 호출이 돌려준 r[:state]
                         resume=nothing,
                         # 패치: dd = true 일 때 DD 를 구하는 방식 (:screen 기본 | :full), 선별 DD 의 MIP 하나 시간 상한
                         #   dd_maxw: 선별 DD 는 활성 좌표의 최대 상자 폭이 이 값 이하인 노드에서만 (넓은 상자에서는 차이가 상수가 아니라 효과 없고 비쌈)
                         #   dd_adapt: DD 를 받은 노드가 dd_warm 개를 넘었는데 상한이 내려간 비율이 dd_minrate 보다 낮으면 dd_probe 번째 노드마다만 선별
                         dd_mode::Symbol=:screen, dd_mip_tl::Float64=2.0, dd_maxw::Float64=25.0, dd_sigbd::Bool=true,
                         dd_adapt::Bool=true, dd_warm::Int=200, dd_minrate::Float64=0.25, dd_probe::Int=10,
                         #   dd_inc (2026-11-27, md 11.24): 선별 후보를 나눠 인증하고 다시 풀어 가지치기되면 멈춤. :off (후보 전부, 예전) | :ratio (모임 비율 순, 양쪽 MIP)
                         #     | :viol1 (지금 LP 해가 꼭짓점 구간에서 벗어난 양 순, 벗어난 쪽 MIP 만) | :viol1s (:viol1 을 시나리오 하나씩, 가지치기되면 선별도 멈춤). dd_batch = 한 번에 인증할 후보 수
                         #   dd_objstop: :viol1·:viol1s 에서 노드 LP 해를 자를 수 없다고 판명되면 MIP 를 멈춤 (BestObjStop = δ̂), dd_gap: DD MIP 의 MIPGap (NaN = 그대로)
                         dd_inc::Symbol=:viol1, dd_batch::Int=2, dd_objstop::Bool=true, dd_gap::Float64=NaN)
    branch_point in (:alphahat, :mid) || error("branch_point = :alphahat | :mid")
    config = (dual_range=dual_range, comp=dual_range && comp, vcut=dual_range && vcut, cslift=dual_range && cslift,
              branch_point=branch_point, child_rounds=child_rounds, maxrounds=maxrounds, dive_max=dive_max,
              sep_maxadd=sep_maxadd, presolve=presolve, heuristic=heuristic,
              fwd=dual_range && cslift && fwd, kmc=dual_range && cslift && kmc, dd=dual_range && dd,
              cbranch=dual_range && comp && cbranch, sticky=sticky, lp_tol=lp_tol, resp=resp, range_mip=range_mip, range_nodes=range_nodes, range_time=range_time,
              range_lp=range_lp, resp_fix=resp_fix, farkas=farkas, lp_method=lp_method, lp_presolve=lp_presolve, lp_crossover=lp_crossover, infeas_pre=infeas_pre, infeas_pre_retry=infeas_pre && infeas_pre_retry, parent_basis=parent_basis, share_rows=share_rows, node_lp_cap=node_lp_cap, hucap=hucap,
              corner_reuse=range_lp && corner_reuse, ev_adapt=ev_adapt, ev_warm=ev_warm, ev_probe=ev_probe, lp_scale=lp_scale)
    lp_scale in (-1, 0, 1, 2, 3) || error("nz_alpha_bnb_dr: lp_scale = -1 | 0 | 1 | 2 | 3 (Gurobi ScaleFlag)")
    resp_fix && !range_lp && error("nz_alpha_bnb_dr: resp_fix = true 는 range_lp = true 와 함께")
    range_lp &&(isfinite(range_time) || range_nodes != typemax(Int) || !range_mip || !dual_range) &&   # 패치: dd 허용
        error("nz_alpha_bnb_dr: range_lp = true 는 dual_range = true, dd = false 와 함께 쓰고 range_mip·range_time·range_nodes 는 기본값으로")
    _nz_dr_announce(config)
    n_rnode = Threads.Atomic{Int}(0)                                         # 범위 MIP 를 푼 노드 수 (range_nodes)
    n_cbranch = Threads.Atomic{Int}(0)                                       # 상보성 분기 횟수
    n_infeas = Threads.Atomic{Int}(0); n_infrec = Threads.Atomic{Int}(0)     # 실행 불가능 노드 (1e-6 재확인 통과) / 재확인에서 되살아난 노드
    n_fk = Threads.Atomic{Int}(0); n_fkdis = Threads.Atomic{Int}(0)          # Farkas 로 증명된 노드 / (NZ_FARKAS_CHECK) 재확인이 실행가능이라 한 노드
    n_fkto = Threads.Atomic{Int}(0)                                          # (NZ_FARKAS_CHECK) 재확인이 시간 제한으로 판정 못 한 노드
    n_pre = Threads.Atomic{Int}(0); n_pretry = Threads.Atomic{Int}(0)        # infeas_pre: 부분 LP 로 증명된 노드 / 부분 LP 를 푼 노드
    t_pre = Threads.Atomic{Float64}(0.0); n_preretry = Threads.Atomic{Int}(0)  # infeas_pre_retry: 재시도에서 증명된 노드
    n_bload = Threads.Atomic{Int}(0); n_bfail = Threads.Atomic{Int}(0)       # parent_basis: 부모 basis 를 넣은 노드 / 못 넣은 노드
    n_cap = Threads.Atomic{Int}(0)
    n_ddnode = Threads.Atomic{Int}(0); n_ddmip = Threads.Atomic{Int}(0); n_ddempty = Threads.Atomic{Int}(0); n_ddcut = Threads.Atomic{Int}(0)   # 패치 (screen)
    n_ddelig = Threads.Atomic{Int}(0); n_ddskip = Threads.Atomic{Int}(0)   # 패치 (dd_adapt)
    for c in (_ABB_DD_T.tlp, _ABB_DD_T.tmip); c[] = 0.0; end; for c in (_ABB_DD_T.nlp, _ABB_DD_T.nprune, _ABB_DD_T.nrechk, _ABB_DD_T.nrechk_inf, _ABB_DD_T.nobjstop); c[] = 0; end   # 패치 (시간 내역)                                           # node_lp_cap 에 걸린 노드
    n_cskip = Threads.Atomic{Int}(0); effcheck = get(ENV, "NZ_EFF_CHECK", "0") == "1"   # corner_reuse: 꼭짓점 하나 이상 건너뛴 노드 / 점검 모드
    n_ev = Threads.Atomic{Int}(0); n_evskip = Threads.Atomic{Int}(0)        # ev_adapt: α̂ 평가한 노드 / 건너뛴 노드
    n_evsince = Threads.Atomic{Int}(0); n_evtick = Threads.Atomic{Int}(0)   # ev_adapt: incumbent 못 바꾼 연속 평가 수 / 건너뛰기 계수기
    bdive = get(ENV, "NZ_BASIS_DIVE", "0") == "1"           # 진단 (parent_basis): dive 노드에도 실어 둔 부모 basis 를 명시적으로 넣음
    skeys = Tuple{Int,Int,Bool}[]; skset = Set{Tuple{Int,Int,Bool}}(); sk_lk = ReentrantLock()   # parent_basis: worker 공유 분리 행 키
    precheck = get(ENV, "NZ_PRE_CHECK", "0") == "1"; n_predis = Threads.Atomic{Int}(0); n_preto = Threads.Atomic{Int}(0)
    fkcheck = get(ENV, "NZ_FARKAS_CHECK", "0") == "1"
    prchk = get(ENV, "NZ_PRUNE_CHECK", "0") == "1"; n_prchk = Threads.Atomic{Int}(0); n_prdis = Threads.Atomic{Int}(0)
    nfix = 0
    if presolve
        hUp, nfix = nz_presolve_hU(nd, x̄)
        nfix > 0 && (nd = nz_with_hU(nd, hUp))
    end
    Threads.nthreads(:interactive) >= 1 || error("nz_alpha_bnb_dr: julia -t $(nworkers + 2),1 로 실행하세요")
    t0 = time(); t_end = t0 + time_limit
    gcw0 = Base.gc_num().total_time_to_safepoint       # GC safepoint 대기 (전체 스레드가 멈춘 wall 시간) — :dd_screen => (nodes=n_ddnode[], mips=n_ddmip[], empty=n_ddempty[], cut=n_ddcut[], skip=n_ddskip[], pruned_after=_ABB_DD_T.nprune[], t_lp=round(_ABB_DD_T.tlp[]; digits=1), n_lp=_ABB_DD_T.nlp[], t_mip=round(_ABB_DD_T.tmip[]; digits=1), rechk=_ABB_DD_T.nrechk[], rechk_inf=_ABB_DD_T.nrechk_inf[], objstop=_ABB_DD_T.nobjstop[]), :gc_wait 로 보고
    nh, S = nz_nh(nd), nd.S
    Hr = findall(nd.hrow .> 0)
    mk(env) = () -> (o = Gurobi.Optimizer(env); MOI.set(o, MOI.Silent(), true);
                     MOI.set(o, MOI.RawOptimizerAttribute("Threads"), 1); o)
    envs = envs === nothing ? _nz_envs(nworkers + 1) : envs
    length(envs) >= nworkers + (heuristic ? 1 : 0) || error("nz_alpha_bnb_dr: envs 가 부족함 (worker $nworkers + heuristic $(heuristic))")
    nhcap = 0
    if hucap
        hUc, nhcap = nz_presolve_hU_feas(nd, x̄; optimizer=mk(envs[1]))
        nhcap > 0 && (nd = nz_with_hU(nd, hUc))
    end
    Ps = [_nz_build_sep(nd, x̄; optimizer=mk(envs[i])) for i in 1:nworkers]
    numerr0 = _NZ_NUMERR[]; numfail0 = _NZ_NUMERR_FAIL[]
    DRs = dual_range ? [_nz_dr_attach!(Ps[i], nd, x̄; optimizer=mk(envs[i]), vcut=vcut, cslift=cslift,
                                       fwd=config.fwd, kmc=config.kmc, dd=config.dd) for i in 1:nworkers] : nothing
    CLs = range_lp ? [[_nz_dr_build_corner(nd, x̄, s; optimizer=mk(envs[i])) for s in 1:nd.S] for i in 1:nworkers] : nothing
    # solo[i]: α_i 를 늘릴 때 걸리는 W 행이 i 하나만 포함 (거점별 한도) → 상자 안에서 늘려도 H 안 (resp_fix 의 위쪽 고정 조건)
    solo = [all(all(nd.W[ll, i2] == 0 || i2 == i for i2 in 1:nh) && nd.W[ll, i] * nd.hU[i] <= nd.wvec[ll] + 1e-9
                for ll in 1:size(nd.W, 1) if nd.W[ll, i] > 0) for i in 1:nh]
    n_rfix = Threads.Atomic{Int}(0)                                          # resp_fix 로 고정한 좌표 수 (노드 합)
    # 노드 LP 허용오차. build_nz_omega 의 기본은 1e-9 + NumericFocus 2 (θ 항의 상쇄 때문에 조였던 것). lp_tol > 1e-9 이면 그 값 + NumericFocus 0 으로 풀고
    #   노드 상한에 상대 1e-6 의 여유를 더한다 (허용오차만큼 덜 수렴한 목적값을 상한으로 쓰는 것에 대한 여유).
    lp_tol > 1e-9 && for P in Ps
        set_optimizer_attribute(P.O.model, "FeasibilityTol", lp_tol); set_optimizer_attribute(P.O.model, "OptimalityTol", lp_tol)
        set_optimizer_attribute(P.O.model, "NumericFocus", 0); P.O.model.ext[:lp_tol] = lp_tol
    end
    zpad(z) = (lp_tol > 1e-9 && isfinite(z)) ? z + 1e-6 * max(1.0, abs(z)) : z
    farkas && for P in Ps; set_optimizer_attribute(P.O.model, "InfUnbdInfo", 1); end   # Farkas 쌍대를 받기 위해
    lp_scale == -1 || for P in Ps; set_optimizer_attribute(P.O.model, "ScaleFlag", lp_scale); end   # 노드 LP 스케일링 (부분 LP 는 이 모델을 복사해 물려받음)
    for P in Ps; P.O.model.ext[:retry_fresh] = true; end          # 노드 LP 수치 오류: NumericFocus 3 전에 처음부터 한 번 (_nz_solve_lp!)
    for P in Ps                                                   # 노드 LP 풀이 방법. ext 값은 재시도·재확인 뒤 되돌릴 값 (기본이면 기존 설정 그대로)
        P.O.model.ext[:lp_method] = lp_method; P.O.model.ext[:lp_presolve] = lp_presolve; P.O.model.ext[:lp_crossover] = lp_crossover
        (lp_method, lp_presolve, lp_crossover) == (1, 0, -1) && continue
        set_optimizer_attribute(P.O.model, "Method", lp_method); set_optimizer_attribute(P.O.model, "Presolve", lp_presolve)
        set_optimizer_attribute(P.O.model, "Crossover", lp_crossover)
    end
    # 진단 (NZ_NODELP_LOG=<파일>): 노드마다 한 줄 — 시각, worker, 종류 (root / dive = 이 worker 가 직전에 푼 노드의 자식 / pop = 열린 노드에서 꺼냄),
    #   LP 횟수, simplex 반복, barrier 반복, 완화 LP 시간, 범위 LP 시간, 결과. 부모에서 이어 푼 LP (dive) 와 그렇지 않은 LP (pop) 의 비용 비교용
    nlog = get(ENV, "NZ_NODELP_LOG", ""); nlog_lk = ReentrantLock()
    if !isempty(nlog)
        for P in Ps; P.O.model.ext[:itlog] = zeros(Int, 3); end
        Base.open(io -> @printf(io, "# t wid kind nlp simplex_it barrier_it t_lp t_range result t_pre | S=%d nworkers=%d lp_method=%d lp_presolve=%d lp_crossover=%d\n",
                           nd.S, nworkers, lp_method, lp_presolve, lp_crossover), nlog, "a")
    end
    # parent_basis: Gurobi 모델을 지금 붙인다 (JuMP 는 첫 optimize 때 붙이며 그때까지 쌓인 행을 제약 종류별로 묶어 복사한다). worker 2~12 는 첫 풀이 전에
    #   루트의 분리 행을 받으므로, 늦게 붙이면 분리 행이 기본 행 사이에 끼어 "기본 행 nr0 개 + 분리 행" 순서가 깨진다 (2026-10-29: 다른 worker 의 부모 basis 무효의 원인)
    parent_basis && for P in Ps; MOI.Utilities.attach_optimizer(P.O.model); end
    if get(ENV, "NZ_BASIS_DEBUG", "0") == "1" && nworkers >= 2       # 진단: worker 1·2 모델의 행·열 순서가 같은가 (행 = 감각·rhs·열 번호·계수)
        sigs = map(Ps[1:2]) do Pk
            MOI.Utilities.attach_optimizer(Pk.O.model); o = JuMP.unsafe_backend(Pk.O.model); Gurobi.GRBupdatemodel(o); ri = Ref{Cint}()
            Gurobi.GRBgetintattr(o, "NumConstrs", ri); nr = Int(ri[]); Gurobi.GRBgetintattr(o, "NumVars", ri); nc = Int(ri[])
            nz = Ref{Cint}(); Gurobi.GRBgetconstrs(o, nz, C_NULL, C_NULL, C_NULL, 0, nr)
            beg = zeros(Cint, nr); ind = zeros(Cint, nz[]); val = zeros(nz[]); Gurobi.GRBgetconstrs(o, nz, beg, ind, val, 0, nr)
            sn = zeros(Cchar, nr); Gurobi.GRBgetcharattrarray(o, "Sense", 0, nr, sn)
            lb = zeros(nc); Gurobi.GRBgetdblattrarray(o, "LB", 0, nc, lb); ub = zeros(nc); Gurobi.GRBgetdblattrarray(o, "UB", 0, nc, ub)
            rows = [(sn[i], sort(collect(zip(ind[(beg[i]+1):(i < nr ? beg[i+1] : nz[])], round.(val[(beg[i]+1):(i < nr ? beg[i+1] : nz[])]; sigdigits=8))))) for i in 1:nr]
            (nr, nc, rows, lb, ub)
        end
        (a, b) = sigs
        nrd = count(i -> a[3][i] != b[3][i], 1:min(a[1], b[1])); ncd = count(j -> a[4][j] != b[4][j] || a[5][j] != b[5][j], 1:min(a[2], b[2]))
        @printf(stderr, "  [basis debug] worker 1·2: 행 %d·%d, 열 %d·%d, 순서가 다른 행 %d, 경계가 다른 열 %d (같은 행 집합: %s)\n",
                a[1], b[1], a[2], b[2], nrd, ncd, sort(a[3]) == sort(b[3]))
    end
    fresh0 = _NZ_FRESH_OK[]
    slow = Tuple{Float64,Float64,Float64,Float64}[]               # 20 s 넘게 걸린 노드: (전체, 범위 MIP, 완화 LP, 실행 불가능 재확인) 초
    Fs = [nz_build_fixed_alpha(nd, x̄; optimizer=mk(envs[i])) for i in 1:nworkers]
    Rs = resp ? [_nz_build_resp(nd, x̄; optimizer=mk(envs[i])) for i in 1:nworkers] : nothing
    n_resp = Threads.Atomic{Int}(0)                    # follower 반응 휴리스틱이 incumbent 를 갱신한 횟수
    tol(v) = rel_gap * (isfinite(v) ? max(1.0, abs(v)) : 1.0)
    boxok(l) = all(nd.W * l .<= nd.wvec .+ 1e-9)

    lk = ReentrantLock()
    t_relax = Threads.Atomic{Float64}(0.0); t_eval = Threads.Atomic{Float64}(0.0); t_range = Threads.Atomic{Float64}(0.0)
    t_recheck = Threads.Atomic{Float64}(0.0)
    lo0, hi0 = dual_range ? _nz_dr_global(nd, DRs[1]) : (zeros(0, 0), zeros(0, 0))
    dlo0, dhi0 = dual_range ? _nz_dr_dglobal(nd, DRs[1]) : (zeros(0, 0), zeros(0, 0))
    # 노드 = (l, u, 상한, lo, hi, witness, φ̄, ext).  ext = (eq: 정상성을 등식으로 둔 좌표 (상보성 분기), dlo·dhi: 쌍대 차이 범위)
    open = Any[(zeros(nh), copy(nd.hU), Inf, lo0, hi0, Dict{Tuple{Int,Int,Bool},Vector{Float64}}(), fill(Inf, S), (eq=falses(nh), dlo=dlo0, dhi=dhi0))]
    if resume !== nothing
        resume.x == x̄ || error("nz_alpha_bnb_dr: resume 상태의 x̄ 가 다름 (Ω 는 x̄ 마다 다른 문제)")
        resume.config == config || error("nz_alpha_bnb_dr: resume 상태의 구성이 다름 ($(resume.config) vs $config)")
        open = Any[resume.open...]                 # 노드 튜플 안의 배열은 worker 가 제자리에서 고치지 않으므로 얕은 복사로 충분
    end
    inflight = fill(-Inf, nworkers)
    LB = Ref(-Inf); best_α = Ref(zeros(nh))
    nodes = Ref(0); done = Ref(false); root_UB = Ref(Inf)
    root_rows = Ref{Any}(nothing)
    if resume !== nothing
        LB[] = resume.LB; best_α[] = copy(resume.α); root_UB[] = resume.root_UB; root_rows[] = resume.root_rows
    end
    node_cands = Tuple{Float64,Vector{Float64}}[]
    ipopt_calls = Ref(0)
    pruned_ub = Ref(resume === nothing ? -Inf : resume.pruned_ub)
    nfixed_sum = Ref(0.0)                      # 노드별 "고정된 쌍대 성분 비율" 의 합 (평균 보고용)

    globalUB() = max(LB[], isempty(open) ? -Inf : maximum(n[3] for n in open), maximum(inflight), pruned_ub[])
    offer!(z, a) = lock(lk) do
        z > LB[] && (LB[] = z; best_α[] = copy(a))
    end

    function worker(wid)
        P = Ps[wid]; F = Fs[wid]; DR = dual_range ? DRs[wid] : nothing
        localnode = nothing; dive_left = 0; synced = false
        nown = 0; nsk = 0                                   # parent_basis: 공유 목록에 올린 자기 분리 행 수 / 읽은 공유 키 수
        while true
            yield()
            node = nothing; fromdive = false
            lock(lk)
            try
                ub_now = globalUB()
                if time() >= t_end || ub_now - LB[] <= tol(LB[]) ||
                   (target !== nothing && (LB[] >= target || ub_now <= target))
                    done[] = true
                end
                if done[]
                    localnode !== nothing && push!(open, localnode)
                    return
                end
                if localnode !== nothing && dive_left > 0 && localnode[3] > LB[] + tol(LB[])
                    node = localnode; dive_left -= 1; fromdive = true
                else
                    if localnode !== nothing
                        localnode[3] > LB[] + tol(LB[]) ? push!(open, localnode) :
                                                          (pruned_ub[] = max(pruned_ub[], localnode[3]))
                    end
                    if !isempty(open)
                        node = popat!(open, argmax([n[3] for n in open]))
                        dive_left = dive_max
                    end
                end
                localnode = nothing
                node !== nothing && (inflight[wid] = node[3])
            finally
                unlock(lk)
            end
            if node === nothing
                sleep(0.05); continue
            end
            l, u, ubp, lop, hip, witp, phibp = node
            # 진단 (NZ_BASIS_ROUNDTRIP=1, parent_basis): dive 노드에서 모델의 부모 basis 를 읽어 두었다가 노드 설정 뒤 그대로 다시 넣음 (넣는 경로의 손실 확인)
            rtbas = (parent_basis && fromdive && get(ENV, "NZ_BASIS_ROUNDTRIP", "0") == "1") ? _nz_dr_get_basis(P) : nothing
            tn0 = time(); tn_rg = 0.0; tn_lp = 0.0; tn_rc = 0.0
            ext = node[8]; eqp = ext.eq; dlo = ext.dlo; dhi = ext.dhi
            if !synced && root_rows[] !== nothing && isempty(P.allrows) && t_end - time() > 30.0
                for (i, j, islo, b) in root_rows[]; sticky ? _nz_sticky_add!(P, i, j, islo, b) : _nz_sep_add!(P, i, j, islo, b); end
                synced = true
            end
            # share_rows: 분리 행 (sticky) 집합을 worker 끼리 맞춤 — 자기가 새로 만든 행의 키를 공유 목록에 올리고, 남이 올린 키 중 없는 행을 추가.
            #   행은 어떤 상자에서도 유효한 RLT 행이라 더해도 완화는 조여지기만 한다. parent_basis 와 함께 시험했으나 다른 worker 의 basis 무효의 원인은
            #   모델을 늦게 붙인 것이었고 (위 attach), 공유는 LP 를 키워 S50 노드 LP 를 느리게 함 (문서 §10.29)
            if share_rows && sticky && root_rows[] !== nothing
                lock(sk_lk) do
                    for n in (nown + 1):length(P.allrows)
                        r = P.allrows[n]; key = (r[1], r[6], r[2]); key in skset || (push!(skset, key); push!(skeys, key))
                    end
                    newk = skeys[(nsk + 1):end]; nsk = length(skeys)
                    have = get(P.O.model.ext, :sticky, Dict{Tuple{Int,Int,Bool},Int}())
                    for (i, j, islo) in newk
                        haskey(have, (i, j, islo)) || _nz_sticky_add!(P, i, j, islo, islo ? l[i] : u[i])
                    end
                end
                nown = length(P.allrows)
            end
            z = -Inf; α̂ = nothing; children = nothing; zE = -Inf; unfinished = false; αE = nothing
            pruned_val = -Inf; lo = lop; hi = hip; wit = witp; phib = phibp; fixfrac = NaN
            boxok(l) && ubp <= LB[] + tol(LB[]) && (pruned_val = ubp)
            if ubp > LB[] + tol(LB[]) && boxok(l)
                sticky ? _nz_sticky_set_box!(P, l, u) : _nz_sep_set_box!(P, l, u)
                _nz_sep_set_wbox!(P, P.Lw, P.Uw)
                empty_box = false
                if dual_range && range_lp                     # 꼭짓점 LP 범위 (모든 노드). 값 상한 φ̄ 는 부모 값
                    skl = config.corner_reuse && get(ext, :sl, false); sku = config.corner_reuse && get(ext, :su, false)
                    (skl || sku) && Threads.atomic_add!(n_cskip, 1)
                    tg = @elapsed ((lo, hi) = _nz_dr_corner_ranges(CLs[wid], DR, nd, x̄, l, u, lop, hip; skip_l=skl, skip_u=sku))
                    if effcheck && (skl || sku)
                        lof, hif = _nz_dr_corner_ranges(CLs[wid], DR, nd, x̄, l, u, lop, hip)
                        dmax = max(maximum(abs.(lof .- lo); init=0.0), maximum(abs.(hif .- hi); init=0.0))
                        dmax <= 1e-5 || error("nz_alpha_bnb_dr: corner_reuse 범위가 전부 계산한 것과 $(dmax) 다름 (skip_l=$skl, skip_u=$sku)")
                    end
                    Threads.atomic_add!(t_range, tg); tn_rg += tg
                    if resp_fix                               # 상자를 줄이면 범위는 그대로 유효 (부분 상자). 줄인 상자로 범위를 한 번 더
                        l, u, nrf = _nz_dr_resp_fix(nd, DR, l, u, lo, hi, solo)
                        if nrf > 0
                            Threads.atomic_add!(n_rfix, nrf)
                            sticky ? _nz_sticky_set_box!(P, l, u) : _nz_sep_set_box!(P, l, u)
                            _nz_sep_set_wbox!(P, P.Lw, P.Uw)
                            tg = @elapsed ((lo, hi) = _nz_dr_corner_ranges(CLs[wid], DR, nd, x̄, l, u, lo, hi))
                            Threads.atomic_add!(t_range, tg); tn_rg += tg
                        end
                    end
                    if config.dd && dd_mode == :full          # 패치 (full): 꼭짓점 범위 뒤 상자별 쌍대 차이 범위 (KKT-MIP 전부), 자식에 물려줌
                        wit = Dict{Tuple{Int,Int,Bool},Vector{Float64}}()
                        tg2 = @elapsed ((dlo, dhi) = _nz_dr_dranges!(DR, nd, l, u, lo, hi, ext.dlo, ext.dhi, witp, wit; deadline=t_end))
                        Threads.atomic_add!(t_range, tg2); tn_rg += tg2
                    end
                    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=comp, eq=eqp, dlo=dlo, dhi=dhi)
                    fixfrac = count(hi .- lo .<= 1e-4) / max(1, length(lo))
                elseif dual_range && !(range_mip && time() - t0 < range_time && Threads.atomic_add!(n_rnode, 1) < range_nodes)   # 범위는 부모 값 그대로, 나머지 강화만 상자에 맞춤
                    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=comp, eq=eqp, dlo=dlo, dhi=dhi)
                    fixfrac = count(hi .- lo .<= 1e-4) / max(1, length(lo))
                elseif dual_range
                    rdl = min(t_end, t0 + range_time)            # range_time 마감 (기본 Inf 면 t_end)
                    tg = @elapsed (rg = _nz_dr_ranges(DR, nd, l, u, lop, hip, witp, phibp; deadline=rdl,
                                                      mip_deadline=(isfinite(range_time) ? rdl : Inf)))
                    Threads.atomic_add!(t_range, tg); tn_rg += tg
                    if rg === nothing
                        empty_box = true                 # 상자 안에 실행가능한 α 가 없음
                    else
                        lo, hi, wit, phib = rg
                        if config.dd
                            tg2 = @elapsed ((dlo, dhi) = _nz_dr_dranges!(DR, nd, l, u, lo, hi, ext.dlo, ext.dhi, witp, wit; deadline=rdl,
                                                                          mip_deadline=(isfinite(range_time) ? rdl : Inf)))
                            Threads.atomic_add!(t_range, tg2); tn_rg += tg2
                        end
                        _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=comp, eq=eqp, dlo=dlo, dhi=dhi)
                        fixfrac = count(hi .- lo .<= 1e-4) / max(1, length(lo))
                    end
                end
                if !empty_box
                    set_time_limit_sec(P.O.model, max(t_end - time(), 0.1))
                    isempty(nlog) || (P.O.model.ext[:itlog] .= 0)
                    # 이 노드의 LP 작업 마감 (_nz_solve_lp! 가 재시도마다 남은 시간으로 TimeLimit 을 다시 둠). 루트는 전체 마감까지
                    ndl = ubp == Inf ? t_end : min(t_end, time() + node_lp_cap); P.O.model.ext[:deadline] = ndl
                    pre_inf = false; tp = 0.0                 # infeas_pre: 부분 LP 로 실행 불가능이 증명되면 노드 LP 를 풀지 않음
                    if infeas_pre && ubp != Inf
                        Threads.atomic_add!(n_pretry, 1)
                        tp = @elapsed ((pre_inf, _, pre_rt) = _nz_dr_pre_infeasible(P.O.model; tlim=min(30.0, ndl - time()), retry=config.infeas_pre_retry))
                        Threads.atomic_add!(t_pre, tp); pre_inf && Threads.atomic_add!(n_pre, 1); pre_rt && Threads.atomic_add!(n_preretry, 1)
                    end
                    bload = false                             # parent_basis: 꺼낸 노드에 부모 basis 가 있으면 시작 basis 로
                    if parent_basis && !pre_inf && (!fromdive || bdive) && get(ext, :bas, nothing) !== nothing
                        bload = _nz_dr_set_basis!(P, ext.bas); Threads.atomic_add!(bload ? n_bload : n_bfail, 1)
                    end
                    rtbas !== nothing && !pre_inf && (bload = _nz_dr_set_basis!(P, rtbas))
                    tr = pre_inf ? 0.0 : @elapsed (z = zpad((sticky ? _nz_sticky_solve! : _nz_sep_solve!)(P; maxrounds=(ubp == Inf ? maxrounds : child_rounds), maxadd=sep_maxadd)))
                    pre_inf && (z = -Inf)
                    dd_empty = false                          # 패치 (screen): 가지치기 못 한 노드에서 선별 DD → 상자가 비었거나 조여지면 다시 풂
                    if config.dd && dd_mode == :screen && range_lp && !pre_inf && ubp != Inf && isfinite(z) && z > LB[] + tol(LB[]) &&
                       maximum((u[i] - l[i] for i in 1:nh if nd.hU[i] > 1e-9); init=0.0) <= dd_maxw &&
                       dd_inc !== :off && _abb_dd_go(dd_adapt, n_ddelig, n_ddnode, n_ddcut, n_ddskip, dd_warm, dd_minrate, dd_probe)
                        # 패치 (dd_inc): 후보를 dd_batch 개씩 인증 → 조여지면 다시 풂 → 가지치기 (또는 비움) 되면 멈춤. 남은 후보는 풀지 않음 (덜 조일 뿐)
                        Threads.atomic_add!(n_ddnode, 1)
                        # :viol1s 는 시나리오 하나씩 선별 (ζF 곱 위반이 큰 시나리오부터) → 그 시나리오 후보를 :viol1 처럼 인증 → 가지치기되면 남은 시나리오는 선별도 안 함
                        sord = Int[]
                        if dd_inc === :viol1s
                            vv_ = P.O.v; α̂d = value.(vv_[:α]); d̂d = value.(vv_[:d]); ζFd = value.(vv_[:ζF])
                            sord = sortperm([sum(abs(ζFd[i, s_] - α̂d[i] * d̂d[s_]) for i in 1:nh) for s_ in 1:nd.S]; rev=true)
                        end
                        tdd = @elapsed (cands = dd_inc === :viol1s ? Tuple{Float64,Int,Int,Float64,Float64}[] : _abb_dd_cands(CLs[wid], DR, nd, x̄, l, u, lo, hi, dlo, dhi; deadline=t_end))
                        dlo = copy(dlo); dhi = copy(dhi); z0dd = z; nmdd = 0; kdd = 0
                        for s_ in sord
                            (dd_empty || !isfinite(z) || z <= LB[] + tol(LB[]) || time() >= t_end) && break
                            tdd += @elapsed (cs = _abb_dd_cands(CLs[wid], DR, nd, x̄, l, u, lo, hi, dlo, dhi; deadline=t_end, only_s=s_))
                            while !isempty(cs) && !dd_empty && isfinite(z) && z > LB[] + tol(LB[]) && time() < t_end
                                sv = [_abb_dd_lpviol(P, DR, c[2], c[3], c[4], c[5]) for c in cs]
                                od = sortperm([x[1] for x in sv]; rev=true); cs = cs[od]; sv = sv[od]
                                sv[1][1] <= 1e-6 && break                              # 이 시나리오에는 지금 LP 해를 자를 후보 없음
                                bt = cs[1:min(dd_batch, end)]; cs = cs[(length(bt)+1):end]; tgt = false
                                tdd += @elapsed for (c, (_, sd_, δ_)) in zip(bt, sv)
                                    sd_ == 0 && continue
                                    rr_, nm_ = _abb_dd_mip!(DR, nd, l, u, lo, hi, dlo, dhi, c[2], c[3], sd_ > 0; mip_tl=dd_mip_tl, deadline=t_end, sigbd=dd_sigbd,
                                                            objstop=(dd_objstop ? δ_ : NaN), gap=dd_gap)
                                    nmdd += nm_; rr_ === :empty && (dd_empty = true; break); rr_ === :tight && (tgt = true)
                                end
                                if dd_empty
                                    z = -Inf; Threads.atomic_add!(n_ddempty, 1)
                                elseif tgt
                                    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=comp, eq=eqp, dlo=dlo, dhi=dhi)
                                    tr += @elapsed (z = zpad((sticky ? _nz_sticky_solve! : _nz_sep_solve!)(P; maxrounds=child_rounds, maxadd=sep_maxadd)))
                                end
                            end
                        end
                        dd_inc === :ratio && sort!(cands; by=c -> c[1])
                        while kdd < length(cands) && !dd_empty && isfinite(z) && z > LB[] + tol(LB[]) && time() < t_end
                            if dd_inc === :viol1                                   # 지금 LP 해 기준으로 남은 후보를 다시 줄 세움
                                rest = cands[(kdd+1):end]; sv = [_abb_dd_lpviol(P, DR, c[2], c[3], c[4], c[5]) for c in rest]
                                od = sortperm([x[1] for x in sv]; rev=true); cands = vcat(cands[1:kdd], rest[od]); sv = sv[od]
                                sv[1][1] <= 1e-6 && break                          # 지금 LP 해를 자를 후보 없음
                            end
                            bt = cands[(kdd+1):min(kdd + dd_batch, end)]; kdd += length(bt); tgt = false
                            tdd += @elapsed for c in bt
                                _, p_, s_, c1_, c2_ = c
                                sides = (false, true)
                                δ1_ = NaN
                                if dd_inc === :viol1
                                    _, sd_, δ1_ = _abb_dd_lpviol(P, DR, p_, s_, c1_, c2_); sd_ == 0 && continue; sides = (sd_ > 0,)
                                end
                                for sd in sides
                                    rr_, nm_ = _abb_dd_mip!(DR, nd, l, u, lo, hi, dlo, dhi, p_, s_, sd; mip_tl=dd_mip_tl, deadline=t_end, sigbd=dd_sigbd,
                                                            objstop=(dd_objstop && dd_inc === :viol1 ? δ1_ : NaN), gap=dd_gap)
                                    nmdd += nm_; rr_ === :empty && (dd_empty = true; break); rr_ === :tight && (tgt = true)
                                end
                                dd_empty && break
                            end
                            if dd_empty
                                z = -Inf; Threads.atomic_add!(n_ddempty, 1)
                            elseif tgt
                                _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=comp, eq=eqp, dlo=dlo, dhi=dhi)
                                tr += @elapsed (z = zpad((sticky ? _nz_sticky_solve! : _nz_sep_solve!)(P; maxrounds=child_rounds, maxadd=sep_maxadd)))
                            end
                        end
                        Threads.atomic_add!(t_range, tdd); tn_rg += tdd; Threads.atomic_add!(n_ddmip, nmdd)
                        (!isnan(z) && z < z0dd) && Threads.atomic_add!(n_ddcut, 1)
                        (!isnan(z) && z <= LB[] + tol(LB[])) && Threads.atomic_add!(_ABB_DD_T.nprune, 1)
                    elseif config.dd && dd_mode == :screen && range_lp && !pre_inf && ubp != Inf && isfinite(z) && z > LB[] + tol(LB[]) &&
                       maximum((u[i] - l[i] for i in 1:nh if nd.hU[i] > 1e-9); init=0.0) <= dd_maxw && dd_inc === :off &&
                       _abb_dd_go(dd_adapt, n_ddelig, n_ddnode, n_ddcut, n_ddskip, dd_warm, dd_minrate, dd_probe)
                        tdd = @elapsed (ddr = _abb_dd_screen!(CLs[wid], DR, nd, x̄, l, u, lo, hi, dlo, dhi; mip_tl=dd_mip_tl, deadline=t_end, sigbd=dd_sigbd))
                        Threads.atomic_add!(t_range, tdd); tn_rg += tdd; Threads.atomic_add!(n_ddnode, 1); Threads.atomic_add!(n_ddmip, ddr[3])
                        if ddr[1] === :empty
                            dd_empty = true; z = -Inf; Threads.atomic_add!(n_ddempty, 1)
                        elseif ddr[1] === :tight
                            dlo, dhi = ddr[2]
                            _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=comp, eq=eqp, dlo=dlo, dhi=dhi)
                            tr += @elapsed (z2 = zpad((sticky ? _nz_sticky_solve! : _nz_sep_solve!)(P; maxrounds=child_rounds, maxadd=sep_maxadd)))
                            (!isnan(z2) && z2 < z) && Threads.atomic_add!(n_ddcut, 1)
                            (!isnan(z2) && z2 <= LB[] + tol(LB[])) && Threads.atomic_add!(_ABB_DD_T.nprune, 1)   # 패치 (시간 내역)
                            z = z2
                        end
                    end
                    if isnan(z) && ndl < t_end && time() < t_end - 1.0  # node_lp_cap 에 걸림 (전체 마감 전): 수치 오류 노드처럼 나눔
                        Threads.atomic_add!(n_cap, 1); z = Inf
                    end
                    if !isempty(nlog)
                        il = P.O.model.ext[:itlog]; kind = ubp == Inf ? "root" : fromdive ? (bload ? "diveb" : "dive") : bload ? "popb$(get(ext, :side, 1))$(ext.bas.wid == wid ? "s" : "o")" : "pop"
                        res = pre_inf ? "preinf" : isnan(z) ? "time" : z == -Inf ? "infeas" : z == Inf ? "numerr" : z <= LB[] + tol(LB[]) ? "prune" : "branch"
                        lock(nlog_lk) do
                            Base.open(io -> @printf(io, "%.2f %d %s %d %d %d %.3f %.3f %s %.3f\n", time() - t0, wid, kind, il[1], il[2], il[3], tr, tn_rg, res, tp), nlog, "a")
                        end
                    end
                    # 진단 (NZ_PRUNE_CHECK=1): LP 값으로 가지칠 노드 (OPTIMAL) 를 기저 없이 처음부터 다시 풀어 기준을 넘으면 출력
                    #   (이어 푼 dual simplex 의 최적값 검증. 2026-10-29 L6 S3 1,031 개·S20 68 개 불일치 0)
                    if prchk && isfinite(z) && z <= LB[] + tol(LB[]) && termination_status(P.O.model) == MOI.OPTIMAL
                        Threads.atomic_add!(n_prchk, 1); cthr = LB[] + tol(LB[])
                        Gurobi.GRBreset(JuMP.unsafe_backend(P.O.model), 0); zc0 = zpad(_nz_solve_lp!(P.O.model))
                        if isfinite(zc0) && zc0 > cthr
                            Threads.atomic_add!(n_prdis, 1)
                            @printf(stderr, "  [가지치기 불일치] 기준 %.10g, 이어 푼 값 %.10g, 처음부터 %.10g (초과 상대 %.2e)\n", cthr, z, zc0, (zc0 - cthr) / max(1.0, abs(cthr)))
                        end
                    end
                    Threads.atomic_add!(t_relax, tr); tn_lp += tr
                    # 실행 불가능 판정 재확인 (_nz_dr_recheck_infeasible!): 그래도 실행 불가능일 때만 가지치기한다. 재확인에서 풀리면
                    #   그 값을 상한으로 쓰고 해는 읽지 않은 채 가장 넓은 좌표를 반으로 나눈다 (수치 오류 노드와 같은 처리).
                    zrec = NaN
                    if z == -Inf && dual_range && !dd_empty      # 패치: DD MIP 가 상자를 비웠으면 (KKT-MIP 실행 불가능) 재확인할 LP 가 없음
                        fk = pre_inf                          # 부분 LP 의 Farkas 증명서로 이미 검증됨
                        if pre_inf && precheck                # 검증용 (NZ_PRE_CHECK=1): 노드 LP 를 처음부터 다시 풀어 대조
                            tn_rc += @elapsed (zchk = _nz_dr_recheck_infeasible!(P.O.model))
                            isfinite(zchk) && Threads.atomic_add!(n_predis, 1)         # 재확인이 해를 찾음 = 불일치
                            isnan(zchk) && Threads.atomic_add!(n_preto, 1)
                        end
                        if farkas && !pre_inf
                            tn_rc += @elapsed (fk = _nz_dr_farkas_certified(P.O.model))
                            fk && Threads.atomic_add!(n_fk, 1)
                            if fk && fkcheck
                                tn_rc += @elapsed (zchk = _nz_dr_recheck_infeasible!(P.O.model))
                                isfinite(zchk) && Threads.atomic_add!(n_fkdis, 1)      # 재확인이 해를 찾음 = 불일치
                                isnan(zchk) && Threads.atomic_add!(n_fkto, 1)          # 재확인이 시간 제한에 걸림 (판정 없음)
                            end
                        end
                        tn_rc += @elapsed (zc = fk ? -Inf : _nz_dr_recheck_infeasible!(P.O.model))
                        Threads.atomic_add!(t_recheck, tn_rc)
                        if zc == -Inf
                            Threads.atomic_add!(n_infeas, 1)
                        else
                            Threads.atomic_add!(n_infrec, 1)
                            zrec = isfinite(zc) ? zpad(zc) : ubp
                            z = Inf
                        end
                    end
                    unfinished = isnan(z)
                    numfail = z == Inf
                    if numfail
                        z = isnan(zrec) ? ubp : min(zrec, ubp)
                        i = argmax(u .- l)
                        if u[i] - l[i] > min_width
                            t = 0.5 * (l[i] + u[i]); u1 = copy(u); u1[i] = t; l2 = copy(l); l2[i] = t
                            children = ((copy(l), u1, z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, sl=true)), (l2, copy(u), z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, su=true)))
                        else
                            pruned_val = z
                        end
                    end
                    z = unfinished ? ubp : min(z, ubp)
                    (!unfinished && !numfail && z <= LB[] + tol(LB[])) && (pruned_val = z)
                    if !unfinished && !numfail && z > LB[] + tol(LB[])
                        v = P.O.v
                        basP = parent_basis ? _nz_dr_get_basis(P) : nothing
                        basP === nothing || (basP = merge(basP, (wid=wid,)))   # 자식에 실을 부모 basis (값을 읽기만 한 뒤라 모델은 그대로)
                        α̂ = clamp.(value.(v[:α]), l, u)
                        # 정확 평가용 사본 (2026-11-23): 아주 작은 성분은 0 으로. α 고정 LP 의 계수 −α̂ 로 들어가 행렬 범위를 4e-6 까지 넓힌다.
                        #   W ≥ 0 이라 0 으로 줄여도 H 안이고, 값은 이 점에서 정확히 평가하므로 incumbent 로 유효 (상자 [l, u] 밖이어도 됨).
                        #   α̂ 자체 (곱 위반, 분기점, 완화 값의 귀속) 는 그대로 둔다. 나무 판 exact_value 와 같은 정리.
                        α̂c = [(α̂[i] < 1e-6 * max(1.0, nd.hU[i])) ? 0.0 : α̂[i] for i in eachindex(α̂)]
                        rv = value.(v[:r]); dv = value.(v[:d])
                        ϖv = zeros(size(v[:ϖ]))
                        for k in Hr, s in 1:S; ϖv[k, s] = value(v[:ϖ][k, s]); end
                        ζLv = value.(v[:ζL]); ζFv = value.(v[:ζF]); ζWv = value.(v[:ζW])
                        set_time_limit_sec(F.model, max(t_end - time(), 0.1))
                        doev = !config.ev_adapt || n_evsince[] < ev_warm || Threads.atomic_add!(n_evtick, 1) % ev_probe == 0   # ev_adapt
                        doev ? Threads.atomic_add!(n_ev, 1) : Threads.atomic_add!(n_evskip, 1)
                        te = @elapsed (zE = doev ? nz_eval_alpha!(F, nd, α̂c) : -Inf)
                        isnan(zE) && (zE = -Inf)
                        αE = α̂c                                 # zE 를 준 점 (incumbent 후보)
                        if resp && doev
                            te += @elapsed begin
                                αr = _nz_resp!(Rs[wid], nd, max.(dv, 0.0), α̂)
                                if αr !== nothing
                                    set_time_limit_sec(F.model, max(t_end - time(), 0.1))
                                    zR = nz_eval_alpha!(F, nd, αr)
                                    (!isnan(zR) && zR > zE) && (zE = zR; αE = αr)
                                end
                            end
                        end
                        Threads.atomic_add!(t_eval, te)
                        viol = zeros(nh)
                        for i in 1:nh
                            u[i] - l[i] > min_width || continue
                            viol[i] = sum(abs(ζLv[i, s] - α̂[i] * rv[s]) + abs(ζFv[i, s] - α̂[i] * dv[s]) for s in 1:S)
                            for (jj, k) in enumerate(Hr)
                                nd.hrow[k] == i || continue
                                viol[i] += sum(abs(ζWv[jj, s] - α̂[i] * ϖv[k, s]) for s in 1:S)
                            end
                        end
                        i = argmax(viol)
                        # 상보성 분기 (cbranch): 아래 끝이 0 인 좌표에서 α̂_i > 0 이면서 정상성 여유 ĝ_i > 0 이면 (상보성 α_i g_i = 0 위반)
                        #   상자를 반으로 나누지 않고 {α_i = 0} | {g_i = 0 (정상성 등식), 상자 그대로} 로 나눈다. 중점 분할은 왼쪽 자식 [0, u/2] 에
                        #   완화된 상보성이 그대로 남아 폭에 비례하는 오차가 반씩만 줄어든다 (2026-10-13 진단: 정체한 상자의 주된 오차).
                        ic = 0
                        if config.cbranch && viol[i] > 1e-9
                            κv = value.(v[:κ]); bestc = 1e-4
                            for i2 in 1:nh
                                (nd.hU[i2] > 0 && u[i2] - l[i2] > min_width && l[i2] <= 1e-9 && !eqp[i2]) || continue
                                g = sum(nd.W[ll, i2] * κv[ll] for ll in 1:size(nd.W, 1)) - nd.c0[i2]
                                for k in DR.Hk
                                    nd.hrow[k] == i2 && (g -= nd.hcoef[k] * sum(value(v[:π̃][k, s]) for s in 1:S))
                                end
                                sc = (α̂[i2] / nd.hU[i2]) * (g / max(1.0, abs(nd.c0[i2])))
                                sc > bestc && (bestc = sc; ic = i2)
                            end
                        end
                        if ic > 0
                            u1 = copy(u); u1[ic] = l[ic]; eqB = copy(eqp); eqB[ic] = true
                            children = ((copy(l), u1, z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, bas=basP, sl=true)), (copy(l), copy(u), z, lo, hi, wit, phib, (eq=eqB, dlo=dlo, dhi=dhi, bas=basP, side=2, sl=true, su=true)))
                            Threads.atomic_add!(n_cbranch, 1)
                        elseif viol[i] > 1e-9
                            width = u[i] - l[i]; t = α̂[i]
                            (branch_point != :alphahat || t - l[i] < 0.1 * width || u[i] - t < 0.1 * width) && (t = 0.5 * (l[i] + u[i]))
                            u1 = copy(u); u1[i] = t; l2 = copy(l); l2[i] = t
                            children = ((copy(l), u1, z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, bas=basP, sl=true)), (l2, copy(u), z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, bas=basP, side=2, su=true)))
                        else
                            z > zE && (zE = z; αE = α̂)     # 곱 위반이 없으면 완화가 정확
                        end
                    end
                end
            end
            lock(lk)
            try
                if unfinished
                    push!(open, node); inflight[wid] = -Inf; dive_left = 0
                    continue
                end
                nodes[] += 1
                time() - tn0 > 20.0 && push!(slow, (time() - tn0, tn_rg, tn_lp, tn_rc))
                isnan(fixfrac) || (nfixed_sum[] += fixfrac)
                pruned_ub[] = max(pruned_ub[], pruned_val)
                if ubp == Inf
                    root_UB[] = z
                    root_rows[] = [(r[1], r[6], r[2], r[3]) for r in P.allrows]
                end
                if zE > LB[]
                    (αE === α̂c || αE === α̂) || Threads.atomic_add!(n_resp, 1)
                    LB[] = zE; best_α[] = copy(αE)
                    n_evsince[] = 0                                                   # ev_adapt: incumbent 가 바뀌면 처음으로
                elseif α̂ !== nothing && doev
                    Threads.atomic_add!(n_evsince, 1)
                end
                if α̂ !== nothing
                    push!(node_cands, (z, copy(α̂)))
                    length(node_cands) > 200 && popfirst!(node_cands)
                end
                if children !== nothing
                    for c in children[2:end]; push!(open, c); end
                    localnode = children[1]
                else
                    dive_left = 0
                end
                inflight[wid] = -Inf
            finally
                unlock(lk)
            end
        end
    end

    function heuristic_task()
        Fh = nz_build_fixed_alpha(nd, x̄; optimizer=mk(envs[end]))
        tried = Set{Vector{Float64}}()
        a0 = _nz_capH(fill(minimum(nd.wvec) / nh, nh), nd)
        while !done[] && t_end - time() > 15.0
            push!(tried, round.(a0; digits=3))
            aloc = nz_ipopt_local_alpha(nd, x̄, Fh, a0; max_time=min(ipopt_time, t_end - time()), deadline=t_end)
            ipopt_calls[] += 1
            if aloc !== nothing
                z = nz_eval_alpha!(Fh, nd, aloc)
                isnan(z) || offer!(z, aloc)
            end
            a0 = nothing
            while a0 === nothing && !done[]
                lock(lk)
                try
                    best = -Inf
                    for (zc, a) in node_cands
                        (zc > best && !(round.(a; digits=3) in tried)) && (best = zc; a0 = a)
                    end
                finally
                    unlock(lk)
                end
                a0 === nothing && sleep(0.5)
            end
            a0 === nothing && break
            a0 = _nz_capH(a0, nd)
        end
    end

    tasks = [Threads.@spawn worker(i) for i in 1:nworkers]
    heuristic && push!(tasks, Threads.@spawn heuristic_task())
    t_log = time()
    while true
        sleep(0.2)
        fin = false; UBn = Inf
        lock(lk)
        try
            UBn = globalUB()
            idle = isempty(open) && all(inflight .== -Inf)
            fin = (time() >= t_end) || (UBn - LB[] <= tol(LB[])) || (idle && (nodes[] > 0 || resume !== nothing)) ||
                  (target !== nothing && (LB[] >= target || UBn <= target))
            any(istaskfailed, tasks) && (fin = true)        # worker 가 예외로 죽으면 바로 멈추고 아래 wait 에서 예외를 올림
            fin && (done[] = true)
        finally
            unlock(lk)
        end
        if verbose && time() - t_log > log_every
            @printf("    [nz-α-B&B-dr] t=%6.1fs nodes=%5d open=%5d LB=%.6f UB=%.6f gap=%.3e\n",
                    time() - t0, nodes[], length(open), LB[], UBn, (UBn - LB[]) / max(1.0, abs(LB[])))
            flush(stdout); t_log = time()
        end
        fin && break
    end
    foreach(wait, tasks)
    UB = max(LB[], isempty(open) ? -Inf : maximum(n[3] for n in open), pruned_ub[])
    return Dict(:LB => LB[], :UB => UB, :α => best_α[], :is_exact => (UB - LB[] <= tol(LB[])),
                :nodes => nodes[], :time => time() - t0, :root_UB => root_UB[], :ipopt_calls => ipopt_calls[],
                :solver => :nz_alpha_bnb_dr, :config => config, :presolve_fixed => nfix, :hucap_tightened => nhcap, :resumed => resume !== nothing,
                :state => (x=copy(x̄), config=config, open=open, LB=LB[], α=copy(best_α[]), pruned_ub=pruned_ub[],
                           root_UB=root_UB[], root_rows=root_rows[]), :t_relax => t_relax[], :t_eval => t_eval[], :t_range => t_range[],
                :t_recheck => t_recheck[], :slow_nodes => sort(slow; rev=true), :numerr_fresh => _NZ_FRESH_OK[] - fresh0,
                :range_mips => dual_range ? sum(D.nmip for D in DRs) : 0, :range_fail => dual_range ? sum(D.nfail for D in DRs) : 0,
                :fixfrac => nodes[] > 0 ? nfixed_sum[] / nodes[] : NaN, :infeas => n_infeas[], :infeas_recovered => n_infrec[],
                :numerr => _NZ_NUMERR[] - numerr0, :numerr_fail => _NZ_NUMERR_FAIL[] - numfail0,
                :cbranch => n_cbranch[], :resp_fix => n_rfix[], :farkas => n_fk[], :farkas_disagree => n_fkdis[], :farkas_check_timeout => n_fkto[], :infeas_pre => n_pre[], :infeas_pre_tried => n_pretry[], :infeas_pre_retry => n_preretry[], :t_pre => t_pre[], :infeas_pre_disagree => n_predis[], :infeas_pre_check_timeout => n_preto[], :basis_loaded => n_bload[], :node_lp_cap_hits => n_cap[], :basis_failed => n_bfail[], :prune_checked => n_prchk[], :prune_disagree => n_prdis[], :resp_hits => n_resp[], :eff => (corner_skip=n_cskip[], ev=n_ev[], ev_skip=n_evskip[]), :gcsafe => _NZ_GCSAFE_STATE[], :dd_screen => (nodes=n_ddnode[], mips=n_ddmip[], empty=n_ddempty[], cut=n_ddcut[], skip=n_ddskip[], pruned_after=_ABB_DD_T.nprune[], t_lp=round(_ABB_DD_T.tlp[]; digits=1), n_lp=_ABB_DD_T.nlp[], t_mip=round(_ABB_DD_T.tmip[]; digits=1), rechk=_ABB_DD_T.nrechk[], rechk_inf=_ABB_DD_T.nrechk_inf[], objstop=_ABB_DD_T.nobjstop[]), :gc_wait => (Base.gc_num().total_time_to_safepoint - gcw0) / 1e9)
end
