# gen_abb_dd_patch.py — 저장소 nonzero_sum/nz_alpha_bnb_dr.jl 의 nz_alpha_bnb_dr 함수만 떼어 "꼭짓점 LP 범위 + 쌍대 차이 범위 (DD)" 를
# 허용하도록 고친 재정의 파일 abb_dd_patch.jl 을 만든다 (2026-11-25 밤, 실험용. 저장소 파일은 고치지 않음).
#   기본 판은 range_lp = true 이면 dd = false 여야 함 (L6 에서 범위 MIP 가 노드 시간을 다 써서). 지리 5 전선 진단에서
#   상자 DD 가 상한 큰 상자를 모두 실행 불가능으로 만들었음 (front_dd.jl, 재확인 통과, incumbent 상자 유효성 통과). 루트 DD 한 번은 효과 없음.
#   dd_mode = :full   꼭짓점 범위 뒤 두 성분 모두 안 고정인 쌍 전부를 KKT-MIP 로 (_nz_dr_dranges!). 상자당 150 ~ 190 s.
#   dd_mode = :screen 노드 LP 로 가지치기 못 한 노드에서만, 두 꼭짓점의 최적 쌍대 면에서 차이가 모인 (쌍, s) 를 골라 KKT-MIP 로 인증하고 다시 풂
#                     (front_dd_lazy.jl SCREEN=corner: 상자 8 개 모두 가지치기, 상자당 약 18 s). MIP 하나 dd_mip_tl 초, 시간 상한이면 objective_bound.
#                     활성 좌표의 최대 폭이 dd_maxw (25) 이하인 노드만. 첫 판 (문턱 없음) 은 489 s 에 노드 170 개 (기본 판 5,032 개) 로 위쪽 넓은 상자에서 시간을 다 씀.
#   유효성: 범위 MIP 판의 DD 와 같은 논증 (상자 안 α 의 recourse 최적 쌍대의 차이 범위). 고른 쌍만 조이는 것은 덜 조일 뿐.
#   자식은 조인 dlo·dhi 를 물려받음 (자식 상자 ⊆ 부모 상자).
#   2026-11-26: dd_sigbd (기본 켬) — 선별 DD 의 KKT-MIP 에 상자의 꼭짓점 LP 범위 [lo, hi] 를 H 행 쌍대의 변수 경계로 넣음 (front_ddmip.jl:
#                MIP 평균 0.87 → 0.19 s, bound 는 같거나 0.01 이하로 더 느슨). 범위는 상자 안 α 의 모든 (정규형) 최적 쌍대를 담아 유효.
#                경계를 넣은 MIP 가 실행 불가능이면 경계를 빼고 다시 풀어 확인한 뒤에만 상자를 비움 (허용오차로 인한 거짓 가지치기 방지).
#   2026-11-26: dd_adapt (기본 켬) — 선별 DD 의 성공률 (상한이 내려간 노드 / DD 받은 노드) 이 낮으면 선별 빈도를 1/dd_probe 로.
#                지리 5 막힌 점은 77% (DD 가 기본 판을 이김), 지리 6 막힌 점은 7% (선별 LP 비용으로 처리량 반) 였음 (md 11.14 (9)).
# 사용: python gen_abb_dd_patch.py  → abb_dd_patch.jl. 스크립트에서 nz_benders.jl 을 include 한 뒤 include("abb_dd_patch.jl"), dd = true 로 부름.
import os
here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, '..', '..', 'nonzero_sum', 'nz_alpha_bnb_dr.jl')
L = open(src, encoding='utf-8').read().split('\n')
start = next(i for i, l in enumerate(L) if l.startswith('function nz_alpha_bnb_dr(nd::NZData, x̄;'))
end = max(i for i, l in enumerate(L) if l == 'end')
assert end > start
body = '\n'.join(L[start:end + 1])

def rep(a, b, cnt=1):
    global body
    n = body.count(a)
    if n != cnt:
        raise SystemExit(f'count {n} != {cnt}: {a[:90]!r}')
    body = body.replace(a, b)

# (0) 인자
rep('                         resume=nothing)',
    '                         resume=nothing,\n'
    '                         # 패치: dd = true 일 때 DD 를 구하는 방식 (:screen 기본 | :full), 선별 DD 의 MIP 하나 시간 상한\n'
    '                         #   dd_maxw: 선별 DD 는 활성 좌표의 최대 상자 폭이 이 값 이하인 노드에서만 (넓은 상자에서는 차이가 상수가 아니라 효과 없고 비쌈)\n'
    '                         #   dd_adapt: DD 를 받은 노드가 dd_warm 개를 넘었는데 상한이 내려간 비율이 dd_minrate 보다 낮으면 dd_probe 번째 노드마다만 선별\n'
    '                         dd_mode::Symbol=:screen, dd_mip_tl::Float64=2.0, dd_maxw::Float64=25.0, dd_sigbd::Bool=true,\n'
    '                         dd_adapt::Bool=true, dd_warm::Int=200, dd_minrate::Float64=0.25, dd_probe::Int=10)')
# (1) range_lp 와 dd 함께 허용
rep('    range_lp &&(dd || isfinite(range_time) || range_nodes != typemax(Int) || !range_mip || !dual_range) &&',
    '    range_lp &&(isfinite(range_time) || range_nodes != typemax(Int) || !range_mip || !dual_range) &&   # 패치: dd 허용')
rep('    n_cap = Threads.Atomic{Int}(0)',
    '    n_cap = Threads.Atomic{Int}(0)\n'
    '    n_ddnode = Threads.Atomic{Int}(0); n_ddmip = Threads.Atomic{Int}(0); n_ddempty = Threads.Atomic{Int}(0); n_ddcut = Threads.Atomic{Int}(0)   # 패치 (screen)\n'
    '    n_ddelig = Threads.Atomic{Int}(0); n_ddskip = Threads.Atomic{Int}(0)   # 패치 (dd_adapt)')
# (2) :full — 꼭짓점 범위 뒤 DD 전부
rep('''                    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=comp, eq=eqp, dlo=dlo, dhi=dhi)
                    fixfrac = count(hi .- lo .<= 1e-4) / max(1, length(lo))
                elseif dual_range && !(range_mip''',
    '''                    if config.dd && dd_mode == :full          # 패치 (full): 꼭짓점 범위 뒤 상자별 쌍대 차이 범위 (KKT-MIP 전부), 자식에 물려줌
                        wit = Dict{Tuple{Int,Int,Bool},Vector{Float64}}()
                        tg2 = @elapsed ((dlo, dhi) = _nz_dr_dranges!(DR, nd, l, u, lo, hi, ext.dlo, ext.dhi, witp, wit; deadline=t_end))
                        Threads.atomic_add!(t_range, tg2); tn_rg += tg2
                    end
                    _nz_dr_set!(P, DR, nd, l, u, lo, hi, phib; comp=comp, eq=eqp, dlo=dlo, dhi=dhi)
                    fixfrac = count(hi .- lo .<= 1e-4) / max(1, length(lo))
                elseif dual_range && !(range_mip''')
# (3) :screen — 노드 LP 뒤 선별 DD
rep('''                    tr = pre_inf ? 0.0 : @elapsed (z = zpad((sticky ? _nz_sticky_solve! : _nz_sep_solve!)(P; maxrounds=(ubp == Inf ? maxrounds : child_rounds), maxadd=sep_maxadd)))
                    pre_inf && (z = -Inf)
''', '''                    tr = pre_inf ? 0.0 : @elapsed (z = zpad((sticky ? _nz_sticky_solve! : _nz_sep_solve!)(P; maxrounds=(ubp == Inf ? maxrounds : child_rounds), maxadd=sep_maxadd)))
                    pre_inf && (z = -Inf)
                    dd_empty = false                          # 패치 (screen): 가지치기 못 한 노드에서 선별 DD → 상자가 비었거나 조여지면 다시 풂
                    if config.dd && dd_mode == :screen && range_lp && !pre_inf && ubp != Inf && isfinite(z) && z > LB[] + tol(LB[]) &&
                       maximum((u[i] - l[i] for i in 1:nh if nd.hU[i] > 1e-9); init=0.0) <= dd_maxw &&
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
                            z = z2
                        end
                    end
''')
rep('''                    if z == -Inf && dual_range
                        fk = pre_inf''', '''                    if z == -Inf && dual_range && !dd_empty      # 패치: DD MIP 가 상자를 비웠으면 (KKT-MIP 실행 불가능) 재확인할 LP 가 없음
                        fk = pre_inf''')
rep(':gc_wait', ':dd_screen => (nodes=n_ddnode[], mips=n_ddmip[], empty=n_ddempty[], cut=n_ddcut[], skip=n_ddskip[]), :gc_wait', cnt=body.count(':gc_wait'))

helper = r'''
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
                _nz_dr_opt!(DR, M, l, u, "dd screen s=$s 쌍 ($(DR.Hk[a]), $(DR.Hk[b]))") || continue
                st = termination_status(M.model)
                if sigbd && st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)   # 경계를 빼고 다시 풀어 확인 (허용오차 거짓 판정 방지)
                    _abb_sigbd!(M, DR, nd, nothing, nothing, s); nmip += 1
                    ok = _nz_dr_opt!(DR, M, l, u, "dd screen 재확인 s=$s")
                    sigbd && _abb_sigbd!(M, DR, nd, lo, hi, s)
                    ok || continue
                    st = termination_status(M.model)
                end
                if st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)       # 상자 안에 recourse 최적 쌍대를 갖는 α 없음
                    set_time_limit_sec(M.model, nothing); sigbd && _abb_sigbd!(M, DR, nd, nothing, nothing, s)
                    return :empty, (dlo, dhi), nmip
                end
                bd = st == MOI.OPTIMAL ? objective_bound(M.model) : (st == MOI.TIME_LIMIT ? _nz_dr_bound_or_nan(M.model) : NaN)
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
'''

hdr = ('# abb_dd_patch.jl — gen_abb_dd_patch.py 가 만든 파일 (손으로 고치지 말 것). 저장소 nz_alpha_bnb_dr 의 재정의:\n'
       '#   꼭짓점 LP 범위 (range_lp) 와 쌍대 차이 범위 (dd = true; dd_mode :screen 기본, :full) 를 함께 쓸 수 있게 함. 나머지는 저장소 판 그대로.\n'
       'println(stderr, "[abb_dd_patch] nz_alpha_bnb_dr 재정의: range_lp + dd 허용 (실험용)")\n')
open(os.path.join(here, 'abb_dd_patch.jl'), 'w', encoding='utf-8').write(hdr + helper + '\n' + body + '\n')
print('ok', start + 1, end + 1)
