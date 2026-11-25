# gen_abb_dd_patch.py — 저장소 nonzero_sum/nz_alpha_bnb_dr.jl 의 nz_alpha_bnb_dr 함수만 떼어 "꼭짓점 LP 범위 + 쌍대 차이 범위 (DD)" 를
# 허용하도록 고친 재정의 파일 abb_dd_patch.jl 을 만든다 (2026-11-25 밤, 실험용. 저장소 파일은 고치지 않음).
#   기본 판은 range_lp = true 이면 dd = false 여야 함 (L6 에서 범위 MIP 가 노드 시간을 다 써서). 지리 5 전선 진단에서
#   상자 DD 가 상한 큰 상자를 모두 실행 불가능으로 만들었음 (front_dd.jl, 재확인 통과, incumbent 상자 유효성 통과). 루트 DD 한 번은 효과 없음.
#   dd_mode = :full   꼭짓점 범위 뒤 두 성분 모두 안 고정인 쌍 전부를 KKT-MIP 로 (_nz_dr_dranges!). 상자당 150 ~ 190 s.
#   dd_mode = :screen 노드 LP 로 가지치기 못 한 노드에서만, 두 꼭짓점의 최적 쌍대 면에서 차이가 모인 (쌍, s) 를 골라 KKT-MIP 로 인증하고 다시 풂
#                     (front_dd_lazy.jl SCREEN=corner: 상자 8 개 모두 가지치기, 상자당 약 18 s). MIP 하나 dd_mip_tl 초, 시간 상한이면 objective_bound.
#   유효성: 범위 MIP 판의 DD 와 같은 논증 (상자 안 α 의 recourse 최적 쌍대의 차이 범위). 고른 쌍만 조이는 것은 덜 조일 뿐.
#   자식은 조인 dlo·dhi 를 물려받음 (자식 상자 ⊆ 부모 상자).
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
    '                         dd_mode::Symbol=:screen, dd_mip_tl::Float64=2.0)')
# (1) range_lp 와 dd 함께 허용
rep('    range_lp &&(dd || isfinite(range_time) || range_nodes != typemax(Int) || !range_mip || !dual_range) &&',
    '    range_lp &&(isfinite(range_time) || range_nodes != typemax(Int) || !range_mip || !dual_range) &&   # 패치: dd 허용')
rep('    n_cap = Threads.Atomic{Int}(0)',
    '    n_cap = Threads.Atomic{Int}(0)\n'
    '    n_ddnode = Threads.Atomic{Int}(0); n_ddmip = Threads.Atomic{Int}(0); n_ddempty = Threads.Atomic{Int}(0); n_ddcut = Threads.Atomic{Int}(0)   # 패치 (screen)')
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
                    if config.dd && dd_mode == :screen && range_lp && !pre_inf && ubp != Inf && isfinite(z) && z > LB[] + tol(LB[])
                        tdd = @elapsed (ddr = _abb_dd_screen!(CLs[wid], DR, nd, x̄, l, u, lo, hi, dlo, dhi; mip_tl=dd_mip_tl, deadline=t_end))
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
rep(':gc_wait', ':dd_screen => (nodes=n_ddnode[], mips=n_ddmip[], empty=n_ddempty[], cut=n_ddcut[]), :gc_wait', cnt=body.count(':gc_wait'))

helper = r'''
"패치 (screen): 두 꼭짓점의 최적 쌍대 면에서 차이가 모인 (쌍, s) 를 골라 KKT-MIP 로 차이 범위를 인증. 반환 (:none | :tight | :empty, (dlo, dhi), MIP 수)"
function _abb_dd_screen!(CL, DR, nd, x̄, l, u, lo, hi, dlo0, dhi0; mip_tl=2.0, deadline=Inf, fixtol=1e-4)
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
                boxset = true
            end
            for ismax in (false, true)
                if time() > deadline
                    set_time_limit_sec(M.model, nothing)
                    return (tight ? :tight : :none), (dlo, dhi), nmip
                end
                @objective(M.model, ismax ? MAX_SENSE : MIN_SENSE, M.σ[DR.Hk[a]] - M.σ[DR.Hk[b]])
                set_time_limit_sec(M.model, max(0.01, min(mip_tl, deadline - time())))
                nmip += 1
                _nz_dr_opt!(DR, M, l, u, "dd screen s=$s 쌍 ($(DR.Hk[a]), $(DR.Hk[b]))") || continue
                st = termination_status(M.model)
                if st in (MOI.INFEASIBLE, MOI.INFEASIBLE_OR_UNBOUNDED)       # 상자 안에 recourse 최적 쌍대를 갖는 α 없음
                    set_time_limit_sec(M.model, nothing)
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
    end
    return (tight ? :tight : :none), (dlo, dhi), nmip
end
'''

hdr = ('# abb_dd_patch.jl — gen_abb_dd_patch.py 가 만든 파일 (손으로 고치지 말 것). 저장소 nz_alpha_bnb_dr 의 재정의:\n'
       '#   꼭짓점 LP 범위 (range_lp) 와 쌍대 차이 범위 (dd = true; dd_mode :screen 기본, :full) 를 함께 쓸 수 있게 함. 나머지는 저장소 판 그대로.\n'
       'println(stderr, "[abb_dd_patch] nz_alpha_bnb_dr 재정의: range_lp + dd 허용 (실험용)")\n')
open(os.path.join(here, 'abb_dd_patch.jl'), 'w', encoding='utf-8').write(hdr + helper + '\n' + body + '\n')
print('ok', start + 1, end + 1)
