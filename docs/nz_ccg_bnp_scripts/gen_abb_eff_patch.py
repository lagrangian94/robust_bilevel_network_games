# -*- coding: utf-8 -*-
# gen_abb_eff_patch.py — 저장소 nonzero_sum/nz_alpha_bnb_dr.jl 의 nz_alpha_bnb_dr 와 _nz_dr_corner_ranges 를 떼어
#   노드 비용을 줄이는 두 가지를 넣은 재정의 파일 abb_eff_patch.jl 을 만든다 (2026-11-27, α-B&B 튜닝 실험용. 저장소 파일은 고치지 않음).
#   근거 (md 11.21): 지리 5 막힌 점 300 s 의 worker 시간 = 완화 LP 51%, 꼭짓점 범위 23%, α̂ 정확 평가 16%. 전선 상자는 몇 노드면 죽지만 열린 상자가 4,041 개.
#   (1) 꼭짓점 재사용 (corner_reuse, 기본 켬): 상자 [l, u] 의 범위는 hi 를 꼭짓점 α = l, lo 를 α = u 의 LP 로 구한다. 왼쪽 자식 [l, u′] 은 부모와 l 이,
#       오른쪽 자식 [l′, u] 는 u 가 같으므로 그 꼭짓점 LP 는 부모와 같은 계산 → 건너뛰고 부모 범위를 그대로 (결과 같음). 상보성 분기의 둘째 자식은 상자가 같아 둘 다 건너뜀.
#       점검: 환경 변수 NZ_EFF_CHECK=1 이면 건너뛴 노드에서도 전부 계산해 비교하고, 1e-5 넘게 다르면 오류.
#   (2) α̂ 평가 적응 (ev_adapt, 기본 켬): 노드마다 하는 α̂ 정확 평가 (incumbent 찾기, resp 반응 평가 포함) 가 ev_warm 번 연속 incumbent 를 못 바꾸면
#       ev_probe 노드에 한 번만 평가. incumbent 가 바뀌면 처음으로. 평가는 하한 (incumbent) 에만 쓰여 상한·가지치기 유효성과 무관.
# 사용: python gen_abb_eff_patch.py → abb_eff_patch.jl. nz_benders.jl 을 include 한 뒤 include. orcl_replay.jl·ccgbp.jl 은 ABB_PATCH 로.
import os
here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, '..', '..', 'nonzero_sum', 'nz_alpha_bnb_dr.jl')
L = open(src, encoding='utf-8').read().split('\n')
start = next(i for i, l in enumerate(L) if l.startswith('function nz_alpha_bnb_dr(nd::NZData, x̄;'))
end = max(i for i, l in enumerate(L) if l == 'end')
assert end > start
body = '\n'.join(L[start:end + 1])
cs = next(i for i, l in enumerate(L) if l.startswith('function _nz_dr_corner_ranges('))
ce = next(i for i in range(cs, len(L)) if L[i] == 'end')
corner = '\n'.join(L[cs:ce + 1])

def rep(txt, a, b, cnt=1):
    n = txt.count(a)
    if n != cnt:
        raise SystemExit(f'count {n} != {cnt}: {a[:90]!r}')
    return txt.replace(a, b)

# ---- (1) 꼭짓점 함수: 건너뛸 꼭짓점 인자
corner = rep(corner, 'function _nz_dr_corner_ranges(CL::Vector{_NZCornerLP}, DR::_NZDR, nd::NZData, x̄, l, u, lo0, hi0; fixtol=1e-4)',
             'function _nz_dr_corner_ranges(CL::Vector{_NZCornerLP}, DR::_NZDR, nd::NZData, x̄, l, u, lo0, hi0; fixtol=1e-4, skip_l::Bool=false, skip_u::Bool=false)\n'
             '    # 패치 (abb_eff_patch): skip_l / skip_u 이면 그 꼭짓점 (α = l 은 hi·ν 최대, α = u 는 lo·ν 최소) 의 LP 를 건너뛰고 lo0·hi0 를 그대로 둔다\n'
             '    skip_l && skip_u && return copy(lo0), copy(hi0)')
corner = rep(corner, '        for (αc, ismax) in ((u, false), (l, true))\n',
             '        for (αc, ismax) in ((u, false), (l, true))\n'
             '            ((ismax && skip_l) || (!ismax && skip_u)) && continue\n')

# ---- 인자
body = rep(body, '                         resume=nothing)',
           '                         resume=nothing,\n'
           '                         # 패치 (abb_eff_patch): corner_reuse = 자식이 부모와 공유하는 꼭짓점의 범위 LP 를 건너뜀 (결과 같음)\n'
           '                         #   ev_adapt = α̂ 정확 평가가 ev_warm 번 연속 incumbent 를 못 바꾸면 ev_probe 노드에 한 번만\n'
           '                         corner_reuse::Bool=true, ev_adapt::Bool=true, ev_warm::Int=300, ev_probe::Int=10)')
body = rep(body, '    n_cap = Threads.Atomic{Int}(0)',
           '    n_ev = Threads.Atomic{Int}(0); n_evskip = Threads.Atomic{Int}(0); n_evsince = Threads.Atomic{Int}(0); n_evtick = Threads.Atomic{Int}(0)   # 패치: 평가 적응\n'
           '    n_cskip = Threads.Atomic{Int}(0); effcheck = get(ENV, "NZ_EFF_CHECK", "0") == "1"                                                  # 패치: 꼭짓점 재사용\n'
           '    n_cap = Threads.Atomic{Int}(0)')
# 꼭짓점 호출 (첫 호출만, resp_fix 의 두 번째 호출은 그대로)
body = rep(body, '                    tg = @elapsed ((lo, hi) = _nz_dr_corner_ranges(CLs[wid], DR, nd, x̄, l, u, lop, hip))\n',
           '                    skl = corner_reuse && get(ext, :sl, false); sku = corner_reuse && get(ext, :su, false)\n'
           '                    (skl || sku) && Threads.atomic_add!(n_cskip, 1)\n'
           '                    tg = @elapsed ((lo, hi) = _nz_dr_corner_ranges(CLs[wid], DR, nd, x̄, l, u, lop, hip; skip_l=skl, skip_u=sku))\n'
           '                    if effcheck && (skl || sku)\n'
           '                        lof, hif = _nz_dr_corner_ranges(CLs[wid], DR, nd, x̄, l, u, lop, hip)\n'
           '                        dmax = max(maximum(abs.(lof .- lo); init=0.0), maximum(abs.(hif .- hi); init=0.0))\n'
           '                        dmax <= 1e-5 || error("abb_eff_patch: 꼭짓점 재사용 범위가 전부 계산한 것과 $(dmax) 다름 (skip_l=$skl, skip_u=$sku)")\n'
           '                    end\n')
# 자식에 공유 꼭짓점 표시: 수치 오류 분할
body = rep(body, '                            children = ((copy(l), u1, z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi)), (l2, copy(u), z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi)))',
           '                            children = ((copy(l), u1, z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, sl=true)), (l2, copy(u), z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, su=true)))')
# 상보성 분기: 첫째 [l, u1] (l 같음), 둘째 같은 상자
body = rep(body, '                            children = ((copy(l), u1, z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, bas=basP)), (copy(l), copy(u), z, lo, hi, wit, phib, (eq=eqB, dlo=dlo, dhi=dhi, bas=basP, side=2)))',
           '                            children = ((copy(l), u1, z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, bas=basP, sl=true)), (copy(l), copy(u), z, lo, hi, wit, phib, (eq=eqB, dlo=dlo, dhi=dhi, bas=basP, side=2, sl=true, su=true)))')
# 중점 분할
body = rep(body, '                            children = ((copy(l), u1, z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, bas=basP)), (l2, copy(u), z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, bas=basP, side=2)))',
           '                            children = ((copy(l), u1, z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, bas=basP, sl=true)), (l2, copy(u), z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, bas=basP, side=2, su=true)))')
# ---- (2) 평가 적응
body = rep(body, '                        te = @elapsed (zE = nz_eval_alpha!(F, nd, α̂c))\n',
           '                        doev = !ev_adapt || n_evsince[] < ev_warm || Threads.atomic_add!(n_evtick, 1) % ev_probe == 0   # 패치: 평가 적응\n'
           '                        doev ? Threads.atomic_add!(n_ev, 1) : Threads.atomic_add!(n_evskip, 1)\n'
           '                        te = @elapsed (zE = doev ? nz_eval_alpha!(F, nd, α̂c) : -Inf)\n')
body = rep(body, '                        if resp\n                            te += @elapsed begin',
           '                        if resp && doev\n                            te += @elapsed begin')
body = rep(body, '                if zE > LB[]\n                    (αE === α̂c || αE === α̂) || Threads.atomic_add!(n_resp, 1)\n                    LB[] = zE; best_α[] = copy(αE)\n                end\n',
           '                if zE > LB[]\n                    (αE === α̂c || αE === α̂) || Threads.atomic_add!(n_resp, 1)\n                    LB[] = zE; best_α[] = copy(αE)\n'
           '                    n_evsince[] = 0                                                   # 패치: incumbent 가 바뀌면 평가 적응을 처음으로\n'
           '                elseif α̂ !== nothing && doev\n'
           '                    Threads.atomic_add!(n_evsince, 1)\n'
           '                end\n')
body = rep(body, ':gc_wait', ':eff => (corner_skip=n_cskip[], ev=n_ev[], ev_skip=n_evskip[]), :gc_wait', cnt=body.count(':gc_wait'))

hdr = ('# abb_eff_patch.jl — gen_abb_eff_patch.py 가 저장소 nonzero_sum/nz_alpha_bnb_dr.jl 에서 만든 재정의 (손으로 고치지 말 것, 2026-11-27 실험용).\n'
       '#  꼭짓점 재사용 (corner_reuse) + α̂ 평가 적응 (ev_adapt). docs/nz_ccg_bnp_2026-11-23.md 11.21.\n'
       'println(stderr, "abb_eff_patch: nz_alpha_bnb_dr 재정의 (corner_reuse, ev_adapt)")\n')
open(os.path.join(here, 'abb_eff_patch.jl'), 'w', encoding='utf-8', newline='\n').write(hdr + corner + '\n\n' + body + '\n')
print('ok')
