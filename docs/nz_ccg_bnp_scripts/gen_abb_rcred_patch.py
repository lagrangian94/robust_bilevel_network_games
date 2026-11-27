# -*- coding: utf-8 -*-
# gen_abb_rcred_patch.py — 노드 LP 의 α 경계 쌍대 (reduced cost) 로 분기 전에 상자를 줄이는 재정의 abb_rcred_patch.jl 을 만든다
#   (2026-11-27, α-B&B 노드 수 튜닝 실험용. 저장소 파일은 고치지 않음). 실험실 front_rcred.jl 과 같은 규칙.
#  근거 (Ryoo & Sahinidis 1996, marginals-based range reduction): 노드 LP R(B) 의 값 z_LP, α_k ≤ u_k 경계의 쌍대 μ > 0 이면 R(B) 에서 그 경계만
#   α_k ≤ t 로 바꾼 LP 는 같은 쌍대해가 실행 가능이라 값 ≤ z_LP − μ (u_k − t). R(B) 의 다른 행은 B 의 부분 상자에서 유효하므로 B ∩ {α_k ≤ t} 의 상한.
#   그래서 t ≤ u_k − (z_LP − 문턱)/μ 인 부분은 문턱 아래 → 자식 상자의 l_k 를 그 값으로 올림 (아래 경계에 있는 좌표는 u_k 를 내림).
#   z_LP 는 zpad 를 더한 노드 LP 값 (부모 상한과의 min 을 하기 전). 문턱 = LB + tol(LB) (그 순간 값, LB 는 늘기만 하므로 보수적).
#   곱 위반·상보성 분기·분할점은 줄인 상자에서 고른다. 꼭짓점 재사용 (corner_reuse) 은 그 쪽 꼭짓점이 바뀌지 않았을 때만.
# 사용: python gen_abb_rcred_patch.py → abb_rcred_patch.jl. orcl_replay.jl 은 ABB_PATCH 로. 인자 rc_reduce (기본 true) 로 끔.
import os
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(here, '..', '..', 'nonzero_sum')

def rep(txt, a, b, cnt=1):
    n = txt.count(a)
    if n != cnt:
        raise SystemExit(f'count {n} != {cnt}: {a[:90]!r}')
    return txt.replace(a, b)

L = open(os.path.join(root, 'nz_alpha_bnb_dr.jl'), encoding='utf-8').read().split('\n')
s = next(i for i, l in enumerate(L) if l.startswith('function nz_alpha_bnb_dr(nd::NZData, x̄;'))
e = max(i for i, l in enumerate(L) if l == 'end')
body = '\n'.join(L[s:e + 1])
body = rep(body, '                         lp_scale::Int=2,\n',
           '                         lp_scale::Int=2,\n'
           '                         rc_reduce::Bool=true,   # 패치 (rcred): 분기 전에 α 경계 쌍대로 상자를 줄임\n')
body = rep(body, '    n_cap = Threads.Atomic{Int}(0)                                           # node_lp_cap 에 걸린 노드\n',
           '    n_cap = Threads.Atomic{Int}(0)                                           # node_lp_cap 에 걸린 노드\n'
           '    n_rc = Threads.Atomic{Int}(0); n_rck = Threads.Atomic{Int}(0); n_rcfr = Threads.Atomic{Float64}(0.0)   # 패치 (rcred): 줄인 노드 / 좌표 / 줄인 폭 비율 합\n')
body = rep(body, '                    pre_inf && (z = -Inf)\n',
           '                    pre_inf && (z = -Inf)\n'
           '                    zlp_rc = z                                                  # 패치 (rcred): 이 노드 LP 의 값 (zpad 포함, 부모 상한과 min 전)\n')
body = rep(body, '                        α̂ = clamp.(value.(v[:α]), l, u)\n',
           '                        α̂ = clamp.(value.(v[:α]), l, u)\n'
           '                        lch = false; uch = false                                 # 패치 (rcred): 이 노드에서 l / u 를 줄였는가\n'
           '                        if rc_reduce && isfinite(zlp_rc) && termination_status(P.O.model) == MOI.OPTIMAL && has_duals(P.O.model)\n'
           '                            thr_rc = LB[] + tol(LB[]); gap_rc = (zlp_rc - thr_rc) * (1 + 1e-6) + 1e-6 * max(1.0, abs(zlp_rc))\n'
           '                            av_rc = value.(v[:α]); rcv = reduced_cost.(v[:α]); l = copy(l); u = copy(u); nrc = 0; frc = 0.0\n'
           '                            for k in 1:nh\n'
           '                                w_rc = u[k] - l[k]; μ_rc = abs(rcv[k]); (w_rc > min_width && μ_rc > 1e-9) || continue\n'
           '                                if av_rc[k] >= u[k] - 1e-9 * max(1.0, w_rc)\n'
           '                                    t_rc = u[k] - gap_rc / μ_rc; t_rc > l[k] && (frc += (t_rc - l[k]) / w_rc; l[k] = t_rc; nrc += 1; lch = true)\n'
           '                                elseif av_rc[k] <= l[k] + 1e-9 * max(1.0, w_rc)\n'
           '                                    t_rc = l[k] + gap_rc / μ_rc; t_rc < u[k] && (frc += (u[k] - t_rc) / w_rc; u[k] = t_rc; nrc += 1; uch = true)\n'
           '                                end\n'
           '                            end\n'
           '                            nrc > 0 && (Threads.atomic_add!(n_rc, 1); Threads.atomic_add!(n_rck, nrc); Threads.atomic_add!(n_rcfr, frc); α̂ = clamp.(α̂, l, u))\n'
           '                        end\n')
# 자식의 꼭짓점 재사용 표시: 바뀐 쪽은 재사용하지 않음
body = rep(body, 'children = ((copy(l), u1, z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, bas=basP, sl=true)), (copy(l), copy(u), z, lo, hi, wit, phib, (eq=eqB, dlo=dlo, dhi=dhi, bas=basP, side=2, sl=true, su=true)))',
           'children = ((copy(l), u1, z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, bas=basP, sl=!lch)), (copy(l), copy(u), z, lo, hi, wit, phib, (eq=eqB, dlo=dlo, dhi=dhi, bas=basP, side=2, sl=!lch, su=!uch)))')
body = rep(body, 'children = ((copy(l), u1, z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, bas=basP, sl=true)), (l2, copy(u), z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, bas=basP, side=2, su=true)))',
           'children = ((copy(l), u1, z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, bas=basP, sl=!lch)), (l2, copy(u), z, lo, hi, wit, phib, (eq=eqp, dlo=dlo, dhi=dhi, bas=basP, side=2, su=!uch)))')
body = rep(body, ':eff => (corner_skip=n_cskip[], ev=n_ev[], ev_skip=n_evskip[]),',
           ':eff => (corner_skip=n_cskip[], ev=n_ev[], ev_skip=n_evskip[]), :rc_reduce => (nodes=n_rc[], coords=n_rck[], frac=n_rcfr[] / max(1, n_rck[])),')
hdr = ('# abb_rcred_patch.jl — gen_abb_rcred_patch.py 가 저장소 nonzero_sum/nz_alpha_bnb_dr.jl 에서 만든 재정의 (손으로 고치지 말 것, 2026-11-27 실험용).\n'
       '#  분기 전 α 경계 쌍대로 상자 줄이기 (rc_reduce). docs/nz_ccg_bnp_2026-11-23.md 11.23.\n'
       'println(stderr, "abb_rcred_patch: nz_alpha_bnb_dr 재정의 (rc_reduce)")\n')
open(os.path.join(here, 'abb_rcred_patch.jl'), 'w', encoding='utf-8', newline='\n').write(hdr + body + '\n')
print('ok')
