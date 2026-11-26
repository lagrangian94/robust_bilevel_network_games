# -*- coding: utf-8 -*-
# gen_abb_cutoff_patch.py — 노드 LP 에 incumbent cutoff 를 주는 재정의 파일 abb_cutoff_patch.jl 을 만든다 (2026-11-27, α-B&B 완화 LP 튜닝 실험용.
#   저장소 파일은 고치지 않음). 저장소 nonzero_sum/nz_alpha_bnb.jl 의 _nz_solve_lp!, nz_alpha_bnb_dr.jl 의 _nz_sticky_solve!·nz_alpha_bnb_dr 를 떼어 고친다.
#   근거 (md 11.22): 지리 5 막힌 점 노드 LP 시간의 52% 가 결국 가지친 노드 (노드당 반복 4,046 vs 분기 노드 1,996).
#   방법: 루트가 아닌 노드 LP 전에 Gurobi Cutoff = c (c 는 zpad(c) < LB + tol(LB) 가 되게 약간 아래). dual simplex 는 목적값 (최대화의 상한) 이
#     c 아래로 내려가면 멈추고 CUTOFF (MOI.OBJECTIVE_LIMIT) — 그 노드의 LP 최적값 < c 가 증명된 것이므로 가지치기 (상한은 c 로 기록, 실제 값 이상이라 유효).
#     분리 라운드 사이에서도 같은 규칙 (첫 LP 가 cutoff 면 둘째 LP 없음). infeas_pre 부분 LP·실행 불가능 재확인 전에는 cutoff 를 끔.
# 사용: python gen_abb_cutoff_patch.py → abb_cutoff_patch.jl. orcl_replay.jl·ccgbp.jl 은 ABB_PATCH 로. 인자 lp_cutoff (기본 true) 로 끔.
import os
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(here, '..', '..', 'nonzero_sum')

def grab(path, head):
    L = open(path, encoding='utf-8').read().split('\n')
    s = next(i for i, l in enumerate(L) if l.startswith(head))
    e = next(i for i in range(s, len(L)) if L[i] == 'end')
    return '\n'.join(L[s:e + 1])

def rep(txt, a, b, cnt=1):
    n = txt.count(a)
    if n != cnt:
        raise SystemExit(f'count {n} != {cnt}: {a[:90]!r}')
    return txt.replace(a, b)

# ---- _nz_solve_lp!: CUTOFF 상태
slp = grab(os.path.join(root, 'nz_alpha_bnb.jl'), 'function _nz_solve_lp!(model)')
slp = rep(slp, '            st in (MOI.OPTIMAL, MOI.INFEASIBLE, MOI.TIME_LIMIT) && Threads.atomic_add!(_NZ_FRESH_OK, 1)',
          '            st in (MOI.OPTIMAL, MOI.INFEASIBLE, MOI.TIME_LIMIT, MOI.OBJECTIVE_LIMIT) && Threads.atomic_add!(_NZ_FRESH_OK, 1)')
slp = rep(slp, '        st in (MOI.OPTIMAL, MOI.INFEASIBLE, MOI.TIME_LIMIT) || for (nf, meth) in ((3, 1), (3, 2))',
          '        st in (MOI.OPTIMAL, MOI.INFEASIBLE, MOI.TIME_LIMIT, MOI.OBJECTIVE_LIMIT) || for (nf, meth) in ((3, 1), (3, 2))')
slp = rep(slp, '            st in (MOI.OPTIMAL, MOI.INFEASIBLE, MOI.TIME_LIMIT) && break',
          '            st in (MOI.OPTIMAL, MOI.INFEASIBLE, MOI.TIME_LIMIT, MOI.OBJECTIVE_LIMIT) && break')
slp = rep(slp, '        st in (MOI.OPTIMAL, MOI.INFEASIBLE, MOI.TIME_LIMIT) || (Threads.atomic_add!(_NZ_NUMERR_FAIL, 1); return Inf)',
          '        st in (MOI.OPTIMAL, MOI.INFEASIBLE, MOI.TIME_LIMIT, MOI.OBJECTIVE_LIMIT) || (Threads.atomic_add!(_NZ_NUMERR_FAIL, 1); return Inf)')
slp = rep(slp, '    st == MOI.INFEASIBLE && return -Inf\n',
          '    st == MOI.INFEASIBLE && return -Inf\n'
          '    # 패치 (abb_cutoff): CUTOFF = LP 최적값이 model.ext[:cutoff] 보다 작음이 증명됨 → 그 값을 (실제 값 이상의) 상한으로\n'
          '    st == MOI.OBJECTIVE_LIMIT && return (haskey(model.ext, :cutoff) ? model.ext[:cutoff] : error("nz_alpha_bnb LP: OBJECTIVE_LIMIT 인데 cutoff 값 없음"))\n')

# ---- _nz_sticky_solve!: cutoff 면 바로 돌려줌
sts = grab(os.path.join(root, 'nz_alpha_bnb_dr.jl'), 'function _nz_sticky_solve!(')
sts = rep(sts, '        (isnan(z) || isinf(z)) && return z\n',
          '        (isnan(z) || isinf(z)) && return z\n'
          '        termination_status(P.O.model) == MOI.OBJECTIVE_LIMIT && return z     # 패치 (abb_cutoff): 해 없음, 분리 안 함\n')

# ---- nz_alpha_bnb_dr
L = open(os.path.join(root, 'nz_alpha_bnb_dr.jl'), encoding='utf-8').read().split('\n')
s = next(i for i, l in enumerate(L) if l.startswith('function nz_alpha_bnb_dr(nd::NZData, x̄;'))
e = max(i for i, l in enumerate(L) if l == 'end')
body = '\n'.join(L[s:e + 1])
body = rep(body, '                         corner_reuse::Bool=true, ev_adapt::Bool=true, ev_warm::Int=300, ev_probe::Int=10,\n',
           '                         corner_reuse::Bool=true, ev_adapt::Bool=true, ev_warm::Int=300, ev_probe::Int=10,\n'
           '                         lp_cutoff::Bool=true,   # 패치 (abb_cutoff): 루트가 아닌 노드 LP 에 incumbent cutoff\n')
body = rep(body, '    n_cap = Threads.Atomic{Int}(0)                                           # node_lp_cap 에 걸린 노드\n',
           '    n_cap = Threads.Atomic{Int}(0)                                           # node_lp_cap 에 걸린 노드\n'
           '    n_cut = Threads.Atomic{Int}(0)                                           # 패치 (abb_cutoff): cutoff 로 끝난 노드\n')
body = rep(body, '                    ndl = ubp == Inf ? t_end : min(t_end, time() + node_lp_cap); P.O.model.ext[:deadline] = ndl\n',
           '                    ndl = ubp == Inf ? t_end : min(t_end, time() + node_lp_cap); P.O.model.ext[:deadline] = ndl\n'
           '                    lp_cutoff && set_optimizer_attribute(P.O.model, "Cutoff", -1e100)   # 패치 (abb_cutoff): 부분 LP (모델 복사) 전에는 끔\n')
body = rep(body, '                    rtbas !== nothing && !pre_inf && (bload = _nz_dr_set_basis!(P, rtbas))\n',
           '                    rtbas !== nothing && !pre_inf && (bload = _nz_dr_set_basis!(P, rtbas))\n'
           '                    cut_on = lp_cutoff && ubp != Inf && !pre_inf && isfinite(LB[])   # 패치 (abb_cutoff): c 는 zpad(c) < LB + tol(LB) 가 되는 값\n'
           '                    if cut_on\n'
           '                        thr = LB[] + tol(LB[]); cval = thr - 3e-6 * max(1.0, abs(thr))\n'
           '                        P.O.model.ext[:cutoff] = cval; set_optimizer_attribute(P.O.model, "Cutoff", cval)\n'
           '                    end\n')
body = rep(body, '                    pre_inf && (z = -Inf)\n',
           '                    pre_inf && (z = -Inf)\n'
           '                    cut_on && !pre_inf && termination_status(P.O.model) == MOI.OBJECTIVE_LIMIT && Threads.atomic_add!(n_cut, 1)\n')
body = rep(body, '                    if z == -Inf && dual_range\n                        fk = pre_inf',
           '                    if z == -Inf && dual_range\n'
           '                        lp_cutoff && set_optimizer_attribute(P.O.model, "Cutoff", -1e100)   # 패치 (abb_cutoff): 재확인은 cutoff 없이\n'
           '                        fk = pre_inf')
body = rep(body, ':eff => (corner_skip=n_cskip[], ev=n_ev[], ev_skip=n_evskip[]),',
           ':eff => (corner_skip=n_cskip[], ev=n_ev[], ev_skip=n_evskip[]), :lp_cutoff => n_cut[],')

hdr = ('# abb_cutoff_patch.jl — gen_abb_cutoff_patch.py 가 저장소 nonzero_sum/ 에서 만든 재정의 (손으로 고치지 말 것, 2026-11-27 실험용).\n'
       '#  노드 LP incumbent cutoff (lp_cutoff). docs/nz_ccg_bnp_2026-11-23.md 11.22.\n'
       'println(stderr, "abb_cutoff_patch: _nz_solve_lp!·_nz_sticky_solve!·nz_alpha_bnb_dr 재정의 (lp_cutoff)")\n')
open(os.path.join(here, 'abb_cutoff_patch.jl'), 'w', encoding='utf-8', newline='\n').write(hdr + slp + '\n\n' + sts + '\n\n' + body + '\n')
print('ok')
