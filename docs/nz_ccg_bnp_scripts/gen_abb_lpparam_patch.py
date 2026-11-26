# -*- coding: utf-8 -*-
# gen_abb_lpparam_patch.py — 노드 LP 모델 (worker 의 분리 LP) 에만 Gurobi 설정을 더 주는 재정의 abb_lpparam_patch.jl 을 만든다
#   (2026-11-27, α-B&B 완화 LP 튜닝 실험용. 저장소 파일은 고치지 않음). 꼭짓점 LP·α̂ 평가·부분 LP 복사본은 그대로가 아님에 주의:
#   부분 LP (infeas_pre) 는 노드 LP 모델을 복사하므로 같은 설정을 물려받는다.
#   인자 lp_params (예 ["ScaleFlag" => 2]) 를 노드 LP 모델 생성 직후 넣는다. 수치 오류 재시도 (_nz_solve_lp!) 는 NumericFocus·Method·Crossover·Presolve 만 되돌리므로
#   여기 준 설정은 재시도에서도 유지된다.
# 사용: python gen_abb_lpparam_patch.py → abb_lpparam_patch.jl. orcl_replay.jl 은 ABB_PATCH 와 KW="(lp_params = [\"ScaleFlag\" => 2],)" 로.
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
body = rep(body, '                         corner_reuse::Bool=true, ev_adapt::Bool=true, ev_warm::Int=300, ev_probe::Int=10,\n',
           '                         corner_reuse::Bool=true, ev_adapt::Bool=true, ev_warm::Int=300, ev_probe::Int=10,\n'
           '                         lp_params=Pair{String,Any}[],   # 패치 (lpparam): 노드 LP 모델에 더 줄 Gurobi 설정\n')
body = rep(body, '    farkas && for P in Ps; set_optimizer_attribute(P.O.model, "InfUnbdInfo", 1); end   # Farkas 쌍대를 받기 위해\n',
           '    farkas && for P in Ps; set_optimizer_attribute(P.O.model, "InfUnbdInfo", 1); end   # Farkas 쌍대를 받기 위해\n'
           '    for P in Ps, (k, v) in lp_params; set_optimizer_attribute(P.O.model, k, v); end      # 패치 (lpparam)\n'
           '    isempty(lp_params) || println(stderr, "abb_lpparam_patch: 노드 LP 설정 ", join(["$k = $v" for (k, v) in lp_params], ", "))\n')
hdr = ('# abb_lpparam_patch.jl — gen_abb_lpparam_patch.py 가 저장소 nonzero_sum/nz_alpha_bnb_dr.jl 에서 만든 재정의 (손으로 고치지 말 것, 2026-11-27 실험용).\n'
       '#  노드 LP 모델에 Gurobi 설정 추가 (lp_params). docs/nz_ccg_bnp_2026-11-23.md 11.22.\n'
       'println(stderr, "abb_lpparam_patch: nz_alpha_bnb_dr 재정의 (lp_params)")\n')
open(os.path.join(here, 'abb_lpparam_patch.jl'), 'w', encoding='utf-8', newline='\n').write(hdr + body + '\n')
print('ok')
