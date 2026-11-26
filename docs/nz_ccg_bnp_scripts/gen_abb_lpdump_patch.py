# -*- coding: utf-8 -*-
# gen_abb_lpdump_patch.py — dive 노드 LP 를 시작 basis 와 함께 저장하는 진단용 재정의 abb_lpdump_patch.jl 을 만든다 (2026-11-27, 완화 LP 튜닝 실험실용.
#   저장소 파일은 고치지 않음, 탐색 경로는 그대로이고 파일만 씀).
#   dive 노드는 직전 노드 (부모) 의 마지막 LP 최적 basis 에서 이어 푼다. 노드 시작 (상자·범위 행을 바꾸기 전) 에 그 basis 를 k_start.bas 로,
#   노드 LP 직전에 바뀐 모델을 k.mps 로 쓴다. 실험실 lp_param_lab.jl 이 둘을 읽어 같은 시작점에서 Gurobi 설정별 반복·시간을 잰다.
#   환경 변수: NZ_LPDUMP (폴더, 없으면 아무것도 안 함), NZ_LPDUMP_EVERY (dive 노드 몇 개에 하나, 기본 5), NZ_LPDUMP_MAX (기본 60), NZ_LPDUMP_T0 (이 초 뒤부터, 기본 60)
# 사용: python gen_abb_lpdump_patch.py → abb_lpdump_patch.jl. orcl_replay.jl 은 ABB_PATCH 로.
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
body = rep(body, '    n_cap = Threads.Atomic{Int}(0)                                           # node_lp_cap 에 걸린 노드\n',
           '    n_cap = Threads.Atomic{Int}(0)                                           # node_lp_cap 에 걸린 노드\n'
           '    ddir = get(ENV, "NZ_LPDUMP", ""); d_every = parse(Int, get(ENV, "NZ_LPDUMP_EVERY", "5")); d_max = parse(Int, get(ENV, "NZ_LPDUMP_MAX", "60"))\n'
           '    d_t0 = parse(Float64, get(ENV, "NZ_LPDUMP_T0", "60")); d_ctr = Threads.Atomic{Int}(0); d_n = Threads.Atomic{Int}(0)   # 패치 (lpdump)\n')
# 노드 시작: dive 노드면 부모의 마지막 basis 를 씀
body = rep(body, '            tn0 = time(); tn_rg = 0.0; tn_lp = 0.0; tn_rc = 0.0\n',
           '            tn0 = time(); tn_rg = 0.0; tn_lp = 0.0; tn_rc = 0.0\n'
           '            dbase = ""                                                       # 패치 (lpdump): 이 노드를 저장하면 파일 이름 앞부분\n'
           '            if !isempty(ddir) && fromdive && time() - t0 > d_t0 && d_n[] < d_max && Threads.atomic_add!(d_ctr, 1) % d_every == 0\n'
           '                k = Threads.atomic_add!(d_n, 1) + 1\n'
           '                if k <= d_max\n'
           '                    b0 = joinpath(ddir, "dv_$(k)")\n'
           '                    Gurobi.GRBwrite(JuMP.unsafe_backend(P.O.model), b0 * "_start.bas") == 0 ? (dbase = b0) : rm(b0 * "_start.bas"; force=true)\n'
           '                end\n'
           '            end\n')
body = rep(body, '                    tr = pre_inf ? 0.0 : @elapsed (z = zpad((sticky ? _nz_sticky_solve! : _nz_sep_solve!)(P; maxrounds=(ubp == Inf ? maxrounds : child_rounds), maxadd=sep_maxadd)))\n',
           '                    if !isempty(dbase) && !pre_inf                                   # 패치 (lpdump): 바뀐 모델 (이 노드의 첫 LP)\n'
           '                        Gurobi.GRBupdatemodel(JuMP.unsafe_backend(P.O.model)); Gurobi.GRBwrite(JuMP.unsafe_backend(P.O.model), dbase * ".mps")\n'
           '                        P.O.model.ext[:itlog] = zeros(Int, 3)\n'
           '                    end\n'
           '                    tr = pre_inf ? 0.0 : @elapsed (z = zpad((sticky ? _nz_sticky_solve! : _nz_sep_solve!)(P; maxrounds=(ubp == Inf ? maxrounds : child_rounds), maxadd=sep_maxadd)))\n'
           '                    if !isempty(dbase) && !pre_inf\n'
           '                        il = P.O.model.ext[:itlog]; isempty(nlog) && delete!(P.O.model.ext, :itlog)\n'
           '                        thr0 = LB[] + tol(LB[])\n'
           '                        write(dbase * ".txt", @sprintf("t %.2f wid %d lp_n %d simplex_it %d wall %.4f z %.10g thr %.10g ubp %.10g\\n", time() - t0, wid, il[1], il[2], tr, z, thr0, ubp))\n'
           '                    end\n')
hdr = ('# abb_lpdump_patch.jl — gen_abb_lpdump_patch.py 가 저장소 nonzero_sum/nz_alpha_bnb_dr.jl 에서 만든 진단용 재정의 (손으로 고치지 말 것, 2026-11-27).\n'
       '#  dive 노드 LP 와 시작 basis 저장 (NZ_LPDUMP). docs/nz_ccg_bnp_2026-11-23.md 11.22.\n'
       'println(stderr, "abb_lpdump_patch: nz_alpha_bnb_dr 재정의 (NZ_LPDUMP = ", get(ENV, "NZ_LPDUMP", ""), ")")\n')
open(os.path.join(here, 'abb_lpdump_patch.jl'), 'w', encoding='utf-8', newline='\n').write(hdr + body + '\n')
print('ok')
