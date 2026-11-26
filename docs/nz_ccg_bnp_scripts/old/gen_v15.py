# -*- coding: utf-8 -*-
# gen_v15.py — ccgbp_v14.jl 에서 ccgbp_v15.jl 을 만든다 (2026-11-26 시험판, md 11.16).
#  (1) oracle 에 실험용 α-B&B 재정의 (ABB_PATCH, 기본 abb_dd_patch.jl = 적응 선별 DD) 와 추가 인자 (ORCL_KW, 기본 "(dd=true, dd_maxw=25.0)").
#  (2) 1 단계에서 oracle 이 결론 없이 (위반도 인증도 아님) 끝나면 1 단계를 끝내지 않고 같은 점을 이어 풂 — 그 점의 1 단계 누적이 PH1_TOT (900 s) 가 될 때까지.
#      v14 회귀 지리 15: 첫 oracle 300 s 가 결론 없음 → belief 3 개로 1 단계 끝 → 나무 2,600 s, 같은 점 이어 풀기가 뒤에 위반을 찾음 (TimeLimit 0.877%).
import os
here = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(here, 'ccgbp_v14.jl'), encoding='utf-8').read()

def rep(a, b, cnt=1):
    global s
    n = s.count(a)
    if n != cnt:
        raise SystemExit(f'count {n} != {cnt}: {a[:90]!r}')
    s = s.replace(a, b)

head = ('# ccgbp_v15.jl — gen_v15.py 가 v14 에서 만든 파일 (손으로 고치지 말 것). v14 + 적응 선별 DD oracle + 1 단계 결론 없는 oracle 이어 풀기 (2026-11-26 시험판).\n'
        '#  docs/nz_ccg_bnp_2026-11-23.md 11.16. (1) ABB_PATCH (기본 abb_dd_patch.jl), ORCL_KW (기본 "(dd=true, dd_maxw=25.0)").\n'
        '#  (2) 1 단계 oracle 이 위반도 인증도 없이 끝나면 같은 점을 누적 PH1_TOT (900 s) 까지 이어 풂. 인증 (상한 ≤ 목표) 이거나 누적이 차면 1 단계 끝.\n'
        '# (아래는 v14 머리말) ')
s = head + s
rep('include(joinpath(root, "nonzero_sum", "nz_belief_ccg.jl")); include(joinpath(root, "nonzero_sum", "nz_belief_ccg_tree.jl"))',
    'include(joinpath(root, "nonzero_sum", "nz_belief_ccg.jl")); include(joinpath(root, "nonzero_sum", "nz_belief_ccg_tree.jl"))\n'
    'const ABB_PATCH = get(ENV, "ABB_PATCH", joinpath(@__DIR__, "abb_dd_patch.jl")); isempty(ABB_PATCH) || include(ABB_PATCH)   # v15: 실험용 α-B&B 재정의 ("" 이면 저장소 판)')
rep('const HUCAP = envf("HUCAP", "1") == "1"   # v11: oracle 정의역 h_i ≤ b_i(x̄)',
    'const HUCAP = envf("HUCAP", "1") == "1"   # v11: oracle 정의역 h_i ≤ b_i(x̄)\n'
    'const ORCL_KW = eval(Meta.parse(envf("ORCL_KW", isempty(ABB_PATCH) ? "NamedTuple()" : "(dd=true, dd_maxw=25.0)")))   # v15: α-B&B 추가 인자\n'
    'const PH1_TOT = parse(Float64, envf("PH1_TOT", "900"))   # v15: 1 단계에서 결론 없는 점의 누적 oracle 시간 상한')
rep('ROOT_EXACT in ("end", "start", "off") || error("ccgbp v14:', 'ROOT_EXACT in ("end", "start", "off") || error("ccgbp v15:')
rep('ORCL_THETA + TREE_SHARE < 1 || error("ccgbp v14:', 'ORCL_THETA + TREE_SHARE < 1 || error("ccgbp v15:')
rep('@printf("--- ccgbp v14 (v13 + ', '@printf("--- ccgbp v15 (v14 + oracle 패치 %s 인자 %s, 1 단계 결론 없으면 누적 %.0fs 까지 이어 풂) | ccgbp v14 (v13 + ')
rep('        PH1_CAP, ORCL_THETA, TREE_SHARE, ROOT_EXACT,', '        isempty(ABB_PATCH) ? "없음" : basename(ABB_PATCH), string(ORCL_KW), PH1_TOT, PH1_CAP, ORCL_THETA, TREE_SHARE, ROOT_EXACT,')
rep('orLB = Dict{Int,Float64}(); orcap = Dict{Int,Float64}()',
    'orLB = Dict{Int,Float64}(); orcap = Dict{Int,Float64}()\n'
    'orcert = Dict{Int,Bool}(); orph1 = Dict{Int,Float64}()   # v15: 점 → 마지막 호출이 인증 (상한 ≤ 목표) 했는지, 1 단계 누적 oracle 시간')
rep('resume=get(orstate, c, nothing), resp=ORCL_RESP))',
    'resume=get(orstate, c, nothing), resp=ORCL_RESP, ORCL_KW...))\n'
    '    orcert[c] = res[:UB] <= tau')
rep('        oracle_at!(r[5]; cap=PH1_CAP) == :none && break   # v14: 1 단계 oracle 은 모두 PH1_CAP 까지 (상태는 남아 2 단계에서 이어 풂)',
    '        tph = time(); hw = oracle_at!(r[5]; cap=PH1_CAP); orph1[r[5]] = get(orph1, r[5], 0.0) + (time() - tph)   # v14: 1 단계 oracle 은 모두 PH1_CAP 까지\n'
    '        if hw == :none                                # v15: 결론 없음이면 같은 점을 누적 PH1_TOT 까지 이어 풂 (다음 반복의 dwsolve 가 같은 점을 줌)\n'
    '            (get(orcert, r[5], false) || orph1[r[5]] >= PH1_TOT) && break\n'
    '            @printf("  [ph1] 결론 없음 → 같은 점 이어 풂 (누적 %.0fs / %.0fs)\\n", orph1[r[5]], PH1_TOT); flush(stdout)\n'
    '        end')
rep('ub_ok() && error("ccgbp v14:', 'ub_ok() && error("ccgbp v15:')
rep('cinc > 0 || error("ccgbp v14:', 'cinc > 0 || error("ccgbp v15:')
rep('ch === nothing && error("ccgbp v14:', 'ch === nothing && error("ccgbp v15:')
rep('isfinite(bnd) || error("ccgbp v14:', 'isfinite(bnd) || error("ccgbp v15:')
open(os.path.join(here, 'ccgbp_v15.jl'), 'w', encoding='utf-8').write(s)
print('ok')
