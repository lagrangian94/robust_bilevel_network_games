# -*- coding: utf-8 -*-
# gen_v16.py — ccgbp_v15.jl 에서 ccgbp_v16.jl 을 만든다 (2026-11-26 시험판, md 11.17).
#  v15 + 1 단계 총량 몫: 1 단계는 경과 PH1_SHARE·TL (기본 0.3, TL 3,600 s 면 1,080 s) 이 넘으면 다음 반복을 시작하지 않는다 (진행 중인 oracle 은 PH1_CAP 까지).
#  v15 DD 끈 판 지리 6: 이어 풀기 뒤 마스터가 다른 점들을 내고 그 위반 oracle 이 300 s 씩 → 1 단계 2,539 s, 나무가 하한을 못 만들어 5.661%.
#  v14 S=10 회귀의 1 단계 174 ~ 1,084 s 라 v14 경로는 거의 그대로 (몫은 폭주 방지용).
import os
here = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(here, 'ccgbp_v15.jl'), encoding='utf-8').read()

def rep(a, b, cnt=1):
    global s
    n = s.count(a)
    if n != cnt:
        raise SystemExit(f'count {n} != {cnt}: {a[:90]!r}')
    s = s.replace(a, b)

head = ('# ccgbp_v16.jl — gen_v16.py 가 v15 에서 만든 파일 (손으로 고치지 말 것). v15 + 1 단계 총량 몫 PH1_SHARE (2026-11-26 시험판).\n'
        '#  docs/nz_ccg_bnp_2026-11-23.md 11.17. 1 단계는 경과 PH1_SHARE·TL (기본 0.3) 이 넘으면 다음 반복을 시작하지 않음.\n'
        '# (아래는 v15 머리말) ')
s = head + s
rep('const PH1_TOT = parse(Float64, envf("PH1_TOT", "900"))   # v15: 1 단계에서 결론 없는 점의 누적 oracle 시간 상한',
    'const PH1_TOT = parse(Float64, envf("PH1_TOT", "900"))   # v15: 1 단계에서 결론 없는 점의 누적 oracle 시간 상한\n'
    'const PH1_SHARE = parse(Float64, envf("PH1_SHARE", "0.3"))   # v16: 1 단계 총량 몫 (경과가 PH1_SHARE·TL 을 넘으면 다음 반복 안 함)')
rep('ORCL_THETA + TREE_SHARE < 1 || error("ccgbp v15:',
    '0 < PH1_SHARE <= 1 || error("ccgbp v16: PH1_SHARE 는 (0, 1]")\n'
    'ORCL_THETA + TREE_SHARE < 1 || error("ccgbp v16:')
rep('ROOT_EXACT in ("end", "start", "off") || error("ccgbp v15:', 'ROOT_EXACT in ("end", "start", "off") || error("ccgbp v16:')
rep('@printf("--- ccgbp v15 (v14 + ', '@printf("--- ccgbp v16 (v15 + 1 단계 총량 몫 %.2f·TL = %.0fs) | ccgbp v15 (v14 + ')
rep('        isempty(ABB_PATCH) ? "없음" : basename(ABB_PATCH),', '        PH1_SHARE, PH1_SHARE * TL, isempty(ABB_PATCH) ? "없음" : basename(ABB_PATCH),')
rep('# ---- 1 단계 (루트, 휴리스틱 열 생성 + 탐색 + oracle): belief 와 UB 모으기 ----\nif PH1\n    while time() - wall < TLB\n        r = dwsolve!(ROOT; exact=false); upd_inc!(r[6])',
    '# ---- 1 단계 (루트, 휴리스틱 열 생성 + 탐색 + oracle): belief 와 UB 모으기 ----\n'
    'ph1_why = "총량 몫"   # v16: 1 단계가 끝난 까닭 (break 가 고쳐 씀)\n'
    'if PH1\n    while time() - wall < min(TLB, PH1_SHARE * TL)   # v16: 1 단계 총량 몫\n        r = dwsolve!(ROOT; exact=false); upd_inc!(r[6])')
rep('            (get(orcert, r[5], false) || orph1[r[5]] >= PH1_TOT) && break',
    '            (get(orcert, r[5], false) || orph1[r[5]] >= PH1_TOT) && (global ph1_why = get(orcert, r[5], false) ? "인증" : "같은 점 누적 PH1_TOT"; break)')
rep('@printf("  [1 단계 끝] %.0fs, belief %d, UB %.2f\\n", t_ph1, count(B -> B.kind == :lp, blks), UB); flush(stdout)',
    '@printf("  [1 단계 끝] %.0fs, belief %d, UB %.2f%s\\n", t_ph1, count(B -> B.kind == :lp, blks), UB,\n'
    '        PH1 ? @sprintf(" (v16: 끝난 까닭 %s, 총량 몫 %.0fs)", ph1_why, PH1_SHARE * TL) : ""); flush(stdout)')
rep('ran_final_cg = false\n',
    '# v16 (2026-11-26 밤 수정): 마지막 노드를 마감 TLB 뒤에 가지치기하면 while 머리의 LB 갱신을 못 거치고 루프가 끝나 나무가 닫혔는데도\n'
    '#   TimeLimit 으로 보고되고 마감 루트 CG 가 쓸모없는 하한을 냄 (N_B 4 지리 12 v17: 3,568 s 에 루트 가지치기, 보고 LB −9,999,422). 루프 머리와 같은 갱신을 한 번 더.\n'
    'global LB = max(LB, min(gapcut(), global_lb()))\n'
    'isfinite(UB) && UB - LB <= TOL * max(1.0, abs(UB)) * (1 + 1e-9) && (global status = :Optimal)\n'
    'ran_final_cg = false\n')
rep('isfinite(bnd) || error("ccgbp v15:', 'isfinite(bnd) || error("ccgbp v16:')
rep('ub_ok() && error("ccgbp v15:', 'ub_ok() && error("ccgbp v16:')
rep('cinc > 0 || error("ccgbp v15:', 'cinc > 0 || error("ccgbp v16:')
rep('ch === nothing && error("ccgbp v15:', 'ch === nothing && error("ccgbp v16:')
assert 'ccgbp v15:' not in s
open(os.path.join(here, 'ccgbp_v16.jl'), 'w', encoding='utf-8').write(s)
print('ok')
