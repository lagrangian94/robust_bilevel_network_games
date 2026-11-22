s = open('ccgdw.jl', encoding='utf-8').read()
head = s[:s.index('# ---------------- 가격 MILP ----------------')]
belief = s[s.index('# ---------------- belief 탐색·확보'):s.index('# ---------------- CCG ----------------')]
head = head.replace('# ccgdw.jl — belief 블록 CCG, 마스터를 Dantzig–Wolfe 열 생성 루트 (Farley 하한) 로 푸는 시험판 (2026-11-23, scratchpad)',
                    '# ccgbp.jl — belief 블록 CCG 를 Dantzig–Wolfe branch-and-price 한 나무로 (2026-11-23 밤, scratchpad). ccgdw.jl 에서 갈라짐.')
a = 'const CGTOL = parse(Float64, envf("CGTOL", "1e-3")); const NRAND = parse(Int, envf("NRAND", "40"))'
assert a in head
head = head.replace(a, a + '\nconst SB_K = parse(Int, envf("SB_K", "3")); const PH1 = envf("PH1", "1") == "1"; const ORCL_TL = parse(Float64, envf("ORCL_TL", "900")); const PSTOP = envf("PSTOP", "1") == "1"')
a = '@printf("--- ccgdw |'
assert a in head
head = head.replace(a, '@printf("--- ccgbp v4 | SB_K %d PH1 %s ORCL_TL %.0f PSTOP %s\\n", SB_K, PH1, ORCL_TL, PSTOP); @printf("')
i0 = head.index('"X 위 선형 최소 min c0 + cᵀz'); i1 = head.index('envs = _nz_envs')
head = head[:i0] + head[i1:]
out = head + open('bp_body.jl', encoding='utf-8').read() + belief + open('bp_main.jl', encoding='utf-8').read()
open('ccgbp_v4.jl', 'w', encoding='utf-8').write(out)
print('written', len(out))
