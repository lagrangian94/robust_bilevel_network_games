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
head = head.replace(a, '@printf("--- ccgbp v5 | SB_K %d PH1 %s ORCL_TL %.0f PSTOP %s\\n", SB_K, PH1, ORCL_TL, PSTOP); @printf("')
a = 'nd = make_loc_tkp(; g=GEO, S=10, seed=1, eps_hat=RR, eps_tilde=RR, delta=RR / 3, beta=0.4)'
assert a in head
head = head.replace(a, 'const LAMU = parse(Float64, envf("LAMU", "100"))   # oracle 의 λᵁ (loc_tkp 기본 100)\n'
                       'nd = make_loc_tkp(; g=GEO, S=10, seed=1, eps_hat=RR, eps_tilde=RR, delta=RR / 3, beta=0.4, lambdaU=LAMU)\n'
                       '@printf("  [설정] η %.1e, μ %.1e (상한 %.1e), oracle λᵁ %.1e, μη = %.3g\\n", ETA, MU, MU_MAX, LAMU, MU * ETA)')
i0 = head.index('"X 위 선형 최소 min c0 + cᵀz'); i1 = head.index('envs = _nz_envs')
head = head[:i0] + head[i1:]
a = '    if Linf >= v - thr && μstar <= MU_MAX'
assert a in belief
belief = belief.replace(a, '    if MU < MU_MAX && Linf >= v - thr && μstar <= MU_MAX      # μ 가 이미 상한이면 올리지 않고 바로 no-good')
out = head + open('bp_body.jl', encoding='utf-8').read() + belief + open('bp_main.jl', encoding='utf-8').read()
open('ccgbp_v5.jl', 'w', encoding='utf-8').write(out)
print('written', len(out))
