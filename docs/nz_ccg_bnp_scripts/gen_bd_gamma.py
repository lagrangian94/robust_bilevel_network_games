# gen_bd_gamma.py (2026-11-28, 사용자 "감마=3으로 해서 서버에서 돌려볼까" → "시범부터 돌려"): zero-sum Benders 실행 사본을 만든다.
#   true_dro/factor_5/run_nig_coupled_scaling.jl 을 읽어 두 가지만 바꾼다. 저장소 파일은 그대로 둔다.
#   (1) include 경로를 절대 경로 (root) 로 바꾼다 (사본이 다른 폴더에 있어도 되게).
#   (2) make_network 의 γ = 2 고정을 환경 변수 NS_GAMMA (기본 2) 로 바꾼다. w = 0.5 γ median 은 원래 식이라 γ 를 따라간다.
#   (3) 반복 상한 max_iter = 1000 고정 (menu·나머지 변형 두 곳) 을 환경 변수 NS_MAXITER (기본 1000) 로 바꾼다. γ = 3 grid 에서 menu cut 만으로 1,000 반복 (317 s) 을 다 써
#       상한이 Inf 로 끝났다 (2026-11-28). B&P 는 시간 제한만 있으므로 비교할 때는 크게 줘서 TL 이 먼저 걸리게 한다.
#   NS_GAMMA = 2, NS_MAXITER = 1000 이면 저장소 판과 같은 실행이다. min-cut VI 는 저장소 판 (2026-11-28 sink 행 수정, _mincut_sink_row).
#   사용: python gen_bd_gamma.py <root> <출력 경로>
import sys
root, out = sys.argv[1], sys.argv[2]
here = __file__.replace('\\', '/').rsplit('/', 1)[0]
t = open(here + '/../../true_dro/factor_5/run_nig_coupled_scaling.jl', encoding='utf-8').read().replace('\r\n', '\n')
def rep(a, b, cnt=1):
    global t
    assert t.count(a) == cnt, (t.count(a), a)
    t = t.replace(a, b)
for f in ('true_dro_data', 'true_dro_build_omp', 'true_dro_build_subproblem', 'true_dro_build_isp_leader', 'true_dro_build_isp_follower',
          'true_dro_benders', 'true_dro_mincut_vi', 'belief_menu_benders'):
    rep('include("../%s.jl")\n' % f, 'include("%s/true_dro/%s.jl")\n' % (root, f))
rep('include("../../network_generator.jl")\n', 'include("%s/network_generator.jl")\n' % root)
rep('NG = NetworkGenerator\n',
    'NG = NetworkGenerator\n'
    'const NS_GAMMA = parse(Int, get(ENV, "NS_GAMMA", "2"))   # gen_bd_gamma.py: 차단 호 수 상한 γ\n'
    'const NS_MAXITER = parse(Int, get(ENV, "NS_MAXITER", "1000"))   # gen_bd_gamma.py: Benders 반복 상한\n'
    'println("[gen_bd_gamma] γ = $(NS_GAMMA) (w = 0.5 γ median), max_iter = $(NS_MAXITER)")\n')
rep('max_iter=1000', 'max_iter=NS_MAXITER', 2)
rep('        return net, 2, net.interdictable_arcs\n', '        return net, NS_GAMMA, net.interdictable_arcs\n')
rep('    return net, 2, intd_arcs\n', '    return net, NS_GAMMA, intd_arcs\n')
t = '# 이 파일은 docs/nz_ccg_bnp_scripts/gen_bd_gamma.py 가 run_nig_coupled_scaling.jl 에서 만든 실행 사본 (γ = NS_GAMMA). 직접 고치지 말 것.\n' + t
open(out, 'w', encoding='utf-8', newline='\n').write(t)
print('ok', out)
