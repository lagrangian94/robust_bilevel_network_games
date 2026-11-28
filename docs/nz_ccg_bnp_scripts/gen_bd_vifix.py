# gen_bd_vifix.py (2026-11-28 아침): true_dro/factor_5/run_nig_coupled_scaling.jl 의 실행 사본을 만든다. 바뀌는 것은 두 가지뿐이다.
#   (1) include 경로를 절대 경로 (root) 로 바꾼다.
#   (2) belief_menu_benders.jl 뒤에 zs_vi_sinkfix.jl 을 include 해서 min-cut VI 의 sink 행을 고친다. 저장소 판은 sink = Ny 마지막 행을 가정하는데 Polska·Abilene·Nobel-US 에서 틀린다.
#   저장소 파일은 그대로 둔다.
#   사용: python gen_bd_vifix.py <root> <zs_vi_sinkfix.jl 경로> <출력 경로>
import sys
root, fix, out = sys.argv[1], sys.argv[2], sys.argv[3]
here = __file__.replace('\\', '/').rsplit('/', 1)[0]
t = open(here + '/../../true_dro/factor_5/run_nig_coupled_scaling.jl', encoding='utf-8').read().replace('\r\n', '\n')
def rep(a, b):
    global t
    assert t.count(a) == 1, a
    t = t.replace(a, b)
for f in ('true_dro_data', 'true_dro_build_omp', 'true_dro_build_subproblem', 'true_dro_build_isp_leader', 'true_dro_build_isp_follower',
          'true_dro_benders', 'true_dro_mincut_vi'):
    rep('include("../%s.jl")\n' % f, 'include("%s/true_dro/%s.jl")\n' % (root, f))
rep('include("../../network_generator.jl")\n', 'include("%s/network_generator.jl")\n' % root)
rep('include("../belief_menu_benders.jl")\n',
    'include("%s/true_dro/belief_menu_benders.jl")\n' % root +
    'include("%s")   # gen_bd_vifix.py: min-cut VI 의 sink 행 수정판 (저장소 판 덮어씀)\n' % fix +
    'println("[gen_bd_vifix] min-cut VI = zs_vi_sinkfix.jl (sink 행을 Nts 에서)")\n')
t = '# 이 파일은 docs/nz_ccg_bnp_scripts/gen_bd_vifix.py 가 run_nig_coupled_scaling.jl 에서 만든 실행 사본 (VI sink 수정). 직접 고치지 말 것.\n' + t
open(out, 'w', encoding='utf-8', newline='\n').write(t)
print('ok', out)
